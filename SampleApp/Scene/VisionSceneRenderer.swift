#if os(visionOS)

import CompositorServices
import Metal
import MetalSplatter
import os
import SampleBoxRenderer
import simd
import SplatIO
import SwiftUI

extension LayerRenderer.Clock.Instant.Duration {
    var timeInterval: TimeInterval {
        let nanoseconds = TimeInterval(components.attoseconds / 1_000_000_000)
        return TimeInterval(components.seconds) + (nanoseconds / TimeInterval(NSEC_PER_SEC))
    }
}

/// VisionSceneRenderer manages rendering for visionOS immersive spaces.
/// It's marked @unchecked Sendable because it manages thread safety manually:
/// - LayerRenderer access is confined to the render thread
/// - Model loading uses async/await
/// - State changes are synchronized through the RendererTaskExecutor
final class VisionSceneRenderer: @unchecked Sendable {
    private static let log =
        Logger(subsystem: Bundle.main.bundleIdentifier!,
               category: "VisionSceneRenderer")

    let layerRenderer: LayerRenderer
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    private var model: ModelIdentifier?
    private var modelRenderer: (any ModelRenderer)?
    private var proceduralSplatController: ProceduralSplatController?

    let inFlightSemaphore = DispatchSemaphore(value: Constants.maxSimultaneousRenders)

    private var lastRotationUpdateTimestamp: Date? = nil
    private var rotation: Angle = .zero

    /// One-shot guard so a persistent per-frame render throw surfaces to the on-screen
    /// log exactly once (not every frame). Re-armed at the top of `load()`.
    private var didReportRenderError = false

    let arSession: ARKitSession
    let worldTracking: WorldTrackingProvider

    init(_ layerRenderer: LayerRenderer) {
        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device
        self.commandQueue = self.device.makeCommandQueue()!

        worldTracking = WorldTrackingProvider()
        arSession = ARKitSession()
    }

    /// Static entry point for starting the renderer.
    static func startRendering(_ layerRenderer: LayerRenderer, model: ModelIdentifier?) {
        let renderer = VisionSceneRenderer(layerRenderer)
        Task {
            do {
                try await renderer.load(model)
            } catch {
                log.error("Error loading model: \(error.localizedDescription)")
                await MainActor.run {
                    SplatLoadModel.shared.addLog("Load failed: \(error.localizedDescription)", isError: true)
                    SplatLoadModel.shared.phase = .failed(error.localizedDescription)
                }
            }
            renderer.startRenderLoop()
        }
    }

    func load(_ model: ModelIdentifier?) async throws {
        guard model != self.model else { return }
        self.model = model

        modelRenderer = nil
        proceduralSplatController = nil
        didReportRenderError = false
        switch model {
        case .gaussianSplat(let url):
            let filename = url.lastPathComponent
            await MainActor.run {
                let m = SplatLoadModel.shared
                m.reset()
                m.filename = filename
                m.phase = .loading
                m.addLog("Opening \(filename)…")
            }

            // Quick PLY header parse to get total vertex count (progress denominator).
            let totalCount = Self.readPLYVertexCount(url: url)
            await MainActor.run {
                SplatLoadModel.shared.totalSplatCount = totalCount
                if let n = totalCount {
                    SplatLoadModel.shared.addLog("PLY header: \(SplatLoadModel.formatCount(n)) splats total")
                } else {
                    SplatLoadModel.shared.addLog("Streaming (format has no header count)…")
                }
            }

            let splat = try SplatRenderer(device: device,
                                          colorFormat: layerRenderer.configuration.colorFormat,
                                          depthFormat: layerRenderer.configuration.depthFormat,
                                          sampleCount: 1,
                                          maxViewCount: layerRenderer.properties.viewCount,
                                          maxSimultaneousRenders: Constants.maxSimultaneousRenders)

            // --- Capture-free sort profiler -------------------------------------------------
            // The hypothesis: the GPU radix sort fails to build its pipeline on the actual
            // Vision Pro device, so every frame falls back to a single-threaded CPU sort of
            // all splats, re-run on every head movement. We can't run Metal capture (it
            // crashes the app on this beta), so we surface the truth in the in-headset log:
            // whether the GPU path is live, and the path + ms + count of each sort.
            let gpuOn = splat.gpuSortAvailable
            Self.log.notice("Sort path at load — GPU sort: \(gpuOn ? "ON" : "OFF", privacy: .public)")
            await MainActor.run {
                SplatLoadModel.shared.addLog(
                    "GPU sort: \(gpuOn ? "ON" : "OFF")\(gpuOn ? "" : "  ← falling back to CPU sort every frame!")",
                    isError: !gpuOn)
            }
            // Throttle the per-sort on-screen line on a WALL CLOCK (≤ once/2s), not a count:
            // post-gate, sorts are rare, so a count stride could hide a mid-session GPU→CPU
            // fallback (the smoking gun this profiler exists to catch) until n hit the stride.
            // A path change (gpu↔cpu) ALWAYS logs, regardless of throttle, prefixed isError on
            // .cpu. State lives in one lock because onSortStats fires on a background thread.
            struct SortLogState {
                var count: Int = 0
                var lastPath: SplatSortPath? = nil
                var lastLogged: Date = .distantPast
            }
            let sortLogThrottle: TimeInterval = 2.0
            let sortLogState = OSAllocatedUnfairLock(initialState: SortLogState())
            splat.onSortStats = { path, duration, count in
                let ms = duration * 1000.0
                let decision = sortLogState.withLock { (s: inout SortLogState) -> (n: Int, log: Bool, pathChanged: Bool)? in
                    s.count += 1
                    let pathChanged = s.lastPath != path           // SplatSortPath: String → Equatable
                    let now = Date()
                    // Always log a path change; otherwise throttle by wall clock.
                    let shouldLog = pathChanged || now.timeIntervalSince(s.lastLogged) >= sortLogThrottle
                    let n = s.count
                    if shouldLog {
                        s.lastLogged = now
                        s.lastPath = path
                    } else {
                        // Still record the path so a later change is detected even when throttled.
                        s.lastPath = path
                    }
                    return shouldLog ? (n, true, pathChanged) : nil
                }
                // os_log every sort (cheap, thread-safe); on-screen log per the decision above.
                Self.log.info("sort #\(count, privacy: .public): \(path.rawValue, privacy: .public) \(ms, privacy: .public) ms, \(count, privacy: .public) splats")
                guard let decision else { return }
                let isError = (path == .cpu)
                Task { @MainActor in
                    let prefix = decision.pathChanged ? "sort path → \(path.rawValue.uppercased()) — " : ""
                    SplatLoadModel.shared.addLog(
                        String(format: "%@sort #%d: %@ %.1f ms, %@ splats",
                               prefix, decision.n, path.rawValue.uppercased(), ms, SplatLoadModel.formatCount(count)),
                        isError: isError)
                }
            }
            // --------------------------------------------------------------------------------

            let reader = try AutodetectSceneReader(url)

            // Stream batches, updating count in the UI every 250k splats.
            var allPoints: [SplatPoint] = []
            if let n = totalCount { allPoints.reserveCapacity(n) }
            var lastLoggedMillion = 0

            // Accumulate the bounding box incrementally during streaming so we never make a
            // second pass over (up to) 20M points just to find min/max.
            var bboxMin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
            var bboxMax = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

            for try await batch in try await reader.read() {
                for p in batch {
                    bboxMin = simd_min(bboxMin, p.position)
                    bboxMax = simd_max(bboxMax, p.position)
                }
                allPoints.append(contentsOf: batch)
                let count = allPoints.count
                // Throttle main-thread hops: update every 250k splats.
                if count & 0x3FFFF == 0 {
                    await MainActor.run { SplatLoadModel.shared.splatCount = count }
                }
                let million = count / 1_000_000
                if million > lastLoggedMillion {
                    lastLoggedMillion = million
                    await MainActor.run {
                        SplatLoadModel.shared.splatCount = count
                        SplatLoadModel.shared.addLog("\(SplatLoadModel.formatCount(count)) splats read…")
                    }
                }
            }

            await MainActor.run {
                SplatLoadModel.shared.splatCount = allPoints.count
                SplatLoadModel.shared.phase = .building
                SplatLoadModel.shared.addLog("Building GPU buffers (\(SplatLoadModel.formatCount(allPoints.count)) splats)…")
            }

            // Auto-orient + smart scale from the bbox accumulated during streaming.
            if !allPoints.isEmpty {
                await Self.applyBoundingBox(min: bboxMin, max: bboxMax)
            }

            let chunk = try SplatChunk(device: device, from: allPoints)
            await splat.addChunk(chunk)

            await MainActor.run {
                SplatLoadModel.shared.phase = .ready(splatCount: allPoints.count)
                SplatLoadModel.shared.addLog("✓ Ready — \(SplatLoadModel.formatCount(allPoints.count)) splats loaded")
            }

            modelRenderer = splat
        case .proceduralSplat:
            let controller = try await ProceduralSplatController(
                device: device,
                colorFormat: layerRenderer.configuration.colorFormat,
                depthFormat: layerRenderer.configuration.depthFormat,
                sampleCount: 1,
                maxViewCount: layerRenderer.properties.viewCount,
                maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            proceduralSplatController = controller
            modelRenderer = controller.splatRenderer
        case .sampleBox:
            modelRenderer = try SampleBoxRenderer(device: device,
                                                  colorFormat: layerRenderer.configuration.colorFormat,
                                                  depthFormat: layerRenderer.configuration.depthFormat,
                                                  sampleCount: 1,
                                                  maxViewCount: layerRenderer.properties.viewCount,
                                                  maxSimultaneousRenders: Constants.maxSimultaneousRenders)
        case .none:
            break
        }
    }

    func startRenderLoop() {
        Task(executorPreference: RendererTaskExecutor.shared) {
            // Request ARKit authorization before starting the session.
            // NSWorldSensingUsageDescription must be in Info.plist or this is silently denied.
            let authResult = await self.arSession.requestAuthorization(for: [.worldSensing])
            for (type, status) in authResult where status != .allowed {
                Self.log.error("ARKit authorization denied — type: \(String(describing: type)), status: \(String(describing: status))")
                await MainActor.run {
                    SplatLoadModel.shared.addLog(
                        "ARKit world sensing denied (\(status)). Rendering may be blank.",
                        isError: true
                    )
                }
            }

            do {
                try await self.arSession.run([self.worldTracking])
            } catch {
                Self.log.error("Failed to initialize ARSession: \(error.localizedDescription)")
                await MainActor.run {
                    SplatLoadModel.shared.addLog("ARKit session failed: \(error.localizedDescription)", isError: true)
                }
            }

            self.renderLoop()
        }
    }

    private func viewports(drawable: LayerRenderer.Drawable, deviceAnchor: DeviceAnchor) -> [ModelRendererViewportDescriptor] {
        let dp = SplatDisplayParams.shared.snapshot

        // Scale → user yaw+auto-rotation → user pitch → user roll → world position
        // commonUpCalibration: 180° Z flip corrects the most common 3DGS PLY orientation.
        let scaleMatrix = matrix4x4_scale(dp.scale, dp.scale, dp.scale)
        let yawMatrix   = matrix4x4_rotation(radians: dp.yaw + Float(rotation.radians),
                                             axis: SIMD3<Float>(0, 1, 0))
        let pitchMatrix = matrix4x4_rotation(radians: dp.pitch, axis: SIMD3<Float>(1, 0, 0))
        let rollMatrix  = matrix4x4_rotation(radians: dp.roll,  axis: SIMD3<Float>(0, 0, 1))
        let translationMatrix = matrix4x4_translation(0.0, dp.positionY, dp.positionZ)
        let commonUpCalibration = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1))

        let simdDeviceAnchor = deviceAnchor.originFromAnchorTransform

        return drawable.views.enumerated().map { (index, view) in
            let userViewpointMatrix = (simdDeviceAnchor * view.transform).inverse
            let projectionMatrix = drawable.computeProjection(viewIndex: index)
            let screenSize = SIMD2(x: Int(view.textureMap.viewport.width),
                                   y: Int(view.textureMap.viewport.height))
            return ModelRendererViewportDescriptor(
                viewport: view.textureMap.viewport,
                projectionMatrix: projectionMatrix,
                viewMatrix: userViewpointMatrix * translationMatrix * scaleMatrix * yawMatrix * pitchMatrix * rollMatrix * commonUpCalibration,
                screenSize: screenSize)
        }
    }

    private func updateRotation() {
        guard SplatDisplayParams.shared.snapshot.autoRotate else {
            // Reset so re-enabling doesn't jump from accumulated angle.
            rotation = .zero
            lastRotationUpdateTimestamp = nil
            return
        }
        let now = Date()
        defer { lastRotationUpdateTimestamp = now }
        guard let lastRotationUpdateTimestamp else { return }
        rotation += Constants.rotationPerSecond * now.timeIntervalSince(lastRotationUpdateTimestamp)
    }

    func renderFrame() {
        guard let frame = layerRenderer.queryNextFrame() else { return }

        frame.startUpdate()
        frame.endUpdate()

        guard let timing = frame.predictTiming() else { return }
        LayerRenderer.Clock().wait(until: timing.optimalInputTime)

        let drawables = frame.queryDrawables()
        guard !drawables.isEmpty else { return }

        // If not ready to render, complete the frame lifecycle with empty
        // command buffers to avoid CompositorServices "too many frames in
        // flight" crash, then return early.
        guard let modelRenderer, modelRenderer.isReadyToRender else {
            frame.startSubmission()
            for drawable in drawables {
                guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                    fatalError("Failed to create command buffer")
                }
                drawable.encodePresent(commandBuffer: commandBuffer)
                commandBuffer.commit()
            }
            frame.endSubmission()
            return
        }

        // Use first drawable for timing/anchor calculations
        let primaryDrawable = drawables[0]
        let time = LayerRenderer.Clock.Instant.epoch.duration(to: primaryDrawable.frameTiming.presentationTime).timeInterval
        let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: time)

        // Without a device anchor the compositor cannot reproject and will drop the frame.
        // Submit empty command buffers to keep the frame lifecycle correct while tracking converges.
        guard let deviceAnchor else {
            frame.startSubmission()
            for drawable in drawables {
                if let commandBuffer = commandQueue.makeCommandBuffer() {
                    drawable.encodePresent(commandBuffer: commandBuffer)
                    commandBuffer.commit()
                }
            }
            frame.endSubmission()
            return
        }

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        frame.startSubmission()

        updateRotation()
        proceduralSplatController?.update()

        for (index, drawable) in drawables.enumerated() {
            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                fatalError("Failed to create command buffer")
            }

            drawable.deviceAnchor = deviceAnchor

            // Signal semaphore when the last drawable's command buffer completes
            if index == drawables.count - 1 {
                let semaphore = inFlightSemaphore
                commandBuffer.addCompletedHandler { _ in
                    semaphore.signal()
                }
            }

            let viewports = self.viewports(drawable: drawable, deviceAnchor: deviceAnchor)

            do {
                try modelRenderer.render(viewports: viewports,
                                          colorTexture: drawable.colorTextures[0],
                                          colorStoreAction: .store,
                                          depthTexture: drawable.depthTextures[0],
                                          rasterizationRateMap: drawable.rasterizationRateMaps.first,
                                          renderTargetArrayLength: layerRenderer.configuration.layout == .layered ? drawable.views.count : 1,
                                          to: commandBuffer)
            } catch {
                Self.log.error("Unable to render scene: \(error.localizedDescription)")
                // Surface a persistent render failure to the in-headset log ONCE, so a black
                // void under "✓ Ready" is legible instead of silent. The pipeline guards make
                // this throw recur every frame, hence the one-shot flag (re-armed in load()).
                if !didReportRenderError {
                    didReportRenderError = true
                    Task { @MainActor in
                        SplatLoadModel.shared.addLog("Render error: \(error.localizedDescription) — scene may appear blank", isError: true)
                    }
                }
            }

            drawable.encodePresent(commandBuffer: commandBuffer)
            commandBuffer.commit()
        }

        frame.endSubmission()
    }

    func renderLoop() {
        while true {
            autoreleasepool {
                if layerRenderer.state == .invalidated {
                    Self.log.warning("Layer is invalidated")
                    return
                } else if layerRenderer.state == .paused {
                    layerRenderer.waitUntilRunning()
                    return
                } else {
                    self.renderFrame()
                }
            }
            if layerRenderer.state == .invalidated {
                return
            }
        }
    }
}

// MARK: - Auto-orient + smart scale from bounding box

extension VisionSceneRenderer {
    /// Derives a suggested orientation and a life-size scale from the loaded scan's bounding box,
    /// then pushes them into `SplatDisplayModel` so the first view is correctly oriented and sized.
    ///
    /// Orientation heuristic: the **shortest** bbox dimension is assumed to be the scan's vertical
    /// thickness (a car is long×wide×short-tall; a building is short in one horizontal axis). We
    /// rotate that axis onto world-up (Y) with an axis-aligned ±90° turn — exactly representable in
    /// the existing pitch/roll Euler controls, so no matrix-chain change is needed in `viewports()`.
    ///
    /// Limitations (by design — bbox alone can't recover these):
    ///  - The **sign** of up is ambiguous; the scan may load upside-down and need a 180° flip.
    ///  - The **spin around Y** (which way the front faces) is not recovered; user adjusts yaw.
    ///  - This is a bbox heuristic, not PCA — a diagonally-posed scan won't be perfectly leveled.
    ///
    /// Note: `viewports()` already applies a fixed 180° Z `commonUpCalibration`; the suggested
    /// pitch/roll here compose on top of that.
    static func applyBoundingBox(min lo: SIMD3<Float>, max hi: SIMD3<Float>) async {
        let size = hi - lo
        let extents = [size.x, size.y, size.z]
        let largest = extents.max() ?? 1.0

        // Index of the shortest axis → the candidate "up/thin" axis.
        var shortestAxis = 0
        var shortestVal = size.x
        if size.y < shortestVal { shortestVal = size.y; shortestAxis = 1 }
        if size.z < shortestVal { shortestVal = size.z; shortestAxis = 2 }

        // Rotate the shortest axis onto +Y. Because it's axis-aligned, the correction is a single
        // ±90° turn (or none), so it maps cleanly onto pitch (about X) / roll (about Z).
        var suggestedPitch: Float = 0   // rotation about X — brings Z up to Y
        var suggestedRoll: Float = 0    // rotation about Z — brings X up to Y
        switch shortestAxis {
        case 1: break                                       // already Y-up
        case 2: suggestedPitch = -Float.pi / 2              // Z is up → tilt forward 90°
        default: suggestedRoll = Float.pi / 2               // X is up → roll 90°
        }

        let extentDesc = String(format: "%.2f×%.2f×%.2f", size.x, size.y, size.z)
        await MainActor.run {
            SplatDisplayModel.shared.applyBoundingBox(
                largestExtent: largest,
                suggestedPitch: suggestedPitch,
                suggestedRoll: suggestedRoll)
            let m = SplatLoadModel.shared
            m.addLog("Bbox \(extentDesc) units — auto-oriented (shortest axis → up)")
            let target = SplatDisplayModel.lifeSizeTargetMeters
            m.addLog(String(format: "Life-size: %.1fm across (scale ×%.3f)",
                            target, target / Swift.max(largest, 1e-5)))
        }
    }
}

// MARK: - PLY header parse

extension VisionSceneRenderer {
    /// Reads only the first 8 KB of a PLY file to extract the vertex count from the header.
    /// Returns nil for non-PLY formats or if the header is malformed.
    static func readPLYVertexCount(url: URL) -> Int? {
        guard let stream = InputStream(url: url) else { return nil }
        stream.open()
        defer { stream.close() }
        var buffer = [UInt8](repeating: 0, count: 8192)
        let n = stream.read(&buffer, maxLength: buffer.count)
        guard n > 10, let header = String(bytes: buffer[0..<n], encoding: .ascii) else { return nil }
        for line in header.components(separatedBy: .newlines) {
            let parts = line.split(separator: " ")
            if parts.count == 3, parts[0] == "element", parts[1] == "vertex" {
                return Int(parts[2])
            }
            if line == "end_header" { break }
        }
        return nil
    }
}

// MARK: -

final class RendererTaskExecutor: TaskExecutor {
    static let shared = RendererTaskExecutor()
    private let queue = DispatchQueue(label: "RenderThreadQueue", qos: .userInteractive)

    func enqueue(_ job: UnownedJob) {
        queue.async {
            job.runSynchronously(on: self.asUnownedSerialExecutor())
        }
    }

    nonisolated func asUnownedSerialExecutor() -> UnownedTaskExecutor {
        UnownedTaskExecutor(ordinary: self)
    }
}

#endif // os(visionOS)

