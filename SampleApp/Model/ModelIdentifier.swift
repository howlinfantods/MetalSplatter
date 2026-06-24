import Foundation
import Observation

// MARK: - Load state shared between the window panel and the immersive renderer

@Observable @MainActor
final class SplatLoadModel {
    static let shared = SplatLoadModel()
    private init() {}

    enum Phase {
        case idle, loading, building
        case ready(splatCount: Int)
        case failed(String)

        var label: String {
            switch self {
            case .idle: return ""
            case .loading: return "Streaming splat data..."
            case .building: return "Uploading to GPU..."
            case .ready(let n): return "\(SplatLoadModel.formatCount(n)) splats loaded"
            case .failed(let m): return m
            }
        }

        var isActive: Bool { if case .idle = self { return false }; return true }
        var isBusy: Bool { switch self { case .loading, .building: return true; default: return false } }
        var isError: Bool { if case .failed = self { return true }; return false }
    }

    struct LogEntry: Identifiable {
        let id = UUID()
        let message: String
        let isError: Bool
    }

    var phase: Phase = .idle
    var filename: String = ""
    var splatCount: Int = 0
    var totalSplatCount: Int? = nil
    var log: [LogEntry] = []

    func addLog(_ message: String, isError: Bool = false) {
        log.append(LogEntry(message: message, isError: isError))
        if log.count > 50 { log.removeFirst() }
    }

    func reset() {
        phase = .idle
        filename = ""
        splatCount = 0
        totalSplatCount = nil
        log = []
    }

    nonisolated static func formatCount(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0fK", Double(n) / 1_000) }
        return "\(n)"
    }
}

// MARK: - Display parameters — written from main actor, read lock-free from render thread

import os

/// Display parameters shared between the main-actor UI and the render thread.
/// Written only from the main actor; read every frame by the render thread.
/// @unchecked Sendable: Float/Bool writes are 4-byte aligned and atomic on ARM.
final class SplatDisplayParams: @unchecked Sendable {
    struct Snapshot {
        var scale: Float = 1.0
        var yaw: Float = 0
        var pitch: Float = 0
        var roll: Float = 0
        var positionZ: Float = -3.0
        var positionY: Float = 0.0
        var autoRotate: Bool = false
    }

    static let shared = SplatDisplayParams()
    private init() {}

    private(set) var snapshot = Snapshot()

    func update(_ f: (inout Snapshot) -> Void) { f(&snapshot) }
}

/// Main-actor observable model that drives ContentView controls and syncs to SplatDisplayParams.
@Observable @MainActor
final class SplatDisplayModel {
    static let shared = SplatDisplayModel()
    private init() {}

    enum Preset { case tabletop, lifeSize, roomScale }

    var scale: Float = 1.0      { didSet { sync() } }
    var yaw: Float = 0          { didSet { sync() } }
    var pitch: Float = 0        { didSet { sync() } }
    var roll: Float = 0         { didSet { sync() } }
    var positionZ: Float = -3.0 { didSet { sync() } }
    var positionY: Float = 0.0  { didSet { sync() } }
    var autoRotate: Bool = false { didSet { sync() } }

    // MARK: - Bounding-box-derived defaults (set once per load)
    //
    // A 3DGS PLY from photogrammetry/SfM is in *arbitrary* units — there is no recoverable
    // real-world scale. So the "Life Size" target below is a heuristic: it normalizes the
    // scan's largest bbox extent to a target physical size, it is NOT a true measurement.

    /// Largest bbox extent (in PLY units) of the loaded scan. 0 until a scan loads.
    /// `scale = targetMeters / largestExtent` makes the scan render at `targetMeters` across.
    private(set) var bboxLargestExtent: Float = 0

    /// Suggested orientation (radians) computed from the bbox at load time. The user can still
    /// override via the sliders; `resetOrientation()` restores *these* values, not zero.
    private(set) var suggestedPitch: Float = 0
    private(set) var suggestedRoll: Float = 0

    /// "Life Size" target: render the scan's largest dimension this many meters across.
    /// 4.5 m ≈ a real car's length, a sensible default for the S2000 hero scene.
    static let lifeSizeTargetMeters: Float = 4.5

    private func sync() {
        SplatDisplayParams.shared.update {
            $0.scale = scale; $0.yaw = yaw; $0.pitch = pitch; $0.roll = roll
            $0.positionZ = positionZ; $0.positionY = positionY; $0.autoRotate = autoRotate
        }
    }

    /// Called once after a scan's points are loaded and its bbox computed (see
    /// `VisionSceneRenderer.computeAndApplyBoundingBox`). Stores bbox-derived defaults and
    /// snaps the live controls to a correctly-oriented, life-size-ish first view.
    func applyBoundingBox(largestExtent: Float, suggestedPitch: Float, suggestedRoll: Float) {
        self.bboxLargestExtent = largestExtent
        self.suggestedPitch = suggestedPitch
        self.suggestedRoll = suggestedRoll
        pitch = suggestedPitch
        roll = suggestedRoll
        applyPreset(.lifeSize)
    }

    /// Scale that makes the scan's largest extent render at `meters` across.
    /// Falls back to a passthrough scale when no bbox is known yet.
    private func scale(forMeters meters: Float) -> Float {
        guard bboxLargestExtent > 1e-5 else { return 1.0 }
        return meters / bboxLargestExtent
    }

    func applyPreset(_ preset: Preset) {
        // Each preset is now derived from the bbox so a 0.3-unit scan and a 300-unit scan both
        // land at a sensible physical size, instead of the old hardcoded 0.15 / 1.0 / 8.0.
        switch preset {
        case .tabletop:
            // ~0.6 m across, floating just below eye line at arm's reach.
            scale = scale(forMeters: 0.6); positionZ = -1.2; positionY = -0.5; autoRotate = false
        case .lifeSize:
            scale = scale(forMeters: Self.lifeSizeTargetMeters); positionZ = -3.0; positionY = 0.0; autoRotate = false
        case .roomScale:
            // ~10 m across — fills the room; step the user back accordingly.
            scale = scale(forMeters: 10.0); positionZ = -8.0; positionY = 0.0; autoRotate = false
        }
    }

    /// Restores the bbox-suggested orientation (not zero) so a reset re-applies the auto-orient
    /// rather than wiping it. Yaw has no meaningful bbox-derived value, so it resets to 0.
    func resetOrientation() { yaw = 0; pitch = suggestedPitch; roll = suggestedRoll }

    func reset() {
        scale = 1.0; yaw = 0
        pitch = suggestedPitch; roll = suggestedRoll
        positionZ = -3.0; positionY = 0.0; autoRotate = false
    }
}

// MARK: -

enum ModelIdentifier: Equatable, Hashable, Codable, CustomStringConvertible {
    case gaussianSplat(URL)
    case proceduralSplat
    case sampleBox

    var description: String {
        switch self {
        case .gaussianSplat(let url):
            "Gaussian Splat: \(url.path)"
        case .proceduralSplat:
            "Procedural Splat"
        case .sampleBox:
            "Sample Box"
        }
    }
}
