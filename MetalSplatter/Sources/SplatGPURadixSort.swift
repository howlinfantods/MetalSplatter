import Metal
import Foundation

/// GPU 4-bit LSD radix sort over `(key: UInt32, value: UInt32)` pairs, ascending by key, stable.
///
/// This is the foundation of the depth-sort overhaul (see `docs/OVERHAUL-PLAN.md`): the current
/// renderer sorts splats with a single-threaded Swift `Array.sort` on the CPU
/// (`SplatSorter.performSort`, ~line 628), re-run every time the camera moves — the dominant
/// bottleneck and the cause of the "clunky" stale-sort popping at 10–20M splats. Moving the sort to
/// the GPU is the order-of-magnitude fix.
///
/// **Status:** the sort kernel itself is unit-tested green against a CPU reference on this Mac
/// (`GPURadixSortTests`). Wiring it into `SplatSorter.performSort`'s per-frame loop (key generation
/// on the GPU, the 3-buffer ring, generation/exclusive-access orchestration) is the device-validated
/// next step.
///
/// **Perf-upgrade targets** (do not change results, only speed; tracked in the overhaul plan):
///  - `radixScanExclusive` runs on a single GPU thread. Fine for correctness and small/mid histograms;
///    for 20M splats replace with a blocked parallel scan.
///  - histogram/scatter use one thread per tile (serial within a tile). For max throughput switch to
///    threadgroup-cooperative per-tile scans (DeviceRadixSort style). Still vastly faster than the CPU.
public final class SplatGPURadixSort {
    public static let radixBits = 4
    /// 32-bit keys / 4 bits per digit.
    public static let passes = 8

    public enum Error: Swift.Error {
        case functionNotFound(String)
        case bufferAllocationFailed
    }

    private let device: MTLDevice
    private let histogramPSO: MTLComputePipelineState
    private let scanPSO: MTLComputePipelineState
    private let scatterPSO: MTLComputePipelineState

    /// Elements per tile. Larger = fewer/longer tiles (smaller histogram, less parallelism).
    public var tileSize: Int = 1024

    // Ping-pong + histogram scratch, grown on demand.
    private var scratchKeys: MTLBuffer?
    private var scratchVals: MTLBuffer?
    private var histBuffer: MTLBuffer?
    private var scratchCapacity = 0
    private var histCapacity = 0

    // Keep in sync with RadixSort.metal : RadixParams
    private struct RadixParams { var count: UInt32; var tileSize: UInt32; var tileCount: UInt32; var shift: UInt32 }

    public init(device: MTLDevice) throws {
        self.device = device
        let library = try Self.loadLibrary(device: device)
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else { throw Error.functionNotFound(name) }
            return try device.makeComputePipelineState(function: fn)
        }
        histogramPSO = try pso("radixHistogram")
        scanPSO      = try pso("radixScanExclusive")
        scatterPSO   = try pso("radixScatter")
    }

    /// Loads the radix-sort kernels in a way that works both on-device (Xcode compiles the `.metal`
    /// files in Resources into a `default.metallib`) and under SwiftPM `swift test` (which copies the
    /// `.metal` as source but builds no metallib — so we compile RadixSort.metal at runtime).
    static func loadLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) { return lib }
        if let url = Bundle.module.url(forResource: "RadixSort", withExtension: "metal"),
           let source = try? String(contentsOf: url, encoding: .utf8) {
            return try device.makeLibrary(source: source, options: nil)
        }
        throw Error.functionNotFound("RadixSort metal library")
    }

    private func ensureScratch(count: Int, tileCount: Int) throws {
        if scratchCapacity < count {
            guard let k = device.makeBuffer(length: max(1, count) * MemoryLayout<UInt32>.stride, options: .storageModeShared),
                  let v = device.makeBuffer(length: max(1, count) * MemoryLayout<UInt32>.stride, options: .storageModeShared)
            else { throw Error.bufferAllocationFailed }
            scratchKeys = k; scratchVals = v; scratchCapacity = count
        }
        let histEntries = tileCount * 16
        if histCapacity < histEntries {
            guard let h = device.makeBuffer(length: max(1, histEntries) * MemoryLayout<UInt32>.stride, options: .storageModeShared)
            else { throw Error.bufferAllocationFailed }
            histBuffer = h; histCapacity = histEntries
        }
    }

    /// Encodes the full 8-pass sort into `commandBuffer`. On completion `keys`/`values` hold the result
    /// (ascending by key); `keys` and `values` must each hold ≥ `count` UInt32 elements.
    ///
    /// Eight passes is even, so after ping-ponging through the internal scratch the data lands back in
    /// the caller's `keys`/`values` buffers. Dispatches use the default serial dispatch type, so each
    /// kernel sees the previous one's writes without explicit barriers.
    public func encode(into commandBuffer: MTLCommandBuffer, keys: MTLBuffer, values: MTLBuffer, count: Int) throws {
        guard count > 1 else { return }
        let tileCount = (count + tileSize - 1) / tileSize
        try ensureScratch(count: count, tileCount: tileCount)
        guard let scratchKeys, let scratchVals, let histBuffer else { throw Error.bufferAllocationFailed }

        var srcK = keys, srcV = values
        var dstK = scratchKeys, dstV = scratchVals

        for pass in 0..<Self.passes {
            var params = RadixParams(count: UInt32(count),
                                     tileSize: UInt32(tileSize),
                                     tileCount: UInt32(tileCount),
                                     shift: UInt32(pass * Self.radixBits))
            guard let enc = commandBuffer.makeComputeCommandEncoder() else { throw Error.bufferAllocationFailed }

            // 1. histogram
            enc.setComputePipelineState(histogramPSO)
            enc.setBuffer(srcK, offset: 0, index: 0)
            enc.setBuffer(histBuffer, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<RadixParams>.stride, index: 2)
            dispatch(enc, pso: histogramPSO, threads: tileCount)

            // 2. exclusive scan (single thread)
            enc.setComputePipelineState(scanPSO)
            enc.setBuffer(histBuffer, offset: 0, index: 0)
            enc.setBytes(&params, length: MemoryLayout<RadixParams>.stride, index: 1)
            enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

            // 3. stable scatter
            enc.setComputePipelineState(scatterPSO)
            enc.setBuffer(srcK, offset: 0, index: 0)
            enc.setBuffer(srcV, offset: 0, index: 1)
            enc.setBuffer(dstK, offset: 0, index: 2)
            enc.setBuffer(dstV, offset: 0, index: 3)
            enc.setBuffer(histBuffer, offset: 0, index: 4)
            enc.setBytes(&params, length: MemoryLayout<RadixParams>.stride, index: 5)
            dispatch(enc, pso: scatterPSO, threads: tileCount)

            enc.endEncoding()
            swap(&srcK, &dstK); swap(&srcV, &dstV)
        }
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder, pso: MTLComputePipelineState, threads: Int) {
        let w = min(pso.maxTotalThreadsPerThreadgroup, 64)
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: max(1, w), height: 1, depth: 1))
    }

    /// Convenience: sort host arrays in place (creates buffers, runs, blocks). Used by tests and tools.
    public func sort(keys: inout [UInt32], values: inout [UInt32]) throws {
        let count = keys.count
        guard count > 1, values.count == count else { return }
        guard let kb = device.makeBuffer(bytes: keys, length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let vb = device.makeBuffer(bytes: values, length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let queue = device.makeCommandQueue(),
              let cmd = queue.makeCommandBuffer()
        else { throw Error.bufferAllocationFailed }
        try encode(into: cmd, keys: kb, values: vb, count: count)
        cmd.commit()
        cmd.waitUntilCompleted()
        let kp = kb.contents().bindMemory(to: UInt32.self, capacity: count)
        let vp = vb.contents().bindMemory(to: UInt32.self, capacity: count)
        for i in 0..<count { keys[i] = kp[i]; values[i] = vp[i] }
    }

    /// Maps a float depth to a monotonic UInt32 sort key.
    ///
    /// IEEE-754 floats don't sort correctly as raw bit patterns (sign + negatives reverse), so we flip
    /// the sign bit for positives and all bits for negatives. With `descending: true` (the default,
    /// matching the renderer's back-to-front order — larger depth drawn first) the key is additionally
    /// inverted so an ascending radix sort yields descending depth.
    public static func sortableDepthKey(_ depth: Float, descending: Bool = true) -> UInt32 {
        let bits = depth.bitPattern
        let mask: UInt32 = (bits & 0x8000_0000) != 0 ? 0xFFFF_FFFF : 0x8000_0000
        let ascending = bits ^ mask
        return descending ? ~ascending : ascending
    }
}
