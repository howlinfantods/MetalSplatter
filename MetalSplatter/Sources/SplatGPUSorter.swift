import Metal
import simd

/// Full GPU splat depth-sort: depth-key generation (from chunk positions) → GPU radix sort →
/// finalize to `ChunkedSplatIndex`. This is the GPU equivalent of `SplatSorter.performSort`'s
/// phases 1–3, replacing the single-threaded CPU path (`SplatSorter.swift:600-640`).
///
/// **Status:** unit-tested green on this Mac's GPU against the CPU reference (`SplatGPUSorterTests` —
/// permutation + back-to-front monotonicity across multiple chunks). Swapping it into
/// `SplatSorter.performSort`'s threading/3-buffer-ring orchestration + device validation is the final
/// step. See `docs/OVERHAUL-PLAN.md`.
public final class SplatGPUSorter {
    /// A chunk to sort: its splat buffer (`EncodedSplatPoint`, 32 B/splat), splat count, and chunk index.
    public struct Chunk {
        public let buffer: MTLBuffer
        public let count: Int
        public let chunkIndex: UInt16
        public init(buffer: MTLBuffer, count: Int, chunkIndex: UInt16) {
            self.buffer = buffer; self.count = count; self.chunkIndex = chunkIndex
        }
    }

    public enum Error: Swift.Error { case functionNotFound(String); case allocationFailed }

    private let device: MTLDevice
    private let radix: SplatGPURadixSort
    private let keyPSO: MTLComputePipelineState
    private let gatherPSO: MTLComputePipelineState

    // Keep in sync with RadixSort.metal
    private struct SortKeyParams {
        var count: UInt32; var baseOffset: UInt32; var byDistance: UInt32
        var camX: Float; var camY: Float; var camZ: Float
        var fwdX: Float; var fwdY: Float; var fwdZ: Float
    }
    private struct GatherParams { var count: UInt32; var chunkCount: UInt32 }

    // scratch (grown on demand)
    private var keys: MTLBuffer?
    private var payload: MTLBuffer?
    private var capacity = 0

    public init(device: MTLDevice) throws {
        self.device = device
        self.radix = try SplatGPURadixSort(device: device)
        let library = try SplatGPURadixSort.loadLibrary(device: device)
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else { throw Error.functionNotFound(name) }
            return try device.makeComputePipelineState(function: fn)
        }
        keyPSO = try pso("splatDepthKeys")
        gatherPSO = try pso("gatherChunkedIndices")
    }

    private func ensureScratch(_ count: Int) throws {
        guard capacity < count else { return }
        guard let k = device.makeBuffer(length: max(1, count) * 4, options: .storageModeShared),
              let p = device.makeBuffer(length: max(1, count) * 4, options: .storageModeShared)
        else { throw Error.allocationFailed }
        keys = k; payload = p; capacity = count
    }

    /// Encodes the full sort into `commandBuffer`. On completion `out` holds the splats as
    /// `ChunkedSplatIndex` (`uint2(chunkIndex, splatIndex)`, 8 B each) in back-to-front order.
    /// `out` must hold ≥ total-splat-count entries.
    public func encode(into commandBuffer: MTLCommandBuffer,
                       chunks: [Chunk],
                       cameraPosition: SIMD3<Float>,
                       cameraForward: SIMD3<Float>,
                       byDistance: Bool,
                       out: MTLBuffer) throws {
        let total = chunks.reduce(0) { $0 + $1.count }
        guard total > 0 else { return }
        try ensureScratch(total)
        guard let keys, let payload else { throw Error.allocationFailed }

        // Per-chunk base offsets + chunk-index table (small, host-built).
        var offsets = [UInt32](repeating: 0, count: chunks.count + 1)
        var chunkIds = [UInt32](repeating: 0, count: max(1, chunks.count))
        var running = 0
        for (i, c) in chunks.enumerated() {
            offsets[i] = UInt32(running); chunkIds[i] = UInt32(c.chunkIndex); running += c.count
        }
        offsets[chunks.count] = UInt32(running)
        guard let offsetsBuf = device.makeBuffer(bytes: offsets, length: offsets.count * 4, options: .storageModeShared),
              let chunkIdsBuf = device.makeBuffer(bytes: chunkIds, length: chunkIds.count * 4, options: .storageModeShared)
        else { throw Error.allocationFailed }

        // 1. Depth keys + global-index payload, one dispatch per chunk (all in one serial encoder).
        guard let keyEnc = commandBuffer.makeComputeCommandEncoder() else { throw Error.allocationFailed }
        keyEnc.setComputePipelineState(keyPSO)
        running = 0
        for c in chunks {
            var kp = SortKeyParams(count: UInt32(c.count), baseOffset: UInt32(running),
                                   byDistance: byDistance ? 1 : 0,
                                   camX: cameraPosition.x, camY: cameraPosition.y, camZ: cameraPosition.z,
                                   fwdX: cameraForward.x, fwdY: cameraForward.y, fwdZ: cameraForward.z)
            keyEnc.setBuffer(c.buffer, offset: 0, index: 0)
            keyEnc.setBuffer(keys, offset: 0, index: 1)
            keyEnc.setBuffer(payload, offset: 0, index: 2)
            keyEnc.setBytes(&kp, length: MemoryLayout<SortKeyParams>.stride, index: 3)
            let w = min(keyPSO.maxTotalThreadsPerThreadgroup, 256)
            keyEnc.dispatchThreads(MTLSize(width: c.count, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: max(1, w), height: 1, depth: 1))
            running += c.count
        }
        keyEnc.endEncoding()

        // 2. Radix sort (keys, payload) — sorted result lands back in keys/payload.
        try radix.encode(into: commandBuffer, keys: keys, values: payload, count: total)

        // 3. Finalize: gather sorted global indices into ChunkedSplatIndex.
        guard let gEnc = commandBuffer.makeComputeCommandEncoder() else { throw Error.allocationFailed }
        gEnc.setComputePipelineState(gatherPSO)
        var gp = GatherParams(count: UInt32(total), chunkCount: UInt32(chunks.count))
        gEnc.setBuffer(payload, offset: 0, index: 0)
        gEnc.setBuffer(offsetsBuf, offset: 0, index: 1)
        gEnc.setBuffer(chunkIdsBuf, offset: 0, index: 2)
        gEnc.setBuffer(out, offset: 0, index: 3)
        gEnc.setBytes(&gp, length: MemoryLayout<GatherParams>.stride, index: 4)
        let w = min(gatherPSO.maxTotalThreadsPerThreadgroup, 256)
        gEnc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: max(1, w), height: 1, depth: 1))
        gEnc.endEncoding()
    }

    /// Convenience for tests/tools: run + block, returning sorted (chunkIndex, splatIndex) pairs.
    public func sort(chunks: [Chunk],
                     cameraPosition: SIMD3<Float>,
                     cameraForward: SIMD3<Float>,
                     byDistance: Bool) throws -> [(chunkIndex: UInt16, splatIndex: UInt32)] {
        let total = chunks.reduce(0) { $0 + $1.count }
        guard total > 0,
              let out = device.makeBuffer(length: total * 8, options: .storageModeShared),
              let queue = device.makeCommandQueue(),
              let cmd = queue.makeCommandBuffer()
        else { return [] }
        try encode(into: cmd, chunks: chunks, cameraPosition: cameraPosition,
                   cameraForward: cameraForward, byDistance: byDistance, out: out)
        cmd.commit(); cmd.waitUntilCompleted()
        let p = out.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: total)
        return (0..<total).map { (UInt16(truncatingIfNeeded: p[$0].x), p[$0].y) }
    }
}
