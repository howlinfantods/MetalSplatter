import XCTest
import Metal
import simd
@testable import MetalSplatter

/// Verifies the full GPU sort (depth-key gen → radix → finalize to ChunkedSplatIndex) across multiple
/// chunks, against the properties the renderer relies on: it's a permutation of every splat, and the
/// splats come out back-to-front (depth non-increasing) — matching the CPU `sort { depth > depth }`.
final class SplatGPUSorterTests: XCTestCase {
    var device: MTLDevice!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device required for tests")
    }

    private func makeChunkBuffer(positions: [SIMD3<Float>]) -> MTLBuffer {
        var splats = positions.map {
            EncodedSplatPoint(position: $0,
                              colorSH0: SIMD3<Float>(1, 1, 1),
                              opacity: 1.0,
                              scale: SIMD3<Float>(0.1, 0.1, 0.1),
                              rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1))
        }
        return device.makeBuffer(bytes: &splats,
                                 length: splats.count * MemoryLayout<EncodedSplatPoint>.stride,
                                 options: .storageModeShared)!
    }

    func testEncodedSplatPointStrideIs32() {
        // The key-gen kernel assumes 8 floats/splat. Guard that invariant.
        XCTAssertEqual(MemoryLayout<EncodedSplatPoint>.stride, 32)
    }

    func testMultiChunkBackToFront() throws {
        var rng = SplitMix64(seed: 11)
        func randPos() -> SIMD3<Float> {
            SIMD3(Float(Int(rng.next() % 4000)) - 2000,
                  Float(Int(rng.next() % 4000)) - 2000,
                  Float(Int(rng.next() % 4000)) - 2000)
        }
        let camera = SIMD3<Float>(5, -3, 10)
        let chunkSizes = [1000, 2500, 777]
        var chunks: [SplatGPUSorter.Chunk] = []
        var positionsByChunk: [[SIMD3<Float>]] = []
        for (ci, n) in chunkSizes.enumerated() {
            let ps = (0..<n).map { _ in randPos() }
            positionsByChunk.append(ps)
            chunks.append(.init(buffer: makeChunkBuffer(positions: ps), count: n, chunkIndex: UInt16(ci)))
        }

        let sorter = try SplatGPUSorter(device: device)
        let result = try sorter.sort(chunks: chunks, cameraPosition: camera,
                                     cameraForward: SIMD3(0, 0, -1), byDistance: true)

        let total = chunkSizes.reduce(0, +)
        XCTAssertEqual(result.count, total)

        // (1) permutation: every (chunkIndex, splatIndex) appears exactly once and is in range.
        var seen = Set<UInt64>()
        for (ci, si) in result {
            XCTAssertLessThan(Int(ci), chunkSizes.count)
            XCTAssertLessThan(Int(si), chunkSizes[Int(ci)])
            seen.insert((UInt64(ci) << 32) | UInt64(si))
        }
        XCTAssertEqual(seen.count, total, "output is not a permutation of all splats")

        // (2) back-to-front: squared distance non-increasing.
        func depthSq(_ ci: UInt16, _ si: UInt32) -> Float {
            let p = positionsByChunk[Int(ci)][Int(si)] - camera
            return p.x*p.x + p.y*p.y + p.z*p.z
        }
        var last = Float.greatestFiniteMagnitude
        for (ci, si) in result {
            let d = depthSq(ci, si)
            XCTAssertLessThanOrEqual(d, last + 1e-1, "splats not back-to-front by distance")
            last = d
        }
    }

    /// Large-N exercise of the dynamic-key-width path (optimization #2). At 2M splats
    /// dynamicNumBits → 20 → 5 passes (odd, so the ping-pong parity blit is on the hot path).
    /// We assert the QUANTIZED key is non-decreasing — the exact invariant the narrowed key
    /// guarantees (intra-bucket depth spread is unbounded by design, so a raw-depth tolerance
    /// would be unreliable; quantized-key monotonicity is correct and tight).
    func testDynamicWidthLargeNQuantizedMonotonic() throws {
        var rng = SplitMix64(seed: 4242)
        let n = 2_000_000
        let numBits = SplatGPUSorter.dynamicNumBits(forCount: n)
        XCTAssertEqual(numBits, 19)                     // round(log2(500_000)) = 19
        let passes = SplatGPURadixSort.passCount(forBits: numBits)
        XCTAssertEqual(passes, 5)                       // ceil(19/4)=5 → odd → blit path
        let shift = UInt32(32 - passes * SplatGPURadixSort.radixBits)

        let camera = SIMD3<Float>(1, 2, 3)
        let ps = (0..<n).map { _ in
            SIMD3<Float>(Float(Int(rng.next() % 8000)) - 4000,
                         Float(Int(rng.next() % 8000)) - 4000,
                         Float(Int(rng.next() % 8000)) - 4000)
        }
        let chunk = SplatGPUSorter.Chunk(buffer: makeChunkBuffer(positions: ps), count: n, chunkIndex: 0)
        let sorter = try SplatGPUSorter(device: device)
        // Explicitly opt into the dynamic (quantized) path — otherwise this runs the exact
        // 32-bit default and proves nothing about optimization #2.
        let result = try sorter.sort(chunks: [chunk], cameraPosition: camera,
                                     cameraForward: SIMD3(0, 0, -1), byDistance: true,
                                     numBits: SplatGPUSorter.dynamicNumBits(forCount: n))
        XCTAssertEqual(result.count, n)

        // Reconstruct the same quantized key the kernel produced and assert non-decreasing.
        func quantizedKey(_ si: UInt32) -> UInt32 {
            let p = ps[Int(si)] - camera
            let depth = p.x*p.x + p.y*p.y + p.z*p.z
            return SplatGPURadixSort.sortableDepthKey(depth, descending: true) >> shift
        }
        var lastKey: UInt32 = 0
        var permutation = Set<UInt32>()
        for (_, si) in result {
            let k = quantizedKey(si)
            XCTAssertGreaterThanOrEqual(k, lastKey, "quantized depth key not monotonic — dynamic-width sort wrong")
            lastKey = k
            permutation.insert(si)
        }
        XCTAssertEqual(permutation.count, n, "output not a permutation of all splats")
    }

    func testDotProductOrdering() throws {
        var rng = SplitMix64(seed: 22)
        let n = 5000
        let ps = (0..<n).map { _ in
            SIMD3<Float>(Float(Int(rng.next() % 2000)) - 1000,
                         Float(Int(rng.next() % 2000)) - 1000,
                         Float(Int(rng.next() % 2000)) - 1000)
        }
        let forward = simd_normalize(SIMD3<Float>(0.2, -0.1, -1))
        let chunk = SplatGPUSorter.Chunk(buffer: makeChunkBuffer(positions: ps), count: n, chunkIndex: 0)
        let sorter = try SplatGPUSorter(device: device)
        let result = try sorter.sort(chunks: [chunk], cameraPosition: .zero,
                                     cameraForward: forward, byDistance: false)
        XCTAssertEqual(result.count, n)
        var last = Float.greatestFiniteMagnitude
        for (_, si) in result {
            let d = simd_dot(ps[Int(si)], forward)
            XCTAssertLessThanOrEqual(d, last + 1e-2, "splats not sorted by descending dot(position, forward)")
            last = d
        }
    }
}
