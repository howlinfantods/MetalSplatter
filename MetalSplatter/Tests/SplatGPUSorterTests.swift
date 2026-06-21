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
