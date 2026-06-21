import XCTest
import Metal
@testable import MetalSplatter

final class GPURadixSortTests: XCTestCase {
    var device: MTLDevice!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device required for tests")
    }

    /// Sorts `count` random keys on the GPU and asserts the result matches a CPU stable sort —
    /// both the key order (ascending) and the payload order (stability: ties broken by original index).
    private func runSortCheck(count: Int, seed: UInt64) throws {
        var rng = SplitMix64(seed: seed)
        var keys = (0..<count).map { _ in UInt32(truncatingIfNeeded: rng.next()) }
        var vals = (0..<count).map { UInt32($0) }

        // CPU reference: stable sort of indices by (key asc, originalIndex asc)
        let refIdx = (0..<count).sorted { a, b in keys[a] != keys[b] ? keys[a] < keys[b] : a < b }
        let refKeys = refIdx.map { keys[$0] }
        let refVals = refIdx.map { vals[$0] }

        let sorter = try SplatGPURadixSort(device: device)
        try sorter.sort(keys: &keys, values: &vals)

        XCTAssertEqual(keys, refKeys, "keys not sorted ascending (count=\(count))")
        XCTAssertEqual(vals, refVals, "payload/stability mismatch (count=\(count))")
    }

    func testTiny() throws { try runSortCheck(count: 5, seed: 1) }
    func testSmall() throws { try runSortCheck(count: 1_000, seed: 2) }
    func testCrossesTileBoundary() throws { try runSortCheck(count: 1_024 * 3 + 7, seed: 3) }
    func testManyCollisions() throws {
        // Force lots of equal keys to stress stability across passes.
        var rng = SplitMix64(seed: 7)
        let count = 50_000
        var keys = (0..<count).map { _ in UInt32(rng.next() % 16) } // only 16 distinct keys
        var vals = (0..<count).map { UInt32($0) }
        let refIdx = (0..<count).sorted { a, b in keys[a] != keys[b] ? keys[a] < keys[b] : a < b }
        let refKeys = refIdx.map { keys[$0] }
        let refVals = refIdx.map { vals[$0] }
        let sorter = try SplatGPURadixSort(device: device)
        try sorter.sort(keys: &keys, values: &vals)
        XCTAssertEqual(keys, refKeys, "collision keys not sorted")
        XCTAssertEqual(vals, refVals, "collision stability broken")
    }
    func testLarge() throws { try runSortCheck(count: 500_000, seed: 4) }

    /// The depth→key encoding must turn descending depth into ascending key order.
    func testDepthKeyDescendingOrder() throws {
        let depths: [Float] = [-3, 0.5, 2.5, 100, 1e9, -1e9, 0.25, -7.5]
        let keys = depths.map { SplatGPURadixSort.sortableDepthKey($0, descending: true) }
        let order = (0..<depths.count).sorted { keys[$0] < keys[$1] }
        let sortedDepths = order.map { depths[$0] }
        XCTAssertEqual(sortedDepths, depths.sorted(by: >), "descending depth key ordering wrong")
    }

    /// End-to-end: GPU-sort splat indices by depth and confirm back-to-front order matches the
    /// CPU reference the renderer currently uses (squared distance, larger first).
    func testSortsSplatIndicesByDepth() throws {
        var rng = SplitMix64(seed: 99)
        let count = 100_000
        let camera = SIMD3<Float>(0, 0, 0)
        let positions = (0..<count).map { _ in
            SIMD3<Float>(Float(Int(rng.next() % 2000)) - 1000,
                         Float(Int(rng.next() % 2000)) - 1000,
                         Float(Int(rng.next() % 2000)) - 1000)
        }
        func depthSq(_ p: SIMD3<Float>) -> Float { let d = p - camera; return d.x*d.x + d.y*d.y + d.z*d.z }

        var keys = positions.map { SplatGPURadixSort.sortableDepthKey(depthSq($0), descending: true) }
        var vals = (0..<count).map { UInt32($0) }
        let sorter = try SplatGPURadixSort(device: device)
        try sorter.sort(keys: &keys, values: &vals)

        // Resulting payload order should be by descending depth (back to front).
        var lastDepth = Float.greatestFiniteMagnitude
        for v in vals {
            let d = depthSq(positions[Int(v)])
            XCTAssertLessThanOrEqual(d, lastDepth + 1e-3, "splats not back-to-front")
            lastDepth = d
        }
    }
}

/// Small deterministic RNG so tests are reproducible.
struct SplitMix64 {
    var s: UInt64
    init(seed: UInt64) { s = seed }
    mutating func next() -> UInt64 {
        s &+= 0x9E3779B97F4A7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
