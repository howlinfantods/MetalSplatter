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

    // MARK: - Dynamic key width (optimization #2)

    /// passCount = ceil(numBits/4), clamped so the radix always runs 1..8 4-bit passes.
    func testPassCountMath() {
        XCTAssertEqual(SplatGPURadixSort.passCount(forBits: 32), 8)
        XCTAssertEqual(SplatGPURadixSort.passCount(forBits: 20), 5)   // 16M scenes
        XCTAssertEqual(SplatGPURadixSort.passCount(forBits: 18), 5)   // ceil(18/4)=5
        XCTAssertEqual(SplatGPURadixSort.passCount(forBits: 16), 4)
        XCTAssertEqual(SplatGPURadixSort.passCount(forBits: 10), 3)   // ceil(10/4)=3 → 12-bit keys
        XCTAssertEqual(SplatGPURadixSort.passCount(forBits: 4), 1)
    }

    /// dynamicNumBits = clamp(round(log2(N/4)), 10, 20). For our 1M–16M target this is 18–20.
    func testDynamicNumBitsMath() {
        XCTAssertEqual(SplatGPUSorter.dynamicNumBits(forCount: 0), 10)        // clamp low
        XCTAssertEqual(SplatGPUSorter.dynamicNumBits(forCount: 4_000), 10)    // round(log2(1000))≈10
        XCTAssertEqual(SplatGPUSorter.dynamicNumBits(forCount: 1_000_000), 18) // round(log2(250000))=18
        XCTAssertEqual(SplatGPUSorter.dynamicNumBits(forCount: 3_000_000), 20) // round(log2(750000))≈20
        XCTAssertEqual(SplatGPUSorter.dynamicNumBits(forCount: 16_000_000), 20) // clamp high
        XCTAssertEqual(SplatGPUSorter.dynamicNumBits(forCount: 64_000_000), 20) // clamp high
    }

    /// A dynamic (odd-pass) key width must still sort correctly AND land the result back in the
    /// caller's keys/values buffers (ping-pong parity blit). Keys are pre-quantized to `numBits`
    /// here (top bits kept), and we assert exact stable order on those quantized keys.
    func testDynamicWidthSortsAndRestoresBuffers() throws {
        for numBits in [10, 13, 16, 18, 20] {            // 3,4,4,5,5 passes — mix of odd & even
            var rng = SplitMix64(seed: UInt64(numBits) &+ 100)
            let count = 40_000
            let shift = UInt32(32 - SplatGPURadixSort.passCount(forBits: numBits) * SplatGPURadixSort.radixBits)
            var keys = (0..<count).map { _ in UInt32(truncatingIfNeeded: rng.next()) >> shift }
            var vals = (0..<count).map { UInt32($0) }

            let refIdx = (0..<count).sorted { a, b in keys[a] != keys[b] ? keys[a] < keys[b] : a < b }
            let refKeys = refIdx.map { keys[$0] }
            let refVals = refIdx.map { vals[$0] }

            let sorter = try SplatGPURadixSort(device: device)
            // Pass the FULL numBits range so all kept bits participate; the keys above are already
            // narrowed to (32-shift) ≥ numBits bits, so this exercises the exact runtime path.
            try sorter.sort(keys: &keys, values: &vals, numBits: 32 - Int(shift))

            XCTAssertEqual(keys, refKeys, "numBits=\(numBits): quantized keys not sorted (parity/blit?)")
            XCTAssertEqual(vals, refVals, "numBits=\(numBits): stability/parity broken")
        }
    }

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
