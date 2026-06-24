#include <metal_stdlib>
using namespace metal;

// GPU 4-bit LSD radix sort over (key: uint32, value: uint32) pairs, ascending by key.
//
// Correctness-first tile decomposition, run once per 4-bit digit (8 passes for 32-bit keys):
//   1. radixHistogram      — per-tile 16-bin histogram of the current digit
//   2. radixScanExclusive  — exclusive prefix sum over the bucket-major histogram → global bases
//   3. radixScatter        — stable scatter of each tile's elements to their global positions
//
// The decomposition is the portable DeviceRadixSort approach (NOT OneSweep — Apple Silicon does not
// expose the forward-progress guarantees OneSweep's decoupled-lookback scan needs). See
// docs/OVERHAUL-PLAN.md. This version favours obvious correctness (one thread per tile, single-thread
// scan) so it can be unit-tested against a CPU reference; the perf upgrades (threadgroup-cooperative
// per-tile scans, blocked global scan) are noted there and do not change the algorithm or its results.

struct RadixParams {
    uint count;      // number of elements to sort
    uint tileSize;   // elements per tile
    uint tileCount;  // ceil(count / tileSize)
    uint shift;      // digit shift: 0, 4, 8, ... 28
};

constant uint kRadixBuckets = 16u; // 1 << 4

// One thread per tile: build a 16-bin histogram of this tile's current digit.
// Output layout is BUCKET-MAJOR: globalHist[bucket * tileCount + tile], so a single
// exclusive scan over the whole array yields correct global base offsets per (bucket, tile).
kernel void radixHistogram(device const uint  *keysIn     [[buffer(0)]],
                           device uint        *globalHist [[buffer(1)]],
                           constant RadixParams &p        [[buffer(2)]],
                           uint tid [[thread_position_in_grid]])
{
    if (tid >= p.tileCount) { return; }
    uint start = tid * p.tileSize;
    uint end   = min(start + p.tileSize, p.count);

    uint hist[kRadixBuckets];
    for (uint b = 0; b < kRadixBuckets; ++b) { hist[b] = 0u; }
    for (uint i = start; i < end; ++i) {
        uint bucket = (keysIn[i] >> p.shift) & (kRadixBuckets - 1u);
        hist[bucket] += 1u;
    }
    for (uint b = 0; b < kRadixBuckets; ++b) {
        globalHist[b * p.tileCount + tid] = hist[b];
    }
}

// Single-thread exclusive prefix sum over globalHist[0 ..< tileCount*16].
// Bucket-major ordering means the running sum gives each (bucket, tile) its global start index.
kernel void radixScanExclusive(device uint        *globalHist [[buffer(0)]],
                               constant RadixParams &p        [[buffer(1)]],
                               uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) { return; }
    uint total = p.tileCount * kRadixBuckets;
    uint running = 0u;
    for (uint i = 0; i < total; ++i) {
        uint v = globalHist[i];
        globalHist[i] = running;
        running += v;
    }
}

// One thread per tile: walk this tile's elements in order, writing each to its global slot.
// Processing in index order with per-bucket incrementing offsets makes each pass STABLE,
// which is what makes multi-pass LSD radix a correct full sort.
kernel void radixScatter(device const uint  *keysIn     [[buffer(0)]],
                         device const uint  *valsIn     [[buffer(1)]],
                         device uint        *keysOut    [[buffer(2)]],
                         device uint        *valsOut    [[buffer(3)]],
                         device const uint  *globalHist [[buffer(4)]],
                         constant RadixParams &p        [[buffer(5)]],
                         uint tid [[thread_position_in_grid]])
{
    if (tid >= p.tileCount) { return; }
    uint offset[kRadixBuckets];
    for (uint b = 0; b < kRadixBuckets; ++b) {
        offset[b] = globalHist[b * p.tileCount + tid];
    }
    uint start = tid * p.tileSize;
    uint end   = min(start + p.tileSize, p.count);
    for (uint i = start; i < end; ++i) {
        uint k = keysIn[i];
        uint bucket = (k >> p.shift) & (kRadixBuckets - 1u);
        uint pos = offset[bucket];
        offset[bucket] = pos + 1u;
        keysOut[pos] = k;
        valsOut[pos] = valsIn[i];
    }
}

// ============================================================================
// Full splat sort: depth-key generation + finalize. The radix kernels above do the middle.
// Keeps these in the same file so the SwiftPM `swift test` fallback (which compiles RadixSort.metal
// as a single source) sees them too.
// ============================================================================

struct SortKeyParams {
    uint count;        // splats in this chunk
    uint baseOffset;   // global write offset for this chunk's splats
    uint byDistance;   // 1 = squared distance to camera; 0 = dot(position, forward)
    float camX, camY, camZ;
    float fwdX, fwdY, fwdZ;
    uint keyShift;     // right-shift applied to the 32-bit key to quantize it to (32-keyShift) bits.
                       // 0 = full 32-bit key (8 passes). For dynamic key width (optimization #2)
                       // keyShift = 32 - passes*4 keeps the TOP bits, which preserves order
                       // (right-shift is monotone non-decreasing) so the radix sort needs fewer passes.
};

// Generates a descending-depth sort key + global-index payload for one chunk.
// `splatFloats` aliases the chunk's EncodedSplatPoint buffer (32 B/splat = 8 floats; position is the
// first 3 floats — MTLPackedFloat3 at offset 0). Larger depth → smaller key, so an ascending radix
// sort yields back-to-front order (matches the CPU `sort { $0.depth > $1.depth }`).
kernel void splatDepthKeys(device const float *splatFloats [[buffer(0)]],
                           device uint        *keysOut     [[buffer(1)]],
                           device uint        *payloadOut  [[buffer(2)]],
                           constant SortKeyParams &p        [[buffer(3)]],
                           uint i [[thread_position_in_grid]])
{
    if (i >= p.count) { return; }
    uint fb = i * 8u;                      // 8 floats per EncodedSplatPoint
    float x = splatFloats[fb + 0];
    float y = splatFloats[fb + 1];
    float z = splatFloats[fb + 2];
    float depth;
    if (p.byDistance != 0u) {
        float dx = x - p.camX, dy = y - p.camY, dz = z - p.camZ;
        depth = dx*dx + dy*dy + dz*dz;
    } else {
        depth = x*p.fwdX + y*p.fwdY + z*p.fwdZ;
    }
    uint bits = as_type<uint>(depth);
    uint mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    uint asc  = bits ^ mask;              // monotonic float→uint (ascending)
    uint key  = ~asc;                     // descending depth → ascending key (full 32-bit)
    // Quantize to (32 - keyShift) bits by keeping the top bits. Right-shift is monotone
    // non-decreasing, so back-to-front order is preserved exactly; ties collapse within a
    // bucket and, since the radix sort is stable, hold their (depth-irrelevant) input order.
    // keyShift == 0 → full 32-bit key, unchanged behaviour.
    uint g = p.baseOffset + i;
    keysOut[g]    = key >> p.keyShift;
    payloadOut[g] = g;                    // payload = global index
}

struct GatherParams { uint count; uint chunkCount; };

// Maps each sorted global index back to a ChunkedSplatIndex (chunkIndex:UInt16, _pad:UInt16,
// splatIndex:UInt32 == uint2(chunkIndex, splatIndex)). Binary-searches the per-chunk base offsets.
kernel void gatherChunkedIndices(device const uint *sortedPayload [[buffer(0)]], // global idx in sorted order
                                 device const uint *offsets       [[buffer(1)]], // [chunkCount+1] global bases
                                 device const uint *chunkIds      [[buffer(2)]], // [chunkCount] chunkIndex values
                                 device uint2      *out           [[buffer(3)]], // ChunkedSplatIndex
                                 constant GatherParams &p          [[buffer(4)]],
                                 uint i [[thread_position_in_grid]])
{
    if (i >= p.count) { return; }
    uint g = sortedPayload[i];
    // binary search: largest c with offsets[c] <= g
    uint lo = 0u, hi = p.chunkCount;      // search in [0, chunkCount)
    while (lo + 1u < hi) {
        uint mid = (lo + hi) >> 1u;
        if (offsets[mid] <= g) { lo = mid; } else { hi = mid; }
    }
    uint localIndex = g - offsets[lo];
    out[i] = uint2(chunkIds[lo], localIndex);
}
