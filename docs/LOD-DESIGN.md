# LOD & Culling Design — 20M-Splat Viewer on M5 Vision Pro

**Status:** design (not yet implemented). Phase 3 of `OVERHAUL-PLAN.md`.
**Date:** 2026-06-22.
**Target:** 20M-splat *total* scenes (Clarte, Kenny iPhone, AmazingComputerRender) holding ≥90 fps,
ideally 120 fps, on the M5 Vision Pro (10-core GPU, ~122 GB/s measured bandwidth, 16 GB shared).

---

## 1. Why LOD is mandatory — the per-frame bandwidth math

The GPU radix sort (Track B, `SplatGPUSorter` + `RadixSort.metal`) is a **4-bit LSD radix = 8 passes**
over 32-bit depth keys. Each pass reads and writes both the key buffer (4 B) and the payload buffer
(4 B) — a scatter that moves ≈ 16 B/element/pass through memory (plus histogram traffic, ignored here
as a lower bound).

For **N = 20M** splats sorted **every frame**:

```
bytes_moved ≈ 16 B × 20e6 × 8 passes        = 2.56 GB / frame  (sort traffic, lower bound)
time_at_122GB/s ≈ 2.56e9 / 122e9            ≈ 21.0 ms / frame  (SORT ALONE)
```

Frame budgets:

| Refresh | Budget / frame | Sort-alone (20M) | Verdict |
|--------:|---------------:|-----------------:|---------|
| 90 Hz   | 11.1 ms        | ~21.0 ms         | **2× over budget before a single splat is drawn** |
| 120 Hz  | 8.3 ms         | ~21.0 ms         | ~2.5× over |

And that is **before** the cull/project pre-pass, the depth-key generation, the rasterizer (2-tri quad
per splat × overdraw × dual eye), and CompositorServices reprojection. Conclusion: **20M splats cannot
be sorted per-frame on the M5 VP at 90 fps.** The fix is threefold and independent:

1. **Sort fewer splats** → cull *before* the sort (Section 3).
2. **Sort less often** → amortize: skip the sort when the camera barely moved (Section 4).
3. **Resident fewer splats** → LOD tree-cut to a per-device budget (Section 5).

**Working budget** (derived, validate on device): to keep sort ≤ ~5 ms we want **≤ ~5M splats actually
sorted per frame** (5M × 16 × 8 / 122e9 ≈ 5.2 ms). That is the design's load-bearing number: *cull and
LOD must reduce the sortable set from 20M-resident to ≤~5M-visible per frame.*

---

## 2. Where this sits in the existing architecture

Current per-frame chain (`SplatSorter.performSort` → `SplatGPUSorter.encode`):

```
[chunk position buffers] → splatDepthKeys (kernel) → 4-bit radix ×8 → gatherChunkedIndices → indexed draw
```

The renderer is already **chunked** (`SplatChunk`, `ChunkedSplatIndex` carries `chunkIndex`+`splatIndex`).
That is the hook for both culling and LOD — we operate at chunk granularity for coarse decisions and at
splat granularity for the fine cull, reusing the existing chunk plumbing instead of replacing it.

Target per-frame chain (3-pass GPU-driven, this doc adds Pass 0 + the LOD tree-cut):

```
LOD tree-cut (CPU, amortized) → resident chunk set
        │
        ▼
Pass 0  cull+project compute  → survivor compaction (atomic counter) ─┐
        │  frustum + behind-cam + sub-pixel reject                    │ writes indirect-draw args
        ▼                                                             │
Pass 1  depth keys (survivors only) → 4-bit radix → gather  ◄─────────┘ (sort the SMALL survivor set)
        │
        ▼
Pass 2  raster (existing 1024-indexed/instanced quad + single-pass stereo amplification)
```

---

## 3. Pass 0 — cull + project (BEFORE the sort)

**Placement: the cull pass runs *before* the radix sort, and the sort consumes only survivors.** This is
the single highest-leverage change — every culled splat is one we don't pay 8 radix passes for. (Culling
*after* the sort would waste the sort on invisible splats — the opposite of what we want.)

New compute kernel `cullAndProject` (add to `RadixSort.metal` or a new `Cull.metal`), one thread per
resident splat:

1. **Frustum reject.** Transform `position` by the per-eye view-projection (already available as the
   `viewMatrix`/`projectionMatrix` in `ModelRendererViewportDescriptor`). Use the **combined** frustum
   of both eyes (union) so a splat visible to either eye survives — cull on the intersection would pop at
   the stereo edges. Test clip-space `-w ≤ x,y,z ≤ w` with a small guard band (≈1.1×) so splats whose
   *footprint* overlaps the screen edge aren't clipped at their center.
2. **Behind-camera reject.** `clip.w ≤ near` → drop (this also removes the today's per-vertex behind-cam
   reject, moving it out of the hot vertex stage).
3. **Sub-pixel (screen-space size) reject.** Project the larger covariance eigen-extent to screen space:
   `pxSize ≈ projectedRadius * (screenHeight / clip.w)`. Drop if `pxSize < kMinPixels` (start at **0.75
   px**; tune). This is the biggest win on dense distant geometry — a 20M car scan seen from across the
   room is mostly sub-pixel. We can approximate `projectedRadius` cheaply from `covA`/`covB` (max of the
   packed covariance diagonal) without a full eigendecomposition.
4. **Survivor compaction.** `atomic_fetch_add` on a global counter → write the splat's global index into a
   compacted `survivors` buffer. The radix sort then runs over `survivorCount`, not `N`.
5. **Indirect draw.** Write `survivorCount` into an `MTLDrawIndexedPrimitivesIndirectArguments` buffer so
   the draw call (and the sort's dispatch sizes) are **GPU-driven** — culled/disabled chunks cost ~0 on
   the CPU side, no per-frame `useResource` loop rebuild.

**Coarse pre-cull (cheap, optional first step):** before the per-splat kernel, frustum-test each chunk's
**bbox** on the CPU (chunks are spatially coherent if authored that way; if not, build a one-time
per-chunk AABB at load). Skip whole chunks that are fully outside the frustum — this removes most splats
with ~chunkCount tests instead of N. The per-splat Pass 0 then refines only the chunks that survive.

---

## 4. Amortized sorting (sort LESS OFTEN)

The sort result is a **back-to-front permutation**; it's only wrong when the camera moves enough to change
ordering. Re-sort only when the camera delta exceeds a threshold:

```
needsResort =  angle(forward, lastSortForward) > kSortAngleEps      // ~2°
            || distance(camPos, lastSortCamPos) > kSortPosEps × sceneScale   // ~2% of bbox extent
            || residentSetChanged                                   // LOD tree-cut changed
```

Between re-sorts, **reuse the last `ChunkedSplatIndex` buffer** and just re-raster. On the M5 VP the head
is rarely perfectly still, but micro-jitter below the threshold is invisible to ordering — this realistically
halves average sort frequency. Keep the existing 3-deep index ring + Mutex/generation orchestration in
`SplatSorter`; this only gates *whether* a new sort is kicked off. (This is a near-free win and should ship
in Phase 1 alongside the GPU sort, ahead of full LOD.)

---

## 5. Hierarchical LOD — bucket tree within the chunked architecture

### 5.1 Structure: a bottom-up spatial LOD tree over chunks

At load (or in a one-time bake), build a **spatial octree-ish hierarchy** whose **leaves are the existing
`SplatChunk`s** (or sub-chunks of ~256k splats each for finer granularity). Each internal node stores:

- an **AABB** (union of children),
- a **representative LOD chunk**: a decimated/merged set of splats approximating the subtree (e.g. 1/4 the
  child splat count, made by importance sampling on `opacity × screen-projected scale`, or by clustering
  nearby splats and merging covariance). Build these coarse levels offline where possible (they're static).

This reuses chunks as the unit of residency and sort — the tree is just an index over chunks plus the
extra coarse-LOD chunks. `ChunkedSplatIndex.chunkIndex` already addresses any chunk, fine or coarse, with
no shader change.

### 5.2 Tree-cut per camera against a resident budget

Each frame (amortized — recompute the cut only when the camera moves past the Section-4 threshold), walk
the tree top-down and choose, per node, fine children vs. the coarse representative based on **projected
screen-space error**:

```
nodeScreenError ≈ nodeWorldSize / distanceToCamera × (screenHeight / 2 / tan(fovY/2))
if nodeScreenError < kErrorPixels:  use this node's coarse LOD, stop descending
else:                               descend to children (finer LOD)
```

Greedily descend the highest-error nodes first until the **resident/sortable budget** (~5M splats, Section 1)
is hit — this is the "lowest-LOD-first, tree-cut to budget" strategy. Distant subtrees collapse to coarse
chunks; the subtree the user is looking at stays full-resolution. The selected node set = the **resident
chunk set** fed to Pass 0.

### 5.3 Streaming (the path to 20M *total* > 16 GB resident)

20M splats × 32 B = 640 MB for positions/cov alone, +1.8 GB for SH3 — plus dual 120 Hz framebuffers, the
index ring, sort scratch, and OS reserve. The full hi-res set may not fit resident. So:

- Keep **coarse levels always resident** (small, cover the whole scene — no popping to empty).
- **Stream fine leaf chunks** in/out on the tree-cut: when a leaf enters the cut, async-load its chunk
  (from a memory-mapped SPZ/SOG on disk) into a heap; when it leaves, evict. Use `MTLResidencySet` /
  `useHeap` (replacing the per-frame per-chunk `useResource` loop) so residency is set-based and cheap.
- Hysteresis on eviction (don't evict a leaf that just left the cut — it'll likely come back as the head
  turns) to avoid thrash.

### 5.4 Memory budget table (model this on device, don't assume)

| Item | Estimate |
|---|---|
| OS + system reserve | ~3–4 GB (validate) |
| Dual framebuffers @ 120 Hz, >4K/eye | ~0.5–1 GB |
| Resident splats (5M @ 32 B + SH) | ~0.5–1 GB |
| Coarse LOD chunks (always resident) | ~0.2 GB |
| Sort scratch (keys+payload @ survivors) | ~40 MB @ 5M |
| Index ring (3× ChunkedSplatIndex @ survivors) | ~120 MB @ 5M |
| Stream staging / in-flight loads | budgeted headroom |

Leaves a comfortable working set well under 16 GB at 5M resident, with disk streaming covering the 20M total.

---

## 6. Thermal / sustained-load ladder

The M5 VP runs ~2.5–3 h; sustained per-frame GPU sort + heavy overdraw is a worst-case load, so steady-state
budget < cold-start ceiling. Tie the LOD budget and `kErrorPixels` to `ProcessInfo.thermalState`:

```
.nominal:  budget 5M,   kErrorPixels 1.5,  target 120 Hz
.fair:     budget 4M,   kErrorPixels 2.0,  target 90 Hz
.serious:  budget 2.5M, kErrorPixels 3.0,  Dynamic Render Quality down
.critical: budget 1.5M, coarse-only,       throttle hard
```

Use CompositorServices **Dynamic Render Quality** as the resolution throttle (MetalFX upscaling cannot
consume a foveated drawable — see OVERHAUL-PLAN).

---

## 7. Implementation order (incremental, each shippable)

1. **Amortized sort gate** (Section 4) — smallest change, immediate average-frametime win. Lands with the
   GPU sort in Phase 1.
2. **Per-chunk AABB + coarse frustum pre-cull** (Section 3 coarse step) — cheap, removes off-screen chunks.
3. **Pass 0 per-splat cull+project + survivor compaction + indirect draw** (Section 3) — sort runs on
   survivors only. This is the headline LOD-adjacent win and unblocks the ≤5M sortable target.
4. **LOD tree build + tree-cut to budget** (Section 5.1–5.2) — coarse levels resident, error-driven cut.
5. **Streaming + residency sets** (Section 5.3) — the 20M-total-on-16GB capstone.
6. **Thermal ladder** (Section 6) — sustained-session correctness.

## 8. Open questions to resolve on device

- Real per-splat `cullAndProject` cost at 20M — is the cull pass itself bandwidth-bound enough to matter
  vs. the sort it saves? (Expected net win, but measure.)
- Best coarse-LOD generation: importance subsampling vs. covariance-merging clusters — quality vs. build cost.
- SPZ/SOG decode-in-shader cost when streaming fine chunks (interacts with Phase 2 compression work).
- Does single-pass stereo amplification complicate the union-frustum cull? (Both eyes share the survivor
  set; confirm no per-eye divergence artifacts at the screen edge.)
