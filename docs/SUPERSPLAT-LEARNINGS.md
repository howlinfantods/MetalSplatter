# SuperSplat / PlayCanvas Learnings — What To Steal For MetalSplatter on M5 Vision Pro

**Status:** synthesis report. Source: six reverse-engineering passes over PlayCanvas engine + SuperSplat editor + splat-transform, cross-checked against our `LOD-DESIGN.md` and `VOS27-API-INVENTORY.md`, and (Rev 2) re-grounded against the actual MetalSplatter fork source.
**Date:** 2026-06-21 (Rev 2 — adversarial review applied).
**Audience:** us, deciding where to spend effort on the MetalSplatter visionOS fork.

> **Baseline note (read first).** The "payoff" column throughout is *relative to the current MetalSplatter fork*, which we verified at `/Users/bryan/Documents/Cairn-Projects/MetalSplatter/`: it re-sorts on **any** camera-pose change (no delta gate — `SplatSorter.updateCameraPose` sets `needsSort = true` unconditionally), the GPU radix is **fixed 4-bit × 8-pass / 32-bit** (`SplatGPURadixSort.radixBits = 4`, `passes = 8`, both hardcoded — and not yet wired into the per-frame loop; the active sorter is the CPU `SplatSorter.performSort`), there is **no LOD and no cull-before-sort**, and **SH is evaluated inline in the vertex shader every frame** with no RGB cache (`SplatProcessing.metal:173 evaluateSH`, called per-splat per-frame). Those four facts are what every win below is measured against.

---

## 0. The one-paragraph answer

SuperSplat is not faster because of a single clever shader. It is faster because of a **work-avoidance quartet** that MetalSplatter does not yet do: (1) it **does not sort every frame** — a camera-delta gate skips the sort entirely when the view barely moved; (2) it **does not touch every splat** — an octree-LOD + per-frame budget + GPU interval cull make per-frame cost scale with *visible* splats, not asset size; (3) when it *does* sort, the sort is **cheaper** — non-uniform binned depth keys with a dynamic 10–20-bit width spend fewer radix passes than our fixed 32-bit / 8-pass; (4) it **does not re-resolve view-dependent color every frame** — SH→RGB re-evaluation is gated on a *separate camera-translation* threshold (rotation doesn't change view-dependent color the way translation does), so a large fraction of frames skip the SH cost entirely over millions of splats. All four are **CPU-side / algorithmic and engine-agnostic** — they are the portable, high-value wins. The flashier piece — their tile-based compute rasterizer — wins on desktop/5090 but **likely loses on Vision Pro** (it bypasses system foveation) — but that conclusion is *contingent on an unmeasured foveation premise* (see §3), so it is flagged-not-settled, and is *not* the thing to port. Be honest: SuperSplat's measured efficiency is a web/desktop result; the part that transfers to VP is the work-avoidance, not the renderer.

---

## 1. Why is SuperSplat so much more efficient than our current viewer?

Ranked by how load-bearing the cause is, with the evidence.

### Root cause #1 — It sorts *far less often* (the single biggest lever)

MetalSplatter today runs the GPU radix sort **every frame** over **all** splats. SuperSplat gates the sort on a camera-delta threshold:

- **Directional sort:** re-sort only when the camera **forward vector** rotates past ~`0.001 rad` (`acos(dot(lastForward, currentForward)) > eps`). Translation does **not** change dot-product depth order, so panning/dollying triggers **zero** sorts.
- **Radial sort (cubemap):** re-sort only when **position** moves past ~`0.001` units.

The engineers' own summary calls this *"the single biggest perf lever — not the sort algorithm at all."* A static or slowly-rotating camera does no sort work. On Vision Pro the head is rarely perfectly still, but micro-jitter below threshold is invisible to ordering — so the realistic win is large.

> Note: this is **already in our `LOD-DESIGN.md` §4** as the "amortized sort gate" — but it is *design-only, unimplemented*. SuperSplat is the proof it works and gives us the exact epsilons.

### Root cause #2 — Per-frame cost scales with *visible* splats, not asset size

SuperSplat's "gsplat-unified" subsystem is a streaming-LOD brain for 16M+ scenes:

- **Octree LOD, flattened to a per-leaf level-select.** (Surprise: there is *no runtime tree walk* — the octree JSON is flattened at load into a flat leaf array; each leaf carries a pyramid of pre-baked density levels. LOD is a per-leaf distance+budget decision, not a hierarchical descent.)
- **A two-stage budget balancer** caps the resident/sortable splat count to an exogenous target (`splatBudget`), so the sort set is bounded *regardless of scene size*.
- **Per-interval GPU frustum cull + prefix-sum + compaction** runs the expensive sort only over survivors. Cull is O(numIntervals) (~one per visible leaf), not O(numSplats).

Net: a 16M scene viewed from across the room sorts ~the budget (e.g. 5M), not 16M. MetalSplatter currently has no LOD and no cull-before-sort, so it pays full asset cost every frame. (Our `LOD-DESIGN.md` §3/§5 designs exactly this; SuperSplat is a working reference for the flat-leaf model — *simpler* than our proposed hierarchical tree-cut.)

### Root cause #3 — The sort itself is cheaper (binned keys + dynamic bit width)

When SuperSplat does sort, it spends fewer bits. **There are two distinct sorters in the engine, and the report previously conflated them — they must be kept separate:**

- **(a) CPU counting-sort worker** (`gsplat-unified/gsplat-unified-sort-worker.js`). This is the sorter the *gsplat-unified* LOD/streaming brain actually uses for its CPU path. It is a **single-pass counting sort** over a `2^numBits + 1`-bucket histogram (`countingSort`, line 155; `bucketCount = 2 ** compareBits + 1`, line 269), run in a Web Worker (`gsplat-unified-sorter.js` wraps it in `new Worker`), posting `order.buffer` back. The dynamic width lives here: `compareBits = clamp(10, 20, round(log2(N/4)))` (line 267), and the **32-bin camera-relative weighting** (tiers `40 / 20 / 8 / 3 / 1`, `gsplat-sort-bin-weights.js` `WEIGHT_TIERS`) packs more key codes near the camera. A counting sort is **not** "`ceil(numBits/4)` four-bit passes" — it is *one* pass; the bit width only sets the histogram size.
- **(b) GPU radix sort** (`engine/src/scene/graphics/radix-sort/`). Two backends selected at runtime: **`ComputeRadixSortMultipass`** (portable 4-bit LSD, `numPasses = numBits / 4`, `sort(keys, n, numBits=16)` → 4 passes, `32` → 8 passes — **this is the Apple-portable reference**) and **`ComputeRadixSortOneSweep`** (NVIDIA-only single-sweep 8-bit). The gsplat-unified *hybrid GPU* path **already feeds dynamic width into the multipass radix**: `numBits = clamp(10,20, round(log2(N/4)))`, `roundedNumBits = ceil(numBits / radixBits) * radixBits`, then `sortIndirect` (`gsplat-manager.js:1695–1732`). So dynamic key width is **proven on SuperSplat's own GPU path**, not just the CPU worker.

MetalSplatter does a **fixed 4-bit × 8-pass = 32-bit** radix. The genuinely portable claim is: *feed a dynamic `numBits` into a Metal multipass radix so it runs `ceil(numBits/4)` of 8 passes*. For the report's own scene range (1M–16M → `numBits = 18–20` → **5 passes**), that is **~37% off sort cost** (5/8 = 0.625). You only reach ~50% (4 passes / 16-bit) for sub-256K scenes. Plus binned keys for free (pure key-gen arithmetic, no API surface). State it as **~37% across our target range**, not "40–50%."

### Root cause #4 — It does not re-resolve view-dependent color every frame (the missed fourth lever)

This is an **independent** work-avoidance lever the earlier draft omitted entirely. View-dependent color (SH→RGB) is re-resolved only when the camera **translates** past a per-node distance-scaled threshold — *rotation is deliberately ignored*, because to first order a pure rotation does not change which view-dependent appearance a splat presents.

- The gate (`gsplat-unified/gsplat-manager.js:978–1036`, `applyWorkBufferUpdates`): `ratio = tan(colorUpdateAngle * DEG_TO_RAD)`; each node/splat accumulates `colorAccumulatedTranslation += translationDelta` and re-resolves only when it exceeds `ratio * max(1, worldDistance)`. Camera delta is **translation-only** (`calculateColorCameraDeltas`, line 1189 — `lastColorUpdateCameraPos.distance(currentCameraPos)`); tracking resets after every color update (`updateColorCameraTracking`, line 1154). A full transform change forces a re-resolve; otherwise SH is left stale. Debug flag `GSPLAT_DEBUG_SH_UPDATE` paint-flashes splats whose color was refreshed.
- The actual evaluation amortized is `evalSH` per-coefficient → packed RGB (`gsplat/gsplat-resolve-sh.js`), run as a quad pass over the centroid texture. The *decision whether to re-run it* is the CPU-side translation accumulator.

Why it matters on M5 VP and why it's a real port candidate: SH re-eval over millions of splats is a genuine per-frame cost, and the VP head **rotates constantly but translates comparatively little** (seated viewing) — exactly the case this gate is built to exploit. **Caveat specific to our fork:** MetalSplatter evaluates SH *inline in the vertex shader every frame with no RGB cache* (`SplatProcessing.metal:173`). So porting this gate is **not** a one-line accumulator — it requires first extracting SH→RGB into a **cached resolve stage** (resolve to a per-splat RGB buffer/texture), *then* gating re-resolution on the translation threshold. That is the honest scope. It **survives native too**: if you pre-resolve SH to feed `GaussianSplatComponent`, Apple will not re-amortize it for you, so the cache-and-gate is yours to own.

### Contributing, but over-credited: the tile rasterizer and Morton/SOG

- **Tile-based compute rasterizer** (count → prefix-sum → scatter-free place → per-tile local bitonic sort → per-tile front-to-back composite with transmittance early-out). On a flat screen / 5090 this is clearly more efficient than instanced-quad: no global sort, far less overdraw, exact FlashGS tile-intersection, per-pixel early-out. **On Vision Pro the advantage likely inverts — but this is a *contingent* conclusion, not a settled one (see §3 "unknowns").** The reasoning: a compute kernel writing a full-res texture **bypasses the hardware rasterizer**, so it would get **zero** benefit from visionOS's system-driven foveation (`VOS27-API-INVENTORY.md §5`: `MTLRasterizationRateMap` is *not* app-controllable on visionOS; foveation applies to the RealityView/fragment path). MetalSplatter's instanced-quad **fragment** path would then get foveated overdraw reduction for free, and the tile path's per-eye work ~doubles for stereo. **The load-bearing premise — "the system foveates the fragment path for free, and a compute tile path genuinely cannot" — is *inferred, not measured*.** Until that is instrumented on device (one-line plan in §3), treat "skip tile-raster on VP" as the *likely* call, not a proven one. If it holds, tile-raster is a desktop win that does not carry to VP; do not port it as a perf play.
- **SOG / Morton ordering** does **not** reduce sort *work* — the engineers explicitly note SOG still sorts every frame; Morton order only makes the sort cache-coherent. SOG's real win is **disk/transfer compression** (~3.5× over compressed-PLY; the 16M KennyRoom in 177 MB). Valuable, but for *loading*, not per-frame efficiency.

**Honest summary:** the durable, transferable efficiency causes are #1, #2, #3, #4 — all CPU-side / algorithmic gates over engine-agnostic work. The renderer architecture itself (#tile) is a desktop win whose VP penalty is *probable but unmeasured*.

---

## 2. Port plan — ranked by payoff ÷ effort on M5 Vision Pro

Each item reconciled against the native `GaussianSplatComponent` (vOS27) option. The reconciliation axis is simple:

- **SURVIVES native:** loaders, LOD bake/format, budget balancer, streaming — Apple gives you a renderer, *not* an importer or an LOD system. Port these regardless of the fork decision.
- **REDUNDANT under native:** GPU radix sort, tile rasterizer, GPU cull/compaction — only matters if we keep the custom renderer. Native does its own sort + foveated render.

| # | Technique | Feasibility | Effort | Payoff on VP | Native verdict |
|---|---|---|---|---|---|
| 1 | **Camera-delta re-sort gate** (forward-only dir; pos for radial; eps ~0.003–0.005 rad on VR) | high | ~1 d | **Very high** — biggest single lever | REDUNDANT under native (Apple sorts); keep if custom |
| 2 | **Dynamic key width + binned keys** (clamp 10–20 bits; 32-bin weight tiers; AABB 8-corner range) → feed dynamic `numBits` into a Metal multipass radix, run `ceil(numBits/4)` of 8 passes | high | ~1–2 d | **High** — ~37% off sort cost in our 1M–16M range (up to ~50% sub-256K) | REDUNDANT under native; keep if custom |
| 2b | **SH→RGB resolve-cache + camera-translation amortization gate** (extract SH eval into a cached resolve stage, then gate re-resolve on `colorAccumulatedTranslation ≥ tan(angle)·dist`) | medium | ~3–5 d | **High** — skips SH re-eval over millions of splats on rotation-dominant VP head motion | **SURVIVES if you pre-resolve SH** — Apple won't amortize your SH cache |
| 3 | **Double-buffered / one-frame-late sort + buffer pooling** | high | ~1–2 d | High — never stall a 90/120 Hz frame | REDUNDANT under native; keep if custom |
| 4 | **CPU LOD bake format + flat-leaf loader** (per-leaf LOD pyramid, packed bounds, geometric distance bands, FOV comp) | high | ~3–5 d | **Very high for 16M** — the part native won't give you | **SURVIVES** — port regardless. **Inert without the #13 bake (hard dependency).** |
| 5 | **Two-stage budget balancer** (feedback scale + sqrt-distance-bucket nudging to a splatBudget; wire to thermalState) | high | ~2–3 d | **Very high for 16M** — bounds the working set | **SURVIVES** — port regardless |
| 6 | **Streaming residency** (ref-count + cooldown hysteresis + underfill; mmap SPZ/SOG + `MTLResidencySet`) | high | ~4–6 d | High for >16 GB scenes | **SURVIVES** — port regardless |
| 7 | **SOG bundle loader + decode** (ZIP + WebP via ImageIO + dequant math) | high (CPU-decode) / med (GPU-resident) | 3–5 d / +1–2 wk | Med-high — unblocks KennyRoom natively + disk win | **SURVIVES** — Apple has no importer |
| 8 | **Per-interval GPU cull + prefix-sum + compaction** feeding sort over survivors | medium | **~2–3 wk** (not 1–1.5: the whole path is GPU-indirect-driven — needs Metal **indirect command buffers / ICB** plumbing, `PrefixSumKernel`, indirect-dispatch arg writes) | High for 16M, **only if custom** | REDUNDANT under native |
| 9 | **Behind-camera count trim** (clamp instanceCount to front-of-camera) | high | <1 d | Low-med fill win | REDUNDANT under native |
| 10 | **Tile compute rasterizer** (bitonic per-tile, FlashGS intersect, transmittance early-out) | high to port | 2–4+ wk | **Likely negative on VP, but CONTINGENT on the foveation premise** (see §3) — no foveation if true, + 2× stereo | Skip *pending one measurement*; native is the better "someone-else-renders" answer |
| 11 | **OneSweep 8-bit radix** | low | 2–3 wk | **Do not port** — unsafe on Apple GPUs. *Corroborated by SuperSplat itself:* `compute-radix-sort.js:_canUseOneSweep()` excludes non-NVIDIA, citing "lack forward-progress guarantees (Apple)" | Skip |
| 12 | **WebGL2 fragment radix (mipmap-prefix-sum)** | low | — | Skip — pure no-compute workaround; Metal has compute | Skip |
| 13 | **KL-style pairwise-Gaussian merge LOD decimation** (`splat-transform/.../decimate.ts`: `momentMatch` + `computeEdgeCost`, mass-weighted; GPU edge-cost in `gpu-edge-cost.ts`. Source labels the pipeline "Moment Matching (MPMM)" in `process.ts:149`.) | medium | 1–2 wk | Offline bake, run on the 5090 — not on-device | **PREREQUISITE for the 16M LOD win** — it produces the per-leaf LOD pyramid that #4 loads. Not orthogonal. |

### Top-tier rationale (the ones to do first)

- **#1 + #2 + #3** are the cheapest, highest-ROI changes and all feed the *existing* Metal sorter. #1 and the pre-sort cull are *already specified in `LOD-DESIGN.md`* — so this is "execute the existing design with SuperSplat's proven specifics" (the `0.001 rad` forward-only gate, the 40/20/8/3/1 bin tiers), not new R&D. For #2, note the multipass radix is the Apple-portable backend and SuperSplat already drives it with dynamic `numBits` on its GPU path — so this is a *proven pattern to port*, not a CPU-only trick to re-derive on GPU.
- **#2b (SH gate)** is the newly-surfaced lever. It is medium-effort *only because our fork evaluates SH inline per-frame* and must first grow a resolve-to-RGB cache; the gate itself is cheap scalar accumulation. It is rotation-tolerant by design, which fits VP head motion, and it survives native if we pre-resolve SH.
- **#4 + #5** are the load-bearing pieces for the 16M target and are **engine-agnostic Swift** — they pay off whether we keep the custom renderer *or* adopt native (you still need to decide *which* splats to feed the native component). Their key correction to our design: SuperSplat proves a **flat per-leaf level-select + sqrt-distance buckets** hits the same budget-bounded result as our proposed hierarchical tree-cut — with *less* runtime machinery. Recommend we **drop the tree-cut descent from `LOD-DESIGN.md §5.2` and adopt the flat model.**
- **#4 has a hard dependency on #13.** The flat-leaf loader is **inert without a pre-baked octree-with-LOD-pyramid asset**, and the only thing in this tree that *produces* that multi-level decimated asset is the #13 decimation pipeline. The ~1-week #4+#5 estimate covers the **runtime loader only**, not the bake toolchain it consumes. To actually ship the 16M-with-LOD win you must also stand up #13 offline (run it on the 5090). Budget the bake toolchain explicitly.

---

## 3. The strategic fork: custom Metal vs native vs hybrid

**Recommendation: HYBRID, split by scene size, with a mandatory bake-off to resolve the one load-bearing unknown.**

### The scene-size math decides it

Using `LOD-DESIGN.md`'s own lower-bound formula (`16 B × N × passes / 122 GB/s`):

| Scene | Brute sort/frame, fixed 8-pass | …with dynamic key width (#2) | 90 Hz budget | Verdict |
|---|---|---|---|---|
| **3M S2000** | ~3.1 ms (8 pass) | ~3.1 ms (18-bit → 5 pass; same f.p.) | 11.1 ms | **Sort is a non-issue either way.** |
| **16M KennyRoom** | ~16.8 ms (8 pass) | **~10.5 ms** (20-bit → 5 pass = **~37% off**) | 11.1 ms | **Sort-alone fits *if we apply #2*.** LOD is needed for *other* reasons (below). |

**Important and easy to miss:** once you apply top-tier optimization #2 (dynamic key width), even 16M's *sort* drops under the 90 Hz budget (20 bits → 5 of 8 passes → ~10.5 ms, i.e. 5/8 = ~37% off). So **sort time is not what forces LOD at 16M.** The real drivers are:

1. **Memory residency.** 16M × 32 B + SH3 is multiple GB resident (our `LOD-DESIGN.md §5.3/§5.4`), independent of sort. The budget balancer + streaming exist to fit the working set, not to speed the sort.
2. **Full-frame cost.** Sort-alone under budget ≠ *frame* under budget once you add cull+project, key-gen, the rasterizer (2-tri quad × overdraw × dual eye), and CompositorServices reprojection.
3. **Can native even ingest 16M?** The open question (see below).

Because **3M is tractable regardless**, the deciding factor for S2000 is *integration convenience* — and that points to **native `GaussianSplatComponent`**: Apple does the sort, foveates for free, composites correctly with passthrough, and it's far less code than maintaining our sort/cull/raster stack. Decode S2000 to `LowLevelBuffer`s, hand it to RealityKit, done.

**16M is the case that justifies the custom pipeline — for memory + full-frame reasons, not sort time — *and only if* the native component cannot size to 16M.** That last clause is **THE open question**, not a settled fact. It is exactly the bake-off the memory notes call for.

### The recommendation, concretely

- **(c) Hybrid.** Keep MetalSplatter's loaders + LOD/budget/streaming as the engine-agnostic core that runs **in both worlds**.
- **Default to native (b)** for standard scenes (≤~3–5M, S2000): least code, free foveation/stereo/passthrough integration, vOS27.
- **Keep custom (a) only where native can't reach:** huge clouds (16M KennyRoom) that need our LOD/budget/streaming to get under budget, *or* features native lacks (custom sort tuning, GPU picking, fisheye, fog, scene-depth occlusion).
- The custom renderer's **GPU radix sort + tile raster are redundant the moment we adopt native** — so do not over-invest there. The **loaders + LOD bake + budget balancer + streaming survive regardless** — invest there first.

### The unknowns that gate the recommendation (be honest)

1. **Can native `GaussianSplatComponent` ingest/render 16M splats at 90 fps on M5 VP?** Unknown. If yes, the custom renderer's reason-to-exist nearly evaporates and the answer collapses to "native + our LOD pre-filter." If no, custom is justified for 16M. **Resolve by bake-off before building more custom renderer.**
2. **Foveation linchpin (gates the entire tile-raster skip in #10).** Two claims must both hold and are currently *inferred, not measured*: (a) MetalSplatter's instanced-quad **fragment** path actually consumes the system-provided `rasterizationRateMap` from the CompositorServices/RealityKit drawable, so it gets foveated overdraw reduction "for free"; and (b) a compute-shader tile rasterizer writing a full-res texture genuinely **cannot** get that benefit. visionOS exposes no app-driven `MTLRasterizationRateMap` and no MetalFX — but "the system auto-foveates the fragment path" must be confirmed **on device**, not assumed. **One-line measurement plan:** instrument fragment-path GPU time (via a Metal counter sample / GPU capture) for an identical scene **with vs without** the system rate map active, on M5 VP; a foveated path shows materially lower fragment cost toward the periphery. Until that delta is measured, treat #10's "skip tile-raster" as the likely-but-unproven call.
3. **Native is vOS27-only.** If we must run vOS26 or Mac, custom is the only option there.

---

## 4. Do these N things next

1. **Run the bake-off (blocks everything).** Decode KennyRoom (16M) to `LowLevelBuffer`s, feed `GaussianSplatComponent` on M5 VP, measure fps + memory. This single result decides whether the custom 16M renderer is worth maintaining. Do this *before* porting any GPU renderer code.
2. **Ship the camera-delta re-sort gate (#1).** ~1 day. Implement `LOD-DESIGN.md §4` with SuperSplat's specifics: forward-only for directional. The epsilon needs **device tuning over a wide range** — sources span 35×: SuperSplat uses `0.001 rad`, our `LOD-DESIGN.md §4` guessed ~2° (`0.035 rad`). Start mid-range, larger than SuperSplat's to absorb VR head micro-motion, and tune on device; also consider a translation threshold, since wide-FOV stereo can pop on pure rotation at the periphery. Biggest win for least code; applies to the custom path now.
3. **Adopt binned keys + dynamic bit width (#2).** ~1–2 days. Port `GSplatSortBinWeights` (32 bins, 40/20/8/3/1 tiers, base/divider) + `numBits = clamp(round(log2(N/4)),10,20)` + AABB 8-corner range into the existing Metal key-gen; round `numBits` up to a multiple of 4 and run that many `/4` passes. **~37% off the current sort across our 1M–16M range** (up to ~50% only for sub-256K scenes). This is a *proven GPU pattern* — SuperSplat already feeds dynamic `numBits` into its multipass radix; we are porting, not inventing.
4. **Add the SH→RGB resolve-cache + camera-translation amortization gate (#2b).** ~3–5 days. First extract `evaluateSH` out of the per-frame vertex shader into a **cached resolve stage** (per-splat RGB buffer/texture); then gate re-resolution on `colorAccumulatedTranslation ≥ tan(colorUpdateAngle·DEG_TO_RAD)·max(1, dist)`, translation-only, resetting after each refresh — port of `gsplat-manager.js:978–1036`. Force a full re-resolve on any transform change. This skips SH cost on the rotation-dominant frames typical of a seated VP head. Survives native iff we pre-resolve SH to feed `GaussianSplatComponent`.
5. **Build the CPU LOD bake format + flat-leaf loader + budget balancer (#4, #5) — and stand up the #13 bake toolchain that feeds it.** ~1 week for the runtime loader (engine-agnostic Swift; pays off under *both* fork branches), **plus** the offline KL-merge decimation bake (#13) that produces the per-leaf LOD pyramid the loader consumes — run that on the 5090. The loader is inert without the bake; do not estimate them as one. **Revise `LOD-DESIGN.md`: replace the hierarchical tree-cut (§5.2) with SuperSplat's flat per-leaf level-select + sqrt-distance buckets.** Wire the budget + dead-zone to `ProcessInfo.thermalState` (our §6 thermal ladder).
6. **Add the double-buffered / one-frame-late sort + buffer pooling (#3).** ~1–2 days. Never block a 90/120 Hz frame; keep only `pendingSorted`, apply ≤1 result/frame, recycle index buffers. (On unified memory, *delete* the WebGL PBO/upload-stream machinery — write straight into a shared `MTLBuffer`.)
7. **SOG loader, decode-to-points first (#7).** 3–5 days. ZIP + WebP via ImageIO + the dequant math. **Decode is straightforward: 16-bit means use `/65535` (`gsplat-sog-data.js:66–68`); all 8-bit channels — quats, V1 scales, sh0 — use `/255`.** The `/257` is only the 8→16-bit mental model (255×257 = 65535); **do not implement a literal `/257`** — it does not appear in the decode. Branch on `meta.version` for V1 (lerp mins/maxs) vs V2 (codebook). Gets KennyRoom viewable natively and captures the disk/transfer win. Defer GPU-resident SOG until the bake-off says we're keeping the custom renderer.
8. **(Footnote, separate from efficiency) Fix S2000 orientation via the DataTable.transform model.** ~1–2 days. This is a *correctness* bug (the 180°-about-Z PLY convention double-flip), **not** an efficiency cause — tag source space at load and apply `delta = targetᵀ⁻¹·source` once, instead of hardcoding a per-format flip. Mentioned here only so it isn't lost; it does not belong in the efficiency thesis.

**Explicitly skip:** OneSweep radix (unsafe on Apple GPUs — *and SuperSplat agrees*: `compute-radix-sort.js:_canUseOneSweep()` excludes non-NVIDIA for the same forward-progress reason, independent corroboration that our instinct is calibrated), and the WebGL2 fragment radix (no-compute workaround). The tile compute rasterizer is the *probable* VP skip — but it is **contingent on the unmeasured foveation premise (§3-unknown-#2)**, so run that one measurement before treating it as settled. **Not a skip:** the KL-merge LOD decimation (#13) — it is an offline producer tool, but it is the **prerequisite bake** for the 16M LOD win (#4), so it must be stood up, just on the 5090, not on-device.

---

## 5. Corrections this forces on our own docs

- **`LOD-DESIGN.md §5.2`:** drop the hierarchical tree-cut with "representative" coarse nodes. Adopt SuperSplat's **flat per-leaf LOD pyramid + sqrt-distance buckets + budget balancer** — same budget-bounded result, less runtime machinery, proven at 16M+.
- **`LOD-DESIGN.md §4` (amortized sort):** upgrade from "skip if camera moved" to SuperSplat's **separate gates** for LOD vs sort vs cull, and **forward-only** sensitivity for directional sorting. **Add a fourth, independent gate not previously in our design: a camera-translation gate on SH→RGB re-resolution** (rotation-tolerant), with a prerequisite SH resolve-cache stage — see root-cause #4 / port item #2b.
- **`LOD-DESIGN.md §3` (Pass 0 sub-pixel cull):** SuperSplat puts sub-pixel/contribution rejection in the **projector** (`minPixelSize`/`minContribution`), *not* the cull pass — keeping cull at O(intervals). Consider moving our per-splat sub-pixel reject likewise.
- **`VOS27-API-INVENTORY.md`:** confirms the fork axis above — native gives renderer+sort+foveation but **no importer and no LOD**, so our loaders + LOD core are load-bearing under every branch.
