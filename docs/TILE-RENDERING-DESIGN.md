# Tile / Overdraw Rendering Design — MetalSplatter on M5 Vision Pro

**Date:** 2026-06-22. **Author:** main session, synthesized from a 4-angle GitHub/literature survey (canonical CUDA/FlashGS + PlayCanvas WGSL; Aras-p UnityGaussianSplatting + StopThePop/Taming-3DGS/VRSplat; Apple TBDR/tile-shading/imageblocks; Brush + wgpu) cross-checked against the fork's actual source.
**Problem:** the S2000 (3M) renders "terribly" and "too opaque/muddy" on device. Root cause is **overdraw**, not the sort.

---

## 0. The verdict (read this first)

**Do NOT rewrite into a compute-tile rasterizer. EVOLVE the imageblock/tile-shader path the fork already has.**

Three facts force this:

1. **Overdraw is ~4× the sort.** Aras-p's measured breakdown on the bicycle scene (6.1M, RTX 3080 Ti): 6.8 ms = **4.5 ms render + 1.1 ms sort + 0.8 ms view-calc**, verdict: *"the main performance culprit is high amount of overdraw, closely followed by an expensive radix sort."* Our fork **is** Aras's architecture in Metal (instanced screenspace quads, fixed-function back-to-front `over` blend, GPU radix sort). So: fix overdraw first; the sort is secondary (and we already added a camera-delta gate + dynamic key width).

2. **Vision Pro foveation is free only in the raster pipeline.** On visionOS, app-driven `MTLRasterizationRateMap` is `API_UNAVAILABLE` (see `VOS27-API-INVENTORY.md §5`); the system foveates *only* content drawn through the RealityView/CompositorLayer **fragment** path. A pure compute-tile rasterizer (Brush / INRIA / PlayCanvas-WGSL) `textureStore`s a full-res eye image from a compute dispatch and **never sees the rate map → forfeits foveation**, the single biggest free perf lever on the M5. Brush proves a tiled compute rasterizer runs on Apple via wgpu→Metal, but it structurally throws away foveation. Wrong trade for VP.

3. **The fork already contains the right substrate, pointed at the wrong job.** `MultiStageRenderPath.metal` already declares an imageblock tile-shader compositor — `FragmentValues { half4 color [[raster_order_group(0)]]; float depth [[raster_order_group(0)]]; }` wrapped in `[[imageblock_data]]`, an `initializeFragmentStore` tile kernel that clears tile memory, a `multiStageSplatFragmentShader` that over-blends by reading `previousFragmentValues` from the imageblock, and a `postprocessFragmentShader` resolve. `SplatRenderer.swift` has a `useMultiStagePipeline` toggle and builds both pipelines. **This is textbook Apple TBDR tile-shading + raster-order-groups + imageblocks — and it stays in the foveated fragment pipeline.** We don't build it; we upgrade what it does per-pixel.

> **Caveat that keeps us honest:** TBDR tile memory / imageblocks cut **bandwidth**, not overdraw. Apple's Hidden Surface Removal cannot cull *blended/transparent* fragments (they don't occlude). So the imageblock scaffold alone does nothing for overdraw — the overdraw fix (front-to-back early-T, bounded-K, tighter quads, fewer primitives) must be layered *into* that fragment shader. The win is that all of those fixes are expressible in the raster pipeline, so we keep foveation while killing overdraw.

**One-line strategy:** keep the foveated raster path → make each pixel stop doing work once it saturates (front-to-back + transmittance early-out), make it order-tolerant (bounded-K MLAB), make each splat touch fewer pixels (opacity-aware tight quads), and feed it fewer splats (LOD). Compute-tile rasterizer stays on the shelf as a documented fallback only.

---

## 1. Measure before building (cheap, do first)

Metal System Trace **crashes this app** (capture interposer vs `MetalCaptureEnabled=false` on this beta — see the trace-crash memory). So measure capture-free:

- **M0 — which path is even running?** Print `useMultiStagePipeline` and confirm the multi-stage (imageblock) path is active on the device for the splat draw. If the device is silently on the single-stage path, that alone may explain the muddiness, and forcing multi-stage is step zero.
- **M1 — baseline GPU ms.** We already added capture-free per-frame timing (`onSortStats` + the render path). Add a render-pass GPU-time readout (`commandBuffer.gpuStartTime/gpuEndTime`) surfaced in the in-headset log, at Bonsai (1.2M) and S2000 (3M). This tells us the render-vs-sort split on *our* hardware, confirming Aras's ratio holds on the M5.
- **M2 — overdraw proxy.** Toggle a debug fragment that outputs blend-count per pixel (heatmap) to *see* overdraw. Confirms the diagnosis visually before we invest.

Gate the build plan on M1: if render ≫ sort (expected), proceed; the steps below are ordered by payoff/effort regardless.

---

## 2. Build path — incremental, each step shippable, all stay foveated

### Step 1 — Opacity-aware tight oriented quads + alpha discard *(cheapest, no pipeline change, big ROI)*
- Replace axis-aligned quads with **oriented** quads sized to the splat's true opacity-weighted extent: cutoff radius from opacity (Brush: `ln(opacity·255)`; WebSplatter: `r = sqrt(ln(255·σ))`). Near-transparent splats stop painting huge empty margins.
- `discard_fragment()` (or early `return`) below ~`1/255` alpha in the fragment shader.
- Pure win, in-raster, foveation kept. Aras and FlashGS both cite this as the first lever. **Effort: ~1–2 d.**

### Step 2 — Front-to-back + per-pixel transmittance early-out *(the core overdraw kill)*
- Flip the depth sort to **front-to-back** and carry running transmittance `T` in the imageblock (`FragmentValues` already holds per-pixel state in tile memory). Once `T < ~1e-3`, the pixel is saturated → early-return; deep stacks of occluded splats cost nothing past saturation. arXiv 2505.18764's "T-Culling" reports ~3×.
- This is the entire reason tiled rasterizers beat instanced quads — and it works **in the fragment/imageblock path** (no compute rasterizer needed), so foveation is kept.
- Requires the imageblock blend to accumulate front-to-back (`C += T·α·c; T·= (1-α)`) instead of fixed-function `over`. The scaffold already reads/writes the imageblock; we change the blend math. **Effort: ~3–5 d.** Interacts with the sort (must be front-to-back).

### Step 3 — Bounded-K MLAB in the imageblock *(correctness + relax the sort + fix popping)*
- Upgrade `FragmentValues` from a single `{color,depth}` to a **K-entry Multi-Layer Alpha Blending** array (Apple's "Order-Independent Transparency with Image Blocks" sample is a near drop-in). Per-pixel insert-and-merge keeps the K frontmost layers; order-independent, so it **fixes the popping** the fork's `sortByDistance` TODO describes (StopThePop's exact problem — a global per-Gaussian z-sort inverts order under head rotation) **and lets us relax the global sort** to a cheap approximate one.
- **Feasibility gate:** verify the imageblock byte budget. `K × (half4 color + float T + float depth)` must fit the tile's imageblock sample length at 32×32; if not, shrink tile to 16×16 or cap K (K≈4–8 is typically plenty). Check `MTLDevice` imageblock limits on the M5 before committing K. **Effort: ~1 wk.**

### Step 4 — LOD / primitive reduction *(biggest lever at 3M+; orthogonal, compounds)*
- Fewer primitives is the most reliable VR win: VRSplat hit Quest-3's 72 fps mainly via Mini-Splatting (~10× fewer Gaussians), not rasterizer tricks. Wire the distance-budget LOD from `LOD-DESIGN.md` (now simplified to SuperSplat's flat per-leaf level-select + budget balancer per `SUPERSPLAT-LEARNINGS.md`). At 3M+ this stacks on Steps 1–3.
- Also the bridge to native: a ≤200k LOD cut is exactly what the native `GaussianSplatComponent` could consume (see the 200k-cap finding) for a future hybrid. **Effort: ~1 wk runtime + the offline decimation bake (run on the 5090).**

### Step 5 — Sort hardening *(secondary, but real)*
- **Apple GPU spin-wait cliff:** web-splat's inter-workgroup spin-wait radix ran **4.5× slower on M1** (Apple schedulers punish workgroups that spin on each other). If our radix uses cross-threadgroup spin, switch to a **wait-free hierarchical** scan (WebSplatter/FidelityFX/`wgpu_sort` pattern). Verify our `RadixSort.metal` doesn't spin-wait.
- Keep the **camera-delta re-sort gate** and **dynamic key width** (already added). Aras: 8-bit radix beat 4-bit (1.1 ms vs 2.4 ms at 6.1M) — consider 8-bit passes. **Morton-reorder** splats so distance-sorted access stays cache-local. With MLAB (Step 3) the sort can be approximate, shrinking it further. **Effort: ~3–5 d.**

### NOT recommended (documented fallback only)
- **Pure compute-tile rasterizer (Brush/INRIA/PlayCanvas-WGSL port).** It's the structural overdraw fix and Brush is a clean blueprint (project → depth radix → tile-pair emit → tile radix → per-tile threadgroup raster with early-out). But on VP it **forfeits foveation**, and to claw foveation back you must re-implement it in-kernel (VRSplat-style variable-resolution tiles driven by ARKit gaze) — a much bigger, riskier job. Only revisit if Steps 1–4 in the raster path can't hit 90 fps at the target splat count *and* we accept building in-kernel foveation.

---

## 3. What the other projects taught us (reference table)

| Project | Camp | On Apple? | Foveation | Take |
|---|---|---|---|---|
| **scier/MetalSplatter** (our base) | instanced-quad + imageblock multi-stage | native Metal/visionOS | **preserved** | Evolve the existing `MultiStageRenderPath` imageblock compositor — don't replace. |
| **Aras-p/UnityGaussianSplatting** | instanced-quad | M1 Max via Metal (21.5 ms bicycle) | preserved | Overdraw = 4× sort. Cheap wins: tight oriented quads, alpha discard. We are its Metal descendant. |
| **Apple OIT-with-ImageBlocks sample** | tile-fragment imageblock | visionOS | preserved | Drop-in reference for Step 3 bounded-K MLAB. |
| **arXiv 2505.18764** (HW diff raster) | fragment + tile-image early-out | ARM/Apple tile GPUs | preserved | Front-to-back + per-pixel T-culling in the fragment path = ~3× (Step 2). |
| **StopThePop** | compute-tile | CUDA (portable) | lost as written | Diagnoses the fork's popping TODO; MLAB (Step 3) is the in-raster fix. |
| **FlashGS** | compute-tile | CUDA | lost as written | Exact ellipse-tile intersection / opacity-aware tight bound → Step 1 (no pipeline change). |
| **Brush** (ArthurBrussee) | compute-tile | Apple via wgpu→Metal | **forfeits** | The blueprint *if* we ever go compute-route; two stable radix sorts (depth then tile) instead of a 64-bit key. |
| **PlayCanvas WGSL (local)** | compute-tile | Apple via WebGPU/Dawn | forfeits | Line-for-line MSL transliteration target for the fallback; intersection + per-tile bitonic. |
| **VRSplat / VR-Splatting** | compute-tile, foveated | CUDA/Quest/Vive | re-implements in-kernel | If forced to compute-route, this is how to get VP foveation back (gaze-driven variable tiles). And: Mini-Splatting primitive reduction → Step 4. |
| **web-splat / WebSplatter** | instanced-quad | Apple via wgpu (M4 63.7 ms, iPhone 38.5 ms) | n/a | Confirms quads work but cap out at 3M+; wait-free radix sort → Step 5. |

---

## 4. Risks / open feasibility questions
- **Imageblock byte budget** for K-layer MLAB at 32×32 (Step 3) — verify against M5 `imageblockSampleLength`; shrink tile or K if needed. **Must check before Step 3.**
- **Is `useMultiStagePipeline` actually true on the device?** (M0) — the whole plan assumes the imageblock path runs on VP. Confirm first.
- **Front-to-back blend correctness** (Step 2) — flipping the sort + accumulation must match a reference render; validate against the current output on a small scene.
- **Sort spin-wait** (Step 5) — confirm `RadixSort.metal` isn't cross-threadgroup spin-waiting (the 4.5× Apple cliff).

## 5. Recommended next action
Steps **1 + 2** are the highest payoff/effort and both stay in the foveated raster path: opacity-aware tight quads + alpha discard, then front-to-back + per-pixel transmittance early-out in the existing imageblock fragment shader. Do **M0/M1/M2** first (a day) to confirm the multi-stage path is live and quantify the render-vs-sort split on the M5, then build 1→2→3→4. No step leaves the renderer broken; each is independently shippable and testable on device.
