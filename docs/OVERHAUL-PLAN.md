# MetalSplatter Overhaul — Huge-Splat Viewer for the M5 Vision Pro

**Fork:** howlinfantods/MetalSplatter (upstream: scier/MetalSplatter, MIT)
**Goal:** a less-clunky splat viewer that performantly displays HUGE (10–20M) Gaussian splats on the **M5 Apple Vision Pro**, using visionOS 26/27 + Metal 4 features the current code misses.
**Date:** 2026-06-17 · derived from a 6-agent research + fact-check pass (all external facts source-checked against live 2026 sources).

---

## TL;DR

1. **The clunkiness is the CPU sort.** MetalSplatter sorts splats by depth on the CPU — a single-threaded `sortTempStorage.sort { $0.depth > $1.depth }` (`SplatSorter.swift:628`) re-run every time the head moves, drawing with a *stale* sort in between. That is the popping/lag, and the hard wall at 10–20M. Moving the sort to the GPU is the **order-of-magnitude** fix; everything else is secondary.
2. **visionOS 27 now has a NATIVE splat renderer.** Apple shipped `GaussianSplatComponent` / `GaussianSplatResource` in RealityKit (WWDC26 session 279) — it did not exist in vOS 26. So the cheapest path may be *don't overhaul MetalSplatter at all* — use Apple's. Its internals (max count, LOD, streaming) are undocumented → **bake-off first** on real 20M scenes.
3. **WebGPU answer: adopt SuperSplat's *ideas*, reject its *runtime*.** WebGPU-in-Safari on visionOS is real (Safari 26.2, Dec 2025) but **VR-only (no passthrough)** and slower than native. Port PlayCanvas's algorithms (GPU radix sort, cull/project compute, SOG/SPZ compression, LOD streaming) into Metal; keep SuperSplat as a desktop authoring tool only.

---

## What the current code does (verified against the clone)

| Aspect | Current implementation | File |
|---|---|---|
| Splat encoding | 32 B/splat: fp32 position, fp16 SH0 RGBA, fp16 packed 3D covariance | `EncodedSplatPoint.swift:11-30` |
| Higher-order SH | separate per-chunk fp16 side buffer, evaluated per-vertex every frame (SH3 ≈ +90 B/splat ≈ 1.8 GB @ 20M) | `SplatChunk.swift:23-28`, `SplatProcessing.metal:23-74` |
| **Depth sort** | **CPU, single-threaded `Array.sort`, full re-sort on every pose change, drawn stale** | **`SplatSorter.swift:600-628`** |
| Rasterization | 2-tri quad per splat, 1024 indexed + instanced; single-stage over-blend + multi-stage imageblock (alpha-weighted depth in `raster_order_group(0)` for clean VP reprojection) | `SplatRenderer.swift:849`, `MultiStageRenderPath.metal` |
| visionOS path | CompositorServices (not RealityKit), single-pass stereo via vertex amplification, system foveation passthrough | `SampleApp/Scene/VisionSceneRenderer.swift` |
| Not used | GPU sort, MetalFX, mesh/object shaders, ICB/indirect draw, residency sets, quantized positions, LOD/culling | — |

**Top bottlenecks at 10–20M on VP:** (1) CPU sort throughput + latency; (2) overdraw / fragment-blend bandwidth; (3) ~640 MB positions + ~1.8 GB SH3 + 3× index buffers, no LOD, in 16 GB shared.

---

## Verified 2026 facts (sources)

- **visionOS 27 `GaussianSplatComponent`** — `BufferResource(count:position:scale:rotation:opacity:sphericalHarmonics:(buffer,degree)) → GaussianSplatResource → GaussianSplatComponent`. "Removes the need for developers to implement splat rendering themselves." vOS27 dev-beta now, GA fall 2026. Internals undocumented. — WWDC26 session 279.
- **M5 Vision Pro** — 10-core GPU w/ per-core Neural Accelerators, **hardware ray tracing** (first VP with RT) + mesh shading, **16 GB**, up to **120 Hz**, +10% pixels. ~153.6 GB/s theoretical / **~122 GB/s measured** bandwidth. — Apple newsroom + M5 roofline analysis.
- **WebGPU on visionOS** — Safari 26.2 (Dec 12 2025) shipped WebXR+WebGPU; but WebXR on visionOS is **immersive-VR only (no passthrough)**. — WebKit blog + Apple forums.
- **PlayCanvas 2.19.0 (Jun 2026)** WebGPU compute splat renderer (cull/project + GPU radix sort): up to **~5.7× FPS** (35M: 13→76 fps on M4 Max). **SOG ~15-20×**, **SPZ ~10×** (full SH). — PlayCanvas blog, Niantic SPZ.
- **GPU sort on Apple Silicon** — use the **portable multi-pass 4-bit radix** (DeviceRadixSort style), **NOT OneSweep** (its decoupled-lookback needs forward-progress guarantees Apple Silicon doesn't expose). simdgroup primitives let native beat the web's 4-bit path. — Linebender wiki, NVIDIA OneSweep paper.
- **MetalFX** — upscaling **cannot** consume a foveated CompositorServices drawable (use CompositorServices **Dynamic Render Quality** instead). Frame Interpolation exists (Metal 4) but its use inside the foveated immersive path is **unverified** — test before relying on it for 120 Hz.

---

## Decision: native Metal, ideas from SuperSplat

Port SuperSplat's **algorithms** to native Metal; keep the proven CompositorServices stereo+foveation path. Reject the WebGPU runtime (VR-only, browser overhead, slower sort on Apple Silicon). Keep SuperSplat/PlayCanvas as a **desktop authoring tool**; use **SOG/SPZ as interchange**.

## Target architecture (GPU-driven, 3-pass)

- **Pass 0 — cull+project compute:** frustum/behind-camera/sub-pixel reject in a pre-pass (move today's per-vertex rejects out of the vertex stage); atomic survivor counter → GPU-driven indirect draw, so disabled/culled chunks cost ~0.
- **Pass 1 — GPU radix sort:** replace `SplatSorter.performSort`'s `Array.sort` with a Metal threadgroup/simdgroup **4-bit radix** over a **monotonic float→uint depth key (sign-bit flip)**. Keep the 3-deep index ring + Mutex/generation orchestration. Per-frame **and amortized** (skip when camera delta is sub-threshold).
- **Pass 2 — raster:** keep both existing pipelines + the 1024-indexed/instanced quad trick + single-pass stereo amplification.
- **LOD/streaming:** bottom-up spatial LOD tree, lowest-LOD-first, tree-cut per camera against a per-device budget — the only way 20M *total* fits in 16 GB.
- **Compression:** SPZ now (already a dependency), then SOG decode in-shader (palette-quantized higher-order SH).
- **Memory/throughput:** quantize positions fp32→fp16; `MTLResidencySet`/`useHeap` instead of the per-frame per-chunk `useResource` loop.
- **Frame rate:** CompositorServices Dynamic Render Quality as the throttle; evaluate (don't assume) MetalFX Frame Interpolation for 120 Hz.

## Phased plan (cheapest-path-to-value ordering)

- **Phase −1 — Native bake-off (do FIRST, ~2-3 days):** thin visionOS-27 `GaussianSplatComponent` viewer; load the real 20M hero scenes (S2000, Kenny iPhone, Clarte) on the M5 VP; measure fps/RAM/quality. **If Apple's native renderer handles them, we may be done** (or it becomes the vOS27 path with MetalSplatter as the vOS26 fallback).
- **Phase 0 — correctness + cheap wins (2-4 days):** fp16 positions; SPZ end-to-end; per-eye depth keys; fix the latent vertex-amplification clamp off-by-one (`kMaxViewCount` → `kMaxViewCount-1`); replace render-thread `Thread.sleep` busy-waits.
- **Phase 1 — GPU depth sort (the headline fix, 2-3 wk):** Metal 4-bit radix replacing `SplatSorter.swift:628`; cull+project pre-pass + indirect draw.
- **Phase 2 — compression + memory (1-2 wk):** SOG decode; residency sets; quantized SH.
- **Phase 3 — LOD streaming (2-3 wk):** spatial LOD tree + budgeted streaming → 20M total.
- **Phase 4 — frame-rate + final bake-off (1-2 wk):** Dynamic Render Quality; test Frame Interpolation; re-benchmark native vs custom.

## Open gating questions (resolve before Phase 0)

1. **Passthrough?** Must splats composite over real-world passthrough, or is full-immersive-VR fine? Passthrough + RealityKit-registered content pushes hard toward the **native** `GaussianSplatComponent` (custom CompositorServices Metal doesn't co-exist with RealityKit content trivially). VR-only keeps the custom-Metal ceiling option open.
2. **Min OS — 27-only or support 26?** 27-only unlocks the native component as a shortcut; 26 support makes the custom Metal pipeline mandatory.
3. **Standalone viewer vs integrated into Drone Harvest tabletop?** Standalone full-immersive viewer for hero scenes is the simplest; integration with the RealityKit tabletop is the harder, passthrough-bound case.
4. Real drone-scene capture profile (total count, SH degree, anisotropy) → how aggressive LOD/quantization must be; validate SOG's lossy SH on real scenes (no PSNR numbers yet).

## Caveats (don't treat as proven)

- The **6-10M effective / 20M total** budget and **120 Hz via frame interpolation** are engineering estimates — no public M5-VP splat benchmark exists. Validate empirically.
- Model a real resident-memory table (OS reserve + dual 120 Hz framebuffers + index ring + work buffers + LOD overhead), not just raw splat bytes.
- Add a **thermal/battery degradation ladder** (`ProcessInfo.thermalState`) — the M5 VP runs ~2.5-3 h; sustained per-frame GPU sort + heavy overdraw at 10M+ is a worst-case load, so steady-state budget < cold-start ceiling.
