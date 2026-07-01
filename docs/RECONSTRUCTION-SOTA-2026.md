# Reconstruction SOTA 2026 — Re-shooting the S2000 into a Clean, Rasterizer-Loadable Splat

**Status:** decision doc. Source: four parallel field surveys (3DGUT/3DGRT caveat verification; "what does Lichtfield use"; latest SOTA filtered by rasterizer-survival; practical solo-creator Windows-5090 pipelines), reconciled against one mechanism test where they disagreed.
**Date:** 2026-06-21.
**Audience:** Bryan, deciding how to re-shoot and re-process the S2000 car scan so it loads cleanly in a **rasterizing** viewer — the MetalSplatter fork (`.ply`/`.spz`, EWA rasterizer + GPU radix sort) or the native visionOS 27 `GaussianSplatComponent` (buffer-only: position/scale/rotation/opacity/SH).

> **The car is glossy. The viewer rasterizes. That pair is the whole problem.** Every "reflective specialist" 3DGS paper wins by swapping the rasterizer for a render-time shader. Those gains die the instant a plain rasterizer loads the PLY. So the doc is organized around one question, applied to every method.

---

## 0. The one-paragraph answer

Re-shoot the car well, fix orientation **at the SfM/alignment stage** (not in the viewer), and train a **standard 3DGS** model — because that is the only thing your two viewers can render. The single test that sorts the entire field is: **where does the appearance gain live?** If it lives in the standard Gaussian + spherical-harmonics (SH) PLY, it survives in MetalSplatter. If it lives in a *renderer-side structure* — a learned environment cubemap, an anisotropic-spherical-Gaussian field, a deferred-shading pass, a 2D-surfel primitive, or a ray-traced secondary-ray path — it does **not** survive, no matter how good the paper's video looks. Apply that test and the noise collapses: **3DGUT** is the right lead/baseline (best rasterizer, robust to capture distortion, fast on Blackwell, exports a normal PLY) but it is **not a reflection fix** — for reflections specifically it is a camera-distortion fix plus a cleaner base. **3DGRT** and all the reflective specialists (3DGS-DR, GaussianShader, Ref-Gaussian, Spec-Gaussian, Ref-DGS, SSR-GS…) are **disqualified** for your viewer because their reflections are render-time effects. The honest meta-answer: **as of mid-2026, no method that survives a vanilla SH-only PLY loader delivers true view-dependent mirror reflections.** (Stated precisely, because it is not that a *rasterizer* cannot do better — SH-replacement methods that stay inside the sortable rasterization pipeline, e.g. Spherical Voronoi, SG-Splatting, Spec-Gaussian's ASG, push the directional-frequency ceiling *above* degree-3 SH and get sharper specular lobes. They still fail the viewer-survival test because they need a custom evaluator / non-SH channels. So the honest ceiling is "what a vanilla SH-only loader can show," not "what a rasterizer can do.") You can have a clean, sharp, correctly-oriented car body with SH-grade view-dependent shimmer; you cannot have a chrome mirror finish in MetalSplatter. Plan the re-shoot to *minimize* the smear (polarizer, even lighting, dense coverage) rather than to *reproduce* the mirror.

---

## 1. The 3DGUT question, resolved head-on

**Is 3DGUT the right lead for a reflective car in a rasterizing viewer, or just a camera-distortion fix?**

**Both, precisely:** it is the right *lead/baseline*, AND for reflections specifically it is effectively only a camera-distortion fix. State the caveat plainly so it never gets misread again:

- **What 3DGUT (3D Gaussian Unscented Transform, NVIDIA, CVPR 2025) actually does:** it replaces EWA-splatting's linearized projection with the Unscented Transform — sigma points projected *exactly* through any nonlinear camera model. That buys robust handling of **distorted / time-dependent cameras**: fisheye, wide-FOV, rolling shutter. It **stays a rasterizer** (fast), and it trains a **standard Gaussian + SH representation**. The unscented transform lives in the *projection*, not in the representation — so the exported `.ply` loads in MetalSplatter and the native `GaussianSplatComponent` like any vanilla 3DGS file.
- **The trap.** The paper's title is "Enabling Distorted Cameras *and* Secondary Rays," and NVIDIA's marketing says 3DGUT "enables reflections and refractions." That reflection capability is **not** something 3DGUT does as a standalone rasterizer. It comes from *aligning* the representation with the ray tracer so secondary rays can be traced by **3DGRT** — the hybrid "3DGRUT" path (primary rays rasterized by 3DGUT, secondary rays ray-traced by 3DGRT). **The reflection gains are a ray-trace-time effect.** Apply the §0 test: those reflections live in the ray-traced path, not in the SH PLY → they **do not survive** rasterization.
- **Therefore, in MetalSplatter, a 3DGUT PLY carries the same SH view-dependent appearance model as vanilla 3DGS.** Any reflection improvement you see is *incidental* (cleaner base, less geometric smear because the camera model was right), not *mechanistic*. Do not read "enables reflections" as "fixes the paint in MetalSplatter."

**Verdict:** Lead with 3DGUT as the clean baseline. Do **not** expect it to render the car's reflections — it has the same ~degree-3 SH ceiling as vanilla 3DGS. No method that survives a vanilla SH-only PLY loader does better on true reflections in mid-2026; that is a viewer-survival limitation, not a 3DGUT shortcoming. (SH-replacement methods can beat the SH ceiling, but only with a custom evaluator your viewer doesn't have — see §0 and §5.)

**Where the surveys disagreed, and the ruling:**
- One survey rated 3DGUT `reflectiveCarFit: high` and said it "natively handles secondary rays within rasterization." **Overclaim — overruled.** Three of four surveys, plus the verified arXiv:2412.12507 abstract and the nv-tlabs/3dgrut README, are careful: in a rasterizer 3DGUT = standard SH appearance.
- One survey rated **3DGS-DR** `reflectiveCarFit: high / exports: yes` ("bakes specular into the rasterizable model — gains survive"). **Overruled.** 3DGS-DR's reflections are computed at render time from a *learned environment cubemap + a deferred shading pass*; a standard SH-PLY carries neither, and neither viewer can interpret them. Apply the §0 test → the mirror vanishes, you get a dull/baked body. The same survey downgraded its own claim to "unproven." **3DGS-DR is a trap, not a fix, for this viewer.**

---

## 2. The "Lichtfield" verdict — USE IT

**"Lichtfield" is your vendored clone of LichtFeld Studio** (German "Lichtfeld" = light field) — the open-source C++23/CUDA 3DGS trainer/editor from MrNeRF (Janusch Patas), formerly `gaussian-splatting-cuda`. This is well-evidenced, not confabulated:
- The cairn directory scan recorded its README opening as `<div align="center"><picture>` — the exact opening of the LichtFeld-Studio README.
- Cairn intake independently names "LichtFeld v0.5.2" and recorded a **live training run on your hardware** (PID, ~7 GB VRAM, training `vector_rs/A_lidar_tuned`). Per CLAUDE.md, "proven" = verified by use — this clears that bar.
- STATE-OF-CAIRN lists Brush and LichtFeld as **distinct** tools, so this is not the Brush trainer.

The OPS_MAP "LLM-regenerated, low value, wipe" label was your own uncertainty about a folder you'd lost track of — the underlying tool is real and current. It bundles real published strategies: **MCMC densification** (3DGS-MCMC, NeurIPS 2024 — the everyday default), **bilateral-grid appearance correction**, its own LFS-densification/PPISP refinements, and — the relevant one — a **3DGUT path**. Input: COLMAP / Nerfstudio datasets. Output: PLY / SOG / SPZ — all of which load in a standard rasterizing viewer. (Upstream LichtFeld-Studio is confirmed to support MCMC + bilateral grid + 3DGUT + pose-opt and PLY/SOG/SPZ export; only the *compiled feature set of your local 051226 snapshot* is unverified — hence the blocking check below.)

**Verdict: USE IT.** It is free, GPLv3, CUDA-12.8 / C++23 (the Blackwell sm_120 toolchain), and already proven on your 5090. **One blocking check before you rely on it for the car:** you (and the surveys) could not read the `LichtfieldStudio051226\` directory from the Mac, so the exact strategy set compiled into *that snapshot* is inferred from upstream v0.5.2. **Open the app's strategy/optimizer selector and confirm the 3DGUT path is actually present** before committing a 3DGUT run. If it isn't, either pull a fresh LichtFeld build or fall to the turnkey #2 pick below — your default MCMC strategy is definitely there regardless.

---

## 3. Top picks to TEST for the S2000, ranked

All three share the same **bookends**, which every survey agreed on and which are not in contention:
- **Front (alignment + orientation fix): RealityScan 2.x.**
- **Back (cleanup + delivery): SuperSplat → SPZ.**

The only real decision is the **trainer** in the middle. Ranked for *best bang, solo creator, on a 5090*:

### #1 — LichtFeld Studio ("Lichtfield"), MCMC baseline → 3DGUT if compiled
*The pragmatic best-bang pick: already proven on your exact hardware, free, no paywall.*

- **Why:** It is the one trainer cairn shows **training live on your 5090** — zero setup risk, the most expensive unknown already retired. Free and GPLv3 (no PLY paywall, unlike Postshot). Bundles the two strategies you want: **MCMC** (cleanest, most artifact-forgiving for a *bounded* object like a car) as the everyday baseline, and a **3DGUT** path for the distorted-camera robustness if your snapshot has it. Consumes COLMAP/Nerfstudio data, so it ingests a RealityScan→COLMAP export directly. Exports PLY **and** SPZ natively.
- **Repo:** https://github.com/MrNeRF/LichtFeld-Studio (your clone: `C:\Users\bryan\Documents\Lichtfield\LichtfieldStudio051226\`).
- **5090/Windows:** Targets CUDA 12.8+ / C++23 = exactly the Blackwell (sm_120) toolchain. Honor the OPS_MAP gotcha: **build/run against CUDA 12.x, not cu13.** Proven training live on your box. Native Windows + Linux.
- **Viewer-loadable export:** **YES — confirmed.** Standard Gaussian+SH PLY and SPZ; both load in MetalSplatter and the native `GaussianSplatComponent`. (3DGUT's unscented transform is projection-side only — representation stays standard.)
- **Effort:** **Low-to-medium.** App already installed and proven; main cost is feeding it a good RealityScan alignment and confirming the strategy selector. **Blocking check:** verify the 3DGUT strategy is actually compiled in the 051226 snapshot before counting on it; MCMC is the safe fallback that is definitely present.

### #2 — Jawset Postshot (RealityScan import → MCMC profile)
*The zero-uncertainty turnkey pick if you want to skip all build/strategy questions.*

- **Why:** Most turnkey trainer in the field. Self-contained CUDA desktop app — **no from-source PyTorch**, so it sidesteps the sm_120 wheel pain entirely. Imports a RealityScan/COLMAP alignment (far higher quality than its internal solve), then trains **Splat3** or **3DGS-MCMC** profiles with simple parameter control. MCMC is the cleanest choice for the bounded reflective car.
- **Repo / site:** https://www.jawset.com/ (overview: radiancefields.com/platforms/postshot).
- **5090/Windows:** Windows-only, NVIDIA required. Self-contained CUDA app → architecturally expected to run on Blackwell/50-series. Vendor floor is **Compute Capability 7.5+ = RTX 2060-class** (per jawset.com system requirements), not 3060. Exact 5090-tested confirmation = unverified but low-risk.
- **Viewer-loadable export:** **YES** — standard rasterizing-3DGS PLY (loads in both viewers). Two caveats: **PLY export is gated behind the paid Indie tier (~€17/mo)**, and a **named "3DGUT" profile is unconfirmed** (it advertises Splat3 + MCMC). For SPZ, run the PLY through SuperSplat.
- **Effort:** **Low.** Most forgiving setup of any option. Cost is the license for PLY export.

### #3 — nerfstudio + gsplat with 3DGUT (`--with_ut --with_eval3d`)
*Highest reflective/distortion quality ceiling; worst ergonomics. Only if #1 and #2 leave the paint visibly worse than you can accept.*

- **Why:** The strongest *open* distortion-/specular-aware trainer. gsplat integrates NVIDIA 3DGUT and is actively extended through 2026 (March 2026 added windshield-style external distortion; PPISP/bilateral-grid color compensation). Trains standard Gaussians+SH → standard PLY. Still bound by the §0 test: the unscented transform is projection-side, so the PLY rasterizes fine; you only lose the live distortion-correction nuance, which is irrelevant for a viewer-facing car.
- **Repo:** https://github.com/nerfstudio-project/gsplat (docs/3dgut.md). Strongest verified Blackwell story upstream (CUDA 12.8.1 support contributed by @johnnynunez in nv-tlabs/3dgrut; Docker build).
- **5090/Windows:** Runs on 5090. Two wheel facts that are easy to conflate — keep them separate:
  - **PyTorch side:** torch 2.7+ ships sm_120 (Blackwell) wheels (2.11 current), so the *torch* dependency is a clean `pip install`, not a build.
  - **gsplat side:** gsplat's own prebuilt-wheel index (`docs.gsplat.studio/whl`) currently tops out at **torch 2.0–2.4 / cu118-cu121-cu124 and does NOT list sm_120**. So a precompiled gsplat wheel for your Blackwell torch combo likely does **not** exist today. The realistic paths, cheapest first: (1) **let gsplat JIT-compile its CUDA kernels at install/first-run** — works if a full CUDA 12.8 toolkit is present, no full source build; (2) **`pip install` from source** as the fallback. NVIDIA's gsplat/3DGUT blog does not promise a prebuilt sm_120 wheel, so **verify your exact python/torch/CUDA against the gsplat wheel index before assuming either path.**
- **Viewer-loadable export:** **YES** — standard Gaussians+SH PLY loads in both viewers. **Distinct from 3DGRT** (ray-traced particles, needs its own RT renderer, does NOT load cleanly — avoid).
- **Effort:** **Medium-high, verify-first** (downgraded from "High / from-source on WSL2"). The pain is the gsplat JIT/from-source step on a Blackwell toolchain, *not* PyTorch itself. If gsplat JIT-builds cleanly against your CUDA toolkit it's medium; if you fall to a full source build on WSL2 it's high. Still the worst ergonomics of the three — only reach for it if #1 and #2 leave the paint visibly worse than you can accept.

> **Ruled out on purpose — 3DGRT and the reflective specialists.** 3DGRT (NVIDIA 3D Gaussian Ray Tracing), 3DGS-DR, GaussianShader, Spec-Gaussian, Ref-Gaussian, Ref-DGS, SSR-GS, IRGS, EVER, and the 2026 reflective wave all fail the §0 test: their appearance gain lives in a render-time path (ray tracing, env cubemap, ASG field, deferred PBR, 2D-surfel primitive). The PLY may load, but the reflections you trained for are gone — you get a dull or smeared body. 2DGS/GOF additionally emit 2D-surfel primitives a standard 3DGS rasterizer can't read at all. Revisit **only** if you later ship a ray-tracing viewer (the M5 VP has hardware RT, but no viewer-loadable export path exists today).
>
> **Current through mid-2026 (newer than everything above) — same verdict:**
> - **Spherical Voronoi** (arXiv:2512.14180, Dec 2025) — the canonical "rasterization-compatible but NOT vanilla-viewer-loadable" case, and the strongest SH-killer in the rasterizer family. It *replaces* SH (its abstract: SH "struggle with high-frequency signals, exhibit Gibbs ringing, and fail to capture specular reflections") with per-Gaussian Voronoi sites/region values, and pairs them with learnable Voronoi light probes in a **deferred 2D-Gaussian pipeline**. No ray tracing — yet it still needs a custom per-view evaluator and non-SH channels, so a vanilla SH-only PLY loader (MetalSplatter / `GaussianSplatComponent`) cannot display it. This is the concrete 2026 instance of the "ASG-field" category the table only gestured at — it cleanly fails the §0/appendix test.
> - **ARS-GS** (Anisotropic Reflective Spherical 3DGS, *J. Imaging* 2026) — ASG reflection + SH diffuse approximation inside a PBR pipeline. Same fate as Spec-Gaussian/GaussianShader: custom PBR shader + non-SH channels → does not survive.
> - **PolarGuide-GSDR** (arXiv:2512.02664, CVPR 2026) — polarization-prior-driven deferred-reflection variant of 3DGS-DR. Still deferred shading → disqualified. *Notably it validates the §4 capture advice:* it leans on polarization to separate reflections, echoing "use a circular polarizer at capture."

---

## 4. End-to-end recipe for the #1 pick (LichtFeld Studio)

**Capture → align (orientation fixed here) → train → PLY/SPZ → Vision Pro.**

**Step 1 — Re-shoot (this is where you beat the glossy paint).**
- 150–300 evenly-spaced photos in a full orbit around the car, plus a second higher/lower ring for top and rocker coverage.
- **Use a circular polarizer if you can** — it physically cuts blown specular highlights at capture time. This is the single most effective reflective-surface mitigation available, because it removes the smear *before* training rather than asking the renderer to reproduce it. (Independent corroboration: the 2026 PolarGuide-GSDR paper leans on polarization priors specifically to separate reflections on real-world glossy scenes — the research is converging on "polarization at capture" as the right lever.)
- **Even, diffuse, unchanging lighting.** Overcast or a large softbox. Do not let highlights move between shots — moving highlights are what 3DGS bakes as floaters/smear.
- Clean, static, textured background (helps SfM); mask the car later in SuperSplat.
- Keep the car and lighting fixed; move only the camera.

**Step 2 — Align in RealityScan 2.x, and FIX ORIENTATION AT THE SOURCE.**
- Import photos → align. RealityScan 2.0's GPU-accelerated solve + default high-quality feature detection gives tighter alignments and fewer disjoint components (less ghosting). Use the **Quality Analysis coverage map** (green→red) to spot capture gaps *before* you train — re-shoot thin spots now, not after a failed train.
- **The S2000 orientation bug was a pose/SfM problem, not a renderer problem.** Fix it here: set the **reconstruction region upright**, and toggle **"Use camera priors for georeferencing" OFF**, then re-align so the model comes out upright. This bakes a correct orientation into the exported poses instead of you rotating the splat in the viewer forever.
- Export the alignment as **COLMAP** format (poses + sparse point cloud) — this is what LichtFeld ingests.

**Step 3 — Train in LichtFeld Studio on the 5090.**
- Load the COLMAP dataset (poses + PLY point-cloud init).
- **Confirm the strategy selector** (the blocking check from §2). Pick **MCMC** as the baseline run — most artifact-forgiving for a bounded object, capped splat budget = cleaner result. If the **3DGUT** path is present and your capture had any wide-FOV/rolling-shutter character, run a second pass with 3DGUT for the distorted-camera robustness.
- Train to convergence. Build against **CUDA 12.x (not cu13)**.
- Accept up front: the paint will show SH-grade view-dependent shimmer, not a true mirror. That ceiling is the field, not the tool.

**Step 4 — Clean up + deliver in SuperSplat.**
- Load the trained PLY in SuperSplat (browser, free, GPU-agnostic — runs on the 5090 box).
- **Crop the car out of the scene; delete background and floaters** — reflective paint produces floater/smear artifacts, and deleting them is the highest-leverage cleanup you have. Recenter/orient if needed (though Step 2 should have orientation right already).
- Export **SPZ** (Niantic, MIT, ~90% smaller than PLY, preserves SH) for Vision Pro delivery — both your viewers need the SH, and SPZ keeps it.

**Step 5 — Load on Vision Pro** via the MetalSplatter fork (`.ply`/`.spz`) or the native `GaussianSplatComponent`. The model is upright (Step 2), clean (Step 4), and standard-SH (Step 3) → it renders.

---

## 5. Reflective-surface handling — what each method actually does IN A RASTERIZER

The car's glossy paint and chrome are the hard part. Be explicit, because every method's *paper* implies a fix that *your viewer* throws away.

| Method | Reflection mechanism | In MetalSplatter / `GaussianSplatComponent` |
|---|---|---|
| **Vanilla 3DGS / MCMC** | View-dependent color via ~degree-3 SH per Gaussian | Survives. SH shimmer only — no true mirror. This is the realistic ceiling. |
| **3DGUT** | Same SH appearance; unscented transform is projection-side | Survives the *geometry/distortion* gain; reflection is **identical to vanilla 3DGS**. Cleaner base → less smear, but **not a reflection mechanism**. |
| **Mip-Splatting / Taming-3DGS / AbsGS / RaDe-GS** | No reflective specialization; anti-alias / densification / geometry | Survives — sharper edges, fewer floaters, better normals. Reflections still SH-ceilinged. Safe quality baselines, not reflective fixes. |
| **3DGRT** | Ray-traced secondary rays at render time | **Lost.** Rasterized, it behaves like an ordinary SH splat. Gains require the RT renderer. |
| **3DGS-DR** | Learned env cubemap + deferred shading pass | **Lost.** PLY loads, mirror vanishes → dull/baked body. The archetypal trap. |
| **GaussianShader** | Per-Gaussian BRDF / shading-function channels read by a custom shader | **Lost.** A standard SH rasterizer ignores the extra channels; highlights collapse. (Spec-Gaussian's anisotropic-SG channels fail the same way — see the SH-replacement row below.) |
| **Ref-Gaussian / SSR-GS / IRGS** | Deferred PBR / split-sum + Mip-cubemap (Ref-Gaussian, SSR-GS); **2D-Gaussian ray tracing** (IRGS, by name) | **Lost** (often triple-disqualified: render-time shading + non-SH material + 2D primitive / RT path). |
| **Ref-DGS** | Deferred shading + learnable **Sph-Mip** env/reflection field + dual Gaussian sets + a specular-mixing shader — **no explicit ray tracing** (rasterization-based) | **Lost.** Rasterizes, but the env/reflection field + custom mixing shader + the second Gaussian set carry the specular — a vanilla SH PLY has none of it. (Note: it is *not* ray-traced — that was a mislabel; its disqualifier is the deferred env-field + custom shader, which is just as fatal for an SH-only viewer.) |
| **Spherical Voronoi / SG-Splatting / Spec-Gaussian (ASG)** | *Replace* SH with a higher-frequency directional model (Voronoi sites + light probes / spherical-Gaussian / anisotropic-SG), evaluated per-view by a custom shader | **Lost** — even though they beat the degree-3 SH ceiling and stay in the sortable rasterization pipeline. The directional channels are non-SH and need a custom evaluator; a vanilla SH-only PLY loader can't read them. The canonical "rasterizer-compatible but NOT vanilla-viewer-loadable" trap. |
| **Car-GS** | *Removes* specular as noise for clean geometry/mesh | Inverse of the goal — strips reflections rather than reproducing them. Useful only if you want car body geometry/mesh. |

**The practical reflective playbook for a rasterizer (mid-2026):**
1. **Fix it at capture, not at render.** Polarizer + even, static lighting + dense coverage. This is the only lever that actually reduces smear in the final rasterized splat.
2. **Pick an artifact-forgiving trainer.** MCMC (capped budget) + clean RealityScan poses give the least floater/smear.
3. **Delete the smear in SuperSplat.** Floater cleanup is cheap and high-impact.
4. **Accept the SH ceiling — defined precisely.** No method that survives a *vanilla SH-only PLY loader* delivers true view-dependent mirror reflections today. SH-replacement methods (Spherical Voronoi, SG-Splatting, Spec-Gaussian ASG) genuinely beat degree-3 SH while staying in the rasterization pipeline — but they need a custom evaluator your viewer doesn't have, so for *your* loader the SH ceiling still holds. Don't burn a week chasing a specialist paper whose gain your viewer cannot display.
5. **Revisit only if you build a ray-tracing viewer.** The M5 VP has hardware RT; if a native RT splat renderer ever ships, 3DGRT/3DGRUT becomes worth re-evaluating. Until then, rasterizer-survival is the gating question for every new release.

---

## Appendix — the one test for any future release

> **Where does the appearance gain live?**
> - In the standard Gaussian + SH PLY → **survives** in MetalSplatter / `GaussianSplatComponent`.
> - In a renderer-side structure (env cubemap, ASG field, deferred-shading pass, 2D-surfel primitive, ray-traced path, **or a non-SH directional model needing a custom evaluator — e.g. Spherical Voronoi sites/probes, spherical-Gaussian channels**) → **does NOT survive**.

The sharpest version of the test: **does it survive a vanilla SH-only PLY loader?** Note that "rasterization-based / no ray tracing" is **not** sufficient — Spherical Voronoi (2512.14180) and Ref-DGS (2603.07664) are both rasterizers with no explicit ray tracing, yet both are disqualified, because the gain lives in non-SH channels + a custom shader. Rasterizer-survival ≠ viewer-survival; SH-loader-survival is the real bar.

Apply it to every "reflective 3DGS" paper before getting excited. It resolves every contradiction in the 2026 field and saves you from training a model whose headline feature your viewer silently discards.
