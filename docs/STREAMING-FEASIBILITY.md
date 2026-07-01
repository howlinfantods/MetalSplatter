# Streaming Feasibility: Render Splats on the 5090, Stream to the M5 Vision Pro

> **Question:** The native MetalSplatter viewer runs poorly on the M5 Apple Vision Pro at 3M splats and worse at 16M. Is rendering on the RTX 5090 and *streaming* the result into the Vision Pro — "like the iRacing plugin" — a viable path?
>
> **Author:** XR streaming architecture review · **Date:** 2026-06-21 · **Status:** decision brief
> **Confidence convention:** `[proven]` = verified by use; `[high]` / `[medium]` / `[low]` = web-grounded but unverified; `[vendor]` = vendor-stated, treat as marketing until measured.
> **Bar:** Bryan's "proven = verified by use." Nothing in the streaming path is `[proven]` yet. That single fact drives the verdict.

---

## VERDICT: CONDITIONAL GO

Streaming the splat viewer from the 5090 into the Vision Pro is **architecturally viable and the pipe is now genuinely open** — but it is **not a download-and-go win, and nothing in it is proven by use yet.** Two things are true at once:

1. **The transport is a near-solved problem.** As of GTC 2026, NVIDIA CloudXR 6.0 shipped a **public Swift client framework** plus a working sample app. You can build a custom visionOS streaming client. This is a real change since the vault doc was written — see "Stale vault assumption" below.
2. **The renderer is the actual project.** Nothing off-the-shelf renders a 16M-splat scene at VR framerate under OpenXR on the 5090 today. That is the part you'd be building, and it does **not** reuse your MetalSplatter (Metal) sort work.

So the verdict is **CONDITIONAL**, not GO, gated on one cheap proof:

> **The discriminator (do this first):** stand up an **off-the-shelf OpenXR splat viewer (Splatapult) under SteamVR on the 5090, streamed to the AVP via ALVR**, with a small scene (S2000 / 3M). If that holds a stable, comfortable image, streaming is proven-by-use for *this hardware* and you can commit to the polished CloudXR build. If it stutters or the renderer can't take the scene, you have your answer for one afternoon's effort instead of a multi-week build.

**Recommended pipe:** **CloudXR 6.0** for the eventual real app (foveated streaming is the decisive advantage for a high-pixel-count splat scene); **ALVR for the throwaway spike** (zero custom client code). See the architecture and effort sections.

**Scene-size call (the part you actually asked for):**

| Scene | Splats | Native on M5 verdict | Streaming verdict |
|---|---|---|---|
| **S2000** | ~3M | **Fix natively.** Within reach of the MetalSplatter overhaul (GPU radix sort + cull-before-sort). Don't stream it. | Overkill; adds latency for no benefit. |
| **KennyRoom** | ~16M (3.6GB PLY / 177MB SOG) | **Hits the M5 bandwidth wall** (see math below). Native is a hard, possibly losing, LOD/streaming fight. | **This is the scene that justifies streaming.** The 5090's ~15× memory-bandwidth headroom renders it without the wall. |

The threshold sits somewhere between 3M and 16M. If the MetalSplatter LOD work lands, native may stretch to ~5–8M sortable. Above that, streaming is the cleaner answer.

---

## Why CONDITIONAL and not GO (against the "proven" bar)

Four unverified links, each individually plausible, none yet `[proven]`:

1. **GeForce-5090-as-CloudXR-server** is NVIDIA-*stated*, not confirmed-by-you. NVIDIA's page says CloudXR runs on "RTX PRO workstations to GeForce RTX GPUs — on PCs or in the cloud" `[vendor]`, and NVIDIA's own X-Plane materials name the **GeForce RTX 5090** as the card doing the lifting `[vendor]`. The CloudXR *docs* still list "RTX 6000 Ada or higher" as the server spec `[medium]` — that reads as the recommended/cloud tier, not a hard GeForce lockout, but **you have not run the runtime on your 5090.** This is the highest-value thing to verify early.
2. **No independent CloudXR latency number exists.** iRacing Connect's pipe is purpose-built and *should* beat ALVR's ~68ms, but every number is vendor-stated `[vendor]`. The one real measurement in the vault is **ALVR: ~108ms out of the box, ~68ms tuned** (MacRumors user) `[medium]`.
3. **visionOS 27 beta compatibility is unconfirmed.** The CloudXR framework targets visionOS 2.4+ and foveated streaming needs 2.6.4+ `[high]`; Bryan is on the **vOS27 beta**. Frame Sixty (a studio that actually built a CloudXR AVP app) reports it works across the 26→27 transition and gaze stays on-device in 27 `[medium]` — encouraging, but beta is beta.
4. **The OpenXR splat renderer on the 5090 does not exist in production-ready form.** Existing OpenXR splat viewers are research-grade (below).

Flip any one of these from `[vendor]`/`[medium]` to `[proven]` and confidence rises. The spike flips #1, #2, and partially #4 in an afternoon.

---

## The five core questions, answered

### Q1 — Is the CloudXR client SDK actually available for a *custom* visionOS app, or did iRacing get privileged access?

**Available to any developer. iRacing was first, not privileged.** `[high]`

CloudXR 6.0 (announced GTC 2026) is a **universal OpenXR-based streaming runtime** with a **public Swift client framework on GitHub**:

- **`NVIDIA/cloudxr-framework`** — the Swift framework (`CloudXRKit`) for building visionOS/iOS clients. Add it via Xcode package dependencies. `[high]`
- **`NVIDIA/cloudxr-apple-generic-viewer`** — a complete sample visionOS client (6DOF head+hand tracking, audio, data channels). Clone, open `CloudXRViewer.xcodeproj`, add the framework, build for visionOS. Described by NVIDIA as "a complete reference implementation for developers building CloudXR client applications on Apple platforms." `[high]`
- Xcode ships a **multi-platform Foveated Streaming template** as a starting point. `[high]`
- **Foveated Streaming** is an Apple *system* framework baked into visionOS 26.4+ (no download); gaze data never leaves the headset. `[high]`

iRacing Connect was simply the **flagship launch app**; X-Plane 12 and Autodesk VRED followed via the same public pipe. There is no evidence of a privileged/partner-only tier for the client framework. `[high]`

### Q2 — The ALVR path: render as an OpenXR/SteamVR app on the 5090 and stream via ALVR?

**Yes, and it's the right tool for the throwaway proof — not the shippable product.** `[medium]`

- ALVR is on the AVP App Store and is the de-facto generic SteamVR→AVP bridge (vault, `[high]`). Setup: install ALVR server on Windows, ALVR client on AVP, run any SteamVR/OpenXR app on the 5090, it streams.
- **The dev story is "free and fiddly."** ALVR gives you generic SteamVR with no custom client work — perfect for testing whether an existing OpenXR splat viewer survives the pipe.
- **Splat-specific gotchas:**
  - **Stereo cost doubles everything.** A splat viewer must render + depth-sort for *two* eyes per frame. Sort is the bottleneck; doing it twice (or once with a shared sort + reprojection) is the core perf question.
  - **Latency + alpha-blended splats.** ALVR's reprojection/timewarp is built for opaque geometry; order-dependent transparent splats can show artifacts on fast head turns. For a *slow-motion inspection* viewer this is largely fine (see Q4).
  - **Input is a non-issue here.** This is a viewer, not a game — no tracked-controller dependency. Bryan also owns PSVR2 Sense controllers, which ALVR now supports on AVP if needed (vault, `[high]`).
- **ALVR's ceiling:** no foveated streaming, higher measured latency (~68ms tuned), more compression artifacts than CloudXR. Good enough to *prove the concept*; CloudXR is what you'd ship.

### Q3 — What actually renders the splats on the 5090 under OpenXR/SteamVR?

**This is the real work. The pipe is a download; the renderer is the build.** `[medium]`

Existing OpenXR Gaussian-splat viewers (all research/hobby-grade — verify scene ceilings yourself):

- **Splatapult** — standalone OpenXR splat viewer, works with SteamVR/Oculus. **Best candidate for the spike** (no engine, runs as an OpenXR app). `[medium]`
- **clarte53/GaussianSplattingVRViewerUnity** — Unity native plugin wrapping the original CUDA rasterizer; pre-compiled Windows build exists. `[medium]`
- **Enndee/Splatviewer_VR** — Unity VR fork of aras-p/UnityGaussianSplatting with runtime loading. `[medium]`
- **Orange Open Source 3DGS OpenXR** — OpenXR layer for the Inria SIBR viewer, Windows+Linux via SteamVR. `[medium]`

**Hard truths:**

- **Most of these were demonstrated at hundreds-of-thousands to low-millions of splats.** None has a public claim of holding **16M at stereo VR framerate.** Expect S2000 (3M) to be plausible off-the-shelf and **KennyRoom (16M) to need real engineering** even on a 5090 (cull-before-sort, LOD, possibly the SOG path for VRAM). `[low]` on 16M working unmodified.
- **None natively reads SOG.** They expect PLY. KennyRoom's 177MB SOG would need a loader; the 3.6GB PLY fits the 5090's 32GB VRAM but is heavy to sort.
- **Your MetalSplatter sort work does NOT transfer.** The GPU radix sort you integrated into `SplatSorter` is **Metal / Apple-GPU**. A 5090 renderer sorts in **CUDA / DX12 / Vulkan** — a separate codebase. Streaming is therefore *not* "reuse the native work"; it's a parallel renderer. (This is exactly why the iRacing analogy is partial: iRacing supplies its *own* render engine as the thing streamed. Bryan must supply the splat-renderer equivalent. CloudXR was never iRacing's hard part — and it won't be yours either.)
- **Build-it option:** a purpose-built OpenXR splat renderer (CUDA sort + tile rasterizer, foveated-aware) is the highest-ceiling path but the largest effort. Most pragmatic middle: harden **Splatapult** or the **clarte53 CUDA plugin** for 16M.

### Q4 — Latency reality for a *slow-motion* splat viewer

**~68ms (ALVR tuned) is acceptable for this use case. CloudXR's foveated streaming materially helps — on bitrate/sharpness more than on latency.** `[medium]`

- The comfort threshold that matters for VR sickness is **motion-to-photon for *head rotation*, target <20ms** — and that is handled on-device by the headset's **reprojection/timewarp**, which corrects the last-rendered frame to the latest head pose regardless of server latency. So the headset stays comfortable even at 68ms render latency. `[high]`
- The 68ms shows up as **content lag** (the splats themselves update slightly late), which is a problem for *twitch* games (Beat Saber, rhythm — the vault's own caveat) and a non-problem for a viewer where you're slowly orbiting/inspecting a static scene. **Slow head motion is the easy case.** `[medium]`
- **CloudXR's foveated streaming** spends bitrate where the eye looks. For a splat scene — which is all fine high-frequency detail and punishing to compress uniformly — this is a **real quality win** (sharp where you look at lower total bitrate), and it trims encode/network cost. It is **not primarily a latency win**; the headline gain is fidelity-per-megabit. The latency edge over ALVR is plausible but **unproven** `[vendor]`.
- Net: for *this* workload, latency is the dimension you have the most margin on. Fidelity and "does the renderer hold framerate" are the real risks.

### Q5 — Network requirements

**PC on Ethernet, headset on a dedicated 6GHz/Wi-Fi 6E link. Plan for >1Gbps.** `[high]`

- **CloudXR generic-viewer sample** states **100 Mbps minimum, 200 Mbps recommended** as the floor `[high]` — but that's a conservative client minimum.
- **iRacing Connect's real-world reqs** (the closest proven analog): **Wi-Fi 6+ router sustaining >1000 Mbps on 5GHz**, driver 580+, up to 4K@120Hz, firewall port 55000 open, network set to "Private," and **ISP combo routers explicitly called "insufficient."** `[high]` (vault)
- **Universal field-report refrain:** PC hardwired to Gigabit Ethernet; headset on a **dedicated 5/6GHz Wi-Fi 6/6E AP** so nothing else shares the air. **Stutter is almost always the network, not the headset.** `[high]` (vault)
- **Splat-specific note:** a detail-dense splat scene is *harder* to compress than a typical game frame (no large flat regions), so it will sit at the **upper end** of the bitrate budget. This is precisely where foveated streaming earns its keep. Budget for the high side, not the 100Mbps floor.
- **Bryan's Wi-Fi situation is unknown** — this is a prerequisite to confirm. A dedicated Wi-Fi 6E AP next to the headset, with the 5090 PC on Ethernet to the same router, is the configuration to target. Without it, the verdict downgrades regardless of how good the renderer is.

---

## Architecture sketch

**Spike path (prove it cheap — do this first):**
```
[5090 / Win11]                                  [Apple Vision Pro / vOS27]
  Splatapult (OpenXR splat viewer)
        |  renders stereo, loads S2000 PLY (3M)
        v
  SteamVR  ──►  ALVR server ──(HEVC/AV1 encode)──►  Wi-Fi 6E ──►  ALVR client (App Store)
                                                                      |
                                              on-device reprojection (head-pose timewarp)
                                                                      v
                                                                   display
```
Custom client code: **zero.** Goal: does a 3M scene hold a stable, comfortable image end-to-end?

**Shippable path (the real app, if the spike passes):**
```
[5090 / Win11]                                          [Apple Vision Pro / vOS27]
  OpenXR splat renderer  ── renders to ──►  CloudXR Runtime (server)
  (Splatapult-derived or                        |  GPU-accelerated HEVC/AV1 encode,
   custom CUDA sort+raster;                      |  foveated (gaze region full-res)
   loads PLY or SOG, 16M)                        v
                                          Wi-Fi 6E / >1Gbps
                                                 |
                                                 v
                                   Custom visionOS client (CloudXRKit)
                                   built from cloudxr-apple-generic-viewer
                                                 |
                                   FoveatedStreaming (system fw) feeds gaze
                                   server-side WITHOUT exposing gaze to the app
                                                 v
                                              display
```

---

## What it would take to build (effort estimate)

`[medium]` — estimates assume one experienced graphics/XR dev; multiply for learning curve on unfamiliar stacks.

| Phase | Work | Effort | Gate |
|---|---|---|---|
| **0. Spike** | ALVR + Splatapult + S2000, dedicated Wi-Fi 6E | **~0.5–1 day** | Does 3M stream comfortably? If no → reconsider whole path. |
| **1. Pipe** | Build the `cloudxr-apple-generic-viewer` sample, confirm CloudXR Runtime runs on the **5090**, connect to a trivial OpenXR server scene | **~3–5 days** | Does CloudXR run on a GeForce 5090 + vOS27 beta? (flips the biggest unknown) |
| **2. Renderer @ 16M** | Get an OpenXR splat renderer to load KennyRoom (PLY first, SOG loader if VRAM-bound) and hold stereo VR framerate — cull-before-sort, LOD, CUDA/Vulkan sort | **~2–6 weeks** | The actual hard part. Highly variable. |
| **3. Polish** | Foveated tuning, controls, SOG loader, comfort | **~1–2 weeks** | Product quality. |

**Total to a real 16M streaming viewer: roughly 1–2 months of focused work**, dominated by Phase 2. The spike is an afternoon and de-risks the whole thing.

---

## For which scene sizes streaming beats fixing native (the quantified call)

The argument is **memory-bandwidth headroom**, straight from your own `docs/LOD-DESIGN.md`:

- **M5 Vision Pro: ~122 GB/s.** LOD-DESIGN's math: an 8-pass radix sort over 20M splats ≈ **2.56 GB/frame ≈ 21ms @122GB/s = 2× over the 90fps budget *before raster.*** The M5 is bandwidth-starved for big splat scenes; that's why 16M is a brutal native fight requiring aggressive LOD/culling/streaming just to *attempt* it.
- **RTX 5090: ~1.8 TB/s (~15× the M5) + 32GB VRAM.** The same 2.56 GB/frame sort is **~1.4ms** — a rounding error. The 16M scene that pins the M5 against the wall is comfortable on the 5090.

That asymmetry *is* the threshold:

- **≤ ~3M (S2000): fix natively.** The MetalSplatter overhaul (GPU radix sort already integrated; cull-before-sort + LOD designed) brings this into range. Streaming adds latency and a 5090 dependency for zero benefit. **Don't stream S2000.**
- **~3–8M: native is plausible *if* the LOD tree-cut work lands** (LOD-DESIGN targets ~5M sortable budget). This is the contested middle; native is worth attempting first.
- **≥ ~16M (KennyRoom): streaming is the cleaner path.** Native requires winning a hard LOD/streaming/thermal fight against a fixed bandwidth wall; the 5090 sidesteps the wall entirely. **This is the scene that justifies the streaming build.**

---

## Stale vault assumption (build on it, don't re-derive)

The vault's `vr-hardware-strategy.md` (May 2026) frames CloudXR as **"integrated by app makers"** / the privileged pipe behind iRacing, implying it was *not* generally available for custom visionOS apps. **That framing is now stale.** At **GTC 2026**, CloudXR **6.0** opened the client side: a public Swift framework (`CloudXRKit`), a generic-viewer sample, and a universal OpenXR server runtime that streams *any* compliant app. A third-party developer can now build a custom visionOS streaming client. The vault's other conclusions (latency shape, network reqs, ALVR as the generic bridge, the input caveats) still hold.

---

## Sources

- [NVIDIA CloudXR 6.0 on Apple visionOS/iOS/iPadOS](https://developer.nvidia.com/topics/ai/xr/cloudxr/apple-platforms)
- [CloudXR 6.0 SDK — NVIDIA Developer](https://developer.nvidia.com/topics/ai/xr/cloudxr-sdk)
- [Stream High-Fidelity Spatial Content with CloudXR 6.0 — NVIDIA Technical Blog](https://developer.nvidia.com/blog/stream-high-fidelity-spatial-computing-content-to-any-device-with-nvidia-cloudxr-6-0/)
- [NVIDIA RTX Computers Now Connect Directly to Apple Vision Pro — NVIDIA Blog](https://blogs.nvidia.com/blog/nvidia-cloudxr-apple-vision-pro/)
- [NVIDIA/cloudxr-framework — GitHub](https://github.com/NVIDIA/cloudxr-framework)
- [NVIDIA/cloudxr-apple-generic-viewer — GitHub](https://github.com/NVIDIA/cloudxr-apple-generic-viewer)
- [Getting CloudXR — NVIDIA CloudXR SDK docs](https://docs.nvidia.com/cloudxr-sdk/latest/getting_cloudxr.html)
- [CloudXR Vision Pro: What Changed in visionOS 27 — Frame Sixty](https://framesixty.com/cloudxr-vision-pro/)
- [X-Plane 12 on Vision Pro via CloudXR — AppleInsider](https://appleinsider.com/articles/26/03/10/x-plane-12-flight-simulator-to-take-advantage-of-nvidia-cloudxr-6-in-visionos-264)
- [Splatapult 3DGS VR and Viewer — Radiance Fields](https://radiancefields.com/splatapult-3dgs-vr-and-viewer)
- [Orange Adds 3DGS OpenXR Support — Radiance Fields](https://radiancefields.com/orange-adds-3dgs-openxr-support)
- [clarte53/GaussianSplattingVRViewerUnity — GitHub](https://github.com/clarte53/GaussianSplattingVRViewerUnity)
- [Enndee/Splatviewer_VR — GitHub](https://github.com/Enndee/Splatviewer_VR)
- Internal: `vault/future-work/vr-hardware-strategy.md`; `MetalSplatter/docs/LOD-DESIGN.md` (M5 bandwidth math)
