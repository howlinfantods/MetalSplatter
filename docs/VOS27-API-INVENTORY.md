# visionOS 27 API Inventory — Gaussian Splats & Spatial Input

> **Purpose:** Decide what to build natively vs. keep in the MetalSplatter custom-Metal pipeline.
> **SDK probed:** Xcode 27.0, Build **27A5194q**, `XROS27.0.sdk` (device seed 27A5194q).
> **Date:** 2026-06-21

## Verification method (read this first)

Symbols were verified by inspecting the **`.swiftinterface`** (the authoritative public Swift API,
emitted by the compiler) and the **Objective-C `.h` headers + `.tbd`** linker stubs in the installed
SDK — *not* literal `nm` on a Mach-O. For a Swift framework, `nm` yields mangled junk and the SDK ships
**no dylib** (only `.tbd` stubs + `.swiftmodule`), so `.swiftinterface`/header inspection is strictly
**stronger** than `nm`. Where I write `[VERIFIED-by-nm]` below, read it as "verified against the installed
SDK's authoritative interface files."

**Load-bearing caveat:** *binary/header presence in the SDK ≠ availability on visionOS.* MetalFX ships a
full `.tbd` in `XROS27.0.sdk` yet every class is `API_UNAVAILABLE(visionos)`. Always confirm the
`@available` / `API_AVAILABLE` annotation, never just symbol presence. (This is exactly why the prior
Cairn note got the splat answer wrong — see §1.)

Verification tags:
- **[VERIFIED-by-nm]** — found in this seed's authoritative SDK interface/header with explicit visionOS availability.
- **[DOC-CONFIRMED]** — corroborated by Apple docs / WWDC26 session content.
- **[ANNOUNCED-UNVERIFIED]** — announced but not confirmable against this SDK.

Authoritative file used for RealityKit symbols:
`XROS27.0.sdk/System/Library/Frameworks/RealityFoundation.framework/Modules/RealityFoundation.swiftmodule/arm64e-apple-xros.swiftinterface`
(RealityKit re-exports RealityFoundation; the `RealityKit.framework` umbrella interface is only ~1k lines and contains **none** of the components.)

---

## 1. GaussianSplatComponent / Native splat support — **PRESENT in 27A5194q** ✅

**[VERIFIED-by-nm] + [DOC-CONFIRMED]** — This **overturns the prior Cairn note** that reported 0 symbols.
The prior probe checked `RealityKit.framework/RealityKit` (a path that doesn't exist — the SDK has no
dylib there, only `RealityKit.tbd`) and/or the thin RealityKit umbrella. The real symbols live in
**`RealityFoundation.framework`**. They exist, fully formed, in this exact seed.

All splat symbols are `@available(visionOS 27.0, macOS 27.0, iOS 27.0, tvOS 27.0, *)`.

### The component (trivial ECS wrapper)
```swift
@available(visionOS 27.0, macOS 27.0, iOS 27.0, tvOS 27.0, macCatalyst 27.0, *)
public struct GaussianSplatComponent : Component {
  public init(_ res: GaussianSplatResource)
  public var splatResource: GaussianSplatResource
}
```

### The resource (the real surface)
```swift
@available(visionOS 27.0, *)
final public class GaussianSplatResource {
  public enum SphericalHarmonicDegree : UInt8 { case zero, first, second, third }   // SH 0..3
  public struct BufferDescriptor {              // one attribute = (buffer, format, stride, offset...)
    public let buffer: LowLevelBuffer
    public let format: MTLAttributeFormat
    public let stride: Int
  }
  public struct BufferResource {                // the splat payload, all GPU buffers
    public let position, scale, rotation, opacity, sphericalHarmonics: BufferDescriptor
    public let count: Int
    public let degree: SphericalHarmonicDegree
  }
  public enum ActivationFunction { case identity, exponential, sigmoid }
  final public var scaleActivation: ActivationFunction      // exp() on scale, sigmoid() on opacity, etc.
  final public var opacityActivation: ActivationFunction
  public enum ProjectionMode { case perspective, tangential }
  final public var projectionMode: ProjectionMode
  public enum SortingMode { case depth, distance }          // <-- see "load-bearing tradeoff" below
  final public var sortingMode: SortingMode
  final public var colorSpace: CGColorSpace
  final public let bufferResource: BufferResource?
  public init(_ bufferResource: BufferResource)
}

extension GaussianSplatResource.BufferResource {            // the convenience ctor you'll actually call
  @MainActor public init(count: Int,
                         position: BufferDescriptor, scale: BufferDescriptor,
                         rotation: BufferDescriptor, opacity: BufferDescriptor,
                         sphericalHarmonics: (BufferDescriptor, SphericalHarmonicDegree)) throws
}
```

### What this means for the build decision

- **Ingestion is buffer-only.** The *only* construction paths are from a `BufferResource` built out of
  `LowLevelBuffer`s — i.e. **you supply Metal buffers**. There is **no file loader** (no `.ply` / `.splat`
  / `.sog` / URL initializer anywhere in the interface). MetalSplatter's PLY/SPLAT/SOG parsers stay
  relevant — they become the "decode to buffers" front end, then you hand RealityKit the buffers.
- **★ Load-bearing tradeoff — RealityKit sorts for you.** `SortingMode { depth, distance }` means the
  native renderer handles back-to-front sorting internally. **This is precisely what the MetalSplatter
  fork's custom GPU radix sort (`SplatSorter.performSort`) was built to do.** If you adopt the native
  component, the GPU radix sort becomes redundant for the on-screen path. The decision is: keep the
  custom Metal pipeline (full control over sort/LOD/culling, works on vOS26 and Mac, your GPU radix sort
  already proven) **vs.** native `GaussianSplatComponent` (Apple does sort + foveated render + composites
  correctly with other RealityKit content and passthrough, but you give up sort control and it's vOS27+).
- **SH up to degree 3** (`.zero ... .third`) — full view-dependent color is supported natively.
- **Activation functions are configurable** (identity/exp/sigmoid) — matches standard 3DGS conventions,
  so raw INRIA-format params map over without pre-activating on CPU.
- **`projectionMode .tangential`** is notable — likely the foveated/curved-projection path tuned for the
  headset's lenses; worth testing for quality vs. `.perspective`.

### WWDC26 / docs corroboration **[DOC-CONFIRMED]**
- Session **"Explore advances in RealityKit"** (WWDC26 #279) introduces native Gaussian splatting.
- Session **#287** "Build next-generation experiences with visionOS 27."
- Apple doc page exists: *Gaussian splats on visionOS*. Apple's own description matches the SDK exactly:
  "render real-world captures as ellipsoids defined by position, scale, rotation, opacity, and spherical
  harmonics, assembled into a `GaussianSplatComponent`… RealityKit does not assume a specific file format;
  you provide buffers." Apple also shipped 3DGS in **Apple Maps** at WWDC26 (same tech).

### LowLevelMesh / LowLevelTexture / LowLevelBuffer **[VERIFIED-by-nm]**
Present in RealityFoundation and these are the bridge for custom GPU data:
`LowLevelBuffer`, `LowLevelMesh`, `LowLevelTexture`, `LowLevelInstanceData`, `LowLevelDeviceResource`.
`GaussianSplatResource.BufferDescriptor.buffer` is a `LowLevelBuffer` — so you allocate a `LowLevelBuffer`
(it wraps `MTLBuffer`-backed storage RealityKit can own), fill it from your decoder, and pass it in.
This is the integration seam between MetalSplatter's loaders and the native renderer.

---

## 2. PSVR2 Sense controllers (spatial accessories) — **PRESENT, landed vOS26** ✅

**[VERIFIED-by-nm]** — via `GameController.framework` headers in `XROS27.0.sdk`.

### Version reconciliation (vault said vOS26 — confirmed)
The vault's "Apple enabled PSVR2 in visionOS 26" is **correct**. The framing depends on which symbol:
- `GCProductCategorySpatialController` → `API_AVAILABLE(..., visionos(26.0))`  ✅ **the controller category is vOS26**
- Spatial input element names (`GCInputThumbstick`, `GCInputGripButton`, `GCInputTrigger`, `GCInputThumbstickButton`) → all `visionos(26.0)`  ✅
- The class `GCSpatialAccessory` itself → `API_AVAILABLE(visionos(27.0))` and the `GCSpatialAccessoryDidConnect/Disconnect` notifications are `visionos(27.0)`.

So: **spatial controllers (incl. PSVR2 Sense) have been usable since vOS26 via `GCController` + the
spatial product category and input names.** vOS27 adds the dedicated **`GCSpatialAccessory`** abstraction
(a cleaner, non-gamepad device class). Both work on this seed.

### Naming caveat — there is NO "PSVR2"/"Sony"/"Sense" string in the SDK
`grep -i psvr|sense|sony|playstation` over the GameController headers returns **nothing PSVR-specific**.
Apple exposes PSVR2 Sense controllers through the **generic** `GCProductCategorySpatialController` /
`GCSpatialAccessory` abstraction, exactly as it does DualSense via `GCProductCategoryDualSense`. The
PSVR2→SpatialController mapping is Apple's documented behavior, not a literal SDK string — treat the
"these are your PSVR2 controllers" identification as **inferred from the category**, confirmed at runtime
by enumerating `GCSpatialAccessory.spatialAccessories` / observing the connect notification.

### How an app reads them
```objc
// vOS27 dedicated class:
@interface GCSpatialAccessory : NSObject <GCDevice>
@property (readonly, class) NSArray<GCSpatialAccessory*> *spatialAccessories;     // discovery
@property (readonly, strong, nullable) id<GCDevicePhysicalInput> input;           // buttons/triggers/sticks
@property (readonly, strong, nullable) GCDeviceHaptics *haptics;                  // <-- haptics, see below
@end
// + GCSpatialAccessoryDidConnectNotification / ...DidDisconnectNotification (visionos 27.0)
```
- **Buttons / triggers / thumbstick:** read via `input` (`GCDevicePhysicalInput`) using element names
  `GCInputTrigger`, `GCInputGripButton` (squeeze), `GCInputThumbstick`, `GCInputThumbstickButton`, A/B/X/Y.
  (Pre-vOS27 path: the same controller surfaces as a `GCController` with these inputs.)

### 6DoF pose — comes from **ARKit**, not GameController **[VERIFIED-by-nm]**
GameController gives you the *buttons*; the *spatial pose* of the controller comes from ARKit's new
**accessory tracking** provider:
`ARKit.framework/Headers/accessory_tracking.h` →
- `ar_accessory_anchor` (subclass of `ar_trackable_anchor`)
- `ar_accessory_tracking_provider` with `ar_accessory_tracking_provider_set_update_handler(...)`
- `ar_accessories_enumerate_accessories(...)`, `ar_accessory_anchors_enumerate_anchors(...)`
These are C entry points marked `AR_REFINED_FOR_SWIFT` (so Swift sees `AccessoryAnchor` /
`AccessoryTrackingProvider`-style names). The update handler returns **rendering-corrected transforms** for
the accessory anchor — that's your 6DoF controller pose, feed it to an `Entity`'s transform.

### Haptics **[VERIFIED-by-nm]**
- `GCSpatialAccessory.haptics` → **`GCDeviceHaptics`** (header `GCDeviceHaptics.h` present; symbol
  `GCDeviceHaptics` in `.tbd`). `GCDeviceHapticsLocality.h` is present too (per-actuator targeting).
- `GCDeviceHaptics` vends a `CHHapticEngine` for the controller's actuators (Core Haptics), so **yes,
  the app can trigger controller haptics**. (`DualSense` adaptive-trigger force is a separate
  `GCDualSenseAdaptiveTrigger` path; not confirmed for PSVR2 Sense.)

---

## 3. Hand-tracking & gesture APIs for grab-to-scale — **rich, cleanest path = ManipulationComponent** ✅

### ★ Recommended path: `ManipulationComponent` (vOS26+) **[VERIFIED-by-nm] + [DOC-CONFIRMED]**
This is the single cleanest way to get "grab a splat with both hands and scale/rotate it" — Apple built it
exactly for this. It is `@available(visionOS 26.0, *)` (visionOS-only; unavailable on iOS/macOS/etc).

```swift
@available(visionOS 26.0, *)
public struct ManipulationComponent : Component {
  public init()
  public var dynamics: Dynamics
  public var releaseBehavior: ReleaseBehavior          // .reset, etc.
  public var audioConfiguration: AudioConfiguration

  // One-call setup: collision shape + hover + which inputs can grab it
  public static func configureEntity(_ entity: Entity,
                                     hoverEffect: HoverEffectComponent.HoverEffect? = nil,
                                     allowedInputTypes: InputTargetComponent.InputType? = nil,
                                     collisionShapes: [ShapeResource]? = nil)

  public struct Dynamics {
    public var translationBehavior: TranslationBehavior
    public var primaryRotationBehavior: RotationBehavior     // .unconstrained / constrained
    public var secondaryRotationBehavior: RotationBehavior   // two-handed roll
    public var scalingBehavior: ScalingBehavior              // .none / .unconstrained  <-- pinch-to-scale
    public var inertia: Inertia                              // .zero/.low/.medium/.high (throw/settle feel)
  }
  public struct InputDevice {
    public enum Kind { case indirectPinch, directPinch, pointer }   // hands + pointer
    public var chirality: Chirality?                                 // left/right hand
    public var kind: Kind
  }
}

public enum ManipulationEvents {     // observe via scene.subscribe(...)
  public struct WillBegin : Event { let inputDeviceSet; let pivotPoint: Point3D }
  public struct WillEnd   : Event { let inputDeviceSet; let wasCancelled: Bool }
  public struct DidUpdateTransform / DidUpdateInputDevices ...
}
```
**Grab-to-scale recipe:**
```swift
var mc = ManipulationComponent()
mc.dynamics.scalingBehavior = .unconstrained          // enable two-handed pinch scale
mc.dynamics.primaryRotationBehavior = .unconstrained
mc.dynamics.inertia = .medium
ManipulationComponent.configureEntity(splatEntity,
    allowedInputTypes: .all, collisionShapes: [.generateBox(size: bboxSize)])
splatEntity.components.set(mc)
```
RealityKit then does two-handed translate + rotate + uniform scale on the entity automatically; subscribe
to `ManipulationEvents` if you need to react (e.g. snap-to-tabletop on `WillEnd`). `InputDevice.Kind` shows
both indirect (eye+pinch) and **directPinch** (reach-out grab) are supported, with `chirality` so you can
tell left/right hand. This is the affordance Apple demoed in WWDC26 #284 ("design review").

### Lower-level fallbacks (more control, more code)
- **ARKit `HandTrackingProvider`** **[VERIFIED-by-nm]** — `ARKit/Headers/hand_tracking.h`,
  `hand_skeleton.h`: `ar_hand_tracking_provider` (subclass of `ar_data_provider`), `ar_hand_anchor`,
  `ar_hand_chirality {right,left}`, `ar_hand_fidelity {nominal,high}` (note: **vOS27 adds a high-fidelity
  mode**), full joint skeleton. Use only if you need raw joints (custom gesture, fingertip painting).
- **`SpatialTrackingSession`** **[VERIFIED-by-nm]** — RealityFoundation; the RealityKit-native way to
  request hand/world anchoring authorization without dropping to raw ARKit. Pairs with `AnchoringComponent`.
- **`GestureComponent`** **[VERIFIED-by-nm]** — `@available(visionOS 26.0, *)`, present but an **empty
  struct** in the interface (marker; configured via SwiftUI `.gesture` attachments /
  `SpatialEventGesture`). For free-form manipulation prefer `ManipulationComponent` over hand-rolling
  gestures.
- **SwiftUI `SpatialEventGesture` / `RealityView` gestures** — the SwiftUI-side path for taps/drags on
  entities; fine for selection, but `ManipulationComponent` is better for 6DoF grab+scale.

---

## 4. Native compressed-splat / SOG ingestion — **none; custom decode required** ⚠️

**[VERIFIED-by-nm]** (absence verified):
- **ModelIO has no splat path.** `grep gaussian|splat` over `ModelIO.tbd` → 0 hits. ModelIO does not load
  `.ply`-as-splats, SOG, or any Gaussian format.
- **RealityKit has no splat file loader.** The only `GaussianSplatResource` initializers take a
  `BufferResource` (see §1). No URL/`named:`/`contentsOf:`/usd path. usdz does **not** carry splats.
- **Conclusion:** SOG / compressed-Gaussian / `.ply` / `.splat` ingestion is **custom-Metal-only**. Your
  pipeline is: *MetalSplatter loaders (PLYIO / SplatIO / SOG decode) → fill `LowLevelBuffer`s →
  `GaussianSplatResource.BufferResource(count:position:scale:rotation:opacity:sphericalHarmonics:)`*.
  The MetalSplatter SOG/PLY/SPLAT parsers remain **load-bearing** even if you adopt native rendering —
  Apple gives you the renderer, not the importer.

---

## 5. Performance levers on vOS27 — mostly NOT what you'd hope ⚠️

> The advisor flagged §5 for over-generalization from `grep`; every item below was re-checked against the
> actual `@available`/`API_AVAILABLE` annotation, not symbol presence.

### MetalFX upscaling — **UNAVAILABLE on visionOS** ❌ **[VERIFIED-by-nm]**
`MetalFX.framework` *ships* in `XROS27.0.sdk`, but **every** class is gated off visionOS:
- `MTLFXTemporalScaler` → `API_UNAVAILABLE(visionos)`
- `MTLFXSpatialScaler` → `API_AVAILABLE(macos(13.0), ios(16.0))` (no visionos — unavailable)
- `MTLFXFrameInterpolator` → `API_UNAVAILABLE(visionos)`
- `MTLFXTemporalDenoisedScaler` → `API_UNAVAILABLE(visionos)`
**Do not plan on MetalFX for splats on Vision Pro.** (This is the canonical "binary present ≠ available"
trap — symbol presence in the tbd would have lied here.)

### Metal 4 (`MTL4*`) — **NOT exposed on visionOS 27 SDK** ❌ **[VERIFIED-by-nm]**
`MTL4*` symbols are all over `Metal.tbd`, but the headers (`MTL4CommandQueue.h`, `MTL4CommandBuffer.h`,
`newSpatialScalerWithDevice:compiler:`, `supportsMetal4FX:`) are annotated
`API_AVAILABLE(macos(26.0), ios(26.0))` — **no `visionos`**. No header carries `API_UNAVAILABLE(visionos)`
either; they simply omit visionOS, which means unavailable. **Use the classic Metal 3 command-buffer path
on Vision Pro.** (Corrects an earlier tbd-only reading that assumed MTL4 was usable.)

### Variable rasterization rate / foveation — **NOT app-controllable on visionOS** ⚠️ **[VERIFIED-by-nm]**
`MTLRasterizationRateMap` and `-[MTLDevice supportsRasterizationRateMapWithLayerCount:]` /
`newRasterizationRateMapWithDescriptor:` are annotated `macos / ios / tvos` only — **no visionOS**.
visionOS applies **system-driven foveation** to your render automatically (you can't author the rate map).
The lever you *do* get is RealityKit's own foveated compositor when you render via `RealityView` /
`CompositorLayer` — i.e. let the system foveate. The native `GaussianSplatComponent` (§1) renders through
that path, which is a real reason to consider it over a fully-custom `CompositorLayer` splat renderer.

### What IS available as a perf lever
- **Native `GaussianSplatComponent`** (§1): sort + foveated render handled by Apple, composites with
  passthrough, almost certainly tuned for the M5 / R1. The biggest single perf lever if you adopt it.
- **`LowLevelMesh` / `LowLevelBuffer` / `LowLevelTexture`** **[VERIFIED-by-nm]**: zero-copy GPU resource
  authoring inside RealityKit — keeps your custom Metal compute (LOD, culling, sort if you stay custom)
  feeding RealityKit without CPU round-trips.
- **`SpatialTrackingSession` high-fidelity hand mode** and `ar_hand_fidelity_high` — relevant only if hand
  tracking is in your hot path.
- **Classic Metal 3 levels:** tile/imageblock memory, indirect command buffers, MSAA control, half/`bf16`
  storage for SH — all standard on visionOS, unchanged. These remain your custom-pipeline knobs.

### M5 note
The vault references the **M5 Vision Pro**. Nothing in the SDK exposes M5-specific API surface; perf gains
are implicit (faster GPU/NPU). Plan against the API availability above, not chip-specific symbols.

---

## TL;DR decision matrix

| Need | Native (vOS27) | Custom (MetalSplatter) | Recommendation |
|---|---|---|---|
| Render splats | `GaussianSplatComponent` ✅ vOS27 | proven, vOS26+/Mac | **Bake-off**: native composites + foveates for free, but vOS27-only and you lose sort/LOD control |
| Sort | RealityKit `SortingMode` does it | your GPU radix sort (proven) | Native makes your radix sort redundant *on-screen* — this is the key tradeoff |
| Load SOG/PLY/SPLAT | ❌ none | ✅ your parsers | **Keep your loaders regardless** — Apple has no importer |
| Grab + 2-hand scale | `ManipulationComponent` ✅ vOS26 | hand-rolled | **Use native** — clearly the cleanest path |
| PSVR2 controllers | `GCSpatialAccessory`/SpatialController ✅ vOS26 + ARKit accessory pose | n/a | **Use native**, pose via ARKit accessory_tracking |
| Controller haptics | `GCDeviceHaptics` (CHHapticEngine) ✅ | n/a | **Use native** |
| Upscaling | MetalFX ❌ unavailable | n/a | Not an option on visionOS |
| Foveation control | system-only (not app-driven) | render via RealityView to get it | Let the system foveate |
| Metal 4 | ❌ not on vOS27 SDK | Metal 3 | Stay on Metal 3 |
