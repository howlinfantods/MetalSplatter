import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var isPickingFile = false

#if os(macOS)
    @Environment(\.openWindow) private var openWindow
#elseif os(iOS)
    @State private var navigationPath = NavigationPath()

    private func openWindow(value: ModelIdentifier) {
        navigationPath.append(value)
    }
#elseif os(visionOS)
    @Environment(\.openImmersiveSpace) var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace

    @State var immersiveSpaceIsShown = false

    // @Observable singleton — SwiftUI auto-tracks property reads in body
    private let loadModel = SplatLoadModel.shared

    private func openWindow(value: ModelIdentifier) {
        Task {
            switch await openImmersiveSpace(value: value) {
            case .opened:
                immersiveSpaceIsShown = true
            case .error, .userCancelled:
                break
            @unknown default:
                break
            }
        }
    }
#endif

    var body: some View {
#if os(visionOS)
        visionOSView
#elseif os(macOS)
        legacyMainView
#elseif os(iOS)
        NavigationStack(path: $navigationPath) {
            legacyMainView
                .navigationDestination(for: ModelIdentifier.self) { modelIdentifier in
                    MetalKitSceneView(modelIdentifier: modelIdentifier)
                        .navigationTitle(modelIdentifier.description)
                }
        }
#endif
    }

    // MARK: - visionOS compact status panel

#if os(visionOS)
    private let displayModel = SplatDisplayModel.shared

    @ViewBuilder
    var visionOSView: some View {
        let bindable = Bindable(displayModel)
        ScrollView(.vertical, showsIndicators: false) {
          VStack(alignment: .leading, spacing: 14) {

            HStack(spacing: 8) {
                Image(systemName: "cube.transparent")
                    .foregroundStyle(.secondary)
                Text("MetalSplatter")
                    .font(.headline)
                Spacer()
            }

            Divider()

            HStack(spacing: 10) {
                Button {
                    isPickingFile = true
                } label: {
                    Label("Open Splat File", systemImage: "doc.badge.plus")
                }
                .buttonStyle(.borderedProminent)
                .disabled(isPickingFile || loadModel.phase.isBusy)

                if immersiveSpaceIsShown {
                    Button("Close Scene") {
                        Task {
                            await dismissImmersiveSpace()
                            immersiveSpaceIsShown = false
                            await MainActor.run { loadModel.reset() }
                        }
                    }
                    .buttonStyle(.bordered)
                }
            }

            if loadModel.phase.isActive {
                Divider()

                VStack(alignment: .leading, spacing: 8) {
                    if !loadModel.filename.isEmpty {
                        Text(loadModel.filename)
                            .font(.subheadline.weight(.medium))
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }

                    if loadModel.phase.isBusy {
                        if let total = loadModel.totalSplatCount, total > 0 {
                            ProgressView(value: Double(loadModel.splatCount), total: Double(total))
                            Text("\(SplatLoadModel.formatCount(loadModel.splatCount)) / \(SplatLoadModel.formatCount(total)) splats")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else {
                            HStack(spacing: 8) {
                                ProgressView()
                                    .scaleEffect(0.75)
                                Text("\(SplatLoadModel.formatCount(loadModel.splatCount)) splats...")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    Text(loadModel.phase.label)
                        .font(.caption)
                        .foregroundStyle(loadModel.phase.isError ? Color.red : Color.secondary)
                }

                if !loadModel.log.isEmpty {
                    Divider()
                    ScrollViewReader { proxy in
                        ScrollView(.vertical, showsIndicators: false) {
                            LazyVStack(alignment: .leading, spacing: 3) {
                                ForEach(loadModel.log) { entry in
                                    Text(entry.message)
                                        .font(.system(size: 11, design: .monospaced))
                                        .foregroundStyle(entry.isError ? Color.red : Color.secondary)
                                        .id(entry.id)
                                }
                            }
                            .padding(6)
                        }
                        .frame(maxHeight: 160)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                        .onChange(of: loadModel.log.count) { _, _ in
                            if let last = loadModel.log.last {
                                withAnimation(.linear(duration: 0.1)) {
                                    proxy.scrollTo(last.id, anchor: .bottom)
                                }
                            }
                        }
                    }
                }
            }

            // Display controls — only show once a splat is loaded
            if case .ready = loadModel.phase {
                Divider()

                VStack(alignment: .leading, spacing: 10) {
                    Text("Display").font(.subheadline.weight(.medium))

                    // Presets
                    HStack(spacing: 8) {
                        Button("Tabletop")  { displayModel.applyPreset(.tabletop) }
                            .buttonStyle(.bordered)
                        Button("Life Size") { displayModel.applyPreset(.lifeSize) }
                            .buttonStyle(.bordered)
                        Button("Room Scale") { displayModel.applyPreset(.roomScale) }
                            .buttonStyle(.bordered)
                    }

                    // Scale — logarithmic so a bbox-derived preset at any computed value is
                    // representable (SfM units are arbitrary) and fine control is even across
                    // orders of magnitude. A linear range would silently clamp the preset.
                    HStack(spacing: 6) {
                        Text("Scale").font(.caption).frame(width: 36, alignment: .leading)
                        Slider(
                            value: Binding(
                                get: { Double(log10(max(displayModel.scale, 0.0001))) },
                                set: { displayModel.scale = Float(pow(10.0, $0)) }
                            ),
                            in: log10(0.005)...log10(200.0)
                        )
                        Text(String(format: "%.2f×", displayModel.scale))
                            .font(.caption.monospacedDigit())
                            .frame(width: 44, alignment: .trailing)
                    }

                    // Depth (Z distance)
                    HStack(spacing: 6) {
                        Text("Depth").font(.caption).frame(width: 36, alignment: .leading)
                        Slider(value: bindable.positionZ, in: -20.0 ... -0.5)
                        Text(String(format: "%.1fm", -displayModel.positionZ))
                            .font(.caption.monospacedDigit())
                            .frame(width: 44, alignment: .trailing)
                    }

                    Divider()

                    // Orientation
                    Text("Orientation").font(.caption).foregroundStyle(.secondary)

                    HStack(spacing: 0) {
                        ForEach([("Pitch", \SplatDisplayModel.pitch),
                                 ("Yaw",   \SplatDisplayModel.yaw),
                                 ("Roll",  \SplatDisplayModel.roll)], id: \.0) { label, kp in
                            VStack(spacing: 4) {
                                Text(label).font(.caption2).foregroundStyle(.secondary)
                                HStack(spacing: 4) {
                                    Button("−") { displayModel[keyPath: kp] -= .pi / 12 }
                                        .buttonStyle(.bordered)
                                    Button("+") { displayModel[keyPath: kp] += .pi / 12 }
                                        .buttonStyle(.bordered)
                                }
                            }
                            .frame(maxWidth: .infinity)
                        }
                    }

                    HStack {
                        Button("Reset") { displayModel.resetOrientation() }
                            .buttonStyle(.bordered)
                        Spacer()
                        Toggle("Auto-Rotate", isOn: bindable.autoRotate)
                            .font(.caption)
                            .toggleStyle(.switch)
                    }
                }
            }

          } // inner VStack
        } // ScrollView
        .padding(20)
        .frame(width: 420)
        .fileImporter(
            isPresented: $isPickingFile,
            allowedContentTypes: [
                UTType(filenameExtension: "ply")!,
                UTType(filenameExtension: "splat")!,
                UTType(filenameExtension: "spz")!,
            ]
        ) { result in
            isPickingFile = false
            switch result {
            case .success(let url):
                _ = url.startAccessingSecurityScopedResource()
                Task {
                    // Hold security-scoped access long enough for multi-GB files to fully load.
                    try await Task.sleep(for: .seconds(600))
                    url.stopAccessingSecurityScopedResource()
                }
                openWindow(value: ModelIdentifier.gaussianSplat(url))
            case .failure(let error):
                Task { @MainActor in
                    loadModel.addLog("File picker error: \(error.localizedDescription)", isError: true)
                    loadModel.phase = .failed("Could not open file")
                }
            }
        }
    }
#endif

    // MARK: - macOS / iOS legacy panel

    @ViewBuilder
    var legacyMainView: some View {
        VStack {
            Spacer()

            Text("MetalSplatter SampleApp")

            Spacer()

            Button("Read Scene File") {
                isPickingFile = true
            }
            .padding()
            .buttonStyle(.borderedProminent)
            .disabled(isPickingFile)
            .fileImporter(isPresented: $isPickingFile,
                          allowedContentTypes: [
                            UTType(filenameExtension: "ply")!,
                            UTType(filenameExtension: "splat")!,
                            UTType(filenameExtension: "spz")!,
                          ]) {
                isPickingFile = false
                switch $0 {
                case .success(let url):
                    _ = url.startAccessingSecurityScopedResource()
                    Task {
                        try await Task.sleep(for: .seconds(600))
                        url.stopAccessingSecurityScopedResource()
                    }
                    openWindow(value: ModelIdentifier.gaussianSplat(url))
                case .failure:
                    break
                }
            }

            Button("Procedural Splat") {
                openWindow(value: ModelIdentifier.proceduralSplat)
            }
            .padding()
            .buttonStyle(.borderedProminent)

            Button("Show Sample Box") {
                openWindow(value: ModelIdentifier.sampleBox)
            }
            .padding()
            .buttonStyle(.borderedProminent)

            Spacer()
        }
    }
}
