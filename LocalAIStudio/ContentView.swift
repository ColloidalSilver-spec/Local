import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var prompt = "a photo of a red apple"
    @State private var negative = ""
    @State private var steps = 4
    @State private var guidance: Float = 1.5
    @State private var size = 512
    @State private var scheduler: SchedulerKind = .lcm

    @State private var status = ""
    @State private var progress: Double = 0
    @State private var image: UIImage?
    @State private var isWorking = false
    @State private var error: String?

    @State private var isDownloading = false
    @State private var dlProgress: Double = 0
    @State private var tokenizer: CLIPTokenizer?
    @State private var engine: DiffusionEngine?

    @State private var importSlot: ModelFile?
    @State private var showImporter = false
    @State private var showDebug = false
    @State private var debugLog = ""

    private let allowedTypes: [UTType] = [.data, .json, .plainText]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    modelSection
                    Divider()
                    promptSection
                    Divider()
                    controlsSection
                    Divider()
                    generateSection
                    if !status.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            ProgressView(value: progress)
                            Text(status).font(.footnote).foregroundStyle(.secondary)
                        }
                    }
                    if let error {
                        Text(error).font(.footnote).foregroundStyle(.red)
                    }
                    if let image {
                        VStack(alignment: .leading, spacing: 8) {
                            Image(uiImage: image)
                                .resizable().scaledToFit()
                                .frame(maxWidth: .infinity)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                            Button {
                                UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
                            } label: {
                                Label("Save to Photos", systemImage: "square.and.arrow.down")
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Local AI Studio")
            .fileImporter(
                isPresented: $showImporter,
                allowedContentTypes: allowedTypes
            ) { result in
                handleImport(result)
            }
        }
    }

    // MARK: - Model section

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Model: Tiny-SD LCM (fits ~1 GB)", systemImage: "cpu")
                .font(.headline)
            if ModelStore.allPresent() {
                Label("All model files present on device", systemImage: "checkmark.circle.fill")
                    .font(.footnote).foregroundStyle(.green)
            } else {
                Text("Downloading the model streams straight to disk — nothing loads into memory until you generate.")
                    .font(.footnote).foregroundStyle(.secondary)
                if isDownloading {
                    ProgressView(value: dlProgress)
                    Text("Downloading… \(Int(dlProgress * 100))%")
                        .font(.footnote).foregroundStyle(.secondary)
                } else {
                    Button {
                        downloadAll()
                    } label: {
                        Label("Download model (~\(ModelStore.totalBytes() / 1_000_000_000) GB)", systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isDownloading)
                }
                DisclosureGroup("Or import files manually") {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Pick each file from the Files app (grab them from the HuggingFace repo \(ModelStore.repo)).")
                            .font(.footnote).foregroundStyle(.secondary)
                        ForEach(ModelStore.files) { f in
                            HStack {
                                Text(f.label).font(.footnote)
                                Spacer()
                                if ModelStore.present(rel: f.relPath) {
                                    Image(systemName: "checkmark").foregroundStyle(.green)
                                } else {
                                    Button("Import…") {
                                        importSlot = f
                                        showImporter = true
                                    }
                                    .font(.footnote)
                                }
                            }
                        }
                    }
                }
                .font(.footnote)
            }
        }
    }

    // MARK: - Prompt section

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Prompt", text: $prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(2...4)
            TextField("Negative prompt (optional)", text: $negative, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...2)
        }
    }

    // MARK: - Controls

    private var controlsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Stepper("Steps: \(steps)", value: $steps, in: 1...16)
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text("Guidance")
                    Spacer()
                    Text(String(format: "%.1f", guidance))
                }
                Slider(value: $guidance, in: 1...10, step: 0.5)
            }
            Picker("Size", selection: $size) {
                Text("256").tag(256)
                Text("384").tag(384)
                Text("512").tag(512)
            }
            .pickerStyle(.segmented)
            Picker("Scheduler", selection: $scheduler) {
                ForEach(SchedulerKind.allCases) { s in
                    Text(s.rawValue).tag(s)
                }
            }
            .pickerStyle(.segmented)
        }
    }

    private var generateSection: some View {
        Button {
            generate()
        } label: {
            if isWorking {
                ProgressView().frame(maxWidth: .infinity)
            } else {
                Text("Generate").frame(maxWidth: .infinity)
            }
        }
        .buttonStyle(.borderedProminent)
        .disabled(isWorking || !ModelStore.allPresent())
    }

    // MARK: - Actions

    private func downloadAll() {
        isDownloading = true
        dlProgress = 0
        let total = ModelStore.totalBytes()
        var done: Int64 = 0
        Task {
            do {
                for f in ModelStore.files {
                    guard !ModelStore.present(rel: f.relPath) else { continue }
                    try await ModelStore.download(file: f) { fp in
                        done = max(done, ModelStore.downloadedBytes())
                        let p = Double(done) / Double(total)
                        Task { @MainActor in
                            self.dlProgress = p
                            self.status = "Downloading \(f.label)… \(Int(p * 100))%"
                        }
                    }
                }
                await MainActor.run {
                    isDownloading = false
                    status = ""
                    error = nil
                }
            } catch {
                let msg = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                await MainActor.run {
                    isDownloading = false
                    error = msg
                }
            }
        }
    }

    private func handleImport(_ result: Result<URL, Error>) {
        guard let f = importSlot else { return }
        importSlot = nil
        guard case .success(let src) = result else { return }
        let gotAccess = src.startAccessingSecurityScopedResource()
        defer { if gotAccess { src.stopAccessingSecurityScopedResource() } }
        do {
            let dest = ModelStore.localURL(f.relPath)
            try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.copyItem(at: src, to: dest)
        } catch {
            self.error = "Import failed: \(error.localizedDescription)"
        }
    }

    private func generate() {
        isWorking = true
        error = nil
        status = "Loading model…"
        progress = 0
        Task.detached(priority: .userInitiated) {
            do {
                try await MainActor.run {
                    if tokenizer == nil {
                        let tokDir = ModelStore.localURL("tokenizer")
                        let vocab = try Data(contentsOf: tokDir.appendingPathComponent("vocab.json"))
                        let merges = try String(contentsOf: tokDir.appendingPathComponent("merges.txt"), encoding: .utf8)
                        tokenizer = CLIPTokenizer(vocabJSON: vocab, mergesText: merges)
                    }
                    if engine == nil { engine = try DiffusionEngine() }
                    try engine!.load()
                }
                let p = GenParams(
                    prompt: prompt, negative: negative, steps: steps,
                    guidance: guidance, size: size, scheduler: scheduler
                )
                let tok = try await MainActor.run { () -> CLIPTokenizer in
                    guard let t = tokenizer else { throw NSError(domain: "gen", code: 1, userInfo: [NSLocalizedDescriptionKey: "no tokenizer"]) }
                    return t
                }
                let eng = try await MainActor.run { () -> DiffusionEngine in
                    guard let e = engine else { throw NSError(domain: "gen", code: 2, userInfo: [NSLocalizedDescriptionKey: "no engine"]) }
                    return e
                }
                let img = try eng.generate(params: p, tokenizer: tok) { msg, pct in
                    Task { @MainActor in
                        self.status = msg
                        self.progress = pct
                    }
                }
                let debug = "Token ids: \(tok.encode(p.prompt).map(String.init).joined(separator: ","))"
                await MainActor.run {
                    image = img
                    status = "Done"
                    isWorking = false
                    debugLog = debug
                }
            } catch {
                await MainActor.run {
                    isWorking = false
                    error = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                }
            }
        }
    }
}

extension ModelFile: Identifiable {
    var id: String { relPath }
}
