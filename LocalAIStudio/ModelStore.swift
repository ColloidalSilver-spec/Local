import Foundation

/// Streaming (bounded-memory) download + storage of the Tiny-SD LCM ONNX model
/// files into the app's Documents directory. Direct port of the web app's OPFS
/// scheme (src/image.js) — same HuggingFace repo, same per-part files, same
/// size verification — but stored as plain files so ORT's CoreML/CPU exec
/// providers can read them, and nothing ever goes through a browser's memory
/// budget.
struct ModelFile {
    let relPath: String   // path under Documents, e.g. "text_encoder/model.onnx"
    let url: String       // full https URL
    let size: Int64
    let label: String
}

enum ModelStore {
    static let repo = "akameswa/lcm-tiny-sd-onnx-fp16"
    static let base = "https://huggingface.co/\(repo)/resolve/main"
    static let rootName = "las-img-tinysd"

    static let files: [ModelFile] = [
        ModelFile(relPath: "text_encoder/model.onnx", url: "\(base)/text_encoder/model.onnx", size: 246537314, label: "Text encoder"),
        ModelFile(relPath: "unet/model.onnx", url: "\(base)/unet/model.onnx", size: 354685, label: "Denoiser (graph)"),
        ModelFile(relPath: "unet/model.onnx_data", url: "\(base)/unet/model.onnx_data", size: 646809600, label: "Denoiser (weights)"),
        ModelFile(relPath: "vae_decoder/model.onnx", url: "\(base)/vae_decoder/model.onnx", size: 157826, label: "Image decoder (graph)"),
        ModelFile(relPath: "vae_decoder/model.onnx_data", url: "\(base)/vae_decoder/model.onnx_data", size: 98965248, label: "Image decoder (weights)"),
        ModelFile(relPath: "tokenizer/tokenizer_config.json", url: "\(base)/tokenizer/tokenizer_config.json", size: 813, label: "Tokenizer config"),
        ModelFile(relPath: "tokenizer/vocab.json", url: "\(base)/tokenizer/vocab.json", size: 1059962, label: "Tokenizer vocab"),
        ModelFile(relPath: "tokenizer/merges.txt", url: "\(base)/tokenizer/merges.txt", size: 524619, label: "Tokenizer merges"),
    ]

    static func dir() -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let d = docs.appendingPathComponent(rootName, isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    static func localURL(_ rel: String) -> URL {
        dir().appendingPathComponent(rel)
    }

    static func present(rel: String) -> Bool {
        let u = localURL(rel)
        guard let a = try? FileManager.default.attributesOfItem(atPath: u.path),
              let s = a[.size] as? Int64, s > 0 else { return false }
        return true
    }

    static func allPresent() -> Bool {
        files.allSatisfy { present(rel: $0.relPath) }
    }

    static func totalBytes() -> Int64 {
        files.reduce(0) { $0 + $1.size }
    }

    static func downloadedBytes() -> Int64 {
        files.reduce(0) { total, f in
            let u = localURL(f.relPath)
            guard let a = try? FileManager.default.attributesOfItem(atPath: u.path),
                  let s = a[.size] as? Int64 else { return total }
            return total + s
        }
    }

    /// Streams one file to disk (writes chunks as they arrive — memory stays
    /// flat regardless of file size), then moves it into place. Reports
    /// 0...1 for this file.
    static func download(file: ModelFile, onFileProgress: @escaping (Double) -> Void) async throws {
        let dest = localURL(file.relPath)
        try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        let tmp = dest.appendingPathExtension("part")
        try? FileManager.default.removeItem(at: tmp)
        FileManager.default.createFile(atPath: tmp.path, contents: nil)
        let fh = try FileHandle(forWritingTo: tmp)
        var req = URLRequest(url: URL(string: file.url)!)
        req.timeoutInterval = 120
        do {
            let (bytes, resp) = try await URLSession.shared.bytes(for: req)
            if let h = resp as? HTTPURLResponse, h.statusCode != 200 {
                throw NSError(domain: "ModelStore", code: h.statusCode,
                              userInfo: [NSLocalizedDescriptionKey: "HTTP \(h.statusCode) downloading \(file.label)"])
            }
            var got: Int64 = 0
            var buf = Data()
            for try await byte in bytes {
                buf.append(byte)
                got += 1
                if buf.count >= 1 << 20 {
                    try fh.write(contentsOf: buf)
                    buf.removeAll(keepingCapacity: true)
                    onFileProgress(file.size > 0 ? Double(got) / Double(file.size) : 0)
                }
            }
            if !buf.isEmpty {
                try fh.write(contentsOf: buf)
                onFileProgress(file.size > 0 ? Double(got) / Double(file.size) : 0)
            }
            try fh.close()
            try FileManager.default.moveItem(at: tmp, to: dest)
        } catch {
            try? fh.close()
            try? FileManager.default.removeItem(at: tmp)
            throw error
        }
    }
}
