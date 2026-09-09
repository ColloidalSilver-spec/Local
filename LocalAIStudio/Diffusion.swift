import Foundation
import UIKit
import OnnxRuntimeBindings

enum SchedulerKind: String, CaseIterable, Identifiable {
    case lcm = "LCM"
    case ddim = "DDIM"
    var id: String { rawValue }
}

struct GenParams {
    var prompt = "a photo of a red apple"
    var negative = ""
    var steps = 4
    var guidance: Float = 1.5
    var size = 512
    var scheduler: SchedulerKind = .lcm
}

/// On-device Stable Diffusion pipeline, a direct port of src/image.js's
/// verified algorithm, running the SAME Tiny-SD LCM ONNX model. Where the web
/// app was confined to Safari's ~1 GB WebContent process budget, this runs in
/// the app's own process (far higher limit) and routes the UNet/VAE through
/// ORT's CoreML execution provider onto the Apple Neural Engine.
final class DiffusionEngine {
    private let env: ORTEnv
    private var te: ORTSession?
    private var unet: ORTSession?
    private var vae: ORTSession?

    init() throws {
        env = try ORTEnv(loggingLevel: .warning)
    }

    private func makeOptions(useCoreML: Bool) throws -> ORTSessionOptions {
        let opts = try ORTSessionOptions()
        try opts.setIntraOpNumThreads(2)
        try opts.setGraphOptimizationLevel(.basic)
        if useCoreML && ORTIsCoreMLExecutionProviderAvailable() {
            let cml = ORTCoreMLExecutionProviderOptions()
            cml.enableOnSubgraphs = true
            try opts.appendCoreMLExecutionProvider(with: cml)
        }
        return opts
    }

    // Try CoreML EP first (ANE/GPU), fall back to CPU-only if the model can't
    // be compiled for CoreML on this device.
    private func makeSession(_ relPath: String) throws -> ORTSession {
        let path = ModelStore.localURL(relPath).path
        var lastErr: Error?
        for useCoreML in [true, false] {
            do {
                let opts = try makeOptions(useCoreML: useCoreML)
                return try ORTSession(env: env, modelPath: path, sessionOptions: opts)
            } catch {
                lastErr = error
            }
        }
        throw lastErr ?? NSError(domain: "DiffusionEngine", code: 1,
                                 userInfo: [NSLocalizedDescriptionKey: "Couldn't create session for \(relPath)"])
    }

    func load() throws {
        if te == nil { te = try makeSession("text_encoder/model.onnx") }
        if unet == nil { unet = try makeSession("unet/model.onnx") }
        if vae == nil { vae = try makeSession("vae_decoder/model.onnx") }
    }

    func freeTextEncoder() { te = nil }
    func freeDecoder() { vae = nil }

    /// Full text-to-image run. `onStatus` is called on the calling queue.
    func generate(params: GenParams, tokenizer: CLIPTokenizer, onStatus: @escaping (String, Double) -> Void) throws -> UIImage {
        onStatus("Encoding prompt…", 0.05)
        let condIds = tokenizer.encode(params.prompt)
        let condHidden = try runTE(condIds)
        var uncondHidden: [Float]? = nil
        if params.guidance > 1.001 {
            let uIds = tokenizer.encode(params.negative.isEmpty ? "" : params.negative)
            uncondHidden = try runTE(uIds)
        }
        // Free the text encoder before the (long) denoise phase — it's served
        // its purpose, and dropping it lowers peak memory.
        freeTextEncoder()

        let latH = params.size / 8
        let latW = params.size / 8
        var latents = [Float](repeating: 0, count: 4 * latH * latW)
        for i in 0..<latents.count { latents[i] = Self.randn() }

        let alphas = Self.alphasCumprod(1000)
        let n = params.steps
        var ts = [Int](repeating: 0, count: n)
        for i in 0..<n {
            ts[i] = Int((999.0 - Double(i) * 999.0 / Double(max(n - 1, 1))).rounded())
        }

        for i in 0..<n {
            let t = ts[i]
            let aT = Float(alphas[t])
            let aPrev = i < n - 1 ? Float(alphas[ts[i + 1]]) : 1.0
            onStatus("Denoising step \(i + 1)/\(n)…", 0.15 + 0.7 * Double(i) / Double(n))
            let epsC = try runUnet(latents, t: t, hidden: condHidden, latH: latH, latW: latW)
            var eps = epsC
            if let u = uncondHidden {
                let epsU = try runUnet(latents, t: t, hidden: u, latH: latH, latW: latW)
                eps = [Float](repeating: 0, count: epsC.count)
                for j in 0..<eps.count { eps[j] = epsU[j] + params.guidance * (epsC[j] - epsU[j]) }
            }
            latents = Self.ddimStep(latents, eps: eps, aT: aT, aPrev: aPrev)
        }

        onStatus("Decoding image…", 0.9)
        let scale: Float = 0.18215
        var scaled = latents.map { $0 / scale }
        let imgFloats = try runVAE(&scaled, latH: latH, latW: latW)
        onStatus("Done", 1.0)
        return Self.imageFromCHW(imgFloats, w: params.size, h: params.size)
    }

    // ---- text encoder ----
    private func runTE(_ ids: [Int32]) throws -> [Float] {
        guard let s = te else {
            throw NSError(domain: "DiffusionEngine", code: 2, userInfo: [NSLocalizedDescriptionKey: "Text encoder not loaded"])
        }
        var idsV = ids
        let m = NSMutableData(bytes: &idsV, length: ids.count * 4)
        let input = try ORTValue(tensorData: m, elementType: .int32, shape: [1, NSNumber(value: ids.count)])
        let res = try s.run(withInputs: ["input_ids": input], outputNames: Set(["last_hidden_state"]), runOptions: nil)
        guard let out = res?["last_hidden_state"] else {
            throw NSError(domain: "DiffusionEngine", code: 3, userInfo: [NSLocalizedDescriptionKey: "No last_hidden_state output"])
        }
        return try Self.floats(of: out)
    }

    // ---- UNet denoiser ----
    private func runUnet(_ latents: [Float], t: Int, hidden: [Float], latH: Int, latW: Int) throws -> [Float] {
        guard let s = unet else {
            throw NSError(domain: "DiffusionEngine", code: 4, userInfo: [NSLocalizedDescriptionKey: "Denoiser not loaded"])
        }
        var l = latents
        let ml = NSMutableData(bytes: &l, length: latents.count * 4)
        let sample = try ORTValue(tensorData: ml, elementType: .float, shape: [NSNumber(value: 1), NSNumber(value: 4), NSNumber(value: latH), NSNumber(value: latW)])
        var tv: [Float] = [Float(t)]
        let mt = NSMutableData(bytes: &tv, length: 4)
        let timestep = try ORTValue(tensorData: mt, elementType: .float, shape: [1])
        var h = hidden
        let mh = NSMutableData(bytes: &h, length: hidden.count * 4)
        let enc = try ORTValue(tensorData: mh, elementType: .float, shape: [1, 77, 768])
        let res = try s.run(withInputs: [
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": enc,
        ], outputNames: Set(["out_sample"]), runOptions: nil)
        guard let out = res?["out_sample"] else {
            throw NSError(domain: "DiffusionEngine", code: 5, userInfo: [NSLocalizedDescriptionKey: "No out_sample output"])
        }
        return try Self.floats(of: out)
    }

    // ---- VAE decoder ----
    private func runVAE(_ scaled: inout [Float], latH: Int, latW: Int) throws -> [Float] {
        guard let s = vae else {
            throw NSError(domain: "DiffusionEngine", code: 6, userInfo: [NSLocalizedDescriptionKey: "Image decoder not loaded"])
        }
        let m = NSMutableData(bytes: &scaled, length: scaled.count * 4)
        let latent = try ORTValue(tensorData: m, elementType: .float, shape: [NSNumber(value: 1), NSNumber(value: 4), NSNumber(value: latH), NSNumber(value: latW)])
        let res = try s.run(withInputs: ["latent_sample": latent], outputNames: Set(["sample"]), runOptions: nil)
        guard let out = res?["sample"] else {
            throw NSError(domain: "DiffusionEngine", code: 7, userInfo: [NSLocalizedDescriptionKey: "No sample output"])
        }
        return try Self.floats(of: out)
    }

    private static func floats(of value: ORTValue) throws -> [Float] {
        let d = try value.tensorData() as Data
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    // ---- math (ported from src/image.js) ----
    static func alphasCumprod(_ num: Int) -> [Double] {
        var betas = [Double](repeating: 0, count: num)
        for i in 0..<num {
            let lin = sqrt(0.00085 + Double(i) / Double(num - 1) * (0.012 - 0.00085))
            betas[i] = lin * lin
        }
        var alphas = [Double](repeating: 0, count: num)
        var cum = 1.0
        for i in 0..<num {
            cum *= 1 - betas[i]
            alphas[i] = cum
        }
        return alphas
    }

    static func ddimStep(_ x: [Float], eps: [Float], aT: Float, aPrev: Float) -> [Float] {
        let sqrtA = sqrt(aT), sqrtAP = sqrt(aPrev)
        let sqrtB = sqrt(1 - aT), sqrtBP = sqrt(1 - aPrev)
        var out = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count {
            let x0 = (x[i] - sqrtB * eps[i]) / sqrtA
            out[i] = sqrtAP * x0 + sqrtBP * eps[i]
        }
        return out
    }

    static func randn() -> Float {
        var u = 0.0, v = 0.0
        while u == 0 { u = Double.random(in: 0..<1) }
        while v == 0 { v = Double.random(in: 0..<1) }
        return Float(sqrt(-2 * log(u)) * cos(2 * .pi * v))
    }

    // CHW float image in [-1,1] → UIImage (same mapping as floatsToCanvas in
    // src/image.js: v*0.5+0.5, clamped).
    static func imageFromCHW(_ data: [Float], w: Int, h: Int) -> UIImage {
        let n = w * h
        var px = [UInt8](repeating: 0, count: n * 4)
        for i in 0..<n {
            px[4 * i] = Self.clampU8(data[i] * 0.5 + 0.5)
            px[4 * i + 1] = Self.clampU8(data[n + i] * 0.5 + 0.5)
            px[4 * i + 2] = Self.clampU8(data[2 * n + i] * 0.5 + 0.5)
            px[4 * i + 3] = 255
        }
        let ctx = CGContext(
            data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        )!
        let img = ctx.makeImage()!
        return UIImage(cgImage: img)
    }

    private static func clampU8(_ x: Float) -> UInt8 {
        let v = (x * 255).rounded()
        return v < 0 ? 0 : v > 255 ? 255 : UInt8(v)
    }
}
