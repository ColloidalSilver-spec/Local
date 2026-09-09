# Local AI Studio — native iOS app (image generation)

This folder is a standalone iOS app that runs the **same Tiny-SD LCM ONNX model**
as the web generator, but as a real native app — escaping Safari's hard
~1 GB WebContent-process memory limit (the "major block" the memory-ceiling
test exposes) that made on-device diffusion crash in the browser.

**Why this works:** Safari (and PWAs / home-screen bookmarks) run web content in
a WebContent process with a ~1 GB jetsam budget that nothing web-side can
raise. A native app runs in its own process with a much higher memory limit,
and ORT's CoreML execution provider additionally routes the UNet/VAE onto the
Apple Neural Engine. No conversion needed — it loads the exact ONNX files the
web app downloads (fp16 weights, fp32 I/O).

## What's here

| File | Role |
| --- | --- |
| `LocalAIStudio/App.swift` | App entry |
| `LocalAIStudio/ContentView.swift` | UI: prompt, negative, steps/guidance, size, scheduler, download/progress, save |
| `LocalAIStudio/ModelStore.swift` | Streams the 8 model files from HuggingFace into Documents (bounded memory), or import from Files |
| `LocalAIStudio/Diffusion.swift` | The pipeline: tokenize → text encoder → LCM/DDIM denoise → VAE decode → UIImage (direct port of `src/image.js`) |
| `LocalAIStudio/CLIPTokenizer.swift` | CLIP BPE tokenizer (port of the web tokenizer; same ids) |

Model repo: `akameswa/lcm-tiny-sd-onnx-fp16` (Tiny-SD LCM, ~0.99 GB total:
text encoder 246 MB, UNet 647 MB, VAE 99 MB, tokenizer).

## Build & run (you need a Mac)

1. Requires **Xcode 16 or newer** (Xcode 26 works) and a recent iPhone (iOS 17+).
2. Double-click `LocalAIStudio.xcodeproj`. Xcode resolves the
   `microsoft/onnxruntime-swift-package-manager` dependency on first open
   (it downloads the ORT iOS framework — needs network, takes a couple of
   minutes).
3. **Signing**: select the `LocalAIStudio` target → Signing & Capabilities →
   choose your Team. A free Apple ID works for running on your own iPhone
   (personal team, 7-day re-sign or just re-run from Xcode; a paid account
   removes the expiry).
4. Connect your iPhone via USB (or use wireless debugging), select it as the
   run destination, press **Run**.
5. First launch: tap **Download model** (~1 GB, streams to disk). Or tap
   *"Or import files manually"* and pick the ONNX files you downloaded from the
   HF repo into the Files app (each slot is labeled).
6. Type a prompt, tap **Generate**. First generation is slower (CoreML
   compiles the model once); afterwards it's fast.

## Build in the cloud (no Mac needed)

This repo includes a GitHub Actions workflow (`.github/workflows/build-ipa.yml`)
that compiles the project on a cloud macOS runner and hands you an **unsigned
IPA** you can sideload with Sideloadly. The workflow file's source of truth
lives at `ci/build-ipa.yml` in this folder; the repo's hidden
`.github/workflows/` copy is assembled from it.

1. Push this folder to a GitHub repo (**public** repos get free macOS runner
   minutes; private repos bill you).
2. GitHub → **Actions** → **Build IPA** → **Run workflow**.
3. When it finishes, download the **LocalAIStudio-ipa** artifact (the build is
   unsigned on purpose — iOS won't run it yet).
4. On Windows, install **Sideloadly** + iTunes (or Apple Devices), plug in your
   iPhone, drag the IPA in, enter your Apple ID, and install.
5. Free Apple ID: the app expires after **7 days** — plug the phone in and let
   Sideloadly re-sign. A paid developer account (US$99/yr) extends that to a
   year.

First cloud build is the real compile test: if the workflow's "Build" step
fails, it lists the Swift errors — the most likely culprits (untested ports)
are called out in the next section.


## Verifying the port (important — I couldn't build this from here)

I have no Mac/Xcode in my environment, so **this code is untested on-device**.
What IS verified: the exact pipeline algorithm (same model, same sampler math)
runs correctly in the web app — it renders a genuine "red apple on a dish" at
4 LCM steps. If something misbehaves on device, in order of likelihood:

1. **Tokenizer mismatch** → garbled/off-prompt images. After your first
   generate, the debug line in the UI shows the token ids. Compare with the
   web app for the same prompt: "a photo of a red apple" should tokenize to
   `49406, 320, 1125, 539, 320, 736, 3055, 49407` (BOS, a, photo, of, a, red,
   apple, EOS — verified against the web app's transformers.js tokenizer).
   If they differ, the BPE/normalizer in `CLIPTokenizer.swift` needs fixing.
2. **CoreML EP rejects a node** → ORT falls back to CPU automatically (session
   creation tries CoreML then CPU-only). A CPU-only run is slow but still
   works — you can watch the status line.
3. **fp16 weight load** → not expected; the ObjC API needs float32 I/O which
   this model provides (verified from the ONNX graphs).

## Deliberate simplifications / future work

- Chat + TTS are intentionally **not** included — they already work fine in the
  web app (Safari can handle them; only diffusion hit the ceiling). Adding a
  `WKWebView` for the chat/voice UI, with image generation bridged to this
  native engine, is a natural next step.
- Only the Tiny-SD LCM preset ships here. The other models in the web app's
  `BUILTIN_MODELS` (SD Turbo, Pony V5, DreamShaper V7…) can be added the same
  way: swap `ModelStore.repo`/`files` and adjust the UNet inputs
  (`timestep_cond` etc.) in `Diffusion.swift`.
- img2img isn't included (Tiny-SD has no vae_encoder anyway).
