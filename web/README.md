# NanoGuard: Client-Side WebGPU AIGC Detector

NanoGuard is a lightweight, high-performance in-browser AI-generated image detection website built for **TechJam 2026**. It runs detection locally on the user's device via **WebGPU** (with automatic WebAssembly fallback) using an optimized student model.

---

## Key Hackathon Highlights

- **Zero Server Inference Cost for TikTok**: By shifting provenance detection from cloud GPU clusters to client hardware (via WebGPU), TikTok can eliminate server-side inference overhead for millions of daily image uploads.
- **Privacy Preserving**: Images never leave the user's device or browser tab.
- **Ultra Low Latency**: Native GPU shader pipelines deliver real-time inference.
- **Graceful Fallback**: Automatically falls back to WebAssembly (WASM CPU) if WebGPU is not supported by the client hardware/browser.

---

## Quickstart: Running the Web App Locally

1. Open your terminal in this repository:
   ```bash
   cd "C:\repos\techjam 26"
   ```

2. Start the server:
   ```bash
   python web/serve.py --host 0.0.0.0 --port 8000
   ```
   The positional form `python web/serve.py 8000` remains supported.

3. Open your browser:
   Navigate to [http://localhost:8000](http://localhost:8000) (Google Chrome or Microsoft Edge recommended for native WebGPU support).

## Opening the Demo on an iPhone

WebGPU requires a secure context on iPhone Safari. `http://localhost` is
secure-context eligible on the same device, but an IP address reached through
LAN access or port forwarding is not. Use HTTPS for a phone or public URL.

For a trusted local certificate, install [mkcert](https://github.com/FiloSottile/mkcert),
then create a certificate containing the PC's LAN address:

```bash
mkcert -install
mkcert -key-file web/nanoguard-key.pem -cert-file web/nanoguard-cert.pem 192.168.1.50 localhost 127.0.0.1
python web/serve.py --host 0.0.0.0 --port 8443 \
  --certfile web/nanoguard-cert.pem --keyfile web/nanoguard-key.pem
```

Install mkcert's local CA on the iPhone before opening
`https://192.168.1.50:8443`; a self-signed certificate that the phone does not
trust will be rejected. For Internet port forwarding, use a public DNS name
with a publicly trusted certificate (or a tunnel such as Cloudflare Tunnel)
instead of forwarding plain HTTP.

The ONNX model is tracked with Git LFS. After cloning, install Git LFS and
pull the binary before starting the server:

```bash
git lfs install
git lfs pull
```
---

## Directory Structure

```text
web/
├── index.html                 # NanoGuard frontend UI matching design screenshot
├── app.js                     # WebGPU & ONNX Runtime Web client-side engine
├── serve.py                   # HTTP server with COOP/COEP isolation headers
├── model/
│   ├── metadata.json          # Model configuration (updated by orchestration)
│   └── (student_*.onnx)       # ONNX exports populated by the orchestration pipeline
└── samples/                   # Pre-bundled authentic & AIGC test images
    ├── sample_authentic_imagenet.jpg
    └── sample_aigc_krea.jpg
```

---

## Orchestrated Model Export

The student models and metadata shown in the UI are generated automatically.
Do not manually edit `metadata.json` with fabricated paths.

The full orchestration pipeline (run via the root training orchestrator) will:
1. Distill each float student and independently promote only passing checkpoints.
2. Train and gate ATT checkpoints, retaining the promoted float checkpoint when ATT fails.
3. Export the selected float/ATT checkpoint to ONNX, then create and evaluate static INT8 PTQ.
4. Copy the verified artifacts into `web/model/` and populate `metadata.json` from their sidecars.

Before orchestration runs, the web application fails closed and displays a pending state.

The metadata JSON schema enforces strict properties for validation:
- \`default_model\`: `<id>`
- \`students\`: Array of objects matching the artifact schema from \`PolishExportPTQ\`.
  - \`id\`: string
  - \`name\`: string
  - \`path\`: string (relative path)
  - \`model_family\`: string
  - \`variant\`: string
  - \`quantization\`: "float32" | "static_int8"
  - \`calibrated_threshold\`: float (from actual passed evaluation, no fallback)
  - \`manifest_digest\`: string
  - \`evaluation_status\`: "promoted" | "experimental" (never "promoted" without evidence)
  - \`input_size\`: array of ints (e.g. \`[3, 224, 224]\`)
  - \`preprocessing_version\`: string (e.g. "2")
  - \`artifact_sha256\`: string
  - \`artifact_size_bytes\`: integer
