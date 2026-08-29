# NanoGuard: Client-Side WebGPU AIGC Detector

NanoGuard is a lightweight, high-performance in-browser AI-generated image detection website built for **TechJam 2026**. It runs detection locally on the user's device via **WebGPU** (with automatic WebAssembly fallback) using an optimized ONNX export of **Checkpoint 2** (`models/teachers/iteration1/checkpoint2`).

---

## Key Hackathon Highlights

- **Zero Server Inference Cost for TikTok**: By shifting provenance detection from cloud GPU clusters to client hardware (via WebGPU), TikTok can eliminate server-side inference overhead for millions of daily image uploads.
- **Privacy Preserving**: Images never leave the user's device or browser tab.
- **Ultra Low Latency**: Native GPU shader pipelines deliver real-time inference (sub-50ms).
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
│   └── checkpoint2.onnx       # Calibrated ONNX model for WebGPU execution
└── samples/                   # Pre-bundled authentic & AIGC test images
    ├── sample_authentic_imagenet.jpg
    └── sample_aigc_krea.jpg
```

---

## Exporting the Full DINOv3 ViT-H+ Checkpoint 2 to ONNX

To export or re-quantize the full teacher checkpoint directly into ONNX format:

```bash
python scripts/export_onnx_webgpu.py \
  --config configs/teacher_dinov3_checkpoint2_full_data.yaml \
  --checkpoint models/teachers/iteration1/checkpoint2/model-weights.safetensors \
  --output web/model/checkpoint2.onnx
```
