# MechaDetect Browser Demo

The demo runs the selected MechaDetect ONNX model inside the browser. WebGPU is
preferred; ONNX Runtime Web falls back to WebAssembly when WebGPU is unavailable.
Inference does not require an image-upload API, so the selected image remains in
the browser process.

What this means in practice:

- no server-side model process for the demo;
- the image is not sent to this repository's server;
- WebAssembly keeps the interface usable on browsers without WebGPU;
- latency depends on the selected model, browser, and local hardware.

These properties describe deployment, not detector accuracy. TechJam-set and
state-of-the-art comparisons remain pending.

---

## Quickstart: Running the Web App Locally

1. Install the Git LFS payloads and project dependencies.

   ```bash
   git lfs install
   git lfs pull
   uv sync --locked
   npm ci
   ```

2. Start the server.

   ```bash
   uv run python web/serve.py --host 0.0.0.0 --port 8000
   ```

3. Open [http://localhost:8000](http://localhost:8000) in Chrome or Edge.

## Opening the Demo on an iPhone

WebGPU requires a secure context on iPhone Safari. `http://localhost` is
secure-context eligible on the same device, but an IP address reached through
LAN access or port forwarding is not. Use HTTPS for a phone or public URL.

For a trusted local certificate, install [mkcert](https://github.com/FiloSottile/mkcert),
then create a certificate containing the PC's LAN address:

```bash
mkcert -install
mkcert -key-file web/mechadetect-key.pem -cert-file web/mechadetect-cert.pem 192.168.1.50 localhost 127.0.0.1
uv run python web/serve.py --host 0.0.0.0 --port 8443 \
  --certfile web/mechadetect-cert.pem --keyfile web/mechadetect-key.pem
```

Install mkcert's local CA on the iPhone before opening
`https://192.168.1.50:8443`; a self-signed certificate that the phone does not
trust will be rejected. For Internet port forwarding, use a public DNS name
with a publicly trusted certificate (or a tunnel such as Cloudflare Tunnel)
instead of forwarding plain HTTP.

The ONNX model is tracked with Git LFS. A clone containing only the pointer file
cannot run inference; `git lfs pull` must complete before the server starts.

---

## Directory Structure

```text
web/
├── index.html                 # MechaDetect frontend
├── app.js                     # ONNX Runtime Web inference controller
├── serve.py                   # Static server with COOP/COEP headers
├── model/
│   ├── metadata.json          # Portable, relative model manifest
│   └── *.onnx                 # Selected Git LFS model artifacts
└── samples/                   # Optional local demonstration images
    ├── sample_authentic_imagenet.jpg
    └── sample_aigc_krea.jpg
```

---

## Orchestrated Model Export

Model metadata is produced from export sidecars. Do not write machine-local
checkpoint or artifact paths into `web/model/metadata.json`.

The delivery path:

1. selects a final student checkpoint;
2. exports float32 ONNX and calibrated static INT8 candidates;
3. checks graph structure and browser-provider behaviour;
4. copies only the selected browser artifact into `web/model/`; and
5. writes portable metadata using relative paths.

The current release choice is Atom Super float32. The INT8 candidates are
experimental because WebGPU and WebAssembly did not agree numerically on the
runtime comparison input.

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
