# MechaDetect Browser Demo

The demo runs six MechaDetect Float32 ONNX artifacts fetched from the immutable
Hugging Face revision recorded in `web/model/metadata.json`. Model weights are
not stored in this Git repository. WebGPU is preferred; ONNX Runtime Web falls
back to WebAssembly when WebGPU is unavailable or when Force WASM is checked.
Inference remains client-side after the model download: images are never sent
to a server. Lattice keeps the spectral-free `forward_tensor` path and loads
its weights through ONNX external-data shard URLs.

---

## Quickstart

1. Install project dependencies.

   ```bash
   uv sync --locked
   ```

2. Start the local server. The Python server sets the COOP/COEP headers that
   browsers require for cross-origin isolation and WASM multi-threading.

   ```bash
   uv run python web/serve.py --host 0.0.0.0 --port 8000
   ```

3. Open **Chrome** or **Edge** at [http://localhost:8000](http://localhost:8000).
   Firefox has partial WebGPU support; it will run on WASM automatically.
   Safari on macOS 15+ supports WebGPU.

The Hugging Face repository must be publicly browser-readable before external
users can load the models. A private repository returns HTTP 401 because the
demo intentionally contains no access token or other credential.

---

## Model Catalog

Six usable Float32 artifacts are hosted in `zye2/mechadetect-models`; the local
catalog contains immutable resolve URLs. The selector groups them by family and
training scope. Lattice is labelled for workstation-class hardware.

| Family | Scope  | Precision | Size | Notes |
|--------|--------|-----------|------|-------|
| Quark   | Super  | Float32   | 96 MB | **Default.** Recommended. |
| Quark   | Normal | Float32   | 96 MB | |
| Atom  | Super  | Float32   | 341 MB | Larger backbone; slower to load. |
| Atom  | Normal | Float32   | 341 MB | |
| Lattice | Normal | Float32 · Workstation | ~3.3 GB + shards | WebGPU preferred; spectral-free browser path. |
| Lattice | Super  | Float32 · Workstation | ~3.3 GB + shards | WebGPU preferred; spectral-free browser path. |

**Quark** uses a ViT-Small backbone (~25 M parameters).
**Atom** uses a ViT-Base backbone (~87 M parameters).
**Lattice** uses DINOv3 ViT-H+/16 (872,606,180 parameters, 32 layers, 1280
encoder dimensions). Its checkpoint config enables a spectral expert, but the
exported browser graph follows `ProvenanceModel.forward_tensor`, which does not
consume the PIL image sequence required by that expert. The sidecars and
catalog explicitly set `spectral_expert_omitted: true`; parity is only claimed
against this spectral-free path.
**Super** training scope includes an extended post-attention head.

The four withdrawn Static INT8 artifacts are absent from the usable catalog and
the Hugging Face release inventory.

**Extensibility:** upload an ONNX protobuf and any external-data shards to the
model repository, then append immutable artifact URLs to
`web/model/metadata.json`. For an external-data model, each record supplies the
location embedded in the ONNX graph as `path` and its browser-fetchable `url`;
`app.js` passes those descriptors directly to ONNX Runtime Web without
concatenating the weights into one JavaScript buffer.

---

## Provider Selection

### WebGPU (default)
The session is created with `executionProviders: ['webgpu']`. If the browser
supports WebGPU (`navigator.gpu` exists) and the Force WASM checkbox is
unchecked, WebGPU is attempted first. A failed GPU session is discarded and a
fresh WASM session is created — no residual GPU state is reused.

The status pill shows the provider that was actually used, not the one that was
requested.

### WebAssembly fallback
WASM is used automatically when:
- `navigator.gpu` is absent (no WebGPU support);
- Force WASM is checked;
- WebGPU session creation throws (driver error, low memory, etc.).

### Threading
WASM multi-threading requires `SharedArrayBuffer`, which is only available
when the page is cross-origin isolated (COOP + COEP response headers). The
bundled `serve.py` sets these headers automatically. When cross-origin
isolation is active, the thread count is set to `max(1, min(8, cores − 1))`.
Without it, single-threaded WASM is used and ORT's verbose threading warnings
are suppressed.

### Force WASM checkbox
Selecting this recreates the session on WASM immediately. De-selecting it
recreates the session on WebGPU. Page reload is not required for any provider
or model switch.

---

## Session and Memory Behaviour

- **One session at a time.** The previous session is released before a new
  model or provider is loaded so the browser never holds two large models in
  GPU/WASM memory simultaneously.
- **External-data loading.** Lattice protobufs are small model graphs. Their
  <=1-GB external shards are supplied to ORT Web as URL descriptors, so the
  controller never assembles the multi-gigabyte weights into one JavaScript
  `ArrayBuffer`. Shards are relocatable with their protobuf.
- **Race-safe switching.** A monotonic `loadGeneration` counter marks in-flight
  loads stale if a newer switch request arrives. The stale load discards its
  result without touching shared state.
- **Warm-up.** One zero-tensor inference runs after every session creation to
  trigger WebGPU shader compilation (or WASM JIT) before user images arrive.
  First real inference latency is therefore closer to steady-state.
- **Inference lock.** Concurrent uploads are serialised through a promise lock,
  so images always run in order against the same session.

---

## First-Load Engine Compilation

WebGPU compiles shaders on first use. For Quark Float32 expect 5–20 s on first
load depending on browser GPU driver caching behaviour. Subsequent loads of the
same model on the same browser benefit from the driver's shader cache. WASM
single-threaded inference for Quark Float32 is typically 800–1500 ms per image.

---

## Required HTTP Headers

The server must set these headers on every response for WASM threading and
full WebGPU access:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

`serve.py` sets both headers automatically. If you serve the files from a
different host (nginx, Caddy, S3 CloudFront), add these headers there.

---

## Opening the Demo on an iPhone

WebGPU requires a secure context on iPhone Safari. `http://localhost` is
eligible on the same device, but a LAN IP address is not. Use HTTPS.

```bash
# Generate a trusted local certificate (requires mkcert installed)
mkcert -install
mkcert -key-file web/mechadetect-key.pem -cert-file web/mechadetect-cert.pem \
  192.168.1.50 localhost 127.0.0.1

uv run python web/serve.py --host 0.0.0.0 --port 8443 \
  --certfile web/mechadetect-cert.pem --keyfile web/mechadetect-key.pem
```

Install mkcert's root CA on the iPhone before opening
`https://192.168.1.50:8443`.

---

## Directory Structure

```text
web/
├── index.html                      # MechaDetect frontend
├── app.js                          # ONNX Runtime Web inference controller
├── serve.py                        # Static server with COOP/COEP headers
├── benchmark.js                    # Puppeteer headless benchmark harness
├── model/
│   └── metadata.json               # Six-model remote Float32 catalog
└── samples/
    ├── sample_authentic_imagenet.jpg
    └── sample_aigc_krea.jpg
```

---

## Orchestrated Model Export

`web/model/metadata.json` is the single source of truth for the browser demo.
Each entry supplies an immutable Hugging Face `path`, an honest
`evaluation_status`, and either a verified `calibrated_threshold` or an
explicit `temporary_ui_threshold`. The temporary threshold is used only to
render a UI verdict; it is not a promotion claim.

Lattice sidecars in the Hugging Face model repository record the exact
checkpoint SHA-256, manifest digest, parameter count, preprocessing/output
contracts, ONNX Runtime parity numbers, and every external shard filename, byte
count, and SHA-256. The ONNX protobuf stores matching relative external-data
locations; the catalog maps those locations to immutable remote URLs.

The Lattice browser export is experimental and spectral-free by design:
`spectral_expert_omitted` is true and `parity_scope` names
`ProvenanceModel.forward_tensor`. Do not interpret its probabilities as
full-model (spectral-fused) parity.
