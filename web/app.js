/**
 * MechaDetect browser inference controller.
 * Supports all eight ONNX artifacts via WebGPU (preferred) or WebAssembly.
 *
 * Session lifecycle contract:
 *  - One active session at a time; the previous is released before a new one
 *    is created, bounding GPU/WASM memory.
 *  - loadGeneration counter ensures that a stale async load that races a
 *    newer request discards its result without touching shared state.
 *  - Upload, model selector, and provider checkbox are disabled while loading.
 *  - One warm-up inference runs after session creation to trigger shader
 *    compilation (WebGPU) or JIT (WASM) before user images arrive.
 *  - activeProvider reflects the session actually created, never assumed.
 *
 * Extensibility: add any model by appending an entry to metadata.json and
 * dropping its ONNX file in web/model/. No code change required.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  ortSession:      null,   // ort.InferenceSession in use
  activeProvider:  null,   // 'WebGPU' | 'WebAssembly' — actual session provider
  isReady:         false,
  isLoading:       false,
  loadError:       null,
  metadata:        null,
  activeModelInfo: null,
  // Monotonic counter — each load request bumps it; completion checks equality
  // so a stale async load never clobbers a newer one.
  loadGeneration:  0,
};
window.state = state;

// ---------------------------------------------------------------------------
// DOM element cache
// ---------------------------------------------------------------------------

const elements = {
  fileInput:            document.getElementById('fileInput'),
  uploadBtn:            document.getElementById('uploadBtn'),
  statusPill:           document.getElementById('statusPill'),
  statusPillText:       document.getElementById('statusPillText'),
  emptyState:           document.getElementById('emptyState'),
  cardsGrid:            document.getElementById('cardsGrid'),
  processingBanner:     document.getElementById('processingBanner'),
  processingText:       document.getElementById('processingText'),
  modelSelect:          document.getElementById('modelSelect'),
  forceWasmCheckbox:    document.getElementById('forceWasmCheckbox'),
  // Modal
  modalBackdrop:        document.getElementById('modalBackdrop'),
  modalCloseBtn:        document.getElementById('modalCloseBtn'),
  modalImg:             document.getElementById('modalImg'),
  modalVerdictText:     document.getElementById('modalVerdictText'),
  modalLatency:         document.getElementById('modalLatency'),
  modalIdentity:        document.getElementById('modalIdentity'),
  modalQuantization:    document.getElementById('modalQuantization'),
  modalThreshold:       document.getElementById('modalThreshold'),
  modalStatus:          document.getElementById('modalStatus'),
  modalScoreNum:        document.getElementById('modalScoreNum'),
  modalBarFill:         document.getElementById('modalBarFill'),
  modalTarget:          document.getElementById('modalTarget'),
  modalDot:             document.getElementById('modalDot'),
  int8Warning:          document.getElementById('int8Warning'),
};

// ---------------------------------------------------------------------------
// URL-param provider override (applies at page load only, before init)
// ---------------------------------------------------------------------------

if (new URLSearchParams(window.location.search).get('provider') === 'wasm') {
  elements.forceWasmCheckbox.checked = true;
}

// ---------------------------------------------------------------------------
// Cross-origin isolation — required for SharedArrayBuffer / WASM threading
// ---------------------------------------------------------------------------

function isCrossOriginIsolated() {
  return typeof self !== 'undefined' && !!self.crossOriginIsolated;
}

// ---------------------------------------------------------------------------
// ORT environment — configured once before the first session is created
// ---------------------------------------------------------------------------

function configureOrtEnv() {
  if (typeof ort === 'undefined') return;

  ort.env.logLevel = 'error';

  // Explicit WASM binary path prevents ORT from probing relative paths that
  // may 404 on unusual server configurations.
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';

  if (isCrossOriginIsolated()) {
    // SharedArrayBuffer is available: enable hardware-aware threading.
    // Leave at least one core free for the browser main thread; cap at 8.
    const cores = navigator.hardwareConcurrency || 2;
    ort.env.wasm.numThreads = Math.max(1, Math.min(8, cores - 1));
  } else {
    // Without COOP+COEP, SharedArrayBuffer is blocked and ORT's internal
    // thread pool cannot be used. Force single-threaded to avoid the
    // warning burst ORT emits when it detects the mismatch.
    ort.env.wasm.numThreads = 1;
  }
}

// ---------------------------------------------------------------------------
// Session factory — WebGPU first, fresh WASM session on any GPU failure
// ---------------------------------------------------------------------------

/**
 * Create an InferenceSession from the model protobuf and optional external
 * tensor-data descriptors. The protobuf is intentionally the only model
 * buffer assembled by this controller: Lattice's multi-gigabyte weights are
 * fetched by ORT from their individual URLs rather than concatenated into one
 * JavaScript ArrayBuffer.
 *
 * The returned object always has { session, provider }. On WebGPU failure a
 * completely fresh WASM session is created — the failed GPU path is not
 * reused, avoiding any residual GPU state leak.
 */
async function createSession(modelBuffer, externalData = []) {
  const hasWebGPU  = typeof navigator !== 'undefined' && !!navigator.gpu;
  const forceWasm  = elements.forceWasmCheckbox.checked;
  const skipGpu    = forceWasm || !hasWebGPU;

  // ORT Web's full graph optimizer can spend unbounded time/materializing
  // multi-gigabyte external-data graphs.  The validated Lattice bundles use
  // the basic path; ordinary single-file student models retain full
  // optimization.
  const hasExternalData = externalData.length > 0;
  const sessionOptions = {
    graphOptimizationLevel: hasExternalData ? 'basic' : 'all',
    executionMode: 'sequential',    // batch-1 repeated inference; no inter-op par
    ...(hasExternalData ? { externalData } : {}),
  };

  if (!skipGpu) {
    try {
      const session = await ort.InferenceSession.create(modelBuffer, {
        ...sessionOptions,
        executionProviders: ['webgpu'],
      });
      return { session, provider: 'WebGPU' };
    } catch (gpuErr) {
      console.warn('[MechaDetect] WebGPU session failed; retrying with WASM:', gpuErr.message);
      // Fall through — fresh WASM session created below, not a retry on the
      // broken GPU session, so no residual GPU state is carried over.
    }
  }

  const session = await ort.InferenceSession.create(modelBuffer, {
    ...sessionOptions,
    executionProviders: ['wasm'],
  });
  return { session, provider: 'WebAssembly' };
}

/**
 * Convert catalog external-data records to the ORT Web descriptor shape.
 * `path` is the location recorded inside the ONNX protobuf; `url` is the
 * browser-fetchable URL. Keeping both fields explicit makes artifacts
 * relocatable while allowing a static host to serve data files from `model/`.
 */
function resolveExternalData(modelInfo) {
  const records = Array.isArray(modelInfo?.external_data) ? modelInfo.external_data : [];
  return records.map((record, index) => {
    const path = record?.path || record?.filename;
    const url = record?.url || record?.path || record?.filename;
    if (typeof path !== 'string' || typeof url !== 'string' || !path || !url) {
      throw new Error(`Invalid external-data descriptor at index ${index}`);
    }
    return {
      path,
      data: new URL(url, window.location.href).toString(),
    };
  });
}

// ---------------------------------------------------------------------------
// Session release
// ---------------------------------------------------------------------------

async function releaseSession() {
  const prev = state.ortSession;
  state.ortSession    = null;
  state.isReady       = false;
  state.activeProvider = null;
  if (prev) {
    try { await prev.release(); } catch (_) {
      // release() absent in older ORT builds — safe to swallow.
    }
  }
}

// ---------------------------------------------------------------------------
// Warm-up — one zero-input inference to trigger shader/JIT compilation
// ---------------------------------------------------------------------------

async function warmUpSession(session) {
  try {
    const inputName = session.inputNames[0] || 'pixel_values';
    const zeros     = new Float32Array(3 * 224 * 224);
    await session.run({ [inputName]: new ort.Tensor('float32', zeros, [1, 3, 224, 224]) });
  } catch (warmErr) {
    console.warn('[MechaDetect] Warm-up non-fatal:', warmErr.message);
  }
}

// ---------------------------------------------------------------------------
// Spin-wait helper
// ---------------------------------------------------------------------------

async function waitUntilIdle() {
  while (state.isLoading) {
    await new Promise(r => setTimeout(r, 50));
  }
}

// ---------------------------------------------------------------------------
// Model initialisation — race-safe, generation-tracked
// ---------------------------------------------------------------------------

async function initClientModel() {
  // If a load is already in-flight, invalidate it and wait for it to exit.
  if (state.isLoading) {
    state.loadGeneration += 1; // marks the in-flight load as stale
    await waitUntilIdle();
  }

  const myGeneration = ++state.loadGeneration;
  state.isLoading    = true;
  state.isReady      = false;
  state.loadError    = null;

  // Disable controls during the switch
  elements.uploadBtn.disabled         = true;
  elements.modelSelect.disabled       = true;
  elements.forceWasmCheckbox.disabled = true;

  const modelInfo = state.activeModelInfo;

  try {
    const calibratedThreshold = modelInfo?.calibrated_threshold;
    const uiThreshold = Number.isFinite(calibratedThreshold)
      ? calibratedThreshold
      : modelInfo?.temporary_ui_threshold;
    if (!modelInfo || !modelInfo.path || !Number.isFinite(uiThreshold) || uiThreshold < 0 || uiThreshold > 1) {
      setStatus('Model configuration missing', 'error');
      elements.processingBanner.classList.remove('active');
      return;
    }
    const externalData = resolveExternalData(modelInfo);

    // A missing calibrated threshold is allowed only when the catalog marks
    // the fallback explicitly as a temporary UI threshold.
    if (!Number.isFinite(calibratedThreshold)) {
      console.warn(
        `[MechaDetect] ${modelInfo.id} has no calibrated threshold; ` +
        `using temporary UI threshold ${uiThreshold}.`,
      );
    }

    // Keep all external-data URLs as descriptors. They are fetched by ORT
    // during session creation and never assembled into this JS model buffer.

    // Release previous session before allocating a new one so we never hold
    // two large models in GPU/WASM memory simultaneously.
    await releaseSession();
    if (myGeneration !== state.loadGeneration) return; // superseded

    const isWasm      = elements.forceWasmCheckbox.checked || !(typeof navigator !== 'undefined' && !!navigator.gpu);
    const providerHint = isWasm ? 'Client Hardware (WASM)' : 'WebGPU';

    setStatus(`Loading ${modelInfo.name}…`, 'loading');
    elements.processingBanner.classList.add('active');
    elements.processingText.textContent = `Fetching ${modelInfo.name} (${modelInfo.size_label || '?'})…`;

    // Streaming fetch with live download progress
    const response = await fetch(modelInfo.path);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} fetching ${modelInfo.path}`);
    }

    const contentLength = response.headers.get('content-length');
    const totalBytes    = contentLength ? parseInt(contentLength, 10) : 0;
    const reader        = response.body.getReader();
    const chunks        = [];
    let received        = 0;

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // Abort as early as possible if a newer request came in
      if (myGeneration !== state.loadGeneration) {
        reader.cancel();
        return;
      }

      chunks.push(value);
      received += value.length;

      if (totalBytes > 0) {
        const pct = Math.round((received / totalBytes) * 100);
        const mb  = (received   / 1048576).toFixed(0);
        const tot = (totalBytes / 1048576).toFixed(0);
        elements.processingText.textContent =
          `Downloading ${modelInfo.name} — ${mb} / ${tot} MB (${pct}%)`;
      }
    }

    if (myGeneration !== state.loadGeneration) return;

    // Assemble only the small ONNX protobuf. External tensor data remains
    // represented by URL descriptors passed to ORT below.
    const modelBuffer = new ArrayBuffer(received);
    const modelView   = new Uint8Array(modelBuffer);
    let   offset      = 0;
    for (const chunk of chunks) {
      modelView.set(chunk, offset);
      offset += chunk.length;
    }

    elements.processingText.textContent =
      externalData.length > 0
        ? `Fetching ${externalData.length} weight shards and compiling ${modelInfo.name} for ${providerHint}…`
        : `Compiling ${modelInfo.name} for ${providerHint}…`;

    const { session, provider } = await createSession(modelBuffer, externalData);

    if (myGeneration !== state.loadGeneration) {
      // Superseded while compiling — discard without exposing to state
      try { await session.release(); } catch (_) {}
      return;
    }

    state.ortSession    = session;
    state.activeProvider = provider;

    elements.processingText.textContent = `Warming up ${modelInfo.name} on ${provider}…`;
    await warmUpSession(session);

    if (myGeneration !== state.loadGeneration) return;

    state.isReady = true;
    elements.processingBanner.classList.remove('active');
    setStatus(`${modelInfo.name} · ${provider}`, 'ready');

    console.log(
      `[MechaDetect] Ready: ${modelInfo.id} on ${provider}` +
      ` | threads=${ort.env.wasm.numThreads}` +
      ` | crossOriginIsolated=${isCrossOriginIsolated()}`
    );
  } catch (err) {
    if (myGeneration !== state.loadGeneration) return;
    state.loadError = err;
    elements.processingBanner.classList.remove('active');
    setStatus('Model load failed — see console', 'error');
    console.error('[MechaDetect] Initialization error:', err);
  } finally {
    if (myGeneration === state.loadGeneration) {
      state.isLoading = false;
      elements.uploadBtn.disabled         = false;
      elements.modelSelect.disabled       = false;
      elements.forceWasmCheckbox.disabled = false;
    }
  }
}

// ---------------------------------------------------------------------------
// Status pill
// ---------------------------------------------------------------------------

function setStatus(text, kind) {
  elements.statusPillText.textContent = text;
  const pill = elements.statusPill;
  pill.classList.remove('status-loading', 'status-ready', 'status-error');
  if (kind) pill.classList.add(`status-${kind}`);
}

// ---------------------------------------------------------------------------
// ImageNet preprocessing — NCHW Float32 [1, 3, 224, 224]
// ---------------------------------------------------------------------------

function preprocessImage(img) {
  const canvas  = document.createElement('canvas');
  canvas.width  = 224;
  canvas.height = 224;
  const ctx     = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, 224, 224);
  const raw = ctx.getImageData(0, 0, 224, 224).data;

  const mean  = [0.485, 0.456, 0.406];
  const std   = [0.229, 0.224, 0.225];
  const N     = 224 * 224;
  const buf   = new Float32Array(3 * N);

  for (let i = 0; i < N; i++) {
    buf[i]         = (raw[i * 4]     / 255.0 - mean[0]) / std[0]; // R
    buf[N + i]     = (raw[i * 4 + 1] / 255.0 - mean[1]) / std[1]; // G
    buf[2 * N + i] = (raw[i * 4 + 2] / 255.0 - mean[2]) / std[2]; // B
  }

  return new ort.Tensor('float32', buf, [1, 3, 224, 224]);
}

// ---------------------------------------------------------------------------
// Inference — serialised so concurrent uploads queue without racing the session
// ---------------------------------------------------------------------------

let inferenceLock = Promise.resolve();

async function detectInBrowser(img, filename) {
  const currentLock = inferenceLock;
  let releaseLock;
  inferenceLock = new Promise(resolve => { releaseLock = resolve; });
  try {
    await currentLock;
    return await executeInference(img, filename);
  } finally {
    releaseLock();
  }
}

async function executeInference(img, filename) {
  // Wait out any in-flight model switch
  if (state.isLoading) {
    elements.processingBanner.classList.add('active');
    elements.processingText.textContent = 'Waiting for model…';
    await waitUntilIdle();
  }

  if (!state.ortSession || !state.isReady) {
    const reason = state.loadError
      ? (state.loadError.message || String(state.loadError))
      : 'Model not ready';
    throw new Error(reason);
  }
  const t0 = performance.now();

  const inputTensor = preprocessImage(img);
  const inputName   = state.ortSession.inputNames[0] || 'pixel_values';
  const results     = await state.ortSession.run({ [inputName]: inputTensor });

  const t1      = performance.now();
  const latency = Math.round(t1 - t0);

  const outputName   = state.ortSession.outputNames[0] || 'probabilities';
  const outputTensor = results[outputName] || Object.values(results)[0];

  if (!outputTensor || outputTensor.data.length < 2) {
    throw new Error('Unexpected output tensor shape from model');
  }

  const pAuth      = Number(outputTensor.data[0]);
  const pAigc      = Number(outputTensor.data[1]);
  const calibratedThreshold = state.activeModelInfo.calibrated_threshold;
  const threshold = Number.isFinite(calibratedThreshold)
    ? calibratedThreshold
    : state.activeModelInfo.temporary_ui_threshold;
  const isAigc     = pAigc >= threshold;
  const confidence = isAigc ? pAigc : pAuth;
  const scorePercent = Math.round(confidence * 100);
  const label      = isAigc ? 'AIGC' : 'Original';

  console.log(`[MechaDetect] "${filename}":`, {
    P_Authentic: pAuth.toFixed(4),
    P_AIGC:      pAigc.toFixed(4),
    Verdict:     label,
    Confidence:  `${scorePercent}%`,
    Latency:     `${latency}ms`,
    Provider:    state.activeProvider,
    Model:       state.activeModelInfo.id,
  });

  return {
    isAigc,
    pAuth,
    pAigc,
    modelInfo:    state.activeModelInfo,
    scorePercent,
    label,
    latency,
    device: `${state.activeProvider} (100% Client-Side)`,
  };
}

// ---------------------------------------------------------------------------
// Card rendering
// ---------------------------------------------------------------------------

function insertCard(result, imageUrl, altText) {
  if (elements.emptyState) elements.emptyState.style.display = 'none';
  elements.cardsGrid.style.display = 'grid';

  const card  = document.createElement('article');
  const kind  = result.isAigc ? 'red' : 'green';
  card.className = `upload-card card-${kind} card-new`;
  card.setAttribute('data-verdict', result.label);
  card.setAttribute('data-score',   result.scorePercent);

  card.innerHTML = `
    <img class="card-image" src="${imageUrl}" alt="${altText || result.label + ' image'}" />
    <div class="card-meta">
      <div class="result-tag">${result.label}</div>
      <div class="score">${result.scorePercent}</div>
    </div>
  `;

  card.addEventListener('click', () => openModal({
    imageUrl,
    verdict:   result.label,
    score:     result.scorePercent,
    isAigc:    result.isAigc,
    latency:   result.latency,
    device:    result.device,
    modelInfo: result.modelInfo,
  }));

  elements.cardsGrid.prepend(card);
}

// ---------------------------------------------------------------------------
// Detail modal
// ---------------------------------------------------------------------------

function openModal(data) {
  elements.modalImg.src                 = data.imageUrl;
  elements.modalVerdictText.textContent = data.isAigc ? 'AI-Generated Content' : 'Authentic / Original';
  elements.modalScoreNum.textContent    = `${data.score}%`;
  elements.modalBarFill.style.width     = `${data.score}%`;

  const color = data.isAigc ? 'var(--red)' : 'var(--green)';
  elements.modalDot.style.background      = color;
  elements.modalScoreNum.style.color      = color;
  elements.modalBarFill.style.background  = color;
  elements.modalTarget.textContent        = data.device;
  elements.modalLatency.textContent       = `${data.latency} ms`;
  elements.modalIdentity.textContent      = data.modelInfo.model_family || '--';
  elements.modalQuantization.textContent  = data.modelInfo.precision_label || data.modelInfo.quantization || '--';
  const hasCalibratedThreshold = Number.isFinite(data.modelInfo.calibrated_threshold);
  const displayedThreshold = hasCalibratedThreshold
    ? data.modelInfo.calibrated_threshold
    : data.modelInfo.temporary_ui_threshold;
  elements.modalThreshold.textContent = displayedThreshold == null
    ? '--'
    : `${displayedThreshold}${hasCalibratedThreshold ? '' : ' (temporary UI)'}`;
  elements.modalStatus.textContent = data.modelInfo.evaluation_status   || '--';

  elements.modalBackdrop.classList.add('open');
}

function closeModal() {
  elements.modalBackdrop.classList.remove('open');
}

// ---------------------------------------------------------------------------
// Image decoding
// ---------------------------------------------------------------------------

async function loadImage(file) {
  if (typeof createImageBitmap === 'function') {
    try { return await createImageBitmap(file); } catch (_) {}
  }
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload  = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error(`Failed to decode "${file.name}"`)); };
    img.src = url;
  });
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = () => resolve(reader.result);
    reader.onerror = () => reject(new Error(`Failed to read "${file.name}"`));
    reader.readAsDataURL(file);
  });
}

// ---------------------------------------------------------------------------
// File upload handler
// ---------------------------------------------------------------------------

async function handleFiles(files) {
  if (!files || files.length === 0) return;
  const imageFiles = Array.from(files).filter(
    f => f.type.startsWith('image/') || /\.(webp|jpe?g|png|avif|heic)$/i.test(f.name)
  );
  if (imageFiles.length === 0) return;

  elements.processingBanner.classList.add('active');

  for (let i = 0; i < imageFiles.length; i++) {
    const file = imageFiles[i];
    if (!file.size) {
      alert(`"${file.name}" is empty. Please check the download.`);
      continue;
    }
    elements.processingText.textContent =
      `Running inference on "${file.name}" (${i + 1}/${imageFiles.length})…`;
    try {
      const [img, dataUrl] = await Promise.all([loadImage(file), readFileAsDataURL(file)]);
      const result = await detectInBrowser(img, file.name);
      insertCard(result, dataUrl, file.name);
    } catch (err) {
      console.error(`[MechaDetect] Error analyzing ${file.name}:`, err);
      alert(`Could not analyze ${file.name}: ${err?.message || String(err)}`);
    }
  }

  elements.processingBanner.classList.remove('active');
}

// ---------------------------------------------------------------------------
// Model selector — grouped by family × scope, with precision/size/INT8 warning
// ---------------------------------------------------------------------------

/**
 * Build <optgroup>-organised selector from the catalog.
 * Grouping key: "${family} ${scope}" (e.g. "Atom Super").
 * Groups appear in catalog declaration order; models within each group do too.
 * Extensible: any catalog entry with a family+scope key gets its own group
 * automatically — no code change needed for new families (e.g. Lattice).
 */
function populateModelSelector(students) {
  const groups = new Map();
  for (const m of students) {
    const key = `${m.family} ${m.scope}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(m);
  }

  let html = '';
  for (const [groupLabel, members] of groups) {
    html += `<optgroup label="${groupLabel}">`;
    for (const m of members) {
      const warn = m.is_experimental_int8 ? ' ⚠ degraded accuracy' : '';
      html += `<option value="${m.id}">${m.precision_label} · ${m.size_label}${warn}</option>`;
    }
    html += '</optgroup>';
  }
  elements.modelSelect.innerHTML = html;
}

/** Show or hide the INT8 degraded-accuracy callout based on the selected model. */
function updateInt8Warning(modelInfo) {
  if (!elements.int8Warning) return;
  if (modelInfo?.is_experimental_int8) {
    elements.int8Warning.classList.add('visible');
  } else {
    elements.int8Warning.classList.remove('visible');
  }
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

function setupListeners() {
  elements.uploadBtn.addEventListener('click', () => elements.fileInput.click());

  if (elements.emptyState) {
    elements.emptyState.addEventListener('click', () => elements.fileInput.click());
  }

  elements.fileInput.addEventListener('change', e => {
    handleFiles(e.target.files);
    e.target.value = '';
  });

  window.addEventListener('dragover', e => {
    e.preventDefault();
    document.body.classList.add('dragover');
  });
  window.addEventListener('dragleave', e => {
    if (e.clientX <= 0 || e.clientY <= 0) document.body.classList.remove('dragover');
  });
  window.addEventListener('drop', e => {
    e.preventDefault();
    document.body.classList.remove('dragover');
    if (e.dataTransfer?.files) handleFiles(e.dataTransfer.files);
  });

  // Model switch: update activeModelInfo then trigger a fresh session load.
  // Generation counter ensures a stale in-flight load cannot win.
  elements.modelSelect.addEventListener('change', e => {
    const selected = state.metadata?.students.find(m => m.id === e.target.value);
    if (selected) {
      state.activeModelInfo = selected;
      updateInt8Warning(selected);
      initClientModel();
    }
  });

  // Provider switch: recreate the session on the new provider.
  elements.forceWasmCheckbox.addEventListener('change', () => {
    initClientModel();
  });

  elements.modalCloseBtn.addEventListener('click', closeModal);
  elements.modalBackdrop.addEventListener('click', e => {
    if (e.target === elements.modalBackdrop) closeModal();
  });
  window.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function loadMetadataAndInit() {
  configureOrtEnv();

  try {
    const res  = await fetch('model/metadata.json');
    if (!res.ok) throw new Error(`HTTP ${res.status} loading metadata.json`);
    const data = await res.json();
    state.metadata = data;

    if (!data.students || data.students.length === 0) {
      elements.modelSelect.innerHTML = '<option value="">No models available</option>';
      elements.modelSelect.disabled  = true;
      setStatus('No models configured', 'error');
      return;
    }

    populateModelSelector(data.students);

    const defaultModel =
      data.students.find(m => m.id === data.default_model) || data.students[0];
    state.activeModelInfo     = defaultModel;
    elements.modelSelect.value = defaultModel.id;
    elements.modelSelect.disabled = false;
    updateInt8Warning(defaultModel);

    await initClientModel();
  } catch (err) {
    state.loadError = err;
    elements.modelSelect.disabled = true;
    setStatus('Metadata failed to load', 'error');
    console.error('[MechaDetect] Failed to load metadata:', err);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  setupListeners();
  loadMetadataAndInit();
});
