/**
 * NanoGuard - 100% In-Browser Client Inference Controller
 * Powered by WebGPU and ONNX Runtime Web.
 * Zero server-side inference: All detection executes on the user's client hardware.
 */

const state = {
  ortSession: null,
  activeProvider: 'WebGPU',
  isReady: false,
  isLoading: false,
  loadError: null,
  metadata: null,
  activeModelInfo: null,
};
window.state = state;

const elements = {
  fileInput: document.getElementById('fileInput'),
  uploadBtn: document.getElementById('uploadBtn'),
  statusPill: document.getElementById('statusPill'),
  statusPillText: document.getElementById('statusPillText'),
  emptyState: document.getElementById('emptyState'),
  cardsGrid: document.getElementById('cardsGrid'),
  processingBanner: document.getElementById('processingBanner'),
  processingText: document.getElementById('processingText'),
  modelSelect: document.getElementById('modelSelect'),
  forceWasmCheckbox: document.getElementById('forceWasmCheckbox'),
  // Modal Elements
  modalBackdrop: document.getElementById('modalBackdrop'),
  modalCloseBtn: document.getElementById('modalCloseBtn'),
  modalImg: document.getElementById('modalImg'),
  modalVerdictText: document.getElementById('modalVerdictText'),
  modalLatency: document.getElementById('modalLatency'),
  modalIdentity: document.getElementById('modalIdentity'),
  modalQuantization: document.getElementById('modalQuantization'),
  modalThreshold: document.getElementById('modalThreshold'),
  modalStatus: document.getElementById('modalStatus'),
  modalScoreNum: document.getElementById('modalScoreNum'),
  modalBarFill: document.getElementById('modalBarFill'),
  modalTarget: document.getElementById('modalTarget'),
  modalDot: document.getElementById('modalDot'),
};

/**
 * Initialize ONNX Runtime Web session in the browser with WebGPU
 */
async function initClientModel() {
  if (state.isLoading) return;
  state.isLoading = true;
  state.isReady = false;
  if (state.ortSession) {
    // Cannot trivially release in JS without reloading, but we just reassign
    state.ortSession = null;
  }
  try {
    const hasWebGPU = typeof navigator !== 'undefined' && !!navigator.gpu;
    
    // Clean console logging from ONNX Runtime internals
    if (typeof ort !== 'undefined') {
      ort.env.logLevel = 'error';
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    }

    const modelInfo = state.activeModelInfo;
    if (!modelInfo || !modelInfo.path || modelInfo.path === 'not_configured' || typeof modelInfo.calibrated_threshold !== 'number') {
      elements.processingBanner.classList.remove('active');
      elements.statusPillText.textContent = 'Model pending orchestration setup';
      console.warn('[NanoGuard] Valid model not configured or evaluated. Waiting for orchestration to populate metadata.json.');
      state.isLoading = false;
      return;
    }
    const useWasmOnly = elements.forceWasmCheckbox.checked || !hasWebGPU;


    elements.statusPillText.textContent = useWasmOnly
      ? 'Detection runs locally with Client Hardware'
      : 'Detection runs locally with WebGPU';
    elements.processingBanner.classList.add('active');
    elements.processingText.textContent = `Loading ${modelInfo.name} into browser memory...`;

    // Stream download model with live progress
    const modelUrl = modelInfo.path;
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
    }

    const contentLength = response.headers.get('content-length');
    const totalBytes = contentLength ? parseInt(contentLength, 10) : 0;
    const reader = response.body.getReader();
    const chunks = [];
    let receivedBytes = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      receivedBytes += value.length;

      if (totalBytes > 0) {
        const pct = Math.round((receivedBytes / totalBytes) * 100);
        const mb = (receivedBytes / (1024 * 1024)).toFixed(0);
        const totalMb = (totalBytes / (1024 * 1024)).toFixed(0);
        elements.processingText.textContent = `Loading ${modelInfo.name} into browser (${mb} / ${totalMb} MB, ${pct}%)...`;
      }
    }

    elements.processingText.textContent = `Compiling compute shaders for ${modelInfo.name}...`;
    const modelBuffer = new Uint8Array(receivedBytes);
    let offset = 0;
    for (const chunk of chunks) {
      modelBuffer.set(chunk, offset);
      offset += chunk.length;
    }

    try {
      if (!useWasmOnly) {
        state.ortSession = await ort.InferenceSession.create(modelBuffer.buffer, {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
        });
        state.activeProvider = 'WebGPU';
      } else {
        throw new Error('Forced WASM or WebGPU adapter not available');
      }
    } catch (epErr) {
      console.warn('[NanoGuard] Primary WebGPU notice, initializing WASM provider:', epErr);
      state.ortSession = await ort.InferenceSession.create(modelBuffer.buffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      state.activeProvider = 'WebAssembly';
    }
    state.isReady = true;
    elements.statusPillText.textContent = `Detection runs locally with ${state.activeProvider}`;
    elements.processingBanner.classList.remove('active');
    console.log(`[NanoGuard] Model initialized successfully on ${state.activeProvider} (Zero server compute)!`);
  } catch (err) {
    state.loadError = err;
    const msg = err?.message || String(err);
    console.error('[NanoGuard] Model initialization error:', err);
    elements.processingBanner.classList.remove('active');
    elements.statusPillText.textContent = 'Model initialization failed';
  } finally {
    state.isLoading = false;
  }
}

/**
 * Preprocess image via HTML5 Canvas into standard ImageNet NCHW Float32 tensor [1, 3, 224, 224]
 */
function preprocessImage(img) {
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  ctx.drawImage(img, 0, 0, 224, 224);
  const imgData = ctx.getImageData(0, 0, 224, 224).data;

  // ImageNet normalization
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  const floatArray = new Float32Array(3 * 224 * 224);
  const channelLength = 224 * 224;

  for (let i = 0; i < channelLength; i++) {
    const r = imgData[i * 4];
    const g = imgData[i * 4 + 1];
    const b = imgData[i * 4 + 2];

    floatArray[i] = (r / 255.0 - mean[0]) / std[0]; // Red channel
    floatArray[channelLength + i] = (g / 255.0 - mean[1]) / std[1]; // Green channel
    floatArray[2 * channelLength + i] = (b / 255.0 - mean[2]) / std[2]; // Blue channel
  }

  return new ort.Tensor('float32', floatArray, [1, 3, 224, 224]);
}

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
  // If model is still loading, await completion
  if (!state.ortSession && state.isLoading) {
    elements.processingBanner.classList.add('active');
    elements.processingText.textContent = `Waiting for ${state.activeModelInfo.name} to finish loading in browser...`;
    while (state.isLoading && !state.ortSession) {
      await new Promise(r => setTimeout(r, 200));
    }
  }

  if (!state.ortSession) {
    const errText = state.loadError ? (state.loadError.message || String(state.loadError)) : 'Model not ready';
    throw new Error(`Model not ready: ${errText}`);
  }

  const t0 = performance.now();
  const inputTensor = preprocessImage(img);
  const inputName = state.ortSession.inputNames[0] || 'pixel_values';
  const feeds = { [inputName]: inputTensor };

  // Run model directly on client hardware
  const results = await state.ortSession.run(feeds);
  const t1 = performance.now();
  const latency = Math.round(t1 - t0);

  const outputName = state.ortSession.outputNames[0] || 'probabilities';
  const outputTensor = results[outputName] || Object.values(results)[0];

  if (!outputTensor || outputTensor.data.length < 2) {
    throw new Error('Unexpected output tensor format from model');
  }

  const pAuth = Number(outputTensor.data[0]);
  const pAigc = Number(outputTensor.data[1]);
  const isAigc = pAigc >= state.activeModelInfo.calibrated_threshold;
  const confidence = isAigc ? pAigc : pAuth;
  const scorePercent = Math.round(confidence * 100);
  const label = isAigc ? 'AIGC' : 'Original';

  console.log(`[NanoGuard] Client result for "${filename}":`, {
    P_Authentic: pAuth.toFixed(4),
    P_AIGC: pAigc.toFixed(4),
    Verdict: label,
    Confidence: `${scorePercent}%`,
    Latency: `${latency}ms`,
    Provider: state.activeProvider,
    ServerCompute: '0%',
  });

  return {
    isAigc,
    pAuth,
    pAigc,
    modelInfo: state.activeModelInfo,
    scorePercent,
    label,
    latency,
    device: `${state.activeProvider} (100% Client-Side)`,
  };
}

/**
 * Prepend card to grid
 */
function insertCard(result, imageUrl, altText) {
  if (elements.emptyState) {
    elements.emptyState.style.display = 'none';
  }
  elements.cardsGrid.style.display = 'grid';

  const card = document.createElement('article');
  const kind = result.isAigc ? 'red' : 'green';
  const label = result.label;

  card.className = `upload-card card-${kind} card-new`;
  card.setAttribute('data-verdict', label);
  card.setAttribute('data-score', result.scorePercent);

  card.innerHTML = `
    <img class="card-image" src="${imageUrl}" alt="${altText || label + ' image'}" />
    <div class="card-meta">
      <div class="result-tag">${label}</div>
      <div class="score">${result.scorePercent}</div>
    </div>
  `;

  card.addEventListener('click', () => {
    openModal({
      imageUrl,
      verdict: label,
      score: result.scorePercent,
      isAigc: result.isAigc,
      latency: result.latency,
      device: result.device,
      modelInfo: result.modelInfo,
    });
  });

  elements.cardsGrid.prepend(card);
}

/**
 * Open detail preview modal
 */
function openModal(data) {
  elements.modalImg.src = data.imageUrl;
  elements.modalVerdictText.textContent = data.isAigc ? 'AI-Generated Content' : 'Authentic / Original';
  elements.modalScoreNum.textContent = `${data.score}%`;
  elements.modalBarFill.style.width = `${data.score}%`;

  const color = data.isAigc ? 'var(--red)' : 'var(--green)';
  elements.modalDot.style.background = color;
  elements.modalScoreNum.style.color = color;
  elements.modalBarFill.style.background = color;
  elements.modalTarget.textContent = data.device;
  elements.modalLatency.textContent = `${data.latency} ms`;
  elements.modalIdentity.textContent = data.modelInfo.model_family || '--';
  elements.modalQuantization.textContent = data.modelInfo.quantization || '--';
  elements.modalThreshold.textContent = data.modelInfo.calibrated_threshold || '--';
  elements.modalStatus.textContent = data.modelInfo.evaluation_status || '--';

  elements.modalBackdrop.classList.add('open');
}

/**
 * Close detail preview modal
 */
function closeModal() {
  elements.modalBackdrop.classList.remove('open');
}

/**
 * Asynchronously decode image using native browser decoders (supports WebP, PNG, JPEG, AVIF, HEIC)
 */
async function loadImage(file) {
  if (typeof createImageBitmap === 'function') {
    try {
      return await createImageBitmap(file);
    } catch (bitmapErr) {
      console.warn('[NanoGuard] createImageBitmap notice, falling back to ObjectURL:', bitmapErr);
    }
  }

  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error(`Failed to decode image "${file.name}". File format may be corrupted.`));
    };
    img.src = objectUrl;
  });
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error(`Failed to read file: ${file.name}`));
    reader.readAsDataURL(file);
  });
}

/**
 * Handle image file upload
 */
async function handleFiles(files) {
  if (!files || files.length === 0) return;
  const imageFiles = Array.from(files).filter(f => f.type.startsWith('image/') || /\.(webp|jpe?g|png|avif|heic)$/i.test(f.name));
  if (imageFiles.length === 0) return;

  elements.processingBanner.classList.add('active');

  for (let i = 0; i < imageFiles.length; i++) {
    const file = imageFiles[i];

    if (!file.size || file.size === 0) {
      alert(`"${file.name}" is an empty (0 bytes) file. Please ensure the image downloaded completely.`);
      continue;
    }

    elements.processingText.textContent = `Running local inference on "${file.name}" (${i + 1}/${imageFiles.length})...`;

    try {
      const [img, dataUrl] = await Promise.all([
        loadImage(file),
        readFileAsDataURL(file),
      ]);

      // Execute 100% client-side inference on WebGPU
      const result = await detectInBrowser(img, file.name);
      insertCard(result, dataUrl, file.name);
    } catch (err) {
      const errorMsg = err?.message || (typeof err === 'string' ? err : JSON.stringify(err)) || String(err);
      console.error(`[NanoGuard] Error analyzing ${file.name}:`, err);
      alert(`Could not analyze ${file.name}: ${errorMsg}`);
    }
  }

  elements.processingBanner.classList.remove('active');
}

/**
 * Setup listeners
 */
function setupListeners() {
  elements.uploadBtn.addEventListener('click', () => elements.fileInput.click());

  if (elements.emptyState) {
    elements.emptyState.addEventListener('click', () => elements.fileInput.click());
  }

  elements.fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
    e.target.value = '';
  });

  window.addEventListener('dragover', (e) => {
    e.preventDefault();
    document.body.classList.add('dragover');
  });

  window.addEventListener('dragleave', (e) => {
    if (e.clientX <= 0 || e.clientY <= 0) {
      document.body.classList.remove('dragover');
    }
  });

  window.addEventListener('drop', (e) => {
    e.preventDefault();
    document.body.classList.remove('dragover');
    if (e.dataTransfer && e.dataTransfer.files) {
      handleFiles(e.dataTransfer.files);
    }
  });

  elements.modalCloseBtn.addEventListener('click', closeModal);
  elements.modalBackdrop.addEventListener('click', (e) => {
    if (e.target === elements.modalBackdrop) closeModal();
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });
}

async function loadMetadataAndInit() {
  try {
    const res = await fetch('model/metadata.json');
    const data = await res.json();
    state.metadata = data;

    if (!data.students || data.students.length === 0) {
      console.warn('[NanoGuard] Metadata is empty. Pending orchestration.');
      elements.modelSelect.innerHTML = '<option value="">No models available</option>';
      elements.modelSelect.disabled = true;
      elements.statusPillText.textContent = 'Model pending orchestration setup';
      state.isReady = false;
      return;
    }

    elements.modelSelect.innerHTML = data.students.map(m =>
      `<option value="${m.id}">${m.name}</option>`
    ).join('');

    const defaultId = data.default_model;
    state.activeModelInfo = data.students.find(m => m.id === defaultId) || data.students[0];
    elements.modelSelect.value = state.activeModelInfo.id;
    elements.modelSelect.disabled = false;

    elements.modelSelect.addEventListener('change', (e) => {
      const selectedId = e.target.value;
      state.activeModelInfo = state.metadata.students.find(m => m.id === selectedId);
      initClientModel();
    });
    elements.forceWasmCheckbox.addEventListener('change', () => {
      initClientModel();
    });

    initClientModel();
  } catch (err) {
    state.loadError = err;
    state.isReady = false;
    elements.modelSelect.disabled = true;
    elements.statusPillText.textContent = 'Model metadata failed to load';
    console.error('Failed to load metadata:', err);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  setupListeners();
  loadMetadataAndInit();
});
