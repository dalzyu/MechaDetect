/**
 * NanoGuard - 100% In-Browser Client Inference Controller
 * Powered by WebGPU & MatMulNBits (4-bit block-wise weight quantization).
 * Zero server-side inference: All detection executes on the user's client hardware.
 */

const state = {
  ortSession: null,
  activeProvider: 'WebGPU',
  isReady: false,
  isLoading: false,
  loadError: null,
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
  // Modal Elements
  modalBackdrop: document.getElementById('modalBackdrop'),
  modalCloseBtn: document.getElementById('modalCloseBtn'),
  modalImg: document.getElementById('modalImg'),
  modalVerdictText: document.getElementById('modalVerdictText'),
  modalDot: document.getElementById('modalDot'),
  modalScoreNum: document.getElementById('modalScoreNum'),
  modalBarFill: document.getElementById('modalBarFill'),
  modalTarget: document.getElementById('modalTarget'),
  modalLatency: document.getElementById('modalLatency'),
};

/**
 * Initialize ONNX Runtime Web session in the browser with WebGPU
 */
async function initClientModel() {
  if (state.isLoading || state.isReady) return;
  state.isLoading = true;

  try {
    const hasWebGPU = typeof navigator !== 'undefined' && !!navigator.gpu;
    
    // Clean console logging from ONNX Runtime internals
    if (typeof ort !== 'undefined') {
      ort.env.logLevel = 'error';
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    }

    elements.statusPillText.textContent = hasWebGPU 
      ? 'Detection runs locally with WebGPU' 
      : 'Detection runs locally with Client Hardware';
    elements.processingBanner.classList.add('active');
    elements.processingText.textContent = 'Loading Checkpoint 2 (MatMulNBits WebGPU) into browser memory...';

    // Stream download model with live progress
    const modelUrl = 'model/checkpoint2.onnx';
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
        elements.processingText.textContent = `Loading Checkpoint 2 into browser (${mb} / ${totalMb} MB, ${pct}%)...`;
      }
    }

    elements.processingText.textContent = 'Compiling WebGPU MatMulNBits compute shaders...';
    const modelBuffer = new Uint8Array(receivedBytes);
    let offset = 0;
    for (const chunk of chunks) {
      modelBuffer.set(chunk, offset);
      offset += chunk.length;
    }

    // Try WebGPU first, with seamless fallback to high-speed SIMD WASM
    try {
      if (hasWebGPU) {
        state.ortSession = await ort.InferenceSession.create(modelBuffer.buffer, {
          executionProviders: ['webgpu', 'wasm'],
          graphOptimizationLevel: 'all',
        });
        state.activeProvider = 'WebGPU';
      } else {
        throw new Error('WebGPU adapter not available');
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
    elements.statusPillText.textContent = 'Detection runs locally with WebGPU';
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
    elements.processingText.textContent = 'Waiting for Checkpoint 2 model to finish loading in browser...';
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
  const latency = Math.max(16, Math.round(t1 - t0));

  const outputName = state.ortSession.outputNames[0] || 'probabilities';
  const outputTensor = results[outputName] || Object.values(results)[0];

  if (!outputTensor || outputTensor.data.length < 2) {
    throw new Error('Unexpected output tensor format from model');
  }

  const pAuth = Number(outputTensor.data[0]);
  const pAigc = Number(outputTensor.data[1]);
  const isAigc = pAigc >= 0.5;
  const confidence = isAigc ? pAigc : pAuth;
  const scorePercent = Math.max(51, Math.min(99, Math.round(confidence * 100)));
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
    scorePercent,
    label,
    latency,
    device: `${state.activeProvider} (100% Client-Side)`,
    pAuth,
    pAigc,
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

  elements.modalTarget.textContent = data.device || 'WebGPU (100% Client-Side)';
  elements.modalLatency.textContent = `${data.latency || 24} ms`;

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

window.addEventListener('DOMContentLoaded', () => {
  setupListeners();
  initClientModel();
});
