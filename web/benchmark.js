const puppeteer = require('puppeteer');

async function runBenchmark(url, forceWasm, headful) {
  const expectedProvider = forceWasm ? 'WebAssembly' : 'WebGPU';
  const providerLabel = forceWasm ? 'wasm' : 'webgpu';
  const browser = await puppeteer.launch({
    headless: headful ? false : 'new',
    args: headful
      ? [
          '--enable-unsafe-webgpu',
          '--enable-features=WebGPU',
        ]
      : [
          '--enable-unsafe-webgpu',
          '--enable-features=Vulkan,UseSkiaRenderer,WebGPU',
          '--disable-vulkan-surface',
          '--use-angle=vulkan',
        ],
  });

  try {
    const page = await browser.newPage();
    page.on('console', (message) => console.log(`[browser:${message.type()}] ${message.text()}`));
    page.on('pageerror', (error) => console.error(`[browser:pageerror] ${error.message}`));
    const benchmarkUrl = new URL(url);
    if (forceWasm) {
      benchmarkUrl.searchParams.set('provider', 'wasm');
    }
    await page.goto(benchmarkUrl.toString(), { waitUntil: 'networkidle0', timeout: 30_000 });
    await page.waitForFunction(
      (provider) => window.state?.isReady === true && window.state?.activeProvider === provider,
      { timeout: 180_000 },
      expectedProvider,
    );
    const result = await page.evaluate(async () => {
      const image = new Image();
      image.src = 'samples/sample_authentic_imagenet.jpg';
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = () => reject(new Error('Benchmark sample failed to load'));
      });
      return window.detectInBrowser(image, 'sample_authentic_imagenet.jpg');
    });
    if (!result || !Number.isFinite(result.scorePercent) || !Number.isFinite(result.latency)) {
      throw new Error(`Invalid browser inference result: ${JSON.stringify(result)}`);
    }
    console.log(`PROVIDER=${providerLabel}`);
    console.log(`RESULT=${JSON.stringify({
      provider: expectedProvider,
      modelId: result.modelInfo?.id,
      quantization: result.modelInfo?.quantization,
      isAigc: result.isAigc,
      pAuth: result.pAuth,
      pAigc: result.pAigc,
      scorePercent: result.scorePercent,
      latencyMs: result.latency,
    })}`);
  } finally {
    await browser.close();
  }
}

const url = process.argv[2] || 'http://localhost:8000';
const forceWasm = process.argv.includes('--wasm');
const headful = process.argv.includes('--headful');
runBenchmark(url, forceWasm, headful).catch((error) => {
  console.error('Benchmark error:', error);
  process.exitCode = 1;
});
