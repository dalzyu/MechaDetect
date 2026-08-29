const puppeteer = require('puppeteer');

async function runBenchmark(url, forceWasm) {
  const expectedProvider = forceWasm ? 'WebAssembly' : 'WebGPU';
  const providerLabel = forceWasm ? 'wasm' : 'webgpu';
  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
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
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30_000 });
    if (forceWasm) {
      await page.click('#forceWasmCheckbox');
    }
    await page.waitForFunction(
      (provider) => window.state?.isReady === true && window.state?.activeProvider === provider,
      { timeout: 90_000 },
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
    if (!result || !Number.isFinite(result.confidence) || !Number.isFinite(result.processingTime)) {
      throw new Error(`Invalid browser inference result: ${JSON.stringify(result)}`);
    }
    console.log(`PROVIDER=${providerLabel}`);
    console.log(`RESULT=${JSON.stringify(result)}`);
  } finally {
    await browser.close();
  }
}

const url = process.argv[2] || 'http://localhost:8000';
const forceWasm = process.argv.includes('--wasm');
runBenchmark(url, forceWasm).catch((error) => {
  console.error('Benchmark error:', error);
  process.exitCode = 1;
});
