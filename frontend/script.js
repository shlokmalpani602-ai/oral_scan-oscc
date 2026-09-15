// ===== OralScan: real backend-connected frontend =====
// Set this to your deployed FastAPI URL before publishing the GitHub Pages frontend.
// Example: https://your-oralscan-api.onrender.com
const API_BASE_URL = window.ORALSCAN_API_URL || 'http://127.0.0.1:8000';
const PREDICT_ENDPOINT = `${API_BASE_URL.replace(/\/$/, '')}/predict`;

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const dzEmpty = document.getElementById('dzEmpty');
const dzPreview = document.getElementById('dzPreview');
const previewImg = document.getElementById('previewImg');
const scanLine = document.getElementById('scanLine');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const statusText = document.getElementById('statusText');
const resultPanel = document.getElementById('resultPanel');
const resultBadge = document.getElementById('resultBadge');
const ringFill = document.getElementById('ringFill');
const confidenceValue = document.getElementById('confidenceValue');
const resultNote = document.getElementById('resultNote');

let currentFile = null;
let currentObjectUrl = null;
let isAnalyzing = false;
let dragCounter = 0;
const RING_CIRCUMFERENCE = 326.73;
const MAX_FILE_SIZE_MB = 15;
const ACCEPTED_TYPES = ['image/jpeg', 'image/png'];

async function analyzeImage(file) {
  const formData = new FormData();
  formData.append('image', file, file.name);

  const response = await fetch(PREDICT_ENDPOINT, { method: 'POST', body: formData });
  let data = null;
  try { data = await response.json(); } catch (_) {}

  if (!response.ok) {
    throw new Error(data?.detail || `Backend error (${response.status})`);
  }
  if (!data || !['Normal', 'OSCC'].includes(data.result) || typeof data.confidence !== 'number') {
    throw new Error('Backend returned an invalid prediction.');
  }
  return data;
}

function handleFile(file) {
  if (!file) return;
  if (!ACCEPTED_TYPES.includes(file.type)) {
    statusText.textContent = 'Please choose a JPG or PNG image file.';
    return;
  }
  const sizeMB = file.size / (1024 * 1024);
  if (sizeMB > MAX_FILE_SIZE_MB) {
    statusText.textContent = `That file is too large (${sizeMB.toFixed(1)} MB). Please choose an image under ${MAX_FILE_SIZE_MB} MB.`;
    return;
  }

  currentFile = file;
  if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
  currentObjectUrl = URL.createObjectURL(file);
  previewImg.src = currentObjectUrl;
  dzEmpty.hidden = true;
  dzPreview.hidden = false;
  analyzeBtn.disabled = false;
  clearBtn.hidden = false;
  statusText.textContent = '';
  resultPanel.hidden = true;
}

function resetUpload() {
  currentFile = null;
  if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
  currentObjectUrl = null;
  fileInput.value = '';
  dzEmpty.hidden = false;
  dzPreview.hidden = true;
  analyzeBtn.disabled = true;
  clearBtn.hidden = true;
  statusText.textContent = '';
  resultPanel.hidden = true;
}

dropzone.addEventListener('click', () => { if (!isAnalyzing) fileInput.click(); });
browseBtn.addEventListener('click', (e) => { e.stopPropagation(); if (!isAnalyzing) fileInput.click(); });
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

dropzone.addEventListener('dragenter', (e) => {
  e.preventDefault(); if (isAnalyzing) return; dragCounter++; dropzone.classList.add('drag-over');
});
dropzone.addEventListener('dragover', (e) => e.preventDefault());
dropzone.addEventListener('dragleave', (e) => {
  e.preventDefault(); dragCounter = Math.max(0, dragCounter - 1);
  if (dragCounter === 0) dropzone.classList.remove('drag-over');
});
dropzone.addEventListener('drop', (e) => {
  e.preventDefault(); dragCounter = 0; dropzone.classList.remove('drag-over');
  if (!isAnalyzing) handleFile(e.dataTransfer.files[0]);
});
clearBtn.addEventListener('click', (e) => { e.stopPropagation(); resetUpload(); });

analyzeBtn.addEventListener('click', async () => {
  if (!currentFile || isAnalyzing) return;
  const fileBeingAnalyzed = currentFile;
  isAnalyzing = true;
  dropzone.classList.add('locked');
  analyzeBtn.disabled = true;
  clearBtn.hidden = true;
  resultPanel.hidden = true;
  statusText.textContent = 'Analyzing image with the BEiT + Random Forest model…';
  scanLine.hidden = false;
  scanLine.classList.add('scanning');

  try {
    const data = await analyzeImage(fileBeingAnalyzed);
    showResult(data);
    statusText.textContent = 'Analysis complete.';
  } catch (err) {
    console.error(err);
    statusText.textContent = `Analysis failed: ${err.message || 'Please try again.'}`;
  } finally {
    isAnalyzing = false;
    dropzone.classList.remove('locked');
    scanLine.classList.remove('scanning');
    scanLine.hidden = true;
    clearBtn.hidden = false;
    analyzeBtn.disabled = false;
  }
});

function showResult(data) {
  const isOSCC = data.result === 'OSCC';
  const confidence = Math.max(0, Math.min(1, data.confidence));
  const pct = Math.round(confidence * 100);

  resultBadge.textContent = data.result;
  resultBadge.className = 'badge ' + (isOSCC ? 'oscc' : 'normal');
  ringFill.classList.remove('normal', 'oscc');
  ringFill.classList.add(isOSCC ? 'oscc' : 'normal');

  const offset = RING_CIRCUMFERENCE * (1 - confidence);
  ringFill.style.transition = 'none';
  ringFill.style.strokeDashoffset = RING_CIRCUMFERENCE;
  void ringFill.getBoundingClientRect();
  ringFill.style.transition = 'stroke-dashoffset 1s ease, stroke 0.3s ease';
  ringFill.style.strokeDashoffset = offset;
  confidenceValue.textContent = pct + '%';

  resultNote.textContent = isOSCC
    ? 'Signs consistent with OSCC were detected in this image.'
    : 'No signs of OSCC were detected in this image.';
  resultPanel.hidden = false;
}
