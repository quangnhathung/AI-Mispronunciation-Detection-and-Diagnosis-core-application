/* ============================================================
   Application — Main controller & event wiring
   ============================================================ */

(function () {
  let currentFile = null;
  let isRecording = false;

  /* ======= DOM references ======= */
  const $ = id => document.getElementById(id);
  const dropZone = $('dropZone');
  const fileInput = $('fileInput');
  const browseLink = $('browseLink');
  const fileRemove = $('fileRemove');
  const analyzeBtn = $('analyzeBtn');
  const inferModelSelect = $('inferModelSelect');
  const thresholdInput = $('thresholdInput');
  const textInput = $('textInput');
  const refreshBtn = $('refreshBtn');
  const sidebarToggle = $('sidebarToggle');
  const sidebar = $('sidebar');
  const themeToggle = $('themeToggle');
  const themeIcon = $('themeIcon');
  const retryBtn = $('retryBtn');
  const clearHistoryBtn = $('clearHistoryBtn');
  const recordBtn = $('recordBtn');
  const recordStopBtn = $('recordStopBtn');
  const recordSection = $('recordSection');

  /* ======= Navigation ======= */
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      const page = item.dataset.page;
      UI.navigateTo(page);
      if (page === 'models') loadModels();
      if (page === 'history') UI.renderHistory();
      if (page === 'dashboard') UI.updateDashboard(UI.getHistory());
      if (window.innerWidth <= 768) sidebar.classList.remove('open');
    });
  });

  /* ======= Sidebar toggle ======= */
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });

  /* ======= Theme toggle ======= */
  const savedTheme = localStorage.getItem('theme') || 'light';
  if (savedTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
  } else {
    themeIcon.innerHTML = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>';
  }

  themeToggle.addEventListener('click', () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    if (isDark) {
      document.documentElement.removeAttribute('data-theme');
      localStorage.setItem('theme', 'light');
      themeIcon.innerHTML = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>';
    } else {
      document.documentElement.setAttribute('data-theme', 'dark');
      localStorage.setItem('theme', 'dark');
      themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
    }
  });

  /* ======= Backend health check ======= */
  async function checkHealth() {
    try {
      const data = await API.getHealth();
      UI.updateBackendStatus(data && data.status === 'ok');
    } catch (_) {
      UI.updateBackendStatus(false);
    }
  }
  checkHealth();
  setInterval(checkHealth, 30000);

  /* ======= Model selector sync ======= */
  function syncModelSelectors(value) {
    inferModelSelect.value = value;
    const topSelect = document.querySelector('.model-select');
    if (topSelect) topSelect.value = value;
  }

  document.querySelector('.model-select')?.addEventListener('change', e => {
    syncModelSelectors(e.target.value);
  });
  inferModelSelect.addEventListener('change', e => {
    syncModelSelectors(e.target.value);
  });

  /* ======= Drag & Drop ======= */
  dropZone.addEventListener('click', e => {
    if (e.target.closest('.file-remove')) return;
    if (e.target.closest('.drop-link')) return;
    fileInput.click();
  });

  browseLink.addEventListener('click', e => {
    e.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileRemove.addEventListener('click', e => {
    e.stopPropagation();
    currentFile = null;
    fileInput.value = '';
    UI.hideFileInfo();
  });

  function handleFile(file) {
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/ogg', 'audio/mp4', 'audio/webm'];
    const validExt = /\.(wav|mp3|flac|ogg|m4a|webm)$/i;
    if (!validTypes.includes(file.type) && !validExt.test(file.name)) {
      UI.showToast('Please select a valid audio file (WAV, MP3, FLAC, OGG, M4A)', 'error');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      UI.showToast('File too large. Max 10 MB.', 'error');
      return;
    }
    currentFile = file;
    UI.hideRecordingUI();
    UI.showFileInfo(file);
  }

  /* ======= Recording ======= */
  Recorder.onTimerUpdate = (elapsed, formatted) => {
    UI.updateRecordTimer(formatted);
  };

  Recorder.onError = (msg) => {
    isRecording = false;
    recordBtn.classList.remove('recording');
    recordBtn.disabled = false;
    UI.hideRecordingUI();
    UI.showRecordError(msg);
  };

  Recorder.onDataReady = (blob, durationSec) => {
    isRecording = false;
    recordBtn.classList.remove('recording');
    recordBtn.disabled = false;
    UI.hideRecordingUI();

    const filename = `recording_${Date.now()}.webm`;
    const file = new File([blob], filename, { type: blob.type || 'audio/webm' });
    handleFile(file);
    UI.showToast(`Recording complete (${durationSec.toFixed(1)}s)`, 'success');
  };

  recordBtn.addEventListener('click', async () => {
    if (isRecording) return;
    try {
      isRecording = true;
      recordBtn.classList.add('recording');
      recordBtn.disabled = true;
      UI.showRecordingUI();
      await Recorder.start();
    } catch (err) {
      isRecording = false;
      recordBtn.classList.remove('recording');
      recordBtn.disabled = false;
      UI.hideRecordingUI();
      UI.showToast(err.message, 'error');
    }
  });

  recordStopBtn.addEventListener('click', () => {
    if (isRecording) {
      Recorder.stop();
    }
  });

  /* ======= Analyze ======= */
  analyzeBtn.addEventListener('click', runInference);
  retryBtn.addEventListener('click', runInference);

  async function runInference() {
    if (!currentFile) {
      UI.showToast('Please upload an audio file first', 'warning');
      return;
    }

    const model = inferModelSelect.value;
    const threshold = parseFloat(thresholdInput.value) || 0.5;
    const text = textInput.value.trim() || '';

    UI.showLoading();
    analyzeBtn.disabled = true;
    analyzeBtn.querySelector('.btn-label').classList.add('hidden');
    analyzeBtn.querySelector('.btn-spinner').classList.remove('hidden');

    try {
      const data = await API.infer(model, currentFile, text, threshold);
      UI.renderResults(data);
      UI.addHistory({
        file_name: currentFile.name,
        model_name: data.model_name || model,
        summary: data.summary,
      });
      UI.updateDashboard(UI.getHistory());
      UI.showToast('Analysis complete!', 'success');
    } catch (err) {
      UI.showError(err.message || 'Inference failed. Please try again.');
      UI.showToast(err.message, 'error');
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.querySelector('.btn-label').classList.remove('hidden');
      analyzeBtn.querySelector('.btn-spinner').classList.add('hidden');
    }
  }

  /* ======= Load models ======= */
  async function loadModels() {
    const grid = document.getElementById('modelsGrid');
    grid.innerHTML = '<p class="text-muted">Loading models...</p>';
    try {
      const data = await API.getModels();
      UI.renderModels(data.models || []);
    } catch (_) {
      grid.innerHTML = '<p class="text-muted">Could not load model info.</p>';
    }
  }

  /* ======= Refresh ======= */
  refreshBtn.addEventListener('click', () => {
    checkHealth();
    loadModels();
    UI.showToast('Refreshed', 'info');
  });

  /* ======= Clear history ======= */
  clearHistoryBtn.addEventListener('click', () => {
    if (confirm('Clear all analysis history?')) {
      UI.clearHistory();
      UI.updateDashboard(UI.getHistory());
    }
  });

  /* ======= Re-run from history ======= */
  document.addEventListener('rerun-history', async (e) => {
    const id = e.detail.id;
    const history = UI.getHistory();
    const entry = history.find(h => h.id === id);
    if (!entry) return;

    UI.navigateTo('inference');
    UI.showToast('Re-running previous analysis...', 'info');

    // Try to re-run with stored model
    inferModelSelect.value = entry.model_name || 'auto';
    document.querySelector('.model-select').value = entry.model_name || 'auto';

    if (entry.file_name) {
      UI.showToast('Please upload the same file again to re-run', 'warning');
    }
  });

  /* ======= Init ======= */
  UI.init();
  loadModels();

  console.log('%c Pronunciation AI 🎤 ', 'background:#4f6ef7;color:white;padding:8px 16px;border-radius:4px;font-size:16px;font-weight:bold;');
  console.log('%c Backend: http://localhost:8000 | Mock fallback enabled', 'color:#64748b;font-size:12px;');
})();
