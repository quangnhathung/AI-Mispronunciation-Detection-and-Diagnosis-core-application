/* ============================================================
   UI Renderer — DOM manipulation & component builders
   ============================================================ */

const UI = (() => {

  /* ---------- Toast ---------- */
  function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; el.style.transform = 'translateX(40px)'; el.style.transition = '0.3s'; }, 3000);
    setTimeout(() => el.remove(), 3500);
  }

  /* ---------- Backend status ---------- */
  function updateBackendStatus(online) {
    const dot = document.querySelector('#backendStatus .status-dot');
    const label = document.querySelector('#backendStatus .status-label');
    if (!dot || !label) return;
    dot.className = `status-dot ${online ? 'online' : 'offline'}`;
    label.textContent = online ? 'Backend Online' : 'Backend Offline (Mock)';
  }

  /* ---------- Navigation ---------- */
  function navigateTo(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    const page = document.getElementById(`page-${pageId}`);
    if (page) page.classList.add('active');

    const navItem = document.querySelector(`.nav-item[data-page="${pageId}"]`);
    if (navItem) navItem.classList.add('active');

    const titles = { dashboard: 'Dashboard', inference: 'Inference', models: 'Model Info', history: 'History' };
    document.getElementById('pageTitle').textContent = titles[pageId] || 'Pronunciation AI';
  }

  /* ---------- File info ---------- */
  function showFileInfo(file) {
    const dropContent = document.getElementById('dropContent');
    const fileInfo = document.getElementById('fileInfo');
    const audioWrapper = document.getElementById('audioPlayerWrapper');
    const audio = document.getElementById('audioPlayer');

    dropContent.classList.add('hidden');
    fileInfo.classList.remove('hidden');
    audioWrapper.classList.add('hidden');

    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatSize(file.size);

    const url = URL.createObjectURL(file);
    audio.src = url;
    audio.onloadedmetadata = () => {
      audioWrapper.classList.remove('hidden');
    };

    document.getElementById('analyzeBtn').disabled = false;
  }

  function hideFileInfo() {
    const dropContent = document.getElementById('dropContent');
    const fileInfo = document.getElementById('fileInfo');
    const audioWrapper = document.getElementById('audioPlayerWrapper');
    const audio = document.getElementById('audioPlayer');

    dropContent.classList.remove('hidden');
    fileInfo.classList.add('hidden');
    audioWrapper.classList.add('hidden');
    audio.src = '';
    document.getElementById('analyzeBtn').disabled = true;
  }

  function showRecordingUI() {
    const btn = document.getElementById('recordBtn');
    btn.classList.add('hidden');
    btn.nextElementSibling?.classList.add('hidden');
    document.getElementById('recordStatus').classList.remove('hidden');
    document.getElementById('recordError').classList.add('hidden');
  }

  function hideRecordingUI() {
    const btn = document.getElementById('recordBtn');
    btn.classList.remove('hidden');
    btn.nextElementSibling?.classList.remove('hidden');
    document.getElementById('recordStatus').classList.add('hidden');
    document.getElementById('recordError').classList.add('hidden');
  }

  function showRecordError(msg) {
    document.getElementById('recordError').classList.remove('hidden');
    document.getElementById('recordErrorMessage').textContent = msg;
  }

  function updateRecordTimer(text) {
    document.getElementById('recordTimer').textContent = text;
  }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  }

  /* ---------- Results rendering ---------- */
  function renderResults(data) {
    const section = document.getElementById('resultSection');
    const loading = document.getElementById('loadingSection');
    const error = document.getElementById('errorSection');
    const empty = document.getElementById('emptyState');

    loading.classList.add('hidden');
    error.classList.add('hidden');
    empty.classList.add('hidden');

    if (!data || !data.success) {
      showToast('Inference failed', 'error');
      return;
    }

    section.classList.remove('hidden');
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Summary
    const s = data.summary || {};
    document.getElementById('sumTotal').textContent = s.total_phonemes || 0;
    document.getElementById('sumCorrect').textContent = s.correct_phonemes || 0;
    document.getElementById('sumIncorrect').textContent = s.incorrect_phonemes || 0;
    document.getElementById('sumAccuracy').textContent = (s.accuracy !== undefined && s.accuracy !== null)
      ? `${(s.accuracy * 100).toFixed(1)}%` : '—';

    // Phoneme table
    const tbody = document.getElementById('phonemeTbody');
    tbody.innerHTML = '';
    const predictions = data.predictions || [];
    if (predictions.length === 0) {
      // Show ASR result if no per-phoneme predictions
      const res = data.result;
      if (res && res.phoneme_sequence) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="7" class="text-muted" style="text-align:center;padding:20px;">
          ASR phoneme sequence: <strong>${res.phoneme_string || res.phoneme_sequence.join(' ')}</strong>
          (confidence: ${(res.overall_confidence * 100).toFixed(0)}%)
        </td>`;
        tbody.appendChild(tr);
      }
    } else {
      predictions.forEach((p, idx) => {
        const tr = document.createElement('tr');
        tr.dataset.idx = idx;
        tr.addEventListener('click', () => {
          document.querySelectorAll('.phoneme-table tbody tr').forEach(r => r.classList.remove('highlight'));
          tr.classList.add('highlight');
        });

        const statusLabel = p.status.charAt(0).toUpperCase() + p.status.slice(1);
        const confPct = (p.confidence * 100).toFixed(0);

        tr.innerHTML = `
          <td>${idx + 1}</td>
          <td><strong>/${p.phoneme}/</strong></td>
          <td><span class="status-badge ${p.status}">${statusLabel}</span></td>
          <td>
            <div class="confidence-bar-wrapper">
              <div class="confidence-bar">
                <div class="confidence-bar-fill" style="width:${confPct}%;background:${p.confidence >= 0.5 ? '#22c55e' : '#ef4444'}"></div>
              </div>
              <span class="confidence-text">${confPct}%</span>
            </div>
          </td>
          <td>${p.expected || '—'}</td>
          <td>${p.actual || '—'}</td>
          <td>
            <span class="tooltip-wrapper">
              ${p.reason || (p.status === 'correct' ? '✓ Good' : '—')}
              ${p.reason ? `<span class="tooltip-text">${p.reason}</span>` : ''}
            </span>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }

    // Charts
    const total = s.total_phonemes || 1;
    const correct = s.correct_phonemes || 0;
    const incorrect = s.incorrect_phonemes || 0;
    ChartManager.renderPie('resultPieChart', correct, incorrect);

    if (predictions.length > 0) {
      const labels = predictions.map(p => `/${p.phoneme}/`);
      const values = predictions.map(p => p.confidence);
      ChartManager.renderBar('resultBarChart', labels, values, 'Confidence');
    } else {
      ChartManager.destroy('resultBarChart');
    }
  }

  function showLoading() {
    document.getElementById('resultSection').classList.add('hidden');
    document.getElementById('errorSection').classList.add('hidden');
    document.getElementById('emptyState').classList.add('hidden');
    document.getElementById('loadingSection').classList.remove('hidden');
  }

  function showError(message) {
    document.getElementById('resultSection').classList.add('hidden');
    document.getElementById('loadingSection').classList.add('hidden');
    document.getElementById('emptyState').classList.add('hidden');
    document.getElementById('errorSection').classList.remove('hidden');
    document.getElementById('errorMessage').textContent = message || 'An unexpected error occurred. Please try again.';
  }

  /* ---------- Models rendering ---------- */
  function renderModels(models) {
    const grid = document.getElementById('modelsGrid');
    if (!models || models.length === 0) {
      grid.innerHTML = '<p class="text-muted">No models available.</p>';
      return;
    }
    grid.innerHTML = models.map(m => `
      <div class="model-card">
        <div class="model-card-header">
          <strong class="model-card-name">${m.display_name || m.name}</strong>
          <span class="model-card-badge ${m.status || 'unloaded'}">${m.status || 'unloaded'}</span>
        </div>
        <div class="model-card-arch">${m.architecture || '—'}</div>
        <div class="model-card-detail"><strong>Task:</strong> ${m.task || '—'}</div>
        <div class="model-card-detail"><strong>Phonemes:</strong> ${m.phoneme_set_size || '—'}</div>
        <div class="model-card-detail"><strong>Sample Rate:</strong> ${m.sample_rate || '—'} Hz</div>
        <div class="model-card-detail"><strong>Version:</strong> ${m.version || '—'}</div>
        <div class="model-card-detail"><strong>Requires Text:</strong> ${m.requires_text ? 'Yes' : 'No'}</div>
      </div>
    `).join('');
  }

  /* ---------- History ---------- */
  function getHistory() {
    try {
      return JSON.parse(localStorage.getItem('mdd_history') || '[]');
    } catch { return []; }
  }

  function addHistory(entry) {
    const list = getHistory();
    list.unshift({
      id: Date.now(),
      timestamp: new Date().toISOString(),
      ...entry,
    });
    if (list.length > 50) list.length = 50;
    localStorage.setItem('mdd_history', JSON.stringify(list));
    renderHistory();
  }

  function renderHistory() {
    const list = getHistory();
    const container = document.getElementById('historyList');
    if (list.length === 0) {
      container.innerHTML = '<p class="text-muted">No history yet. Run an analysis first.</p>';
      return;
    }
    container.innerHTML = list.map(h => {
      const acc = h.summary ? (h.summary.accuracy * 100).toFixed(1) : '—';
      const cls = h.summary && h.summary.accuracy >= 0.6 ? 'good' : 'bad';
      return `
        <div class="history-item" data-id="${h.id}">
          <div class="history-meta">
            <span class="history-name">${h.file_name || 'Unknown file'}</span>
            <span class="history-info">
              <span>🤖 ${h.model_name || 'auto'}</span>
              <span>📅 ${new Date(h.timestamp).toLocaleString()}</span>
              <span class="history-badge ${cls}">${acc}% accuracy</span>
            </span>
          </div>
          <button class="btn-text rerun-btn" data-id="${h.id}">Re-run</button>
        </div>
      `;
    }).join('');

    document.querySelectorAll('.rerun-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const id = parseInt(btn.dataset.id);
        // Re-run stored in app.js via event
        const event = new CustomEvent('rerun-history', { detail: { id } });
        document.dispatchEvent(event);
      });
    });
  }

  function clearHistory() {
    localStorage.removeItem('mdd_history');
    renderHistory();
    showToast('History cleared', 'info');
  }

  /* ---------- Dashboard ---------- */
  function updateDashboard(history) {
    const total = history.length;
    let totalCorrect = 0;
    let totalPhonemes = 0;

    history.forEach(h => {
      if (h.summary) {
        totalCorrect += h.summary.correct_phonemes || 0;
        totalPhonemes += h.summary.total_phonemes || 0;
      }
    });

    document.getElementById('statTotalValue').textContent = total;
    document.getElementById('statCorrectValue').textContent = totalCorrect;
    document.getElementById('statIncorrectValue').textContent = totalPhonemes - totalCorrect;
    document.getElementById('statAccuracyValue').textContent = totalPhonemes > 0
      ? `${((totalCorrect / totalPhonemes) * 100).toFixed(1)}%` : '—';

    ChartManager.renderPie('dashPieChart', totalCorrect, totalPhonemes - totalCorrect);

    // Recent activity
    const recent = document.getElementById('dashRecentList');
    if (history.length === 0) {
      recent.innerHTML = '<p class="text-muted">No analyses yet.</p>';
    } else {
      recent.innerHTML = history.slice(0, 5).map(h => `
        <div class="history-item">
          <div class="history-meta">
            <span class="history-name">${h.file_name || 'Unknown'}</span>
            <span class="history-info">${h.model_name || 'auto'} · ${new Date(h.timestamp).toLocaleString()}</span>
          </div>
        </div>
      `).join('');
    }
  }

  /* ---------- Init ---------- */
  function init() {
    renderHistory();
    updateDashboard(getHistory());
  }

  return {
    init,
    showToast,
    updateBackendStatus,
    navigateTo,
    showFileInfo,
    hideFileInfo,
    showRecordingUI,
    hideRecordingUI,
    showRecordError,
    updateRecordTimer,
    renderResults,
    showLoading,
    showError,
    renderModels,
    getHistory,
    addHistory,
    renderHistory,
    clearHistory,
    updateDashboard,
  };
})();
