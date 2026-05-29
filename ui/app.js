(function () {
  "use strict";

  const API_BASE = "http://localhost:8000/api/v1";

  const ALLOWED_EXTENSIONS = ["wav", "mp3", "flac", "m4a", "ogg"];
  const ALLOWED_TYPES = [
    "audio/wav",
    "audio/mpeg",
    "audio/flac",
    "audio/mp4",
    "audio/ogg",
  ];
  const MAX_SIZE_BYTES = 10 * 1024 * 1024;

  const MODEL_NAMES = {
    wav2vec2: "Wav2Vec2-MDD",
    cnn_bilstm_ctc: "CNN-BiLSTM-CTC",
    dab_transformer: "DAB-Transformer",
  };

  const MODEL_ENDPOINTS = {
    cnn_bilstm_ctc: "/infer/cnn-bilstm-ctc",
    dab_transformer: "/infer/dab-transformer",
    wav2vec2: "/infer/wav2vec2",
  };

  let backendOnline = false;
  let recordingState = {
    stream: null,
    recorder: null,
    chunks: [],
    startTime: null,
    timerInterval: null,
    audioBlob: null,
    pcmData: null,
    sampleRate: 16000,
    isRecording: false,
  };
  let recordedWavBlob = null;
  let uploadedFile = null;
  let analyserNode = null;
  let animationId = null;

  function getApiUrl(path) {
    return API_BASE + path;
  }

  async function apiFetch(path, options = {}) {
    const url = getApiUrl(path);
    const res = await fetch(url, {
      ...options,
      headers: { ...options.headers },
    });
    const data = await res.json();
    if (!res.ok) {
      const errMsg =
        data?.error?.message || `HTTP ${res.status}: ${res.statusText}`;
      const err = new Error(errMsg);
      err.code = data?.error?.code || "HTTP_ERROR";
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  const api = {
    health() {
      return apiFetch("/health");
    },
    ready() {
      return apiFetch("/ready");
    },
    version() {
      return apiFetch("/version");
    },
    models() {
      return apiFetch("/models");
    },
    modelDetail(name) {
      return apiFetch("/models/" + encodeURIComponent(name));
    },
    loadModel(name) {
      return apiFetch("/models/" + encodeURIComponent(name) + "/load", {
        method: "POST",
      });
    },
    unloadModel(name) {
      return apiFetch("/models/" + encodeURIComponent(name) + "/unload", {
        method: "POST",
      });
    },
    labels(modelName) {
      return apiFetch(
        "/labels?model_name=" + encodeURIComponent(modelName || "wav2vec2"),
      );
    },
    async infer(formData, model) {
      var endpointPath = MODEL_ENDPOINTS[model];
      if (!endpointPath) {
        throw new Error("Unknown model: " + model);
      }
      const url = getApiUrl(endpointPath);
      const res = await fetch(url, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) {
        const errMsg = data?.error?.message || `HTTP ${res.status}`;
        const err = new Error(errMsg);
        err.code = data?.error?.code || "HTTP_ERROR";
        err.status = res.status;
        err.data = data;
        throw err;
      }
      return data;
    },
  };

  function toast(message, type, duration) {
    type = type || "info";
    duration = duration || 4000;
    const container = $("toastContainer");
    const icons = {
      success: "\u2713",
      error: "\u2717",
      warning: "\u26A0",
      info: "\u2139",
    };
    const el = document.createElement("div");
    el.className = "toast " + type;
    el.innerHTML =
      '<span class="toast-icon">' +
      (icons[type] || icons.info) +
      "</span>" +
      '<span class="toast-content">' +
      escapeHtml(message) +
      "</span>" +
      '<button class="toast-close" onclick="this.parentElement.remove()">&times;</button>';
    container.appendChild(el);
    setTimeout(function () {
      if (el.parentElement) el.remove();
    }, duration);
  }

  function escapeHtml(str) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
  }

  function formatTime(seconds) {
    var m = Math.floor(seconds / 60);
    var s = Math.floor(seconds % 60);
    return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
  }

  function formatMs(ms) {
    if (ms < 1000) return ms.toFixed(0) + "ms";
    return (ms / 1000).toFixed(2) + "s";
  }

  function getFileExtension(filename) {
    var parts = filename.split(".");
    return parts.length > 1 ? parts.pop().toLowerCase() : "";
  }

  function $(id) {
    return document.getElementById(id);
  }
  function qs(sel, ctx) {
    return (ctx || document).querySelector(sel);
  }
  function qsa(sel, ctx) {
    return (ctx || document).querySelectorAll(sel);
  }

  // === TAB NAVIGATION ===
  function initNavigation() {
    var navBtns = qsa(".nav-btn");
    navBtns.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var tab = btn.dataset.tab;
        navBtns.forEach(function (b) {
          b.classList.remove("active");
        });
        btn.classList.add("active");
        qsa(".tab-content").forEach(function (tc) {
          tc.classList.remove("active");
        });
        var target = $("tab" + tab.charAt(0).toUpperCase() + tab.slice(1));
        if (target) target.classList.add("active");
      });
    });
  }

  // === SOURCE TABS ===
  function initSourceTabs() {
    var btns = qsa(".source-btn");
    btns.forEach(function (btn) {
      btn.addEventListener("click", function () {
        btns.forEach(function (b) {
          b.classList.remove("active");
        });
        btn.classList.add("active");
        var source = btn.dataset.source;
        qsa(".source-panel").forEach(function (p) {
          p.classList.remove("active");
        });
        var panel = $(
          "panel" + source.charAt(0).toUpperCase() + source.slice(1),
        );
        if (panel) panel.classList.add("active");
        updateInferButton();
      });
    });
  }

  // === BACKEND STATUS ===
  async function checkBackendStatus() {
    var dot = $("statusDot");
    var text = $("statusText");
    try {
      var h = await api.health();
      if (h.status === "ok") {
        backendOnline = true;
        dot.className = "status-dot online";
        text.textContent = "Online \u00B7 v" + h.version;
        return true;
      }
      throw new Error("Unexpected response");
    } catch (e) {
      backendOnline = false;
      dot.className = "status-dot offline";
      text.textContent = "Offline";
      return false;
    }
  }

  // === HOME TAB ===
  async function loadHomeData() {
    var loading = $("homeLoading");
    var dataContainer = $("homeCardsData");
    try {
      var online = await checkBackendStatus();
      if (!online) {
        loading.style.display = "";
        loading.querySelector(".card-body").innerHTML =
          '<p style="color:var(--red)">\u2717 Backend unavailable. Make sure the server is running.</p>';
        return;
      }
      var [health, ready, version, models] = await Promise.all([
        api.health(),
        api.ready(),
        api.version(),
        api.models(),
      ]);

      $("healthStatus").textContent = health.status || "-";
      $("healthVersion").textContent = health.version || "-";
      $("readyStatus").textContent = ready.ready
        ? "\u2713 Ready"
        : "\u2717 Not Ready";
      $("readyStatus").style.color = ready.ready
        ? "var(--green)"
        : "var(--red)";
      $("readyLoaded").textContent =
        (ready.models_loaded || []).join(", ") || "None";
      $("readyFailed").textContent =
        (ready.models_failed || []).join(", ") || "None";

      $("versionApp").textContent =
        version.app_name + " v" + version.app_version;
      $("versionPython").textContent = version.python_version || "-";
      var deps = version.dependencies || {};
      var depStr = Object.keys(deps)
        .map(function (k) {
          return k + ": " + deps[k];
        })
        .join(", ");
      $("versionDeps").textContent = depStr || "-";

      $("modelsTotal").textContent = models.total + " registered";
      var list = $("modelsList");
      list.innerHTML = "";
      (models.models || []).forEach(function (m) {
        var item = document.createElement("div");
        item.className = "model-item";
        var statusClass = m.loaded
          ? "loaded"
          : m.status === "error"
            ? "error"
            : "unloaded";
        var statusLabel = m.loaded
          ? "Loaded"
          : m.status === "error"
            ? "Error"
            : "Unloaded";
        item.innerHTML =
          '<div class="model-item-info">' +
          '<div class="model-item-name">' +
          escapeHtml(m.display_name || m.name) +
          "</div>" +
          '<div class="model-item-desc">' +
          escapeHtml(m.description || "") +
          "</div>" +
          '<div class="model-item-meta">' +
          "<span>" +
          escapeHtml(m.architecture || "") +
          "</span>" +
          "<span>&#9679;</span>" +
          "<span>" +
          m.phoneme_set_size +
          " phonemes</span>" +
          "<span>&#9679;</span>" +
          "<span>" +
          m.sample_rate +
          " Hz</span>" +
          "</div></div>" +
          '<span class="model-status-badge ' +
          statusClass +
          '">' +
          statusLabel +
          "</span>";
        list.appendChild(item);
      });

      loading.style.display = "none";
      dataContainer.style.display = "";
    } catch (e) {
      loading.style.display = "";
      loading.querySelector(".card-body").innerHTML =
        '<p style="color:var(--red)">\u2717 Error: ' +
        escapeHtml(e.message) +
        "</p>";
    }
  }

  // === MODELS TAB ===
  async function loadModelsTab() {
    var loading = $("modelsLoading");
    var container = $("modelsListContainer");
    try {
      var data = await api.models();
      loading.style.display = "none";
      container.style.display = "";
      var html = "";
      (data.models || []).forEach(function (m, i) {
        var statusClass = m.loaded
          ? "loaded"
          : m.status === "error"
            ? "error"
            : "unloaded";
        var statusLabel = m.loaded
          ? "Loaded"
          : m.status === "error"
            ? "Error"
            : "Unloaded";
        var requiresText = m.requires_text ? "Requires text" : "Text optional";
        html +=
          '<div class="card" style="margin-bottom:12px;">' +
          '<div class="card-header"><h3>' +
          escapeHtml(m.display_name || m.name) +
          "</h3>" +
          '<span class="model-status-badge ' +
          statusClass +
          '">' +
          statusLabel +
          "</span></div>" +
          '<div class="card-body">' +
          '<div class="stat-row"><span>Name</span><span class="stat-value">' +
          escapeHtml(m.name) +
          "</span></div>" +
          '<div class="stat-row"><span>Architecture</span><span class="stat-value">' +
          escapeHtml(m.architecture || "-") +
          "</span></div>" +
          '<div class="stat-row"><span>Task</span><span class="stat-value">' +
          escapeHtml(m.task || "-") +
          "</span></div>" +
          '<div class="stat-row"><span>Phonemes</span><span class="stat-value">' +
          (m.phoneme_set_size || 0) +
          "</span></div>" +
          '<div class="stat-row"><span>Sample Rate</span><span class="stat-value">' +
          (m.sample_rate || "-") +
          " Hz</span></div>" +
          '<div class="stat-row"><span>Text Requirement</span><span class="stat-value">' +
          requiresText +
          "</span></div>" +
          '<div class="stat-row"><span>Checkpoint</span><span class="stat-value" style="font-size:11px;word-break:break-all;">' +
          escapeHtml(m.checkpoint_path || "Not found") +
          "</span></div>" +
          '<div class="stat-row"><span>GPU Required</span><span class="stat-value">' +
          (m.requires_gpu ? "Yes" : "No") +
          "</span></div>" +
          "</div>" +
          '<div class="card-list"><div class="model-item"><div class="model-actions">' +
          (m.loaded
            ? '<button class="btn btn-small btn-unload" data-model="' +
              escapeHtml(m.name) +
              '">Unload</button>'
            : '<button class="btn btn-small btn-load" data-model="' +
              escapeHtml(m.name) +
              '">Load</button>') +
          "</div></div></div></div>";
      });
      container.innerHTML = html;

      container.querySelectorAll(".btn-load").forEach(function (btn) {
        btn.addEventListener("click", async function () {
          var name = btn.dataset.model;
          try {
            toast("Loading " + name + "...", "info");
            await api.loadModel(name);
            toast(name + " loaded successfully", "success");
            loadModelsTab();
          } catch (e) {
            toast("Failed to load " + name + ": " + e.message, "error");
          }
        });
      });
      container.querySelectorAll(".btn-unload").forEach(function (btn) {
        btn.addEventListener("click", async function () {
          var name = btn.dataset.model;
          try {
            toast("Unloading " + name + "...", "info");
            await api.unloadModel(name);
            toast(name + " unloaded", "success");
            loadModelsTab();
          } catch (e) {
            toast("Failed to unload " + name + ": " + e.message, "error");
          }
        });
      });
    } catch (e) {
      loading.style.display = "";
      loading.querySelector(".card-body").innerHTML =
        '<p style="color:var(--red)">\u2717 Error: ' +
        escapeHtml(e.message) +
        "</p>";
    }
  }

  // === RECORDER ===
  function initRecorder() {
    $("btnRecord").addEventListener("click", startRecording);
    $("btnStop").addEventListener("click", stopRecording);
    $("btnReset").addEventListener("click", resetRecording);
  }

  async function startRecording() {
    if (recordingState.isRecording) return;
    try {
      var stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordingState.stream = stream;

      var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      var source = audioCtx.createMediaStreamSource(stream);
      var analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserNode = analyser;
      startVisualizer(analyser);

      var options = {};
      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
        options = { mimeType: "audio/webm;codecs=opus" };
      }

      var recorder = new MediaRecorder(stream, options);
      recordingState.recorder = recorder;
      recordingState.chunks = [];
      recordingState.pcmData = null;
      recordedWavBlob = null;

      recorder.ondataavailable = function (e) {
        if (e.data.size > 0) recordingState.chunks.push(e.data);
      };

      recorder.onstop = function () {
        var blob = new Blob(recordingState.chunks, {
          type: recorder.mimeType || "audio/webm",
        });
        recordingState.audioBlob = blob;
        convertToWav(blob);
        stopVisualizer();
        if (recordingState.stream) {
          recordingState.stream.getTracks().forEach(function (t) {
            t.stop();
          });
          recordingState.stream = null;
        }
      };

      recorder.start(100);
      recordingState.isRecording = true;
      recordingState.startTime = Date.now();
      recordingState.timerInterval = setInterval(function () {
        var elapsed = (Date.now() - recordingState.startTime) / 1000;
        $("recorderTimer").textContent = formatTime(elapsed);
      }, 100);
      $("recorderTimer").classList.add("recording");
      btnRecord.disabled = true;
      btnStop.disabled = false;
      btnReset.disabled = true;
      toast("Recording started", "info");
    } catch (e) {
      if (e.name === "NotAllowedError" || e.name === "PermissionDeniedError") {
        toast(
          "Microphone access denied. Please allow microphone permissions.",
          "error",
        );
      } else {
        toast("Cannot start recording: " + e.message, "error");
      }
    }
  }

  function stopRecording() {
    if (!recordingState.isRecording || !recordingState.recorder) return;
    recordingState.isRecording = false;
    recordingState.recorder.stop();
    clearInterval(recordingState.timerInterval);
    $("btnRecord").disabled = false;
    $("btnStop").disabled = true;
    $("btnReset").disabled = false;
    toast("Recording stopped", "success");
  }

  function resetRecording() {
    if (recordingState.stream) {
      recordingState.stream.getTracks().forEach(function (t) {
        t.stop();
      });
    }
    clearInterval(recordingState.timerInterval);
    recordingState.isRecording = false;
    recordingState.recorder = null;
    recordingState.chunks = [];
    recordingState.audioBlob = null;
    recordingState.pcmData = null;
    recordedWavBlob = null;
    $("recorderTimer").textContent = "00:00";
    $("recorderTimer").classList.remove("recording");
    $("btnRecord").disabled = false;
    $("btnStop").disabled = true;
    $("btnReset").disabled = true;
    var preview = $("recorderPreview");
    preview.style.display = "none";
    preview.querySelector("audio").src = "";
    stopVisualizer();
    if (analyserNode) analyserNode = null;
    updateInferButton();
    toast("Recording reset", "info");
  }

  function convertToWav(blob) {
    var fileReader = new FileReader();
    fileReader.onload = async function () {
      try {
        var arrayBuffer = fileReader.result;
        var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        var audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        var pcm = audioBuffer.getChannelData(0);
        recordingState.pcmData = pcm;
        recordingState.sampleRate = audioBuffer.sampleRate;
        var wavBlob = encodeWav(pcm, audioBuffer.sampleRate);
        recordedWavBlob = wavBlob;

        var preview = $("recorderPreview");
        var audioEl = preview.querySelector("audio");
        audioEl.src = URL.createObjectURL(wavBlob);
        preview.style.display = "";
        updateInferButton();
        toast("Audio ready for analysis", "success");
      } catch (e) {
        toast("Audio conversion failed: " + e.message, "error");
      }
    };
    fileReader.onerror = function () {
      toast("Failed to read recorded audio", "error");
    };
    fileReader.readAsArrayBuffer(blob);
  }

  function encodeWav(samples, sampleRate) {
    var numChannels = 1;
    var bitsPerSample = 16;
    var byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
    var blockAlign = (numChannels * bitsPerSample) / 8;
    var dataSize = samples.length * blockAlign;
    var bufferSize = 44 + dataSize;
    var buffer = new ArrayBuffer(bufferSize);
    var view = new DataView(buffer);

    function writeString(offset, str) {
      for (var i = 0; i < str.length; i++)
        view.setUint8(offset + i, str.charCodeAt(i));
    }
    writeString(0, "RIFF");
    view.setUint32(4, bufferSize - 8, true);
    writeString(8, "WAVE");
    writeString(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(36, "data");
    view.setUint32(40, dataSize, true);

    var offset = 44;
    for (var i = 0; i < samples.length; i++) {
      var s = Math.max(-1, Math.min(1, samples[i]));
      s = s < 0 ? s * 0x8000 : s * 0x7fff;
      view.setInt16(offset, s, true);
      offset += 2;
    }
    return new Blob([buffer], { type: "audio/wav" });
  }

  // === VISUALIZER ===
  function startVisualizer(analyser) {
    var canvas = $("visualizer");
    var ctx = canvas.getContext("2d");
    if (!ctx) return;
    var w = canvas.width;
    var h = canvas.height;
    var bufferLength = analyser.frequencyBinCount;
    var dataArray = new Uint8Array(bufferLength);

    function draw() {
      if (!analyser) return;
      animationId = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);
      ctx.fillStyle = "#12141e";
      ctx.fillRect(0, 0, w, h);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#6366f1";
      ctx.beginPath();
      var sliceWidth = w / bufferLength;
      var x = 0;
      for (var i = 0; i < bufferLength; i++) {
        var v = dataArray[i] / 128.0;
        var y = (v * h) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(w, h / 2);
      ctx.stroke();
    }
    draw();
  }

  function stopVisualizer() {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
    var canvas = $("visualizer");
    var ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.fillStyle = "#12141e";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  }

  // === UPLOAD ===
  function initUpload() {
    var zone = $("uploadZone");
    var fileInput = $("fileInput");

    zone.addEventListener("click", function () {
      fileInput.click();
    });

    zone.addEventListener("dragover", function (e) {
      e.preventDefault();
      zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", function () {
      zone.classList.remove("dragover");
    });
    zone.addEventListener("drop", function (e) {
      e.preventDefault();
      zone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });
    fileInput.addEventListener("change", function () {
      if (fileInput.files.length > 0) {
        handleFileSelect(fileInput.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    var ext = getFileExtension(file.name);
    var validExt = ALLOWED_EXTENSIONS.indexOf(ext) !== -1;
    var validType = ALLOWED_TYPES.indexOf(file.type) !== -1;
    if (!validExt && !validType) {
      toast(
        "Invalid format: ." +
          ext +
          ". Allowed: " +
          ALLOWED_EXTENSIONS.join(", "),
        "error",
      );
      return;
    }
    if (file.size > MAX_SIZE_BYTES) {
      toast("File exceeds 10MB limit", "error");
      return;
    }
    uploadedFile = file;
    $("uploadFileName").textContent = file.name;
    $("uploadFileSize").textContent = formatSize(file.size);
    $("uploadFileInfo").style.display = "";
    $("uploadZone").querySelector(".upload-icon").style.display = "none";
    $("uploadZone").querySelector(".upload-text").textContent = file.name;
    $("uploadZone").querySelector(".upload-hint").style.display = "none";
    updateInferButton();
    toast("File selected: " + file.name, "success");
  }

  function clearUploadedFile() {
    uploadedFile = null;
    $("uploadFileInfo").style.display = "none";
    $("fileInput").value = "";
    var zone = $("uploadZone");
    zone.querySelector(".upload-icon").style.display = "";
    zone.querySelector(".upload-text").textContent =
      "Drop audio file here or click to browse";
    zone.querySelector(".upload-hint").style.display = "";
    updateInferButton();
  }

  $("btnClearFile").addEventListener("click", function (e) {
    e.stopPropagation();
    clearUploadedFile();
  });

  // === INFER BUTTON ===
  function updateInferButton() {
    var btn = $("btnInfer");
    var source = qs(".source-btn.active");
    var isRecord = source && source.dataset.source === "record";
    var isUpload = source && source.dataset.source === "upload";
    var hasAudio = (isRecord && recordedWavBlob) || (isUpload && uploadedFile);
    var model = $("modelSelect").value;
    var text = $("textInput").value.trim();
    var wav2vec2RequiresText = model === "wav2vec2";

    if (wav2vec2RequiresText && !text) {
      btn.disabled = true;
      btn.textContent = "Text required for Wav2Vec2";
      return;
    }
    if (!hasAudio) {
      btn.disabled = true;
      btn.textContent = isRecord
        ? "Record audio first"
        : "Select audio file first";
      return;
    }
    btn.disabled = false;
    btn.innerHTML =
      '<span class="btn-icon">\u25B6</span> Analyze Pronunciation';
  }

  $("modelSelect").addEventListener("change", updateInferButton);
  $("textInput").addEventListener("input", updateInferButton);

  // === THRESHOLD SLIDER ===
  $("thresholdSlider").addEventListener("input", function () {
    $("thresholdValue").textContent = parseFloat(this.value).toFixed(2);
  });

  var _processing = false;

  // === INFER ===
  async function runInference() {
    if (_processing) return;
    _processing = true;

    var btn = $("btnInfer");
    var cardResults = $("cardResults");
    var cardProcessing = $("cardProcessing");
    var cardEmpty = $("cardEmpty");

    var source = qs(".source-btn.active");
    var isRecord = source && source.dataset.source === "record";

    if (isRecord && !recordedWavBlob) {
      toast("No recording available. Please record audio first.", "warning");
      _processing = false;
      return;
    }
    if (!isRecord && !uploadedFile) {
      toast("No file selected. Please upload an audio file.", "warning");
      _processing = false;
      return;
    }

    var model = $("modelSelect").value;
    var text = $("textInput").value.trim() || null;
    var threshold = parseFloat($("thresholdSlider").value);
    var topK = 10;
    var returnDetails = true;

    if (model === "wav2vec2" && !text) {
      toast("Wav2Vec2 model requires ground-truth text.", "error");
      _processing = false;
      return;
    }

    btn.disabled = true;
    btn.innerHTML =
      '<div class="spinner" style="width:16px;height:16px;border-width:2px;"></div> Processing...';
    cardResults.style.display = "none";
    cardEmpty.style.display = "none";
    cardProcessing.style.display = "";
    $("processingStatus").textContent = "Analyzing audio...";
    $("processingModel").textContent =
      "Model: " + (MODEL_NAMES[model] || model);

    var progressFill = $("progressFill");
    progressFill.style.animation = "none";
    progressFill.style.width = "30%";

    try {
      var formData = new FormData();
      if (isRecord && recordedWavBlob) {
        var filename = "recording_" + Date.now() + ".wav";
        formData.append("file", recordedWavBlob, filename);
      } else if (uploadedFile) {
        formData.append("file", uploadedFile);
      }
      if (text) formData.append("text", text);
      formData.append("top_k", String(topK));
      formData.append("threshold", String(threshold));
      formData.append("return_details", String(returnDetails));

      progressFill.style.width = "60%";
      $("processingStatus").textContent = "Running model inference...";

      var result = await api.infer(formData, model);

      progressFill.style.width = "100%";
      $("processingStatus").textContent = "Analysis complete!";

      cardProcessing.style.display = "none";
      displayResults(result);
    } catch (e) {
      cardProcessing.style.display = "none";
      cardEmpty.style.display = "";
      toast("Inference failed: " + e.message, "error");
    }

    btn.disabled = false;
    btn.innerHTML =
      '<span class="btn-icon">\u25B6</span> Analyze Pronunciation';
    _processing = false;
  }

  function displayResults(data) {
    var card = $("cardResults");
    card.style.display = "";

    var badge = $("resultBadge");
    var acc = data.summary ? data.summary.accuracy : 0;
    if (acc >= 0.8) {
      badge.textContent = "Good";
      badge.className = "result-badge good";
    } else if (acc >= 0.5) {
      badge.textContent = "Needs Work";
      badge.className = "result-badge needs-work";
    } else {
      badge.textContent = "Needs Improvement";
      badge.className = "result-badge poor";
    }

    var meta = $("resultMeta");
    meta.innerHTML =
      "Model: <strong>" +
      escapeHtml(data.model_name || "-") +
      "</strong>" +
      " \u00B7 File: <strong>" +
      escapeHtml(data.input_file || "-") +
      "</strong>" +
      " \u00B7 Time: <strong>" +
      formatMs(data.processing_time_ms || 0) +
      "</strong>" +
      (data.request_id
        ? ' \u00B7 ID: <span style="font-size:11px;color:var(--text-muted)">' +
          escapeHtml(data.request_id) +
          "</span>"
        : "");

    var summary = data.summary || {};
    var summaryHtml = $("resultSummary");
    summaryHtml.innerHTML =
      '<div class="summary-stats">' +
      '<div class="summary-stat total"><span class="summary-stat-value">' +
      (summary.total_phonemes || 0) +
      '</span><span class="summary-stat-label">Total</span></div>' +
      '<div class="summary-stat correct"><span class="summary-stat-value">' +
      (summary.correct_phonemes || 0) +
      '</span><span class="summary-stat-label">Correct</span></div>' +
      '<div class="summary-stat incorrect"><span class="summary-stat-value">' +
      (summary.incorrect_phonemes || 0) +
      '</span><span class="summary-stat-label">Incorrect</span></div>' +
      '<div class="summary-stat accuracy"><span class="summary-stat-value">' +
      (summary.accuracy * 100).toFixed(1) +
      '%</span><span class="summary-stat-label">Accuracy</span></div>' +
      "</div>";

    if (
      summary.precision != null ||
      summary.recall != null ||
      summary.f1_score != null
    ) {
      summaryHtml.innerHTML +=
        '<div class="summary-stats" style="margin-top:8px;">' +
        (summary.precision != null
          ? '<div class="summary-stat"><span class="summary-stat-value">' +
            (summary.precision * 100).toFixed(1) +
            '%</span><span class="summary-stat-label">Precision</span></div>'
          : "") +
        (summary.recall != null
          ? '<div class="summary-stat"><span class="summary-stat-value">' +
            (summary.recall * 100).toFixed(1) +
            '%</span><span class="summary-stat-label">Recall</span></div>'
          : "") +
        (summary.f1_score != null
          ? '<div class="summary-stat"><span class="summary-stat-value">' +
            (summary.f1_score * 100).toFixed(1) +
            '%</span><span class="summary-stat-label">F1 Score</span></div>'
          : "") +
        "</div>";
    }

    var predicts = data.predictions || [];
    var phonemesEl = $("resultPhonemes");
    phonemesEl.innerHTML = "";

    if (predicts.length === 0) {
      phonemesEl.innerHTML =
        '<p style="color:var(--text-dim);font-size:13px;text-align:center;padding:20px;">No detailed phoneme predictions returned. Set "Return Details" to true for per-phoneme analysis.</p>';
    } else {
      if (data.result && data.result.phoneme_string) {
        var seqEl = document.createElement("div");
        seqEl.style.cssText =
          "margin-bottom:10px;font-size:13px;color:var(--text-dim);word-break:break-all;";
        seqEl.innerHTML =
          'Phoneme sequence: <strong style="color:var(--text);font-family:monospace;">' +
          escapeHtml(data.result.phoneme_string) +
          "</strong>";
        phonemesEl.appendChild(seqEl);
      }

      predicts.forEach(function (p) {
        var row = document.createElement("div");
        row.className = "phoneme-row status-" + (p.status || "unknown");
        var statusDisplay = p.status || "unknown";
        var expectedStr = "";
        if (p.status === "substitution" && p.expected && p.actual) {
          expectedStr =
            '<span class="phoneme-expected">Expected: <strong>' +
            escapeHtml(p.expected) +
            "</strong> | Actual: <strong>" +
            escapeHtml(p.actual) +
            "</strong></span>";
        } else if (p.status === "deletion" && p.expected) {
          expectedStr =
            '<span class="phoneme-expected">Missing: <strong>' +
            escapeHtml(p.expected) +
            "</strong></span>";
        } else if (p.status === "insertion" && p.actual) {
          expectedStr =
            '<span class="phoneme-expected">Extra: <strong>' +
            escapeHtml(p.actual) +
            "</strong></span>";
        }
        var reasonStr = p.reason
          ? '<span class="phoneme-reason">' + escapeHtml(p.reason) + "</span>"
          : "";
        row.innerHTML =
          '<span class="phoneme-symbol">' +
          escapeHtml(p.phoneme || "?") +
          "</span>" +
          '<span class="phoneme-status">' +
          statusDisplay +
          "</span>" +
          '<span class="phoneme-confidence">' +
          (p.confidence * 100).toFixed(0) +
          "%</span>" +
          (expectedStr || reasonStr
            ? '<span style="flex:1;min-width:0;">' +
              expectedStr +
              reasonStr +
              "</span>"
            : "");
        phonemesEl.appendChild(row);
      });
    }

    var resultsArea = qs(".inference-output-area");
    resultsArea.scrollIntoView({ behavior: "smooth", block: "nearest" });
    toast("Analysis complete", "success");

    addHistory(data);
  }

  // === HISTORY ===
  var historyEntries = [];
  var MAX_HISTORY = 20;

  function addHistory(data) {
    historyEntries.unshift({
      timestamp: new Date().toISOString(),
      model: data.model_name,
      file: data.input_file,
      accuracy: data.summary ? data.summary.accuracy : 0,
      total: data.summary ? data.summary.total_phonemes : 0,
      correct: data.summary ? data.summary.correct_phonemes : 0,
      incorrect: data.summary ? data.summary.incorrect_phonemes : 0,
      time: data.processing_time_ms,
      requestId: data.request_id,
    });
    if (historyEntries.length > MAX_HISTORY) historyEntries.pop();
    renderHistory();
  }

  function renderHistory() {
    var container = $("historyContainer");
    if (!container) return;
    container.style.display = historyEntries.length > 0 ? "" : "none";
    var list = container.querySelector(".history-list");
    if (!list) return;
    list.innerHTML = "";
    historyEntries.forEach(function (entry) {
      var date = new Date(entry.timestamp);
      var timeStr = date.toLocaleTimeString();
      var item = document.createElement("div");
      item.className = "model-item";
      var accClass =
        entry.accuracy >= 0.8
          ? "good"
          : entry.accuracy >= 0.5
            ? "needs-work"
            : "poor";
      item.innerHTML =
        '<div class="model-item-info">' +
        '<div class="model-item-name" style="font-size:13px;">' +
        escapeHtml(entry.model) +
        " \u00B7 " +
        escapeHtml(entry.file) +
        "</div>" +
        '<div class="model-item-meta">' +
        timeStr +
        " \u00B7 " +
        entry.total +
        " phonemes \u00B7 " +
        formatMs(entry.time) +
        "</div></div>" +
        '<div><div class="result-badge ' +
        accClass +
        '">' +
        (entry.accuracy * 100).toFixed(0) +
        "%</div></div>";
      list.appendChild(item);
    });
  }

  function addHistoryUI() {
    var parent = $("tabInference");
    var container = document.createElement("div");
    container.id = "historyContainer";
    container.style.cssText = "margin-top:20px;display:none;";
    container.innerHTML =
      '<div class="card"><div class="card-header"><h3>Session History</h3><span style="font-size:11px;color:var(--text-muted)">Last ' +
      MAX_HISTORY +
      '</span></div><div class="history-list"></div></div>';
    parent.appendChild(container);
  }

  // === EVENT BINDING ===
  $("btnInfer").addEventListener("click", runInference);

  // === POLLING ===
  var statusInterval = null;

  function startStatusPolling() {
    statusInterval = setInterval(function () {
      checkBackendStatus().catch(function () {});
    }, 15000);
  }

  // === INIT ===
  function init() {
    initNavigation();
    initSourceTabs();
    initRecorder();
    initUpload();
    addHistoryUI();

    updateInferButton();

    loadHomeData();
    loadModelsTab();
    startStatusPolling();

    qsa(".source-btn").forEach(function (btn) {
      btn.addEventListener("click", updateInferButton);
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
