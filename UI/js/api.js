/* ============================================================
   API Layer — Fetch with fallback mock
   ============================================================ */

const API = (() => {
  const BASE_URL = 'http://localhost:8000';
  const TIMEOUT_MS = 30000;

  function getBaseUrl() {
    const stored = localStorage.getItem('api_base_url');
    return stored || BASE_URL;
  }

  async function request(method, path, body, isFormData = false) {
    const url = getBaseUrl() + path;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

    const options = {
      method,
      signal: controller.signal,
    };

    if (body) {
      options.body = isFormData ? body : JSON.stringify(body);
      if (!isFormData) {
        options.headers = { 'Content-Type': 'application/json' };
      }
    }

    try {
      const res = await fetch(url, options);
      clearTimeout(timeout);

      if (!res.ok) {
        let errData = null;
        try { errData = await res.json(); } catch (_) {}
        const msg = errData?.error?.message || errData?.detail || `HTTP ${res.status}`;
        throw new Error(msg);
      }

      return await res.json();
    } catch (err) {
      clearTimeout(timeout);
      if (err.name === 'AbortError') throw new Error('Request timed out');
      throw err;
    }
  }

  /* ---------- Mock data for offline demo ---------- */
  function generateMockResponse(modelName, text) {
    const phonemePool = [
      'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
      'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
      'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
      'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH',
    ];
    const words = text ? text.split(/\s+/) : ['hello', 'world'];
    const phonemes = [];
    for (const w of words) {
      const count = Math.max(2, Math.ceil(w.length / 2));
      for (let i = 0; i < count; i++) {
        phonemes.push(phonemePool[Math.floor(Math.random() * phonemePool.length)]);
      }
    }

    const predictions = [];
    let correctCount = 0;
    for (let i = 0; i < phonemes.length; i++) {
      const isCorrect = Math.random() > 0.35;
      const conf = isCorrect ? 0.7 + Math.random() * 0.3 : Math.random() * 0.5;
      const statuses = ['correct', 'incorrect', 'substitution', 'deletion', 'insertion'];
      const status = isCorrect ? 'correct' : statuses[Math.floor(Math.random() * 4) + 1];
      if (isCorrect) correctCount++;

      predictions.push({
        phoneme: phonemes[i],
        status: status,
        confidence: Math.round(conf * 1000) / 1000,
        start_time: null,
        end_time: null,
        reason: status === 'correct' ? null : `Possible mispronunciation of /${phonemes[i]}/`,
        expected: phonemes[i],
        actual: isCorrect ? phonemes[i] : phonemePool[Math.floor(Math.random() * phonemePool.length)],
      });
    }

    return {
      success: true,
      model_name: modelName,
      input_file: 'sample_audio.wav',
      predictions,
      result: {
        phoneme_sequence: phonemes,
        phoneme_string: phonemes.join(' '),
        overall_confidence: Math.round((correctCount / phonemes.length) * 1000) / 1000,
      },
      summary: {
        total_phonemes: phonemes.length,
        correct_phonemes: correctCount,
        incorrect_phonemes: phonemes.length - correctCount,
        accuracy: Math.round((correctCount / phonemes.length) * 1000) / 1000,
      },
      processing_time_ms: Math.round(200 + Math.random() * 800),
      request_id: Math.random().toString(36).slice(2, 10),
    };
  }

  function generateMockModels() {
    return {
      models: [
        { name: 'cnn_bilstm_ctc', display_name: 'CNN-BiLSTM-CTC', version: '1.0.0', architecture: 'CNN + BiLSTM + CTC', task: 'ASR + MDD', requires_text: false, sample_rate: 16000, phoneme_set_size: 42, loaded: false, status: 'unloaded' },
        { name: 'dab_transformer', display_name: 'DAB-Transformer', version: '1.0.0', architecture: 'Conv1d + Transformer + CTC', task: 'ASR + MDD', requires_text: false, sample_rate: 16000, phoneme_set_size: 41, loaded: false, status: 'unloaded' },
        { name: 'wav2vec2', display_name: 'Wav2Vec2-MDD', version: '4.0.0', architecture: 'Wav2Vec2 + Bi-GRU + Cross-Attention', task: 'Phoneme scoring', requires_text: true, sample_rate: 16000, phoneme_set_size: 46, loaded: false, status: 'unloaded' },
      ],
      total: 3,
    };
  }

  /* ---------- Public API ---------- */
  return {
    async getHealth() {
      return request('GET', '/api/v1/health');
    },

    async getModels() {
      try {
        return await request('GET', '/api/v1/models');
      } catch (_) {
        return generateMockModels();
      }
    },

    async infer(modelName, file, text, threshold) {
      const formData = new FormData();
      formData.append('file', file, file.name);
      formData.append('model_name', modelName);
      formData.append('threshold', String(threshold || 0.5));
      formData.append('return_details', 'true');
      if (text) formData.append('text', text);

      let path = '/api/v1/infer';
      if (modelName && modelName !== 'auto') {
        const routeMap = {
          cnn_bilstm_ctc: '/api/v1/infer/cnn-bilstm-ctc',
          dab_transformer: '/api/v1/infer/dab-transformer',
          wav2vec2: '/api/v1/infer/wav2vec2',
        };
        path = routeMap[modelName] || path;
      }

      try {
        return await request('POST', path, formData, true);
      } catch (_) {
        return generateMockResponse(modelName || 'auto', text || 'hello world');
      }
    },

    async getLabels(modelName) {
      return request('GET', `/api/v1/labels?model_name=${modelName || 'wav2vec2'}`);
    },
  };
})();
