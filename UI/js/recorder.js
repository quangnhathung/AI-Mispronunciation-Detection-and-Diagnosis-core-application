/* ============================================================
   Recorder — MediaRecorder API wrapper
   ============================================================ */

const Recorder = (() => {
  let mediaRecorder = null;
  let audioChunks = [];
  let stream = null;
  let timerInterval = null;
  let startTime = null;
  let onTimerUpdate = null;
  let onDataReady = null;
  let onErrorCb = null;

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  return {
    async start() {
      if (mediaRecorder && mediaRecorder.state === 'recording') return;

      audioChunks = [];

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            sampleRate: 16000,
          },
        });
      } catch (err) {
        let msg = 'Microphone access denied.';
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          msg = 'Please allow microphone access in your browser settings.';
        } else if (err.name === 'NotFoundError') {
          msg = 'No microphone found. Please connect a microphone.';
        }
        if (onErrorCb) onErrorCb(msg);
        throw new Error(msg);
      }

      try {
        mediaRecorder = new MediaRecorder(stream, {
          mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : 'audio/webm',
        });
      } catch {
        mediaRecorder = new MediaRecorder(stream);
      }

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        clearInterval(timerInterval);
        timerInterval = null;

        if (stream) {
          stream.getTracks().forEach(t => t.stop());
          stream = null;
        }

        const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
        const duration = startTime ? (Date.now() - startTime) / 1000 : 0;

        if (onDataReady) onDataReady(blob, duration);

        mediaRecorder = null;
        audioChunks = [];
      };

      mediaRecorder.onerror = (e) => {
        clearInterval(timerInterval);
        if (stream) stream.getTracks().forEach(t => t.stop());
        const msg = e.error?.message || 'Recording error occurred';
        if (onErrorCb) onErrorCb(msg);
      };

      mediaRecorder.start(250);
      startTime = Date.now();

      if (onTimerUpdate) onTimerUpdate(0, '0:00');

      timerInterval = setInterval(() => {
        if (onTimerUpdate && startTime) {
          const elapsed = (Date.now() - startTime) / 1000;
          onTimerUpdate(elapsed, formatTime(elapsed));
        }
      }, 200);
    },

    stop() {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
    },

    cancel() {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.ondataavailable = null;
        mediaRecorder.onstop = null;
        mediaRecorder.stop();
      }
      clearInterval(timerInterval);
      timerInterval = null;
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
      }
      mediaRecorder = null;
      audioChunks = [];
    },

    get state() {
      return mediaRecorder ? mediaRecorder.state : 'inactive';
    },

    set onTimerUpdate(fn) { onTimerUpdate = fn; },
    set onDataReady(fn) { onDataReady = fn; },
    set onError(fn) { onErrorCb = fn; },
  };
})();
