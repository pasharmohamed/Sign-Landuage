/**
 * Main application controller
 * Handles: Speech Recognition, Text input, Avatar animation queue
 */
import { SkeletonAvatar } from './avatar.js';

class SignLanguageApp {
  constructor() {
    this.avatar = null;
    this.signs = {};
    this.recognition = null;
    this.isMicActive = false;
    this.lastLatency = 0;

    this._init();
  }

  async _init() {
    await this._loadSignDictionary();
    this._initAvatar();
    this._initSpeechRecognition();
    this._bindUI();
    this._updateStatus('ready');
  }

  /* ── Sign Dictionary ─────────────────────────────── */
  async _loadSignDictionary() {
    try {
      const res = await fetch('../data/poses/arabic_signs.json');
      const data = await res.json();
      this.signs = data.signs;
      console.log(`[Dictionary] Loaded ${Object.keys(this.signs).length} signs`);
    } catch (e) {
      console.error('[Dictionary] Failed to load signs:', e);
    }
  }

  /* ── Avatar ──────────────────────────────────────── */
  _initAvatar() {
    const canvas = document.getElementById('avatar-canvas');
    this.avatar = new SkeletonAvatar(canvas);

    this.avatar.onSignStart = (label) => {
      this._updateQueueDisplay();
      this._flashSign(label);
    };

    this.avatar.onSignComplete = (label) => {
      this._updateQueueDisplay();
    };

    this.avatar.onQueueEmpty = () => {
      this._updateQueueDisplay();
      this._setAvatarStatus('idle');
    };
  }

  /* ── Speech Recognition ──────────────────────────── */
  _initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn('[Speech] Not supported in this browser');
      document.getElementById('btn-mic').disabled = true;
      document.getElementById('btn-mic').title = 'غير مدعوم في هذا المتصفح';
      return;
    }

    this.recognition = new SpeechRecognition();
    this.recognition.lang = 'ar-SA';
    this.recognition.continuous = true;
    this.recognition.interimResults = true;

    this.recognition.onresult = (e) => {
      let interim = '';
      let final = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) final += t;
        else interim += t;
      }

      if (interim) this._showRecognized(interim, false);
      if (final.trim()) {
        this._showRecognized(final.trim(), true);
        this._processText(final.trim());
      }
    };

    this.recognition.onend = () => {
      if (this.isMicActive) this.recognition.start(); // Auto-restart
    };

    this.recognition.onerror = (e) => {
      console.error('[Speech] Error:', e.error);
      if (e.error === 'not-allowed') {
        this._showRecognized('❌ لا يوجد إذن للميكروفون', true);
        this._toggleMic(false);
      }
    };
  }

  /* ── Text Processing ─────────────────────────────── */
  _processText(text) {
    const t0 = performance.now();
    const words = text.trim().split(/\s+/);
    let foundAny = false;

    // Try multi-word phrases first, then individual words
    const phrases = this._extractPhrases(words);
    
    phrases.forEach(phrase => {
      if (this.signs[phrase]) {
        this.avatar.queueSign(this.signs[phrase]);
        this._setAvatarStatus('signing');
        foundAny = true;
      }
    });

    if (!foundAny) {
      this._showRecognized(`"${text}" — لم يُوجد في القاموس`, true);
    }

    this.lastLatency = Math.round(performance.now() - t0);
    document.getElementById('latency-val').textContent = `${this.lastLatency}ms`;
    this._updateQueueDisplay();
  }

  _extractPhrases(words) {
    const result = [];
    // Try 2-word combinations first
    for (let i = 0; i < words.length - 1; i++) {
      result.push(`${words[i]} ${words[i + 1]}`);
    }
    // Then individual words
    words.forEach(w => result.push(w));
    return result;
  }

  /* ── UI Bindings ─────────────────────────────────── */
  _bindUI() {
    // Text input
    const input = document.getElementById('text-input');
    const sendBtn = document.getElementById('btn-send');

    const sendText = () => {
      const val = input.value.trim();
      if (!val) return;
      this._showRecognized(val, true);
      this._processText(val);
      input.value = '';
    };

    sendBtn.addEventListener('click', sendText);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') sendText();
    });

    // Mic button
    document.getElementById('btn-mic').addEventListener('click', () => {
      this._toggleMic();
    });

    // Camera button
    document.getElementById('btn-camera').addEventListener('click', () => {
      this._toggleCamera();
    });

    // Quick sign buttons
    document.querySelectorAll('.quick-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const word = btn.dataset.sign;
        this._showRecognized(word, true);
        this._processText(word);
      });
    });

    // Speed slider
    const speedSlider = document.getElementById('speed-slider');
    if (speedSlider) {
      speedSlider.addEventListener('input', (e) => {
        const speed = parseFloat(e.target.value);
        this.avatar.setLerpSpeed(speed);
        document.getElementById('speed-val').textContent = Math.round(speed * 100) + '%';
      });
    }

    // Clear button
    document.getElementById('btn-clear')?.addEventListener('click', () => {
      this.avatar.clearQueue();
      this._updateQueueDisplay();
      this._setAvatarStatus('idle');
    });
  }

  /* ── Camera ──────────────────────────────────────── */
  async _toggleCamera() {
    const btn = document.getElementById('btn-camera');
    const video = document.getElementById('camera-video');
    const placeholder = document.getElementById('camera-placeholder');

    if (this.cameraStream) {
      this.cameraStream.getTracks().forEach(t => t.stop());
      this.cameraStream = null;
      video.srcObject = null;
      video.style.display = 'none';
      placeholder.style.display = 'flex';
      btn.classList.remove('active');
      document.getElementById('dot-camera').classList.remove('active');
      return;
    }

    try {
      this.cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      video.srcObject = this.cameraStream;
      video.style.display = 'block';
      placeholder.style.display = 'none';
      btn.classList.add('active');
      document.getElementById('dot-camera').classList.add('active');
      this._startHandTracking(video);
    } catch (e) {
      console.error('[Camera]', e);
    }
  }

  _startHandTracking(video) {
    // Placeholder for MediaPipe integration
    // Phase 2: Will use @mediapipe/holistic for real-time hand tracking
    document.getElementById('chip-hands').classList.add('on');
    document.getElementById('chip-hands').textContent = '✋ يد محددة';
  }

  /* ── UI State Helpers ────────────────────────────── */
  _toggleMic(force = null) {
    if (!this.recognition) return;
    
    this.isMicActive = force !== null ? force : !this.isMicActive;
    const btn = document.getElementById('btn-mic');
    const dot = document.getElementById('dot-mic');

    if (this.isMicActive) {
      this.recognition.start();
      btn.classList.add('active');
      btn.querySelector('.btn-text').textContent = 'إيقاف';
      dot.classList.add('active');
    } else {
      this.recognition.stop();
      btn.classList.remove('active');
      btn.querySelector('.btn-text').textContent = 'ميكروفون';
      dot.classList.remove('active');
    }
  }

  _showRecognized(text, isFinal) {
    const el = document.getElementById('recognized-text');
    el.textContent = text;
    if (isFinal) {
      el.classList.add('flash');
      setTimeout(() => el.classList.remove('flash'), 400);
    }
  }

  _setAvatarStatus(state) {
    const dot = document.getElementById('dot-avatar');
    const label = document.getElementById('avatar-status-label');
    
    dot.className = 'status-dot';
    if (state === 'signing') {
      dot.classList.add('processing');
      label.textContent = 'يؤدي الإشارة';
    } else {
      dot.classList.add('active');
      label.textContent = 'جاهز';
    }
  }

  _flashSign(label) {
    const el = document.getElementById('current-sign-label');
    if (el) {
      el.textContent = label;
      el.style.opacity = '1';
    }
  }

  _updateQueueDisplay() {
    const queueEl = document.getElementById('anim-queue');
    if (!queueEl) return;
    
    const total = this.avatar.queueLength;
    queueEl.innerHTML = total === 0
      ? '<div class="queue-item">لا توجد إشارات في الانتظار</div>'
      : `<div class="queue-item current">🤟 ${this.avatar.currentSign?.label || '...'}</div>
         ${total > 1 ? `<div class="queue-item">+ ${total - 1} إشارة في الانتظار</div>` : ''}`;
  }

  _updateStatus(state) {
    const dot = document.getElementById('dot-avatar');
    if (state === 'ready') {
      dot.classList.add('active');
    }
  }
}

// Start the app
window.addEventListener('DOMContentLoaded', () => {
  window.app = new SignLanguageApp();
});
