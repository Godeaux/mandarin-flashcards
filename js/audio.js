/* ============================================
   Audio Module - Playback
   ============================================ */

import { CONFIG } from './config.js';

export const Audio = {
  _currentAudio: null,

  // --- Main Pronunciation ---

  /** Play main character audio (pre-generated MP3) */
  async play(char) {
    this.stop();

    return new Promise((resolve, reject) => {
      const audioPath = `${CONFIG.AUDIO_DIR}${encodeURIComponent(char)}.mp3`;
      const audio = new window.Audio(audioPath);

      audio.onerror = () => {
        // Fallback: show "no audio file yet" toast if file not found
        this._showNoAudioToast();
        resolve();
      };

      audio.addEventListener('ended', () => {
        this._currentAudio = null;
        this.stop();
        resolve();
      });

      this._currentAudio = audio;
      this._showStopBtn();

      audio.play().catch((err) => {
        console.error('Failed to play audio:', err);
        this._showNoAudioToast();
        reject(err);
      });
    });
  },

  /** Fallback: show "no audio file yet" toast (3 seconds, then fade) */
  _showNoAudioToast() {
    if (typeof document === 'undefined') return;

    // Create temporary toast element
    const toast = document.createElement('div');
    toast.className = 'audio-fallback-toast';
    toast.textContent = 'No audio file yet';
    toast.style.cssText = `
      position: fixed;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 12px 24px;
      border-radius: 8px;
      font-size: 14px;
      z-index: 9999;
      opacity: 1;
      transition: opacity 3s ease-out;
    `;
    document.body.appendChild(toast);

    // Fade out after 3 seconds
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 3000);
    }, 3000);
  },

  // --- Utilities ---

  /** Stop any currently playing audio */
  stop() {
    if (this._currentAudio) {
      this._currentAudio.pause();
      this._currentAudio.currentTime = 0;
      this._currentAudio = null;
    }

    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      speechSynthesis.cancel();
    }

    // Reset variant button playing states (if DOM available)
    if (typeof document !== 'undefined') {
      try {
        document.querySelectorAll('.btn-variant').forEach((btn) => btn.classList.remove('playing'));

        // Hide stop button
        const stopBtn = document.getElementById('btn-audio-stop');
        if (stopBtn) stopBtn.classList.add('hidden');
      } catch (e) {
        // DOM not ready yet
      }
    }
  },

  /** Show the stop button */
  _showStopBtn() {
    if (typeof document === 'undefined') return;

    const stopBtn = document.getElementById('btn-audio-stop');
    if (stopBtn) stopBtn.classList.remove('hidden');
  },
};