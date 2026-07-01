/* ============================================
   Audio Module - Playback & variant curation
   ============================================ */

import { CONFIG } from './config.js';
import { Storage } from './storage.js';

export const Audio = {
  _currentAudio: null,
  lastPlayedVariant: 0,

  // --- Main Pronunciation ---

  /** Play main character audio (pre-generated MP3) */
  async play(char) {
    this.stop();

    return new Promise((resolve, reject) => {
      const audioPath = `${CONFIG.AUDIO_DIR}${encodeURIComponent(char)}.mp3`;
      const audio = new window.Audio(audioPath);
      
      audio.onerror = () => {
        // Fallback to Web Speech API if file not found
        this._speakFallback(char);
        resolve();
      };

      audio.addEventListener('ended', () => {
        this._currentAudio = null;
        this.stop();
        resolve();
      });

      this._currentAudio = audio;
      this._showStopBtn();
      
      audio.play().catch(err => {
        console.error('Failed to play audio:', err);
        this._speakFallback(char);
        reject(err);
      });
    });
  },

  /** Fallback: use browser's speech synthesis */
  _speakFallback(text) {
    if (!('speechSynthesis' in window)) {
      console.warn('Audio not available');
      return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'zh-CN';
    utterance.rate = 0.8;

    const voices = speechSynthesis.getVoices();
    const zhVoice = voices.find(v => v.lang.startsWith('zh'));
    if (zhVoice) utterance.voice = zhVoice;

    speechSynthesis.cancel();
    speechSynthesis.speak(utterance);
  },

  // --- Variant Picker (audio curation) ---

  /** Check if a card has variants and return state */
  getVariantState(char, variantData) {
    const variants = variantData[char];
    
    if (!variantData || !variants || variants.length === 0) {
      return { hasVariants: false };
    }

    return {
      hasVariants: true,
      variants,
      selectedVariant: this.lastPlayedVariant
    };
  },

  /** Play a specific variant */
  playVariant(char, variant) {
    this.stop();

    const url = `${CONFIG.SERVER_URL}/audio/variants/${encodeURIComponent(char)}_v${variant}.mp3`;
    const audio = new window.Audio(url);
    
    this._currentAudio = audio;
    this.lastPlayedVariant = variant;

    audio.addEventListener('ended', () => {
      this.stop();
    });

    this._showStopBtn();
    
    return audio.play().catch(err => {
      console.error(`Failed to play variant ${char}_v${variant}:`, err);
      throw new Error(`Failed to play variant`);
    });
  },

  /** Promote a variant as the main audio for this character */
  async promoteVariant(char, variant) {
    if (!CONFIG.USE_SERVER || !Storage.serverAvailable) return false;

    try {
      const result = await Storage.promoteVariant(char, variant);
      
      if (result) {
        // Clear variants from local cache since this char now has main audio only
        delete this.variantData?.[char];
        
        return true;
      }
    } catch (err) {
      console.error('Failed to promote variant:', err);
    }

    return false;
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
        document.querySelectorAll('.btn-variant').forEach(btn => 
          btn.classList.remove('playing')
        );
        
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

  /** Get last played variant number */
  getLastPlayedVariant() {
    return this.lastPlayedVariant;
  }
};
