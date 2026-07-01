/* ============================================
   Storage Module - LocalStorage wrapper + server sync
   ============================================ */

import { CONFIG } from './config.js';

const STORAGE_KEYS = {
  deck: "hanzi_deck",
  srs: "hanzi_srs",
  session: "hanzi_session",
  streak: "hanzi_streak",
  theme: "hanzi_theme",
  category: "hanzi_category"
};

export const Storage = {
  serverAvailable: false,

  // --- LocalStorage Operations ---

  load(key) {
    try {
      const data = localStorage.getItem(STORAGE_KEYS[key]);
      return data ? JSON.parse(data) : null;
    } catch (e) {
      console.error(`Failed to load ${key}:`, e);
      return null;
    }
  },

  save(key, data) {
    try {
      localStorage.setItem(STORAGE_KEYS[key], JSON.stringify(data));
    } catch (e) {
      console.error(`Failed to save ${key}:`, e);
    }
  },

  // --- Server Sync (optional) ---

  async detectServer() {
    if (!CONFIG.USE_SERVER) return false;

    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 1500);
      
      const resp = await fetch(`${CONFIG.SERVER_URL}/progress`, { 
        signal: ctrl.signal,
        method: 'GET'
      });
      clearTimeout(timer);
      
      if (resp.ok) {
        this.serverAvailable = true;
        return true;
      }
    } catch {
      // Server not available
    }
    
    this.serverAvailable = false;
    return false;
  },

  async loadFromServer() {
    if (!this.serverAvailable) return null;

    try {
      const resp = await fetch(`${CONFIG.SERVER_URL}/progress`);
      if (resp.ok) {
        return await resp.json();
      }
    } catch (e) {
      console.error("Failed to load from server:", e);
    }
    
    return null;
  },

  async saveToServer(srsData, session, streak) {
    if (!this.serverAvailable) return;

    try {
      await fetch(`${CONFIG.SERVER_URL}/progress`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ srsData, session, streak })
      });
    } catch (e) {
      console.error("Failed to save to server:", e);
    }
  },

  async loadAudioSelections() {
    if (!this.serverAvailable) return {};

    try {
      const resp = await fetch(`${CONFIG.SERVER_URL}/audio-selections`);
      if (resp.ok) {
        return await resp.json();
      }
    } catch (e) {
      console.error("Failed to load audio selections:", e);
    }
    
    return {};
  },

  async saveAudioSelections(selections) {
    if (!this.serverAvailable) return;

    try {
      await fetch(`${CONFIG.SERVER_URL}/audio-selections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(selections)
      });
    } catch (e) {
      console.error("Failed to save audio selections:", e);
    }
  },

  async loadVariantData() {
    if (!this.serverAvailable) return {};

    try {
      const resp = await fetch(`${CONFIG.SERVER_URL}/audio/variants`);
      if (resp.ok) {
        return await resp.json();
      }
    } catch (e) {
      console.error("Failed to load variant data:", e);
    }
    
    return {};
  },

  async promoteVariant(char, variant) {
    if (!this.serverAvailable) return false;

    try {
      const resp = await fetch(`${CONFIG.SERVER_URL}/audio/promote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ char, variant })
      });
      
      if (resp.ok) {
        const result = await resp.json();
        return result.ok;
      }
    } catch (e) {
      console.error("Failed to promote variant:", e);
    }
    
    return false;
  }
};
