/* ============================================
   Storage Module - LocalStorage wrapper + server sync
   ============================================ */

import { CONFIG, getAuthToken } from './config.js';

const STORAGE_KEYS = {
  deck: 'hanzi_deck',
  srs: 'hanzi_srs',
  session: 'hanzi_session',
  streak: 'hanzi_streak',
  theme: 'hanzi_theme',
  category: 'hanzi_category',
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

       // Check health endpoint first (no auth required)
       const healthResp = await fetch(`${CONFIG.SERVER_URL}/health`, {
         signal: ctrl.signal,
       });

       if (healthResp.ok) {
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
      console.error('Failed to load from server:', e);
    }

    return null;
  },

   async saveToServer(srsData, session, streak) {
     if (!this.serverAvailable) return;

     try {
       const token = getAuthToken();
       await fetch(`${CONFIG.SERVER_URL}/progress`, {
         method: 'POST',
         headers: { 
           'Content-Type': 'application/json',
           'Authorization': token ? `Bearer ${token}` : ''
         },
         body: JSON.stringify({ srsData, session, streak }),
       });
     } catch (e) {
       console.error('Failed to save to server:', e);
     }
  },
};