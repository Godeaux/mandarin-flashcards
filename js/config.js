/* ============================================
   Configuration - App settings and URLs
   ============================================ */

export const CONFIG = {
  // Server URL for future licensing/sync (optional, falls back to localStorage)
  SERVER_URL: 'http://localhost:8787',

  // Enable server features (set to false to use localStorage-only)
  USE_SERVER: false,

  // Audio settings
  AUDIO_DIR: 'audio/',
  VARIANTS_DIR: 'audio/variants/',

  // Data files
  CARDS_FILE: 'data/cards.json',
  PROGRESS_FILE: 'data/progress.json',
  AUDIO_SELECTIONS_FILE: 'data/audio-selections.json',

  // UI settings
  DEFAULT_STREAK: 0,
  DEFAULT_EASE_FACTOR: 2.5,
  TOAST_DURATION: 2500,
};
