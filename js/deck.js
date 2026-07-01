/* ============================================
   Deck Module - Card data loading & filtering
   ============================================ */

import { CONFIG } from './config.js';
import { Storage } from './storage.js';

export const Deck = {
  cards: [],
  
  // --- Load Cards ---

  async load() {
    try {
      // Try to load from data/cards.json first
      const resp = await fetch(CONFIG.CARDS_FILE);
      
      if (!resp.ok) {
        throw new Error(`Failed to load ${CONFIG.CARDS_FILE}`);
      }
      
      this.cards = await resp.json();
      
      // Ensure each card has a clean copy (no mutations)
      this.cards = this.cards.map(c => ({ ...c }));
      
      console.log(`Loaded ${this.cards.length} cards from ${CONFIG.CARDS_FILE}`);
      return this.cards;
    } catch (err) {
      console.error('Failed to load cards from JSON, falling back to localStorage:', err);
      
      // Fallback: try localStorage
      const savedDeck = Storage.load('deck');
      if (savedDeck && savedDeck.length > 0) {
        this.cards = savedDeck;
        console.log(`Loaded ${this.cards.length} cards from localStorage`);
        return this.cards;
      }
      
      throw new Error('No card data available. Ensure data/cards.json exists.');
    }
  },

  // --- Getters ---

  getAll() {
    return [...this.cards];
  },

  getById(id) {
    return this.cards.find(c => c.id === id);
  },

  getByCategory(category) {
    if (category === 'all') return this.getAll();
    return this.cards.filter(c => c.category === category);
  },

  getCategories() {
    const cats = new Set(this.cards.map(c => c.category));
    return Array.from(cats).sort();
  },

  // --- Save Deck (for custom cards) ---

  save() {
    Storage.save('deck', this.cards);
  },

  addCard(card) {
    this.cards.push(card);
    this.save();
  },

  updateCard(id, updates) {
    const idx = this.cards.findIndex(c => c.id === id);
    if (idx !== -1) {
      this.cards[idx] = { ...this.cards[idx], ...updates };
      this.save();
    }
  },

  // --- Reset to defaults ---

  reset() {
    localStorage.removeItem('hanzi_deck');
    return this.load();
  }
};
