/* ============================================
   Main Bootstrapper - Initializes app & wires modules
   ============================================ */

import { CONFIG } from './config.js';
import { Storage } from './storage.js';
import { Deck } from './deck.js';
import { SM2 } from './srs.js';
import { Audio } from './audio.js';
import { UI } from './ui.js';

const App = {
  // --- State ---
  deck: [],
  srsData: {},
  session: { date: SM2.todayStr(), reviewed: 0, correct: 0 },
  streak: { count: 0, lastDate: null },
  activeCategory: 'all',
  studyQueue: [],
  currentCardIndex: 0,
  isFlipped: false,
  selectedCardId: null,
  writingCard: null,
  serverAvailable: false,
  variantData: {},
  audioSelections: {},
  lastPlayedVariant: 0,
  _currentAudio: null,

  // --- Initialization ---

  async init() {
    try {
      // Load core data
      await this.loadData();

      // Bind event listeners
      this.bindEvents();

      // Apply saved theme
      this.applyTheme();

      // Show initial view
      this.switchView('study');
    } catch (err) {
      console.error('Failed to initialize app:', err);
      UI.toast('Failed to load app. Check console for details.');
    }
  },

  // --- Data Loading ---

  async loadData() {
    // Load deck (from JSON or localStorage)
    this.deck = await Deck.load();

    // Load SRS data
    const savedSRS = Storage.load('srs');
    if (savedSRS) {
      this.srsData = savedSRS;
    }

    // Ensure every card has SRS data
    this.deck.forEach((card) => {
      if (!this.srsData[card.id]) {
        this.srsData[card.id] = SM2.defaultState();
      }
    });

    // Load session
    const savedSession = Storage.load('session');
    if (savedSession) {
      this.session = savedSession;
      if (this.session.date !== SM2.todayStr()) {
        this.session = { date: SM2.todayStr(), reviewed: 0, correct: 0 };
      }
    }

    // Load streak
    const savedStreak = Storage.load('streak');
    if (savedStreak) {
      this.streak = savedStreak;
    }

    // Load active category
    const savedCat = Storage.load('category');
    if (savedCat) this.activeCategory = savedCat;

    // Detect server availability (optional)
    if (CONFIG.USE_SERVER) {
      this.serverAvailable = await Storage.detectServer();
      if (this.serverAvailable) {
        // Load server data if available
        await this.syncFromServer();
      }
    }

    // Save initial state
    this.saveAll();
  },

  // --- Server Sync (optional) ---

  async syncFromServer() {
    const serverData = await Storage.loadFromServer();
    if (serverData) {
      // Override local data with server data (basic merge strategy)
      if (serverData.srsData) {
        // Merge SRS data, preferring server values
        this.srsData = { ...this.srsData, ...serverData.srsData };
      }
      if (serverData.session) this.session = serverData.session;
      if (serverData.streak) this.streak = serverData.streak;
      if (serverData.audioSelections) this.audioSelections = serverData.audioSelections;
    }

    // Ensure all cards still have SRS data after merge
    this.deck.forEach((card) => {
      if (!this.srsData[card.id]) {
        this.srsData[card.id] = SM2.defaultState();
      }
    });
  },

  async saveToServer() {
    if (!this.serverAvailable || !CONFIG.USE_SERVER) return;

    await Storage.saveToServer(this.srsData, this.session, this.streak);
    await Storage.saveAudioSelections(this.audioSelections);
  },

  // --- Save All State ---

  saveAll() {
    Storage.save('deck', this.deck);
    Storage.save('srs', this.srsData);
    Storage.save('session', this.session);
    Storage.save('streak', this.streak);
    Storage.save('category', this.activeCategory);

    // Optional server sync
    if (CONFIG.USE_SERVER && this.serverAvailable) {
      this.saveToServer();
    }
  },

  // --- Event Binding ---

  bindEvents() {
    // Navigation tabs
    document.querySelectorAll('.nav-tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        const view = tab.dataset.view;
        if (view) this.switchView(view);
      });
    });

    // Theme toggle
    const themeBtn = document.getElementById('btn-theme');
    if (themeBtn) {
      themeBtn.addEventListener('click', () => UI.toggleTheme());
    }

    // Search
    const searchBtn = document.getElementById('btn-search');
    if (searchBtn) {
      searchBtn.addEventListener('click', () => UI.openSearch());
    }

    const searchClose = document.getElementById('search-close');
    if (searchClose) {
      searchClose.addEventListener('click', () => UI.closeSearch());
    }

    // Search input (Enter to search, Esc to close)
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
      searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          // TODO: Implement search functionality
          e.preventDefault();
        }
      });
      searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
          UI.closeSearch();
        }
      });
    }

    // Study view controls
    this.bindStudyEvents();
    this.bindBrowseEvents();
    this.bindStatsEvents();
    this.bindManageEvents();
    this.bindWritingPracticeEvents();
  },

  // --- Study View Events ---

  bindStudyEvents() {
    // Card tap to reveal + swipe to rate
    this.initSwipeHandler();

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.target !== document.body) return;
      if (e.code === 'Space') {
        e.preventDefault();
        this.revealCard(); // toggle
      } else if (e.code === 'ArrowLeft') {
        this.rateCard(0); // Again
      } else if (e.code === 'ArrowDown') {
        this.rateCard(1); // Hard
      } else if (e.code === 'ArrowUp') {
        this.rateCard(2); // Good
      } else if (e.code === 'ArrowRight') {
        this.rateCard(3); // Easy
      }
    });

    // Audio controls
    const mainAudioBtn = document.getElementById('btn-audio');
    if (mainAudioBtn) {
      mainAudioBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const card = this.studyQueue[this.currentCardIndex];
        if (card) this.speakCardAudio(card.char);
      });
    }

    const stopAudioBtn = document.getElementById('btn-audio-stop');
    if (stopAudioBtn) {
      stopAudioBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.stopAudio();
      });
    }

    // Skip & Learn new
    document.getElementById('btn-skip')?.addEventListener('click', () => this.skipCard());
    document.getElementById('btn-learn-new')?.addEventListener('click', () => this.learnNewCards());

    // Category toggle button
    const catToggle = document.getElementById('btn-toggle-cat');
    if (catToggle) {
      catToggle.addEventListener('click', () => {
        const filters = document.getElementById('category-filters');
        if (filters) {
          filters.classList.toggle('cat-collapsed');
          catToggle.textContent = filters.classList.contains('cat-collapsed')
            ? 'Filter ▾'
            : 'Filter ▴';
        }
      });
    }

    // Category filter buttons
    document.querySelectorAll('.cat-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.cat-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        this.activeCategory = btn.dataset.cat;
        Storage.save('category', this.activeCategory);
        this.refreshStudyView();
      });
    });
  },

  // --- Browse View Events ---

  bindBrowseEvents() {
    const browseGrid = document.getElementById('browse-grid');
    if (browseGrid) {
      // Card clicks handled by UI.renderBrowse via delegation
      // But we need to bind detail view controls

      const detailClose = document.getElementById('detail-close');
      if (detailClose) {
        detailClose.addEventListener('click', () => UI.closeCardDetail());
      }

      const detailAudio = document.getElementById('btn-detail-audio');
      if (detailAudio) {
        detailAudio.addEventListener('click', () => {
          const card = this.deck.find((c) => c.id === this.selectedCardId);
          if (card) this.speakCardAudio(card.char);
        });
      }

      const toggleKnownBtn = document.getElementById('btn-toggle-known');
      if (toggleKnownBtn) {
        toggleKnownBtn.addEventListener('click', () => {
          if (this.selectedCardId) {
            this.toggleKnownCard();
          }
        });
      }

      const writePracticeBtn = document.getElementById('btn-practice-write');
      if (writePracticeBtn) {
        writePracticeBtn.addEventListener('click', () => {
          if (this.selectedCardId) {
            this.openWritingPractice();
            UI.closeCardDetail();
          }
        });
      }
    }

    // Filter & sort
    document.getElementById('browse-filter')?.addEventListener('change', () => {
      this.activeCategory = document.getElementById('browse-filter').value;
      Storage.save('category', this.activeCategory);
      UI.renderBrowse(this.deck, this.srsData, this.activeCategory);
    });

    document.getElementById('browse-sort')?.addEventListener('change', () => {
      UI.renderBrowse(this.deck, this.srsData, this.activeCategory);
    });
  },

  // --- Stats View Events ---

  bindStatsEvents() {
    // Stats view is read-only, no interactions needed beyond navigation
  },

  // --- Manage View Events ---

  bindManageEvents() {
    // Reset buttons
    document.getElementById('btn-reset-progress')?.addEventListener('click', () => {
      if (confirm('Reset all progress? This cannot be undone.')) {
        this.resetProgress();
      }
    });

    document.getElementById('btn-reset-all')?.addEventListener('click', () => {
      if (confirm('Reset EVERYTHING? Cards, progress, settings? This cannot be undone.')) {
        this.resetEverything();
      }
    });
  },

  // --- Writing Practice Events ---

  bindWritingPracticeEvents() {
    document
      .getElementById('btn-clear-canvas')
      ?.addEventListener('click', () => this.clearCanvas());
    document
      .getElementById('btn-show-strokes')
      ?.addEventListener('click', () => this.showStrokeHints());
    document
      .getElementById('btn-close-writing')
      ?.addEventListener('click', () => this.closeWritingPractice());
  },

  // --- Core Logic Methods (delegated to modules where possible) ---

  getFilteredDeck() {
    return this.activeCategory === 'all'
      ? this.deck
      : this.deck.filter((c) => c.category === this.activeCategory);
  },

  refreshStudyView() {
    const filtered = this.getFilteredDeck();
    const due = [];
    const newCards = [];

    filtered.forEach((card) => {
      const srs = this.srsData[card.id];
      if (srs.known) return;
      if (SM2.isDue(srs)) due.push(card);
      else if (SM2.isNew(srs)) newCards.push(card);
    });

    // Study queue: due cards first, then up to 5 new cards
    this.studyQueue = [...due, ...newCards.slice(0, 5)];
    this.currentCardIndex = 0;
    this.isFlipped = false;

    const container = document.getElementById('flashcard-container');
    const empty = document.getElementById('study-empty');

    if (this.studyQueue.length === 0) {
      container.classList.add('hidden');
      empty.classList.remove('hidden');
    } else {
      container.classList.remove('hidden');
      empty.classList.add('hidden');
      this.showCard();
    }

    this.updateDashboard();
  },

  skipCard() {
    this.stopAudio();
    this.currentCardIndex++;
    if (this.currentCardIndex >= this.studyQueue.length) {
      this.refreshStudyView();
    } else {
      this.showCard();
    }

    this.updateDashboard();
  },

  learnNewCards() {
    const filtered = this.getFilteredDeck();
    const newCards = filtered.filter((c) => {
      const srs = this.srsData[c.id];
      return !srs.known && SM2.isNew(srs);
    });

    if (newCards.length === 0) {
      UI.toast('No new cards available in this category');
      return;
    }

    this.studyQueue = newCards.slice(0, 10);
    this.currentCardIndex = 0;
    document.getElementById('flashcard-container').classList.remove('hidden');
    document.getElementById('study-empty').classList.add('hidden');
    this.showCard();
  },

  updateStreak() {
    const today = SM2.todayStr();
    const yesterday = SM2.dateStr(new Date(Date.now() - 86400000));

    if (this.streak.lastDate === today) {
      // Already counted today
    } else if (this.streak.lastDate === yesterday) {
      this.streak.count++;
      this.streak.lastDate = today;
      this.saveAll();
    } else {
      this.streak.count = 1;
      this.streak.lastDate = today;
      this.saveAll();
    }
  },

  rateCard(quality) {
    this.stopAudio();
    const card = this.studyQueue[this.currentCardIndex];
    if (!card) return;

    const srs = this.srsData[card.id];
    this.srsData[card.id] = SM2.review(srs, quality);
    this.saveSRS();

    // Update session
    this.session.reviewed++;
    if (quality >= 2) this.session.correct++;
    this.saveSession();

    // Update streak
    this.updateStreak();

    // Next card
    this.currentCardIndex++;
    if (this.currentCardIndex >= this.studyQueue.length) {
      this.refreshStudyView();
    } else {
      this.showCard();
    }

    this.updateDashboard();
  },

  // --- Audio Wrapper ---

  speakCardAudio(char) {
    if (Audio) {
      Audio.play(char).catch(() => {
        // Fallback handled inside Audio module
      });
    }
  },

  stopAudio() {
    if (Audio) Audio.stop();
  },

  // --- Custom Card Management ---

  addCustomCard() {
    const char = document.getElementById('new-char').value.trim();
    const pinyin = document.getElementById('new-pinyin').value.trim();
    const meaning = document.getElementById('new-meaning').value.trim();
    const mnemonic = document.getElementById('new-mnemonic').value.trim();
    const visual = document.getElementById('new-visual').value.trim();

    if (!char || !pinyin || !meaning) {
      UI.toast('Please fill in character, pinyin, and meaning');
      return;
    }

    // Generate unique id
    const id = 'custom_' + Date.now();

    const card = {
      id,
      char,
      pinyin,
      meaning,
      category: 'custom',
      mnemonic: mnemonic || '',
      visual: visual || '',
      soundBridge: '',
      radicals: '',
      exampleSentence: '',
      exampleBreakdown: '',
    };

    this.deck.push(card);
    this.srsData[id] = SM2.defaultState();
    this.saveAll();

    // Reset form
    document.getElementById('add-card-form')?.reset();
    UI.toast(`Added "${char}" to your deck!`);

    // Refresh views
    this.refreshStudyView();
    UI.renderBrowse(this.deck, this.srsData, this.activeCategory);
  },

  exportDeck() {
    const data = {
      version: 1,
      deck: this.deck,
      srs: this.srsData,
      exportDate: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hanzi-flashcards-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    UI.toast('Deck exported!');
  },

  importDeck(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);

        if (data.version && data.deck && data.srs) {
          // Merge or replace based on user preference (for now, replace)
          // (replace not implemented, so we'll merge)
          // For simplicity: replace deck, merge SRS

          this.deck = data.deck;
          this.srsData = { ...this.srsData, ...data.srs };

          // Ensure all cards have SRS data
          this.deck.forEach((card) => {
            if (!this.srsData[card.id]) {
              this.srsData[card.id] = SM2.defaultState();
            }
          });

          this.saveAll();
          UI.toast('Deck imported successfully!');

          // Refresh views
          this.refreshStudyView();
          UI.renderBrowse(this.deck, this.srsData, this.activeCategory);
          UI.switchView('study');
        } else {
          UI.toast('Invalid deck file format');
        }
      } catch (err) {
        UI.toast('Failed to import deck: ' + err.message);
        console.error(err);
      }
    };

    reader.onerror = () => {
      UI.toast('Failed to read file');
    };

    reader.readAsText(file);

    // Reset file input
    event.target.value = '';
  },

  toggleKnownCard() {
    if (!this.selectedCardId) return;

    const srs = this.srsData[this.selectedCardId];
    srs.known = !srs.known;

    this.saveSRS();
    UI.closeCardDetail();
    UI.renderBrowse(this.deck, this.srsData, this.activeCategory);
    UI.toast(srs.known ? 'Marked as known' : 'Unmarked for review');
  },

  // --- Writing Practice (simplified - delegate to existing canvas logic) ---

  openWritingPractice() {
    const wp = document.getElementById('writing-practice');
    if (wp) wp.classList.remove('hidden');

    // Focus canvas
    const canvas = document.getElementById('writing-canvas');
    if (canvas) canvas.focus();
  },

  closeWritingPractice() {
    const wp = document.getElementById('writing-practice');
    if (wp) wp.classList.add('hidden');

    this.writingCard = null;
    this.clearCanvas();
  },

  clearCanvas() {
    const canvas = document.getElementById('writing-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid guide
    ctx.strokeStyle = getComputedStyle(document.documentElement)
      .getPropertyValue('--border')
      .trim();
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);

    // Cross guide lines
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, 0);
    ctx.lineTo(canvas.width / 2, canvas.height);
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    // Diagonal guides
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(canvas.width, canvas.height);
    ctx.moveTo(canvas.width, 0);
    ctx.lineTo(0, canvas.height);
    ctx.stroke();
  },

  showStrokeHints() {
    if (!this.writingCard) return;

    const hints = document.getElementById('stroke-hints');
    const charHints = {
      山: '① left vertical ② top-right corner ③ inner horizontal ④ inner horizontal ⑤ bottom closing',
      水: '① left vertical ② top horizontal ③ vertical hook',
      火: '① left dot ② left-falling ③ right dot ④ enclosing',
      木: '① horizontal ② vertical hook ③ left-falling ④ right dot',
      日: '① top horizontal ② vertical ③ bottom horizontal ④ enclosing',
      月: '① left-falling ② right-falling ③ enclosing',
      雨: '① top horizontal ② enclosing ③ four dots inside',
      田: '① vertical ② horizontal ③ vertical ④ enclosing',
      石: '① cliff ② mouth shape',
      风: '① sail frame ② swirl inside',
      人: '① left leg ② right leg',
      大: '① person ② arms wide',
      小: '① split ② two dots',
      口: '① square opening',
      手: '③ fingers ② palm with hook',
      目: '① eye shape ② pupil details',
      女: '① right-curving ② left-falling ③ horizontal',
      子: '① horizontal hook ② curved vertical ③ horizontal',
      心: '① left dot ② center curve ③ center dot ④ right dot',
      一: '① horizontal stroke',
      二: '① top short ② bottom long',
      三: '① top ② middle ③ bottom',
      四: '① left ② top-right ③ inner-left ④ inner-curve ⑤ bottom',
      五: '① top ② left turn ③ inner ④ bottom',
      好: 'write 女 then 子',
      中: '① left ② top-right turn ③ bottom ④ center pierce',
      天: '① top ② second horizontal ③ left-falling ④ right-falling',
      王: '① top ② middle ③ vertical ④ bottom',
      马: '① horizontal fold ② vertical hook ③ horizontal',
      门: '① left dot ② top-left turn ③ horizontal fold hook',
      力: '① horizontal fold hook ② left-falling',
    }[this.writingCard?.char];

    if (charHints) {
      hints.textContent = charHints;
      hints.classList.remove('hidden');
    } else {
      hints.textContent = `No stroke hints for ${this.writingCard?.char}`;
      hints.classList.remove('hidden');
    }
  },

  // --- Reset Functions ---

  resetProgress() {
    // Reset only SRS, session, streak
    this.srsData = {};
    this.session = { date: SM2.todayStr(), reviewed: 0, correct: 0 };
    this.streak = { count: 0, lastDate: null };

    // Reinitialize SRS data for all cards
    this.deck.forEach((card) => {
      if (!this.srsData[card.id]) {
        this.srsData[card.id] = SM2.defaultState();
      }
    });

    this.saveAll();
    this.refreshStudyView();
    UI.toast('Progress reset!');
  },

  resetEverything() {
    // Reset ALL data to defaults
    localStorage.clear();

    // Reload everything from scratch
    this.deck = [];
    this.srsData = {};
    this.session = { date: SM2.todayStr(), reviewed: 0, correct: 0 };
    this.streak = { count: 0, lastDate: null };
    this.activeCategory = 'all';
    this.studyQueue = [];
    this.currentCardIndex = 0;
    this.isFlipped = false;
    this.selectedCardId = null;
    this.writingCard = null;
    this.serverAvailable = false;
    this.variantData = {};
    this.audioSelections = {};
    this.lastPlayedVariant = 0;
    this._currentAudio = null;

    this.init();
    UI.toast('Everything reset to factory defaults!');
  },

  // --- Missing Methods ---

  applyTheme() {
    const saved = localStorage.getItem('hanzi_theme');
    if (saved) {
      document.documentElement.setAttribute('data-theme', saved);
      const btn = document.getElementById('btn-theme');
      if (btn) btn.textContent = saved === 'light' ? '☀️' : '🌙';
    }
  },

  showCard() {
    const card = this.studyQueue[this.currentCardIndex];
    if (!card) return;

    const srs = this.srsData[card.id];

    // Front (character only)
    document.getElementById('card-char').textContent = card.char;

    // Back (condensed)
    document.getElementById('card-char-back').textContent = card.char;
    document.getElementById('card-pinyin-back').textContent = card.pinyin;
    document.getElementById('card-meaning').textContent = card.meaning;
    document.getElementById('card-mnemonic').textContent = card.mnemonic || '';
    document.getElementById('card-visual').textContent = card.visual || '';
    document.getElementById('card-sound-bridge').textContent = card.soundBridge || '';
    document.getElementById('card-radicals').textContent = card.radicals || '';

    // Progress
    document.getElementById('card-progress-text').textContent =
      `${this.currentCardIndex + 1} / ${this.studyQueue.length}`;

    // Reset to front face
    this.isFlipped = false;
    const front = document.querySelector('.card-front');
    const back = document.querySelector('.card-back');
    if (front) front.classList.remove('hidden');
    if (back) back.classList.add('hidden');

    // Reset card position
    const flashcard = document.getElementById('flashcard');
    if (flashcard) {
      flashcard.style.transform = '';
      flashcard.classList.remove('animating', 'exit-left', 'exit-right', 'exit-up', 'exit-down');
    }

    // Reset glows
    document.querySelectorAll('.swipe-glow').forEach((g) => (g.style.opacity = 0));
    document.querySelectorAll('.swipe-label').forEach((l) => (l.style.opacity = 0));

    // Show hint
    const hint = document.getElementById('swipe-hint');
    if (hint) hint.textContent = 'swipe to rate · tap to reveal';
  },

  updateDashboard() {
    const filtered = this.getFilteredDeck();
    let due = 0,
      newCount = 0,
      learned = 0;

    filtered.forEach((card) => {
      const srs = this.srsData[card.id];
      if (srs?.known) {
        learned++;
        return;
      }
      if (SM2.isDue(srs)) due++;
      else if (SM2.isNew(srs)) newCount++;
      else if (srs?.repetitions > 0) learned++;
    });

    document.getElementById('stat-due').textContent = due;
    document.getElementById('stat-new').textContent = newCount;
    document.getElementById('stat-learned').textContent = learned;
    document.getElementById('stat-streak').textContent = this.streak.count;
    document.getElementById('stat-today-count').textContent = this.session.reviewed;
    document.getElementById('stat-today-acc').textContent =
      this.session.reviewed > 0
        ? Math.round((this.session.correct / this.session.reviewed) * 100) + '%'
        : '—';
  },

  saveSRS() {
    Storage.save('srs', this.srsData);
    if (CONFIG.USE_SERVER && this.serverAvailable) {
      this.saveToServer();
    }
  },

  saveSession() {
    Storage.save('session', this.session);
    if (CONFIG.USE_SERVER && this.serverAvailable) {
      this.saveToServer();
    }
  },

  // --- View Switching ---

  switchView(viewName) {
    // DOM switching
    UI.switchView(viewName);

    // View-specific rendering with App state
    if (viewName === 'study') {
      this.refreshStudyView();
    } else if (viewName === 'browse') {
      UI.renderBrowse(this.deck, this.srsData, this.activeCategory);
    } else if (viewName === 'stats') {
      UI.renderStats(this.deck, this.srsData, this.session, this.streak);
    }
  },

  // --- Card Reveal ---

  revealCard() {
    this.isFlipped = !this.isFlipped;

    const front = document.querySelector('.card-front');
    const back = document.querySelector('.card-back');
    if (this.isFlipped) {
      if (front) front.classList.add('hidden');
      if (back) back.classList.remove('hidden');
    } else {
      if (front) front.classList.remove('hidden');
      if (back) back.classList.add('hidden');
    }
  },

  // --- Swipe Handler ---

  initSwipeHandler() {
    const card = document.getElementById('flashcard');
    if (!card) return;

    const container = document.getElementById('flashcard-container');
    const THRESHOLD = 80; // px to trigger rating

    let startX = 0,
      startY = 0,
      dx = 0,
      dy = 0,
      isDragging = false;
    let mouseDown = false; // track if mouse button is pressed
    let lastTapTime = 0;
    let tapTarget = null; // track what element was tapped

    const onStart = (e) => {
      tapTarget = e.target;
      // Don't interfere with audio button clicks
      if (e.target.closest('#btn-audio')) return;
      isDragging = false;
      mouseDown = true; // mark that button is pressed
      const touch = e.touches ? e.touches[0] : e;
      startX = touch.clientX;
      startY = touch.clientY;
      dx = 0;
      dy = 0;
      card.classList.remove('animating');
    };

    const onMove = (e) => {
      // Only drag if mouse button is actually pressed
      if (!mouseDown) return;
      if (!startX && !startY) return;
      const touch = e.touches ? e.touches[0] : e;
      dx = touch.clientX - startX;
      dy = touch.clientY - startY;

      // Detect tap vs drag
      if (Math.abs(dx) > 8 || Math.abs(dy) > 8) {
        isDragging = true;
      }
      if (!isDragging) return;

      e.preventDefault();

      // Move card (works on both front and back)
      const rotation = dx * 0.05;
      card.style.transform = `translate(${dx}px, ${dy}px) rotate(${rotation}deg)`;

      // Calculate intensity (0-1)
      const absDx = Math.abs(dx);
      const absDy = Math.abs(dy);
      const intensity = Math.min(1, Math.max(absDx, absDy) / THRESHOLD);

      // Determine primary direction and update glow/label
      document.querySelectorAll('.swipe-glow').forEach((g) => (g.style.opacity = 0));
      document.querySelectorAll('.swipe-label').forEach((l) => (l.style.opacity = 0));

      if (absDx > absDy) {
        if (dx < 0) {
          document.getElementById('swipe-glow-left').style.opacity = intensity;
          document.getElementById('label-again').style.opacity = intensity;
        } else {
          document.getElementById('swipe-glow-right').style.opacity = intensity;
          document.getElementById('label-easy').style.opacity = intensity;
        }
      } else {
        // Vertical
        if (dy < 0) {
          document.getElementById('swipe-glow-up').style.opacity = intensity;
          document.getElementById('label-good').style.opacity = intensity;
        } else {
          document.getElementById('swipe-glow-down').style.opacity = intensity;
          document.getElementById('label-hard').style.opacity = intensity;
        }
      }
    };

    const onEnd = () => {
      // Skip if the tap was on the audio button
      if (tapTarget?.closest('#btn-audio')) {
        tapTarget = null;
        return;
      }
      tapTarget = null;

      // Debounce: ignore rapid double-fires from touch+mouse
      const now = Date.now();
      if (now - lastTapTime < 200) return;
      lastTapTime = now;

      // Reset mouseDown immediately so hover doesn't trigger dragging
      mouseDown = false;

      if (!isDragging) {
        // It was a tap - toggle the card face
        this.revealCard();
        // Reset position state for tap
        startX = 0;
        startY = 0;
        dx = 0;
        dy = 0;
        return;
      }

      // Calculate abs values BEFORE resetting dx/dy
      const absDx = Math.abs(dx);
      const absDy = Math.abs(dy);

      // Check if threshold met
      if (Math.max(absDx, absDy) >= THRESHOLD) {
        let exitClass, rating;
        if (absDx > absDy) {
          if (dx < 0) {
            exitClass = 'exit-left';
            rating = 0;
          } else {
            exitClass = 'exit-right';
            rating = 3;
          }
        } else {
          if (dy < 0) {
            exitClass = 'exit-up';
            rating = 2;
          } else {
            exitClass = 'exit-down';
            rating = 1;
          }
        }

        // Exit animation
        card.classList.add('animating');
        card.classList.add(exitClass);

        // Rate after animation
        setTimeout(() => {
          this.rateCard(rating);
        }, 350);
      } else {
        // Snap back
        card.classList.add('animating');
        card.style.transform = '';
      }

      // Reset glows
      document.querySelectorAll('.swipe-glow').forEach((g) => (g.style.opacity = 0));
      document.querySelectorAll('.swipe-label').forEach((l) => (l.style.opacity = 0));
      
      // Reset position state after all checks are done
      startX = 0;
      startY = 0;
      dx = 0;
      dy = 0;
    };
    // Mouse events (desktop only — touch events call preventDefault to avoid duplicates)
    card.addEventListener('mousedown', onStart);
    card.addEventListener('mousemove', onMove);
    card.addEventListener('mouseup', onEnd);
    card.addEventListener('mouseleave', onEnd);

    // Touch events
    card.addEventListener('touchstart', (e) => {
      onStart(e);
      mouseDown = true;
    }, { passive: true });
    card.addEventListener('touchmove', onMove, { passive: false });
    card.addEventListener('touchend', onEnd);
    card.addEventListener('touchcancel', onEnd);
  },
};

// --- Boot ---
document.addEventListener('DOMContentLoaded', () => {
  // Preload voices for speech synthesis
  if ('speechSynthesis' in window) {
    speechSynthesis.getVoices();
    speechSynthesis.addEventListener('voiceschanged', () => speechSynthesis.getVoices());
  }
  App.init();
});
