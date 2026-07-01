/* ============================================
   UI Module - View rendering & event binding
   ============================================ */

import { SM2 } from './srs.js';
import { Audio } from './audio.js';

export const UI = {
  // --- Study View Rendering ---

  /** Refresh the study view (show queue or empty state) */
  renderStudyView(studyQueue, srsData, activeCategory, deck) {
    const container = document.getElementById('flashcard-container');
    const empty = document.getElementById('study-empty');
    const writing = document.getElementById('writing-practice');

    writing.classList.add('hidden');

    if (studyQueue.length === 0) {
      container.classList.add('hidden');
      empty.classList.remove('hidden');
    } else {
      container.classList.remove('hidden');
      empty.classList.add('hidden');
      this.showCard(studyQueue, srsData);
    }

    this.updateDashboard(studyQueue.length, activeCategory, deck);
  },

  /** Show the current flashcard */
  showCard(studyQueue, srsData) {
    if (studyQueue.length === 0 || this.currentCardIndex >= studyQueue.length) {
      this.renderStudyView(studyQueue, srsData, this.activeCategory, this.deck);
      return;
    }

    const card = studyQueue[this.currentCardIndex];
    const srs = srsData[card.id];

    // Front
    document.getElementById('card-char').textContent = card.char;
    document.getElementById('card-pinyin').textContent = card.pinyin;
    document.getElementById('card-cat-tag').textContent = card.category;

    // Back
    document.getElementById('card-char-back').textContent = card.char;
    document.getElementById('card-pinyin-back').textContent = card.pinyin;
    document.getElementById('card-meaning').textContent = card.meaning;
    document.getElementById('card-mnemonic').textContent = card.mnemonic || '';
    document.getElementById('card-visual').textContent = card.visual || '';
    document.getElementById('card-sound-bridge').textContent = card.soundBridge || '';
    document.getElementById('card-radicals').textContent = card.radicals || '';
    document.getElementById('card-example-sentence').textContent = card.exampleSentence || '';
    document.getElementById('card-example-breakdown').textContent = card.exampleBreakdown || '';

    // Progress
    document.getElementById('card-progress-text').textContent =
      `${this.currentCardIndex + 1} / ${studyQueue.length}`;

    // Rating time previews
    const intervals = SM2.previewIntervals(srs);
    document.getElementById('rate-again-time').textContent = SM2.formatInterval(intervals[0]);
    document.getElementById('rate-hard-time').textContent = SM2.formatInterval(intervals[1]);
    document.getElementById('rate-good-time').textContent = SM2.formatInterval(intervals[2]);
    document.getElementById('rate-easy-time').textContent = SM2.formatInterval(intervals[3]);

    // Audio picker
    this.updateAudioPicker(card);

    // Reset flip state
    this.isFlipped = false;
    document.getElementById('flashcard').classList.remove('flipped');
    document.getElementById('rating-buttons').classList.add('hidden');
  },

  /** Flip the flashcard */
  flipCard() {
    const flashcard = document.getElementById('flashcard');
    if (!flashcard || flashcard.closest('.hidden')) return;

    const isFlipped = flashcard.classList.toggle('flipped');
    document.getElementById('rating-buttons').classList.toggle('hidden', !isFlipped);
    this.resizeCardInner(isFlipped);
  },

  /** Resize card inner content based on flip state */
  resizeCardInner(isFlipped) {
    const inner = document.querySelector('.card-inner');
    if (isFlipped) {
      const back = document.querySelector('.card-back');
      requestAnimationFrame(() => {
        const h = back.scrollHeight;
        inner.style.minHeight = Math.max(400, h) + 'px';
      });
    } else {
      inner.style.minHeight = '400px';
    }
  },

  /** Update dashboard stats */
  updateDashboard(dueCount, activeCategory, deck) {
    document.getElementById('stat-due').textContent = dueCount;

    const categories = this.getCategories(deck);
    const catSelect = document.getElementById('category-filter');
    if (catSelect && catSelect.value !== activeCategory) {
      catSelect.value = activeCategory;
    }

    // Update category stats in browse view if visible
    const catStats = document.getElementById('browse-category-stats');
    if (catStats) {
      catStats.innerHTML = categories
        .map((cat) => {
          const catCards = deck.filter((c) => c.category === cat);
          const learned = catCards.filter((c) => srsData[c.id]?.known).length;
          const pct = catCards.length ? Math.round((learned / catCards.length) * 100) : 0;

          return `
          <div class="cat-progress-row">
            <span class="cat-progress-label">${this.capitalize(cat)}</span>
            <div class="cat-progress-bar">
              <div class="cat-progress-fill" style="width: ${pct}%"></div>
            </div>
            <span class="cat-progress-text">${pct}%</span>
          </div>
        `;
        })
        .join('');
    }
  },

  // --- Browse View Rendering ---

  renderBrowse(deck, srsData, activeCategory) {
    const grid = document.getElementById('browse-grid');
    if (!grid) return;

    let filtered = deck.filter((c) =>
      activeCategory === 'all' ? true : c.category === activeCategory
    );

    // Sort
    const sortMode = document.getElementById('browse-sort')?.value || 'id';
    filtered.sort((a, b) => {
      if (sortMode === 'id') return a.id.localeCompare(b.id);
      if (sortMode === 'learned') {
        const aLearned = srsData[a.id]?.known ? 1 : 0;
        const bLearned = srsData[b.id]?.known ? 1 : 0;
        return bLearned - aLearned;
      }
      if (sortMode === 'pinyin') return a.pinyin.localeCompare(b.pinyin);
      return 0;
    });

    grid.innerHTML = filtered
      .map((card) => {
        const srs = srsData[card.id];
        const learned = srs?.known ? 'learned' : '';

        return `
        <div class="browse-card ${learned}" data-id="${card.id}">
          <div class="browse-char">${card.char}</div>
          <div class="browse-pinyin">${card.pinyin}</div>
          <div class="browse-meaning">${card.meaning}</div>
        </div>
      `;
      })
      .join('');

    // Bind click events
    grid.querySelectorAll('.browse-card').forEach((el) => {
      el.addEventListener('click', () => this.showCardDetail(el.dataset.id, deck, srsData));
    });
  },

  /** Show card detail modal */
  showCardDetail(cardId, deck, srsData) {
    const card = deck.find((c) => c.id === cardId);
    if (!card) return;

    this.selectedCardId = cardId;

    document.getElementById('detail-char').textContent = card.char;
    document.getElementById('detail-pinyin').textContent = card.pinyin;
    document.getElementById('detail-meaning').textContent = card.meaning;
    document.getElementById('detail-mnemonic').textContent = card.mnemonic || '';
    document.getElementById('detail-visual').textContent = card.visual || '';
    document.getElementById('detail-sound-bridge').textContent = card.soundBridge || '';
    document.getElementById('detail-radicals').textContent = card.radicals || '';
    document.getElementById('detail-example').textContent = card.exampleSentence || '';
    document.getElementById('detail-breakdown').textContent = card.exampleBreakdown || '';

    const srs = srsData[card.id];
    const learned = srs?.known ? 'Known' : srs?.nextReview ? 'Learning' : 'New';
    document.getElementById('detail-status').textContent = learned;

    document.getElementById('card-detail-modal').classList.remove('hidden');
  },

  closeCardDetail() {
    this.selectedCardId = null;
    document.getElementById('card-detail-modal').classList.add('hidden');
  },

  toggleKnown(cardId, deck, srsData, activeCategory) {
    const card = deck.find((c) => c.id === cardId);
    if (!card) return;

    const srs = srsData[cardId];
    srs.known = !srs.known;

    // Update UI
    this.closeCardDetail();
    this.renderBrowse(deck, srsData, activeCategory);
  },

  // --- Stats View Rendering ---

  renderStats(deck, srsData, session, streak) {
    const total = deck.length;
    let learned = 0,
      known = 0,
      due = 0;

    deck.forEach((card) => {
      const srs = srsData[card.id];
      if (srs?.known) {
        known++;
        learned++;
      } else if (SM2.isDue(srs)) {
        due++;
        learned++;
      } else if (!SM2.isNew(srs)) {
        learned++;
      }
    });

    document.getElementById('stats-total').textContent = total;
    document.getElementById('stats-learned').textContent = learned;
    document.getElementById('stats-known').textContent = known;
    document.getElementById('stats-due').textContent = due;
    document.getElementById('stats-streak').textContent = streak.count + ' days';
    document.getElementById('stats-session-count').textContent = session.reviewed;
    document.getElementById('stats-session-correct').textContent = session.correct;
    document.getElementById('stats-session-acc').textContent =
      session.reviewed > 0 ? Math.round((session.correct / session.reviewed) * 100) + '%' : '—';

    // Category progress
    const categories = this.getCategories(deck);
    document.getElementById('stats-category-progress').innerHTML = categories
      .map((cat) => {
        const catCards = deck.filter((c) => c.category === cat);
        const learnedInCat = catCards.filter((c) => srsData[c.id]?.known).length;
        const pct = catCards.length ? Math.round((learnedInCat / catCards.length) * 100) : 0;

        return `
          <div class="cat-progress-row">
            <span class="cat-progress-label">${this.capitalize(cat)}</span>
            <div class="cat-progress-bar">
              <div class="cat-progress-fill" style="width: ${pct}%"></div>
            </div>
            <span class="cat-progress-text">${pct}%</span>
          </div>
        `;
      })
      .join('');
  },

  // --- Navigation & Theme ---

  switchView(viewName) {
    document.querySelectorAll('.view').forEach((v) => {
      v.classList.remove('active');
      v.classList.add('hidden');
    });
    document.querySelectorAll('.nav-tab').forEach((t) => t.classList.remove('active'));

    const viewEl = document.getElementById(`view-${viewName}`);
    if (viewEl) {
      viewEl.classList.add('active');
      viewEl.classList.remove('hidden');
    }
    const tab = document.querySelector(`.nav-tab[data-view="${viewName}"]`);
    if (tab) tab.classList.add('active');
  },

  toggleTheme() {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    document.documentElement.setAttribute('data-theme', isLight ? 'dark' : 'light');

    // Save preference
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('hanzi_theme', isLight ? 'dark' : 'light');
    }

    // Update button icon
    const btn = document.getElementById('btn-theme');
    if (btn) btn.textContent = isLight ? '🌙' : '☀️';
  },

  openSearch() {
    document.getElementById('search-overlay').classList.remove('hidden');
    setTimeout(() => document.getElementById('search-input')?.focus(), 10);
  },

  closeSearch() {
    document.getElementById('search-overlay').classList.add('hidden');
  },

  // --- Utilities ---

  toast(message) {
    const el = document.getElementById('toast');
    if (!el) return;

    el.textContent = message;
    el.classList.remove('hidden');

    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => el.classList.add('hidden'), 2500);
  },

  capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
  },

  getCategories(deck) {
    if (!deck) return [];
    const cats = new Set(deck.map((c) => c.category));
    return Array.from(cats).sort();
  },
};
