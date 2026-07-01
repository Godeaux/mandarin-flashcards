/* ============================================
   SRS Module - SM-2 spaced repetition algorithm
   ============================================ */

export const SM2 = {
  /** Default state for a new card */
  defaultState() {
    return {
      easeFactor: 2.5,
      interval: 0,
      repetitions: 0,
      nextReview: null, // null means never reviewed = new card
      known: false,
    };
  },

  /**
   * Process a review rating.
   * @param {object} state - current SRS state
   * @param {number} quality - 0=Again, 1=Hard, 2=Good, 3=Easy (mapped to SM-2 0-5)
   * @returns {object} updated state
   */
  review(state, quality) {
    // Map our 0-3 scale to SM-2's 0-5 scale
    const q = [0, 2, 4, 5][quality];
    const s = { ...state };

    if (q < 3) {
      // Failed — reset
      s.repetitions = 0;
      s.interval = 0;
    } else {
      if (s.repetitions === 0) {
        s.interval = 1;
      } else if (s.repetitions === 1) {
        s.interval = 6;
      } else {
        s.interval = Math.round(s.interval * s.easeFactor);
      }
      s.repetitions += 1;
    }

    // Update ease factor
    s.easeFactor = s.easeFactor + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02));
    if (s.easeFactor < 1.3) s.easeFactor = 1.3;

    // Calculate next review date
    const now = new Date();
    if (s.interval === 0) {
      // Review again in 1 minute (for this session) — stored as today
      s.nextReview = this.todayStr();
    } else {
      const next = new Date(now.getTime() + s.interval * 86400000);
      s.nextReview = this.dateStr(next);
    }

    return s;
  },

  /** Get next interval preview for each rating */
  previewIntervals(state) {
    return [0, 1, 2, 3].map((q) => {
      const result = this.review(state, q);
      return result.interval;
    });
  },

  /** Format interval as human string */
  formatInterval(days) {
    if (days === 0) return '< 1m';
    if (days === 1) return '1d';
    if (days < 30) return days + 'd';
    if (days < 365) return Math.round(days / 30) + 'mo';
    return (days / 365).toFixed(1) + 'y';
  },

  todayStr() {
    return this.dateStr(new Date());
  },

  dateStr(d) {
    return d.toISOString().slice(0, 10);
  },

  isDue(state) {
    if (!state.nextReview) return false; // new card, not due
    return state.nextReview <= this.todayStr();
  },

  isNew(state) {
    return state.nextReview === null;
  },
};
