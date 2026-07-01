# 漢字 Study — Minimal Viable Product Architecture

**Goal:** Sellable lifetime-license flashcard app (one-time $20 purchase).  
**Focus:** Flashcards + example sentences + audio playback.  
**Philosophy:** Slim, fast, maintainable, offline-first, no bloat.

---

## 🎯 Core Features (MVP)

1. **Study Mode**
   - Flashcard front: character + pinyin
   - Flip → back: meaning + example sentence + audio button
   - Rating buttons (Again/Hard/Good/Easy) with SM-2 intervals
   - Category filters (Nature, People, Numbers, Common)

2. **Browse Mode**
   - Grid of all cards
   - Click to see details + play audio

3. **Stats**
   - Cards learned / due / streak
   - Today's session count & accuracy

4. **Audio**
   - Pre-generated MP3s for each character
   - Example sentence audio (single variant per card)
   - Play/stop controls

5. **Persistence**
   - LocalStorage-only (no server, no sync)
   - Export/Import deck + progress as JSON file
   - No user accounts, no cloud, no subscriptions

---

## 🏗 High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Frontend (Pure JS)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │   UI     │  │   SRS    │  │ Storage  │         │
│  │ Module   │  │ Module   │  │ Adapter  │         │
│  └──────────┘  └──────────┘  └──────────┘         │
│  ┌──────────┐  ┌──────────┐                        │
│  │  Audio   │  │   Deck   │                        │
│  │ Player   │  │ Manager  │                        │
│  └──────────┘  └──────────┘                        │
└─────────────────────────────────────────────────────┘
                     ↓
          ┌──────────────────────┐
          │   data/ (static)     │
          │  - cards.json        │
          │  - audio/*.mp3       │
          └──────────────────────┘
```

---

## 📁 Project Structure (Post-Refactor)

```
mandarin-flashcards/
├── index.html              # Single entry point
├── style.css               # Minimal, utility-first CSS
├── js/
│   ├── main.js             # Bootstrapper + event wiring
│   ├── ui.js               # View rendering (study/browse/stats)
│   ├── srs.js              # SM-2 algorithm + queue management
│   ├── deck.js             # Card data + filtering
│   ├── audio.js            # Audio playback (Web Audio API)
│   └── storage.js          # LocalStorage wrapper + import/export
├── data/
│   ├── cards.json          # All card definitions (31+ chars)
│   └── progress.json       # User progress (backup, optional)
├── audio/
│   ├── 山.mp3              # Character pronunciation
│   ├── 山_sentence.mp3     # Example sentence audio
│   └── ...                 # One file per card
├── tools/                  # Development tools (audio gen, server)
│   ├── generate_audio.py   # Audio generation (keep latest)
│   └── server.py           # Dev server for testing/audio curation
├── LICENSE                 # Commercial license text
└── README.md               # User-facing documentation
```

---

## 🧱 Module Responsibilities

### `js/deck.js`
- Load `data/cards.json` (async)
- Filter by category
- Expose: `getAll()`, `getByCategory(cat)`, `getById(id)`, `addCard()`, `updateCard()`, `reset()`

### `js/srs.js`
- SM-2 review logic (review(card, rating) → newInterval)
- Queue management (due cards + new cards)
- Stats tracking (session count, accuracy, streak)
- Expose: `getNextCard()`, `rateCard(rating)`, `getStats()`, `previewIntervals()`, `formatInterval()`

### `js/ui.js`
- Render flashcard (front/back)
- Render study queue, rating buttons with interval previews
- Render browse grid + card detail modal
- Render stats dashboard
- Handle view switching (study/browse/stats/manage)
- Handle navigation, theme toggle, search, toast notifications
- Expose: `renderStudyView()`, `showCard()`, `flipCard()`, `updateDashboard()`, `renderBrowse()`, `showCardDetail()`, `switchView()`, `toggleTheme()`, `toast()`

### `js/audio.js`
- Preload audio files (lazy load on demand)
- Play/stop sentence audio with fallback to Web Speech API
- Handle audio variant picker UI
- Promote selected variant to main audio
- Expose: `play()`, `stop()`, `playVariant()`, `promoteVariant()`, `updateAudioPicker()`

### `js/storage.js`
- Wrapper around localStorage
- Save/load SRS state, session, streak, category, theme
- Export full deck + progress to JSON file
- Import from JSON file (merge strategy)
- Optional server sync for future licensing/sync features
- Expose: `load()`, `save()`, `detectServer()`, `loadFromServer()`, `saveToServer()`, `saveAll()`

### `js/main.js`
- Initialize app on DOMContentLoaded
- Load deck → init SRS → render study view
- Wire up navigation, keyboard shortcuts, audio controls
- Handle errors gracefully (show toast messages)
- Export/Import deck functionality
- Reset progress/reset everything

---

## 🗄 Data Model

### Card (`data/cards.json`)
```json
{
  "id": "shan1",
  "char": "山",
  "pinyin": "shān",
  "meaning": "mountain",
  "category": "nature",
  "mnemonic": "Three peaks rising up...",
  "exampleSentence": "我喜欢爬山。",
  "sentencePinyin": "wǒ xǐhuān pá shān.",
  "sentenceMeaning": "I like mountain climbing."
}
```

### SRS State (`localStorage`)
```json
{
  "srsData": {
    "shan1": {
      "easeFactor": 2.5,
      "interval": 1,
      "repetitions": 1,
      "nextReview": "2026-04-03"
    }
  },
  "session": {
    "date": "2026-06-29",
    "reviewed": 5,
    "correct": 4
  },
  "streak": {
    "count": 3,
    "lastDate": "2026-06-29"
  }
}
```

---

## 🚀 Build & Deploy Flow

1. **Dev:** Run local static server (Python `http.server` from root or `tools/` directory)
2. **Build:** Minify JS/CSS, optimize audio files (optional)
3. **Deploy:** Push to GitHub Pages / Netlify / sell as ZIP download
4. **License:** Simple license key check on first run (optional, can be skipped for MVP)

### Development Tools (`tools/` directory)
- `generate_audio.py` - Batch generate Qwen3 TTS audio for flashcard characters
- `server.py` - Local dev server with progress sync + audio curation endpoints

These tools are kept for future development/audio generation but are **not** required for runtime.
The app runs 100% client-side with LocalStorage-only persistence.

---

## 💰 Monetization Strategy (Lifetime License)

- **Option A: No DRM** — Sell ZIP file with app + instructions. Trust-based.
- **Option B: Basic License Key** — Generate unique key at purchase, validate against local storage (can be bypassed but deters casual pirates).
- **Option C: Hardware-bound** — Tie license to device fingerprint (browser fingerprinting).

For MVP: Start with **Option A**, add license check later if needed.

---

## 🛠 Tech Stack Choices

| Layer | Choice | Why |
|-------|--------|-----|
| Frontend | Vanilla JS (ES6 modules) | No build step, zero dependencies, small bundle |
| Styling | Plain CSS + CSS variables | Simple theming, no framework bloat |
| Storage | LocalStorage | Offline-first, no server costs |
| Audio | Web Audio API | Native, no external libs |
| Build | None (or Vite for minification) | Keep it simple |
| Hosting | GitHub Pages / Netlify | Free static hosting |

---

## 📅 Refactor Roadmap

### Phase 1: Strip Down (Week 1)
- Remove dead code (writing canvas, audio variants, server sync)
- Delete `server.py`, `generate_audio_v*.py` (keep only pre-generated MP3s)
- Consolidate card data into `data/cards.json`
- Remove global constants from `app.js`

### Phase 2: Modularize (Week 2)
- Split `app.js` → 5 modules (`deck`, `srs`, `ui`, `audio`, `storage`)
- Convert to ES modules (`<script type="module">`)
- Add proper error handling + toast notifications

### Phase 3: Polish (Week 3)
- Optimize CSS (remove unused rules, add animations)
- Improve mobile responsiveness
- Add onboarding tour / tooltips for first-time users
- Export/Import UI polish

### Phase 4: Ship (Week 4)
- Minify JS/CSS
- Test on multiple browsers + devices
- Write README + license file
- Create ZIP bundle + landing page

---

## 🎯 Success Metrics

- **Bundle size:** < 500KB total (JS + CSS + 31 cards + audio)
- **Load time:** < 2s on slow 3G
- **Code complexity:** No file > 300 lines, no global variables
- **Maintainability:** New feature can be added in < 1 day

---

## 🚫 Out of Scope (For Now)

- User accounts / cloud sync
- Multi-user support
- Advanced analytics
- Gamification (achievements, leaderboards)
- Writing practice canvas
- Audio variant curation
- Mobile app wrapper (PWA only if time permits)

---

**Next Step:** Start Phase 1 — delete dead code + consolidate data.  
Want me to begin? 🛠️
