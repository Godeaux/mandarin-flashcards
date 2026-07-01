# Immediate Next Steps — Stabilize the Project

## 🎯 Priority Order

### 1. Fix Git Index (Critical)
The Git repo is confused about file locations (moved/deleted files). Reset to match reality:
```bash
cd mandarin-flashcards
git add -A
git reset HEAD
git add .
git commit -m "Reset index to match current file structure"
```
**Why:** Prevents data loss, ensures clean history for future commits.

---

### 2. Delete Test Files (Cleanup)
Remove the 11 test HTML files in `tools/tests/` — they're dev artifacts, not needed for production:
```bash
rm -rf tools/tests/
git add -A
git commit -m "Remove test files — keep Git history if needed"
```
**Why:** Reduces clutter, prevents confusion about what's part of the app.

---

### 3. Add Build + Prettier + Linting (Professional Quality)
Set up esbuild for minification, Prettier for code formatting, ESLint for catching bugs:

#### a. Create `package.json`
```bash
npm init -y
```

#### b. Install dev dependencies
```bash
npm install -D esbuild prettier eslint
```

#### c. Create config files
**`.prettierrc`** (auto-format code):
```json
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 100
}
```

**`.eslintrc.json`** (catch bugs):
```json
{
  "env": {
    "browser": true,
    "es2022": true
  },
  "parserOptions": {
    "ecmaVersion": "latest",
    "sourceType": "module"
  },
  "rules": {
    "no-unused-vars": "warn",
    "no-console": "off"
  }
}
```

**`.gitignore`** (add these lines):
```
node_modules/
dist/
audio/variants/
```

#### d. Add npm scripts to `package.json`
```json
{
  "scripts": {
    "build": "esbuild app.js --bundle --minify --outfile=dist/app.min.js",
    "build:css": "esbuild style.css --bundle --minify --outfile=dist/style.min.css",
    "format": "prettier --write 'js/**/*.js' 'style.css'",
    "lint": "eslint js/**/*.js",
    "dev": "python3 -m http.server 8080 --bind 0.0.0.0"
  }
}
```

#### e. Run it
```bash
npm run format    # Auto-fix code style
npm run build     # Minify JS to dist/app.min.js
```

**Why:** Standardizes code, catches errors early, produces production-ready bundles.

---

### 4. Fix Server Path Issues
The current `server.py` uses relative paths that break if you run it from different directories.

#### Option A: Simple fix (recommended)
Edit `tools/server.py` to use absolute paths based on the script location:
```python
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
AUDIO_DIR = os.path.join(PROJECT_ROOT, "audio")
```

#### Option B: Just use Python's built-in server (simpler)
For development, skip `server.py` entirely and serve from root:
```bash
cd mandarin-flashcards
python3 -m http.server 8080 --bind 0.0.0.0
```
This serves everything (HTML, JS, CSS, audio) and works perfectly for local dev.

**Why:** Prevents "file not found" errors, makes the server portable.

---

### 5. Add Error Handling for Missing Audio
Currently, if an MP3 file doesn't exist, the app fails silently.

#### In `js/audio.js`, add a fallback:
```javascript
async playAudio(audioPath) {
  try {
    const audio = new Audio(audioPath);
    await audio.play();
  } catch (err) {
    console.warn('Audio not found:', audioPath);
    UI.toast('Audio unavailable — check your connection');
    // Optional: play a fallback sound or vibrate
  }
}
```

**Why:** Better user experience, prevents broken playback states.

---

## 🔧 What's the Server Actually Doing?

Let me clarify the server confusion:

### You have **two servers**:

1. **`python3 -m http.server 8080`** (what's currently running)
   - Serves **static files only** from the project root
   - `index.html` → HTML
   - `app.js` → JavaScript
   - `style.css` → CSS
   - `audio/*.mp3` → Sound files
   - **No backend logic** — just file serving

2. **`tools/server.py`** (the custom server with endpoints)
   - Also serves static files, PLUS:
   - `GET/POST /progress` → Saves SRS progress (which cards you've learned)
   - `GET/POST /audio-selections` → Tracks which audio variant you prefer per character
   - `POST /audio/promote` → Promotes a chosen variant to main audio
   - **Requires:** Running from the project root, not from `tools/`

**For now, just use option 1** (`python3 -m http.server 8080`) — it's simpler and handles 90% of your needs. The custom server is only needed for progress sync across devices.

---

## ✅ Checklist

- [ ] Fix Git index
- [ ] Delete test files
- [ ] Set up esbuild + Prettier + ESLint
- [ ] Fix server paths (or just use built-in server)
- [ ] Add audio fallback handling
- [ ] Run `npm run format` on all JS
- [ ] Commit everything
- [ ] Deploy to GitHub Pages

Start with **step 1 (Git fix)** — it takes 30 seconds and prevents future headaches. Then tackle the tooling setup (step 3) — it'll pay dividends as you add more features. 🎯
