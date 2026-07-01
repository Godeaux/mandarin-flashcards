# Mandarin Flashcards — Project Setup & Scope

## 🎯 Vision
A focused, solo-developer Mandarin learning app. **Lifetime purchase ($5–10)**, clean UI, no social bloat. Built with AI assistance, scope remains manageable for one person.

---

## 🚀 Development Architecture (Mac Studio)

### Core Principle: Single-Source Filesystem
**Everything runs on Mac Studio.** No repo syncs. Laptop connects via **VS Code Remote-SSH**. Both agents (Cline + OpenClaw) see live files instantly.

| Component | Location | Access |
|-----------|----------|--------|
| Repo | Mac Studio only | Remote-SSH (VS Code) + OpenClaw FS tools |
| Model Server (`vllm-mlx`) | `localhost:8000` | Cline → localhost, OpenClaw → localhost |
| Cline UI | Laptop (remote window) | VS Code Remote-SSH |
| OpenClaw | Mac Studio gateway | WebChat (Tailscale) or Telegram/Slack |

### Quick Setup Steps

**1. Model Server (Mac Studio)**
```bash
pip install vllm-mlx
vllm-mlx serve mlx-community/Qwen3-Coder-Next-8bit \
  --host 0.0.0.0 --port 8000 --continuous-batching
```
*Why vllm-mlx?* Fixes KV cache bugs, enables prefix caching (reuses context), handles concurrent requests.

**2. Tailscale Networking**
```bash
# Mac Studio — enable SSH
sudo tailscale up --ssh

# Laptop — test connectivity
curl http://mac-studio.tailxxxx.ts.net:8000/v1/models
```

**3. VS Code Remote-SSH (Laptop)**
- Install "Remote - SSH" extension
- Connect to `mac-studio.tailxxxx.ts.net`
- Open repo folder → Install Cline in remote window
- Cline config: Base URL = `http://localhost:8000/v1`, Model = `mlx-community/Qwen3-Coder-Next-8bit`

**4. OpenClaw (Mac Studio)**
```bash
npm install -g openclaw@latest
openclaw onboard
```
Edit `~/.openclaw/openclaw.json`:
```json
{ "agent": { "model": "openai-compatible/mlx-community/Qwen3-Coder-Next-8bit" } }
```
Set workspace root to repo path (check `openclaw doctor` for config key).

**Contention:** Cline + OpenClaw share model queue. Continuous batching handles load; serialized execution is fine if both run simultaneously.

---

## 🧠 Key Learning Mechanics

### Signature Features
- **Mnemonic Character Aids:** Card backs include visual descriptions, sound-based associations, or combos to remember characters intuitively
- **Smart Audio Repetition (Qwen3-TTS):**  
  `Sentence → Word-by-word breakdown (Chinese + English) → Full sentence repeat`  
  Proven effective for retention.

### Future Features (Post-MVP)
- Category filtering (directions, food, entertainment, etc.)
- Basic login/streak tracking (only after accounts system is solid)
- **No social features planned** — focus on learning efficacy

---

## 📏 Scope Guardrails

| Do | Don't |
|----|-------|
| Start minimal, add categories incrementally | Match competitor feature bloat |
| Leverage AI for content generation | Build complex online/social systems |
| Keep UI clean and intentional | Add engagement hooks (leaderboards, leagues) |
| One-time purchase model | Subscriptions or freemium tiers |

**Solo-dev rule:** Every feature must be understandable/maintainable by one person. If it feels too big, cut scope.

---

## 🔑 Differentiators vs Competitors

Unlike Duolingo/HelloChinese/LingoDee:
- ✅ No ads, no subscriptions, no social pressure
- ✅ Mnemonic-based character learning (not just flashcards)
- ✅ Smart audio repetition for speaking/listening
- ✅ Category-based real-world contexts
- ✅ Lifetime purchase ($5–10)

**Own this niche:** Effective Mandarin learning without the engagement theater.
