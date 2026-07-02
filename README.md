# 漢字 Study — Mandarin SRS

A spaced-repetition flashcard app for learning Chinese characters.

## Quick Start

```bash
# Install backend dependencies and start server
cd backend && npm install && npm start

# In another terminal, serve the frontend
npm run dev
```

## Features

- **31 curated beginner characters** with mnemonics, radicals, and example sentences
- **SM-2 spaced repetition algorithm** for optimal memorization
- **Swipe-to-rate** interface with keyboard shortcuts (Space/Arrows)
- **Dark/Light theme** toggle
- **Category filters** and progress tracking
- **Writing practice** with stroke hints

## Development Tools

### Audio Generation (using Qwen3-TTS)

The app includes a local web interface for generating Mandarin audio files:

```bash
# Start the TTS generator server
cd tools/tts-generator && python server.py

# Then open http://localhost:8765 in your browser
```

See `tools/tts-generator/README.md` for more details.

### Audio Files

Pre-generated MP3 audio files are stored in `audio/` directory. Character pronunciations use the Qwen3-TTS model locally.

## Migration to Supabase (Optional)

To switch from SQLite to Supabase:

1. Create a [Supabase project](https://supabase.com)
2. Copy schema from `backend/db/sqlite.js` to Supabase SQL Editor
3. Update `backend/config.js`:
   ```javascript
   export const DB_TYPE = 'supabase';
   ```
4. Add environment variables for Supabase credentials
5. Uncomment supabase import and update adapter export in `backend/config.js`
## License

This app is available for a one-time payment of $5 for lifetime access.