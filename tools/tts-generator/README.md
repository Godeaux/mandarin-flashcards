# Qwen3-TTS Web Interface

A simple web interface for generating Mandarin audio files using Qwen3-TTS locally.

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   pip install flask flask-cors
   ```

2. **Start the server**:
   ```bash
   cd /Users/nut/.openclaw/workspace/qwen3-tts-interface
   python server.py
   ```

3. **Open in browser**:
   ```
   http://localhost:8765
   ```

## Usage

1. Enter Chinese text in the text area (e.g., `你好，今天天气很好。`)
2. Optionally set a filename (auto-generated if empty)
3. Choose a voice style (default works fine)
4. Click **Generate Audio**
5. Wait 10-30 seconds for generation
6. Play back the audio
7. Click **Save to File** to download the MP3

## Features

- ✅ Simple web interface
- ✅ Local generation (no API needed)
- ✅ Instant playback
- ✅ Download generated files
- ✅ Auto-filename from text
- ✅ MP3 output (192kbps, 24kHz)

## Output Location

Generated files are saved to:
```
/Users/nut/.openclaw/workspace/qwen3-tts-interface/output/
```

## Notes

- First generation will take longer (model loading)
- Subsequent generations are faster
- Model runs on MPS (M4 Mac)
- Timeout: 120 seconds per generation

## Troubleshooting

**"Model not found"**: Make sure Qwen3-TTS is installed and the path is correct in `simple_generate.py`.

**"Generation timed out"**: Complex sentences take longer. Try shorter text or increase timeout in `server.py`.

**"FFmpeg not found"**: Install ffmpeg: `brew install ffmpeg`
