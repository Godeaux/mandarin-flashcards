#!/usr/bin/env python3
"""Batch-generate Qwen3 TTS audio for all flashcard characters.

Uses max_new_tokens cap to prevent hallucinated extra audio.
Instructive tone for language learners.
"""
import os
import subprocess
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# All 31 characters with pinyin for the instruction text
CHARS = [
    ("山", "shān", "mountain"),
    ("水", "shuǐ", "water"),
    ("火", "huǒ", "fire"),
    ("木", "mù", "tree"),
    ("日", "rì", "sun"),
    ("月", "yuè", "moon"),
    ("雨", "yǔ", "rain"),
    ("田", "tián", "field"),
    ("石", "shí", "stone"),
    ("风", "fēng", "wind"),
    ("人", "rén", "person"),
    ("大", "dà", "big"),
    ("小", "xiǎo", "small"),
    ("口", "kǒu", "mouth"),
    ("手", "shǒu", "hand"),
    ("目", "mù", "eye"),
    ("女", "nǚ", "woman"),
    ("子", "zǐ", "child"),
    ("心", "xīn", "heart"),
    ("一", "yī", "one"),
    ("二", "èr", "two"),
    ("三", "sān", "three"),
    ("四", "sì", "four"),
    ("五", "wǔ", "five"),
    ("好", "hǎo", "good"),
    ("中", "zhōng", "middle"),
    ("天", "tiān", "sky"),
    ("王", "wáng", "king"),
    ("马", "mǎ", "horse"),
    ("门", "mén", "door"),
    ("力", "lì", "power"),
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(OUT_DIR, exist_ok=True)

INSTRUCT = (
    "Speak clearly and slowly, as if teaching a student who is learning Mandarin Chinese for the first time. "
    "Pronounce each tone precisely and distinctly. Be calm and instructive."
)

# Max tokens: ~12 tokens/sec at 12Hz, cap at 3 seconds of audio = ~36 tokens
# Give some headroom: 60 tokens max
MAX_NEW_TOKENS = 18

print("Loading Qwen3-TTS model...", flush=True)
tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="mps",
    dtype="float32",
)

success = 0
failed = []

for char, pinyin, meaning in CHARS:
    wav_path = os.path.join(OUT_DIR, f"{char}.wav")
    mp3_path = os.path.join(OUT_DIR, f"{char}.mp3")

    print(f"Generating: {char} ({pinyin}, {meaning})...", flush=True)
    try:
        wavs, sr = tts.generate_custom_voice(
            text=char,
            language="chinese",
            speaker="serena",
            instruct=INSTRUCT,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        wav = np.asarray(wavs[0], dtype=np.float32)
        dur = len(wav) / sr
        print(f"  -> {dur:.1f}s", flush=True)

        # Save as wav first, then convert to mp3
        sf.write(wav_path, wav, sr)

        # Convert to mp3 with ffmpeg
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-b:a", "128k", "-ar", "24000", mp3_path],
            capture_output=True,
        )

        # Remove intermediate wav
        if os.path.exists(mp3_path):
            os.remove(wav_path)
            success += 1
        else:
            failed.append(char)
            print(f"  ⚠️  ffmpeg conversion failed for {char}", flush=True)

    except Exception as e:
        failed.append(char)
        print(f"  ❌ Failed: {e}", flush=True)

print(f"\nDone! {success}/{len(CHARS)} generated successfully.", flush=True)
if failed:
    print(f"Failed: {', '.join(failed)}", flush=True)
