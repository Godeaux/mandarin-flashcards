#!/usr/bin/env python3
"""
Simple Qwen3-TTS generation for the web interface.
Usage: python simple_generate.py "你的文本" [voice description]
Output: output/<uuid>.mp3
"""

import os
import sys
import time
import uuid
import subprocess
import numpy as np
import soundfile as sf

from qwen_tts import Qwen3TTSModel

def generate_audio(text, output_path=None, voice_desc=None):
    """
    Generate audio from text using Qwen3-TTS.
    
    Args:
        text: Chinese text to synthesize
        output_path: Output MP3 path (optional, auto-generated if not provided)
        voice_desc: Custom voice description string (e.g., "a warm female voice speaking slowly")
    
    Returns:
        Path to generated MP3 file
    """
    if output_path is None:
        output_dir = os.path.expanduser("~/openclaw/workspace/qwen3-tts-interface/output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4().hex[:8]}.mp3")
    
    print(f"Generating: {text}", flush=True)
    if voice_desc:
        print(f"Voice style: {voice_desc}", flush=True)
    
    try:
        # Initialize model (lazy load)
        if not hasattr(generate_audio, 'model'):
            print("Loading Qwen3-TTS model...", flush=True)
            generate_audio.model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map="mps",  # Use device_map instead of device
                dtype="float32"
            )
            print("Model loaded!", flush=True)
        
        model = generate_audio.model
        
        # Generate audio with custom voice description if provided
        start = time.time()
        if voice_desc:
            # Use generate_custom_voice with the voice description as instruct
            wav_list, sr = model.generate_custom_voice(
                text=text,
                language="Auto",
                speaker="serena",  # Default speaker
                instruct=voice_desc,
            )
        else:
            # Use default generation
            wav_list, sr = model.generate_custom_voice(
                text=text,
                language="Auto",
                speaker="serena",
                instruct="Speak naturally.",
            )
        
        # Get first audio sample
        wav = np.asarray(wav_list[0], dtype=np.float32)
        print(f"Generation took {time.time() - start:.1f}s", flush=True)
        
        # Save as WAV first
        wav_path = output_path.replace('.mp3', '.wav')
        sf.write(wav_path, wav, sr)
        
        # Convert to MP3
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path,
             "-af", "loudnorm=I=-14:TP=-1:LRA=11",
             "-b:a", "192k", "-ar", "24000", output_path],
            capture_output=True,
            text=True
        )
        
        # Clean up WAV
        if os.path.exists(output_path):
            os.remove(wav_path)
            print(f"Saved: {output_path}", flush=True)
            return output_path
        else:
            raise Exception(f"FFmpeg failed: {result.stderr}")
            
    except Exception as e:
        print(f"Error: {e}", flush=True)
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_generate.py \"your Chinese text here\" [output_path] [voice description]")
        sys.exit(1)
    
    text = sys.argv[1]
    
    # Check if second arg is a path (contains /) or voice description
    if len(sys.argv) > 2 and '/' in sys.argv[2]:
        output_path = sys.argv[2]
        voice_desc = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        output_path = None
        voice_desc = sys.argv[2] if len(sys.argv) > 2 else None
    
    output = generate_audio(text, output_path=output_path, voice_desc=voice_desc)
    print(f"Generated: {output}")
