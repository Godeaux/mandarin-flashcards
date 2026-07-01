#!/usr/bin/env python3
"""
Simple HTTP server for Qwen3-TTS generation.
Run: python server.py
Then open: http://localhost:8765
"""

import os
import sys
import json
import tempfile
import subprocess
import uuid
from flask import Flask, request, send_file, jsonify, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Track the latest generated audio file
latest_audio_path = None

def generate_audio_with_qwen(text, voice_style="default"):
    """
    Generate audio using Qwen3-TTS via simple_generate.py
    Returns path to generated audio file.
    """
    # Create unique filename
    unique_id = str(uuid.uuid4())[:8]
    output_path = os.path.join(OUTPUT_DIR, f"{unique_id}.mp3")
    
    # Use the qwen-tts-env Python interpreter
    QWEN_PYTHON = os.path.expanduser("~/.openclaw/qwen-tts-env/bin/python")
    
    # Build command with voice description if provided
    cmd = [
        QWEN_PYTHON,
        os.path.join(SCRIPT_DIR, "simple_generate.py"),
        text,
        output_path
    ]
    
    # Add voice description if not "default"
    if voice_style and voice_style.lower() != "default" and voice_style.strip():
        cmd.append(voice_style)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=SCRIPT_DIR
        )
        
        if result.returncode != 0:
            raise Exception(result.stderr or result.stdout or "Generation failed")
        
        # The script outputs the path on success
        output_lines = result.stdout.strip().split('\n')
        generated_path = None
        for line in output_lines:
            if line.startswith("Generated: "):
                generated_path = line.split("Generated: ")[1].strip()
                break
        
        if not generated_path or not os.path.exists(generated_path):
            raise Exception("Output file not created")
        
        return generated_path
        
    except subprocess.TimeoutExpired:
        raise Exception("Generation timed out (120s)")
    except Exception as e:
        raise e

@app.route('/')
def index():
    """Serve the frontend."""
    return send_file('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate audio from text."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        voice_style = data.get('voice', 'default')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate audio
        output_path = generate_audio_with_qwen(text, voice_style)
        
        # Track the latest generated file
        global latest_audio_path
        latest_audio_path = output_path
        
        # Return the audio file
        return send_file(
            output_path,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name=os.path.basename(output_path)
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-latest-audio')
def get_latest_audio():
    """Get the path of the latest generated audio file."""
    global latest_audio_path
    if latest_audio_path and os.path.exists(latest_audio_path):
        return jsonify({'audio_path': latest_audio_path})
    return jsonify({'error': 'No audio generated yet'}), 404

@app.route('/list')
def list_audio():
    """List all generated audio files."""
    files = []
    for f in sorted(os.listdir(OUTPUT_DIR), reverse=True):
        if f.endswith('.mp3'):
            path = os.path.join(OUTPUT_DIR, f)
            files.append({
                'name': f,
                'size': os.path.getsize(path),
                'created': os.path.getctime(path)
            })
    return jsonify(files)

@app.route('/download/<filename>')
def download(filename):
    """Download a specific audio file."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(path, as_attachment=True)

@app.route('/trim', methods=['POST'])
def trim():
    """Trim an audio file based on start/end timestamps."""
    try:
        data = request.get_json()
        audio_path = data.get('audio_path', '')
        start_ms = int(data.get('start_ms', 0))
        end_ms = int(data.get('end_ms', 0))
        
        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'error': 'Invalid audio path'}), 400
        
        if start_ms < 0 or end_ms <= start_ms:
            return jsonify({'error': 'Invalid trim timestamps'}), 400
        
        # Create output path
        filename = os.path.basename(audio_path)
        name, ext = os.path.splitext(filename)
        trimmed_path = os.path.join(OUTPUT_DIR, f"{name}_trimmed{ext}")
        
        # Convert ms to seconds for ffmpeg
        start_sec = start_ms / 1000.0
        duration_sec = (end_ms - start_ms) / 1000.0
        
        # Use ffmpeg to trim
        cmd = [
            'ffmpeg', '-y',
            '-i', audio_path,
            '-ss', str(start_sec),
            '-t', str(duration_sec),
            '-c', 'copy',
            trimmed_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({'error': f'FFmpeg failed: {result.stderr}'}), 500
        
        if not os.path.exists(trimmed_path):
            return jsonify({'error': 'Trimmed file not created'}), 500
        
        # Return the trimmed file
        return send_file(
            trimmed_path,
            mimetype='audio/mpeg',
            as_attachment=False,
            download_name=f"{name}_trimmed{ext}"
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎙️ Qwen3-TTS Generator Server")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Open http://localhost:8765 in your browser")
    print("Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=8765, debug=True)
