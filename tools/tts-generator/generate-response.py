#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_generate import generate_audio

text = "收到你的语音消息！从现在开始，在这个会话中，当你发送语音消息时，我会自动用语音回复。要我继续这样做吗？"
output_path = "/tmp/tts-response.mp3"
voice_desc = "a warm, friendly voice speaking naturally"

output = generate_audio(text, output_path=output_path, voice_desc=voice_desc)
print(f"Generated: {output}")
