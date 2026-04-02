#!/usr/bin/env python3
"""Local dev server for Mandarin flashcards — progress sync + audio curation.

Run: python server.py
Endpoints:
  GET/POST /progress           — SRS state, known flags, session, streak
  GET/POST /audio-selections   — which audio variant is picked per card
  GET      /audio/variants/*   — serve variant audio files
  POST     /audio/promote      — promote a variant to main audio, delete others
"""
import json
import os
import shutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote

PORT = 8787
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
VARIANTS_DIR = os.path.join(AUDIO_DIR, "variants")

# Allowed origins for CORS
ALLOWED_ORIGINS = [
    "https://godeaux.github.io",
    "http://localhost",
    "http://127.0.0.1",
    "null",  # file:// origin
]

os.makedirs(DATA_DIR, exist_ok=True)


def read_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def write_json(filename, data):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def origin_allowed(origin):
    if not origin:
        return True
    return any(origin.startswith(o) for o in ALLOWED_ORIGINS)


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        origin = self.headers.get("Origin", "")
        if origin_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin or "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)

        if path == "/progress":
            self._json_response(read_json("progress.json"))

        elif path == "/audio-selections":
            self._json_response(read_json("audio-selections.json"))

        elif path.startswith("/audio/variants/"):
            filename = path.split("/audio/variants/", 1)[1]
            filepath = os.path.join(VARIANTS_DIR, filename)
            if os.path.isfile(filepath):
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self._cors()
                size = os.path.getsize(filepath)
                self.send_header("Content-Length", size)
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._json_response({"error": "not found"}, 404)

        elif path == "/audio/variants":
            # List available variants grouped by character
            variants = {}
            if os.path.isdir(VARIANTS_DIR):
                for f in sorted(os.listdir(VARIANTS_DIR)):
                    if f.endswith(".mp3"):
                        # e.g. 山_v1.mp3 → char=山, variant=1
                        base = f[:-4]  # strip .mp3
                        if "_v" in base:
                            char, vnum = base.rsplit("_v", 1)
                            variants.setdefault(char, []).append(int(vnum))
            self._json_response(variants)

        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        path = unquote(urlparse(self.path).path)

        if path == "/progress":
            data = json.loads(self._read_body())
            write_json("progress.json", data)
            self._json_response({"ok": True})

        elif path == "/audio-selections":
            data = json.loads(self._read_body())
            write_json("audio-selections.json", data)
            self._json_response({"ok": True})

        elif path == "/audio/promote":
            data = json.loads(self._read_body())
            char = data.get("char")
            variant = data.get("variant")
            if not char or not variant:
                self._json_response({"error": "char and variant required"}, 400)
                return

            src = os.path.join(VARIANTS_DIR, f"{char}_v{variant}.mp3")
            dst = os.path.join(AUDIO_DIR, f"{char}.mp3")

            if not os.path.isfile(src):
                self._json_response({"error": f"variant file not found: {char}_v{variant}.mp3"}, 404)
                return

            # Copy chosen variant to main audio
            shutil.copy2(src, dst)

            # Delete all variants for this character
            removed = 0
            for f in os.listdir(VARIANTS_DIR):
                if f.startswith(f"{char}_v") and f.endswith(".mp3"):
                    os.remove(os.path.join(VARIANTS_DIR, f))
                    removed += 1

            self._json_response({"ok": True, "promoted": f"{char}_v{variant}.mp3", "removed": removed})

        else:
            self._json_response({"error": "not found"}, 404)

    def log_message(self, format, *args):
        print(f"[server] {args[0]}")


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Flashcard server running on http://localhost:{PORT}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Audio dir: {AUDIO_DIR}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
