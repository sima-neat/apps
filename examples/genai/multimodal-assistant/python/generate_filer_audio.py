import os
from pipertts import PiperTTS

# Define output directory and filler phrases
OUTPUT_DIR = "static/fillers"
FILLER_PHRASES = [
    "Let me think about it...",
    "Just a second while I think about it...",
    "One moment please, let me think about it...",
    "I am working on it, please wait..."
]

def sanitize_filename(phrase):
    """Sanitize phrase to create a safe filename."""
    return phrase.replace("…", "").replace(".", "").replace(" ", "_").lower()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tts = PiperTTS()  # Assumes model is at default path

    for phrase in FILLER_PHRASES:
        buffer = tts.synthesize(phrase)
        filename = f"{sanitize_filename(phrase)}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        tts.save_audio(buffer, output_path)

if __name__ == "__main__":
    main()
