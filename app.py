# app.py — Robust Mood Playlist Generator (Hugging Face Space)
# - Uses Google Gemini via google-generativeai
# - Cleans model output, parses JSON safely
# - Adds Retry (variation) for same mood
# - Hyperlinks using spotify_dataset.csv if available (fallback: YouTube search)
# - Clickable links via gr.HTML output
# - UI improvements: dark theme, green buttons, playlist cards, shimmer input, bold white links

import os
import re
import json
import random
from typing import Optional, Tuple

import gradio as gr
import google.generativeai as genai
import pandas as pd

# ----------------------------
# Configuration / Globals
# ----------------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"  # change if needed (use a model from your available list)

try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception:
    # fallback: don't crash on import time — we'll let generate call surface errors
    model = None

# Keep track of last mood + seed used so Retry produces a different variation
_last_mood: Optional[str] = None
_last_seed: Optional[int] = None

# ----------------------------
# Load optional Spotify dataset (for exact hyperlinking)
# ----------------------------
try:
    spotify_df = pd.read_csv("spotify_dataset.csv")
    # normalize columns: expect track_name, artists, track_id or track_url
    # If 'track_id' not present but 'track_url' present, we can use it.
    print("Spotify dataset loaded successfully.")
except FileNotFoundError:
    spotify_df = None
    print("Warning: Spotify dataset not found. Hyperlinking will be best-effort (YouTube fallback).")

# ----------------------------
# Prompt — strict JSON instructions + curator context
# ----------------------------
full_prompt = """
About: 
The easiest and most effective input is a single, descriptive word that represents a mood, activity, or theme, such as "focus," "workout," or "nostalgia." This simple input provides the expert with a clear starting point to build a sophisticated and highly relevant playlist.  
How a Playlist is Created from the Word "Focus"  
Using the single word "focus," the Playlist Curator Expert would:  
Use Input as Mood for AI: [USER_INPUT] 
Analyze the Mood: The expert would start by internally considering the psychology of focus, choosing music with a consistent, low-variance tempo to prevent distraction, like instrumental tracks or lo-fi beats.  
Curate the Music: They would then draw on their extensive music knowledge to select tracks that align with this psychological profile. They'd choose songs with consistent rhythms and minimal lyrical content to maintain a steady, unobtrusive soundscape.  
Determine Length and Order: The expert would set the playlist length to match a typical work session (60-90 minutes) to minimize interruption. They would order the songs to create a subtle arc, starting with a gentle introduction, maintaining a steady core, and ending with a winding-down period.  
Refine with Data: They would create the playlist and then use data from similar successful playlists to refine their choices, ensuring the song selection and order align with what keeps listeners engaged.  
The Optimum Length for a "Focus" Playlist  
The optimum length for a "focus" playlist is 60 to 90 minutes. This length matches the duration of a typical work or study session, which helps users stay in a "flow state" without needing to restart the playlist or search for new music.  
How the Order is Decided  
The order is decided based on the psychological principles of attention and engagement, creating a subtle musical arc: an introduction to ease the listener in, a long core section of consistent tracks to maintain concentration, and a winding-down section to signal the end of the session. The expert ensures this natural flow keeps the user immersed in the experience.  
Output Format: A single JSON object with a key "playlist" whose value is an array of 10 objects. Each object in the array must have two keys: "song" and "artist". The first 8 songs should be in English and the next 2 should be in Hindi. No other text or conversation is allowed. The output must start with `{` and end with `}`.
Now, create a 10-song playlist for the mood: {mood}.
You are a strict JSON-only music playlist generator. **You must respond ONLY with a single valid JSON object.**
Do NOT include any extra text, explanation, code fences, or commentary — ONLY JSON from the first character to the last.
Output format (exact structure required):
{
  "playlist": [
    {"song": "Song Title 1", "artist": "Artist Name 1"},
    {"song": "Song Title 2", "artist": "Artist Name 2"},
    ...
  ]
}
Requirements:
- Return exactly 10 items in "playlist".
- The first 5 items must be English songs; the next 5 items must be Hindi songs.
- Each item must contain keys "song" and "artist" with string values.
- The JSON must start with '{' and end with '}'.
- No trailing commas. No comments. No stray words.
Context (for your internal choice of songs — do NOT output this context):
You are "The Playlist Curator Expert" — The Playlist Curator Expert is a master of music, data, and human behavior. This individual possesses a creative vision for music curation, backed by a deep, academic understanding of genres and their history. They are also a data-driven strategist, using analytics to validate their artistic choices and optimize playlists for maximum listener engagement. Finally, they have a strong grasp of psychology, enabling them to create playlists that genuinely connect with listeners on an emotional and psychological level. This all-in-one professional crafts relevant playlists, manages their brand, and ensures their success from start to finish. 
Create a 10-song playlist for the mood: {mood}.
Now produce the JSON object exactly as specified above.
"""

# ----------------------------
# Utility: sanitize & extract JSON from model output
# ----------------------------
def clean_and_extract_json(raw_text: str) -> str:
    if raw_text is None:
        return ""

    s = raw_text.strip()
    s = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", s, flags=re.IGNORECASE)
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        s = s[first:last + 1]
    s = re.sub(r'(".*?")\s+[A-Za-z0-9][A-Za-z0-9\s\-\&\(\)\.]*\s*(,|\})', r'\1\2', s)
    s = re.sub(r',\s*([\]\}])', r'\1', s)
    s = re.sub(r'\n\s+', '\n', s)
    return s.strip()

# ----------------------------
# Helper: parse cleaned JSON to Python dict safely
# ----------------------------
def parse_playlist_json(clean_text: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        data = json.loads(clean_text)
    except Exception as e:
        return None, f"json.loads failed: {repr(e)}"

    if not isinstance(data, dict) or "playlist" not in data:
        return None, "Parsed JSON does not contain top-level 'playlist' key."
    if not isinstance(data["playlist"], list):
        return None, "'playlist' is not an array."
    for i, item in enumerate(data["playlist"]):
        if not isinstance(item, dict) or "song" not in item or "artist" not in item:
            return None, f"Item {i} missing 'song' or 'artist'."
    return data, None

# ----------------------------
# Helper: hyperlinking using Spotify csv (if available); fallback to YouTube search
# ----------------------------
def get_spotify_link_or_youtube(song_title: str, artist_name: str) -> str:
    def youtube_search_link(song, artist):
        q = (song + " " + artist).strip().replace(" ", "+")
        return f"https://www.youtube.com/results?search_query={q}"
    if spotify_df is None:
        return youtube_search_link(song_title, artist_name)
    try:
        candidate = spotify_df[
            spotify_df['track_name'].str.contains(re.escape(song_title), case=False, na=False) &
            spotify_df['artists'].str.contains(re.escape(artist_name), case=False, na=False)
        ]
        if not candidate.empty:
            row = candidate.iloc[0]
            if 'track_id' in row and pd.notna(row['track_id']):
                return f"https://open.spotify.com/track/{row['track_id']}"
            if 'track_url' in row and pd.notna(row['track_url']):
                return row['track_url']
    except Exception:
        return youtube_search_link(song_title, artist_name)
    return youtube_search_link(song_title, artist_name)

# ----------------------------
# Core: call model and return parsed playlist dict
# ----------------------------
def call_model_and_get_playlist(mood: str, variation_seed: Optional[int] = None) -> Tuple[Optional[dict], Optional[str]]:
    if model is None:
        return None, "Model not configured (model object is None). Check MODEL_NAME and SDK."
    prompt = full_prompt.replace("{mood}", mood)
    if variation_seed is not None:
        prompt += f"\n\n# VariationSeed: {variation_seed}. Please produce a different valid playlist (JSON only)."
    try:
        resp = model.generate_content(prompt)
    except Exception as e:
        return None, f"Model call failed: {repr(e)}"
    raw_text = getattr(resp, "text", None)
    if raw_text is None:
        raw_text = str(resp)
    raw_text = raw_text.strip()
    cleaned = clean_and_extract_json(raw_text)
    data, parse_err = parse_playlist_json(cleaned)
    if data is not None:
        return data, None
    attempt = re.sub(r'(".*?")\s+[A-Za-z][A-Za-z0-9\s\-\&\(\)\.]*\s*(,|\})', r'\1\2', cleaned)
    attempt = re.sub(r',\s*([\]\}])', r'\1', attempt)
    try:
        data2 = json.loads(attempt)
        data2, parse_err2 = parse_playlist_json(json.dumps(data2))
        if data2 is not None:
            return data2, None
    except Exception:
        pass
    return None, f"Could not parse JSON after cleaning. Raw: {raw_text}\n\nCleaned: {cleaned}\n\nParse error: {parse_err}"

# ----------------------------
# UI Changes Start — Custom CSS + Playlist Cards
# ----------------------------
custom_css = """
body, .gradio-container {
    background-color: black !important;
    font-family: Arial, sans-serif !important;
    color: white !important;
}
h1, h2, h3, h4, h5, h6, .gr-markdown { color: white !important; }
.gr-button {
    background-color: #00FF00 !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    border: 2px solid white !important;
    transition: all 0.3s ease;
}
.gr-button:hover {
    background-color: #00cc00 !important;
    transform: scale(1.05);
}
.gr-textbox, .gr-textbox textarea {
    background-color: black !important;
    border: 2px solid white !important;
    color: white !important;
    font-family: Arial, sans-serif !important;
    border-radius: 6px !important;
    box-shadow: 0 0 20px rgba(255,255,255,0.2);
    transition: all 0.5s ease;
}
.gr-textbox:focus, .gr-textbox textarea:focus {
    animation: shimmer 1.5s infinite;
}
@keyframes shimmer {
    0% { box-shadow: 0 0 5px rgba(255,255,255,0.1); }
    50% { box-shadow: 0 0 15px rgba(255,255,255,0.8); }
    100% { box-shadow: 0 0 5px rgba(255,255,255,0.1); }
}
.playlist-card {
    background-color: #111 !important;
    border: 1px solid white !important;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    transition: all 0.3s ease;
}
.playlist-card a {
    color: white !important;
    font-weight: bold !important;
    text-decoration: none !important;
}
.playlist-card:hover {
    box-shadow: 0 0 20px rgba(0,255,0,0.7);
    transform: scale(1.02);
}
"""
# ----------------------------
# UI Changes End

# ----------------------------
# Public wrapper functions for Gradio
# ----------------------------
def _format_playlist_as_cards(playlist_dict: dict) -> str:
    playlist = playlist_dict.get("playlist", [])
    card_lines = []
    for idx, item in enumerate(playlist, start=1):
        song = item.get("song", "Unknown Song")
        artist = item.get("artist", "Unknown Artist")
        link = get_spotify_link_or_youtube(song, artist)
        card_lines.append(f'<div class="playlist-card">{idx}. <a href="{link}" target="_blank">{song} - {artist}</a></div>')
    return "\n".join(card_lines)
    
# def _format_playlist_as_cards(playlist):
#     html = "<div class='playlist-container'>"
#     for idx, item in enumerate(playlist, start=1):
#         song = item.get("song", "Unknown Song")
#         artist = item.get("artist", "Unknown Artist")
#         link = item.get("link", None)

#         html += f"<div class='playlist-card'><p>{idx}. {song} – {artist}</p>"

#         if link and "open.spotify.com/track/" in link:
#             # Extract track ID from Spotify URL
#             track_id = link.split("track/")[-1].split("?")[0]
#             html += f"""
#                 <iframe src="https://open.spotify.com/embed/track/{track_id}"
#                         width="100%" height="80"
#                         frameborder="0" allow="autoplay; clipboard-write; encrypted-media; picture-in-picture"
#                         loading="lazy"></iframe>
#             """
#         elif link:
#             # Non-Spotify direct link (clickable)
#             html += f'<a href="{link}" target="_blank">Listen</a>'
#         else:
#             # Fallback → YouTube search
#             query = f"{song} {artist}".replace(" ", "+")
#             yt_link = f"https://www.youtube.com/results?search_query={query}"
#             html += f'<a href="{yt_link}" target="_blank">Search on YouTube</a>'

#         html += "</div>"
#     html += "</div>"
#     return html

def generate_playlist(mood: str) -> str:
    global _last_mood, _last_seed
    if not mood or not mood.strip():
        return "⚠️ Please enter a mood (e.g., 'nostalgic', 'calm')."
    _last_mood = mood.strip()
    _last_seed = random.randint(1, 10**9)
    data, err = call_model_and_get_playlist(_last_mood, variation_seed=_last_seed)
    if err:
        return f"⚠️ Error: {err}"
    html = _format_playlist_as_cards(data)
    return html

def retry_playlist() -> str:
    global _last_mood, _last_seed
    if not _last_mood:
        return "⚠️ No previous mood found. Please generate a playlist first."
    new_seed = _last_seed
    attempts = 0
    while new_seed == _last_seed and attempts < 5:
        new_seed = random.randint(1, 10**9)
        attempts += 1
    _last_seed = new_seed
    data, err = call_model_and_get_playlist(_last_mood, variation_seed=_last_seed)
    if err:
        return f"⚠️ Error on retry: {err}"
    html = _format_playlist_as_cards(data)
    return html

# ----------------------------
# Gradio UI (Blocks) with custom CSS
# ----------------------------
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🎵 Mood-Based Playlist Generator")
    gr.Markdown("An AI Playlist Curator that generates playlists based on your mood word. Powered by Google Gemini. Output links are clickable cards.")

    with gr.Row():
        input_mood = gr.Textbox(
            label="Enter a mood (e.g., 'nostalgic', 'energetic', 'calm')",
            placeholder="e.g. calm",
            lines=1
        )

        with gr.Column(scale=0.5):
            generate_btn = gr.Button("Generate Playlist")
            retry_btn = gr.Button("Retry Same Mood")

    playlist_output = gr.HTML(label="Generated Playlist")

    generate_btn.click(fn=generate_playlist, inputs=input_mood, outputs=playlist_output)
    retry_btn.click(fn=retry_playlist, inputs=None, outputs=playlist_output)

    gr.Examples([["nostalgic"], ["energetic"], ["calm"], ["jolly"]], inputs=[input_mood])

# ----------------------------
# Launch
# ----------------------------
if __name__ == "__main__":
    demo.launch(share=False, debug=True)
