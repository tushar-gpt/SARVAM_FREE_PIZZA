# Saathi: understand it, teach it, test it

## Run
```
pip install -r requirements.txt
export SARVAM_API_KEY=your_key      # or paste it in the sidebar
streamlit run app.py
```
WAV lectures work out of the box. For MP3/M4A uploads longer than 30 s, install ffmpeg (pydub uses it to split audio).

## Modes
- **Samjho**: record/upload a lecture (+ board photos) → simple explanation, diagram, "I was lost here" deep dives, voice.
- **Samjhao**: pick any topic → explain it to Chintu (confused) and Priya (topper); wrong ideas get corrected.
- **Parkho**: adaptive test, easy → very tough, conceptual / numerical / logical → report.
- **Meri Pragati**: mastery per concept, weak points, personal feedback.

Modes are independent. When topics match (same concept tags), past weak points automatically get extra focus.

## Sarvam models
Saaras v4 · Sarvam Vision · Sarvam-105B · Sarvam-105B Conversations · Bulbul v3 · Sarvam Translate

## Demo flow (2 min)
1. Parkho on "seasons" → get one question wrong.
2. Meri Pragati → weak point + suggestion appears.
3. Samjhao on "seasons" → Chintu's confusion targets exactly that mistake.
4. Samjho with a short lecture clip → simple explanation + diagram.
