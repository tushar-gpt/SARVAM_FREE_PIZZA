"""
Saathi: understand it, teach it, test it, in your own language.

Modes (independent, but connected when topics match):
  Samjho       - record a lecture -> simple explanation + diagram + "I'm lost" deep dives + board photos
  Samjhao      - revise any topic by teaching two AI classmates (confused + topper), with corrections
  Parkho       - adaptive test (easy -> very tough; conceptual / numerical / logical) + report
  Meri Pragati - progress, weak points and personal feedback

Sarvam models used:
  Saaras v4 (speech-to-text), Sarvam Vision (board/notes OCR), Sarvam-105B (reasoning, judging,
  questions), Sarvam-105B Conversations (classmate replies), Bulbul v3 (voices),
  Sarvam Translate (languages beyond the 11 core ones)

Run:  pip install -r requirements.txt  &&  streamlit run app.py
"""

import base64
import io
import json
import os
import re
import time
import wave
import zipfile
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

BASE = "https://api.sarvam.ai"
PROGRESS_FILE = "progress.json"
SEG_SECONDS = 25  # Saaras REST accepts up to 30 s per request

LANGS = {
    # 11 core languages: Sarvam-105B + Bulbul v3 work natively
    "English": "en-IN", "Hindi": "hi-IN", "Telugu": "te-IN", "Tamil": "ta-IN",
    "Kannada": "kn-IN", "Malayalam": "ml-IN", "Marathi": "mr-IN", "Bengali": "bn-IN",
    "Gujarati": "gu-IN", "Punjabi": "pa-IN", "Odia": "od-IN",
    # extra languages: reached through Sarvam Translate (text only)
    "Assamese": "as-IN", "Urdu": "ur-IN", "Nepali": "ne-IN", "Maithili": "mai-IN",
    "Konkani": "kok-IN", "Sanskrit": "sa-IN", "Bodo": "brx-IN", "Dogri": "doi-IN",
    "Kashmiri": "ks-IN", "Manipuri": "mni-IN", "Santali": "sat-IN", "Sindhi": "sd-IN",
}
CORE_LANGS = list(LANGS)[:11]
LEVELS = {1: "Easy", 2: "Basic", 3: "Medium", 4: "Hard", 5: "Very tough"}

st.set_page_config(page_title="Saathi", page_icon="📚", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Baloo 2', system-ui, sans-serif !important; }
.hero { background: linear-gradient(120deg,#3b2fc9,#7b3fe4 55%,#ff9f1c); border-radius: 22px; padding: 22px 28px;
        color: #fff; margin-bottom: 8px; animation: pop .6s ease-out; }
.hero h1 { font-size: 2.6rem; margin: 0; font-weight: 800; color: #fff; }
.hero p { margin: 2px 0 0; font-size: 1.15rem; opacity: .95; }
@keyframes pop { from { transform: scale(.96); opacity: 0 } to { transform: scale(1); opacity: 1 } }
.stTabs [data-baseweb="tab"] { font-size: 1.05rem; font-weight: 700; border-radius: 999px; padding: 6px 16px; }
.stTabs [aria-selected="true"] { background: #ffefd6; color: #3b2fc9 !important; }
.stButton > button { border-radius: 999px; font-weight: 700; transition: transform .12s; }
.stButton > button:hover { transform: translateY(-2px) scale(1.03); }
.stButton > button:active { transform: scale(.97); }
[data-testid="stChatMessage"] { border-radius: 18px; animation: pop .35s ease-out; }
.stProgress > div > div > div > div { background: linear-gradient(90deg,#ff9f1c,#7b3fe4) !important; }
.chip { display:inline-block; background:#ffefd6; color:#3b2fc9; border-radius:999px; padding:2px 12px; font-weight:700; margin-right:6px; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Settings")
    api_key = st.text_input("Sarvam API key", value=os.getenv("SARVAM_API_KEY", ""), type="password")
    lang_name = st.selectbox("Your language", list(LANGS), index=0)
    voice_on = st.toggle("Speak replies aloud", value=True)
    if lang_name not in CORE_LANGS:
        st.caption(f"{lang_name} is shown as text through Sarvam Translate. Voice is available in the 11 core languages.")
    st.divider()
    st.markdown(
        "**Sarvam models in this app**\n"
        "- Saaras v4: lecture and answer transcription\n"
        "- Sarvam Vision: board and notes photos\n"
        "- Sarvam-105B: explaining, judging, questions\n"
        "- Sarvam-105B Conversations: classmates' replies\n"
        "- Bulbul v3: voices\n"
        "- Sarvam Translate: 23 languages"
    )

LANG = LANGS[lang_name]
NEEDS_TRANSLATE = lang_name not in CORE_LANGS
OUT_LANG = "English" if NEEDS_TRANSLATE else lang_name
RULES = (
    f"Write every student-facing text in {OUT_LANG}. Keep technical terms, formulas and units in English, "
    "the way students see them in exams. Use simple words and everyday Indian examples. "
    "Concept tags must be lowercase 'subject/topic/concept' with hyphens instead of spaces "
    "(e.g. 'physics/seasons/earth-tilt'). Reuse an existing tag exactly when it is the same concept."
)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ----------------------------------------------------------------------------- Sarvam API helpers
def _headers(json_body=False):
    if not api_key:
        st.error("Add your Sarvam API key in the sidebar to start.")
        st.stop()
    h = {"api-subscription-key": api_key}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _check(r, what):
    if r.status_code >= 400:
        raise RuntimeError(f"{what} failed ({r.status_code}): {r.text[:300]}")
    return r.json()


def _mime(name):
    ext = name.rsplit(".", 1)[-1].lower()
    return {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4", "ogg": "audio/ogg",
            "webm": "audio/webm", "aac": "audio/aac", "flac": "audio/flac"}.get(ext, "application/octet-stream")


def split_audio(data, filename, seconds=SEG_SECONDS):
    """Split audio into <=25 s WAV chunks. Uses the stdlib for WAV, pydub (needs ffmpeg) for other formats."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            params, per = w.getparams(), w.getframerate() * seconds
            chunks = []
            while True:
                frames = w.readframes(per)
                if not frames:
                    break
                buf = io.BytesIO()
                with wave.open(buf, "wb") as out:
                    out.setparams(params)
                    out.writeframes(frames)
                chunks.append(buf.getvalue())
            return chunks, "wav"
    except (wave.Error, EOFError):
        pass
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(data))
        chunks = []
        for i in range(0, len(seg), seconds * 1000):
            buf = io.BytesIO()
            seg[i:i + seconds * 1000].export(buf, format="wav")
            chunks.append(buf.getvalue())
        return chunks, "wav"
    except Exception:
        return [data], filename.rsplit(".", 1)[-1]  # works if the clip is under 30 s


def stt(audio_bytes, filename="audio.wav", mode="codemix"):
    """Saaras v4 speech-to-text (codemix keeps English technical words in English)."""
    r = requests.post(
        f"{BASE}/speech-to-text", headers=_headers(),
        files={"file": (filename, audio_bytes, _mime(filename))},
        data={"model": "saaras:v4", "mode": mode}, timeout=120,
    )
    return (_check(r, "Saaras speech-to-text").get("transcript") or "").strip()


def llm(messages, model="sarvam-105b", max_tokens=4000, temperature=0.3):
    r = requests.post(
        f"{BASE}/v1/chat/completions", headers=_headers(True),
        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        timeout=180,
    )
    data = _check(r, f"{model} chat")
    text = data["choices"][0]["message"].get("content") or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _parse_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def llm_json(system, user, model="sarvam-105b"):
    msgs = [{"role": "system", "content": system + " Reply with ONLY one valid JSON object, no markdown fences."},
            {"role": "user", "content": user}]
    for _ in range(2):
        text = llm(msgs, model)
        obj = _parse_json(text)
        if obj is not None:
            return obj
        msgs += [{"role": "assistant", "content": text[:2000]},
                 {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object."}]
    raise RuntimeError("Sarvam-105B did not return valid JSON. Please try again.")


def _chunks(text, n):
    parts, cur = [], ""
    for para in re.split(r"\n+", text):
        while len(para) > n:
            parts.append(para[:n])
            para = para[n:]
        if len(cur) + len(para) + 1 > n and cur:
            parts.append(cur)
            cur = para
        else:
            cur = f"{cur}\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return parts


def translate(text, target):
    out = []
    for chunk in _chunks(text, 1800):
        r = requests.post(
            f"{BASE}/translate", headers=_headers(True),
            json={"input": chunk, "source_language_code": "en-IN",
                  "target_language_code": target, "model": "sarvam-translate:v1"},
            timeout=60,
        )
        out.append(_check(r, "Sarvam Translate").get("translated_text", chunk))
    return "\n".join(out)


SKIP_KEYS = {"tag", "where", "verdict", "id", "type", "mermaid", "subject", "topic", "segment", "level", "score"}


def tl(obj, key=None):
    """Translate text (or every text field of a dict/list) into an extra language via Sarvam Translate."""
    if not NEEDS_TRANSLATE or key in SKIP_KEYS:
        return obj
    if isinstance(obj, str) and obj.strip():
        try:
            return translate(obj, LANG)
        except Exception as e:
            st.warning(f"Translation skipped: {e}")
            return obj
    if isinstance(obj, list):
        return [tl(x, key) for x in obj]
    if isinstance(obj, dict):
        return {k: tl(v, k) for k, v in obj.items()}
    return obj


def tts(text, speaker="ritu"):
    """Bulbul v3 voice. Returns a list of WAV clips (long text is split under the 2,500-char limit)."""
    if NEEDS_TRANSLATE or not text:
        return []
    clean = re.sub(r"[#*_`>|]", "", text)
    clips = []
    for chunk in _chunks(clean, 1500)[:3]:
        body = {"text": chunk, "language_code": LANG, "model": "bulbul:v3", "speaker": speaker}
        r = requests.post(f"{BASE}/text-to-speech", headers=_headers(True), json=body, timeout=90)
        if r.status_code in (400, 422):  # older API field name
            body["target_language_code"] = body.pop("language_code")
            r = requests.post(f"{BASE}/text-to-speech", headers=_headers(True), json=body, timeout=90)
        clips += [base64.b64decode(a) for a in _check(r, "Bulbul text-to-speech").get("audios", [])]
    return clips


def play(holder, flag="played"):
    clips = holder.get("audio") or []
    for i, clip in enumerate(clips):
        st.audio(clip, format="audio/wav", autoplay=(i == 0 and not holder.get(flag)))
    holder[flag] = True


def vision_ocr(img_bytes, filename, lang_code):
    """Sarvam Vision (Document AI digitise job) -> markdown text of the board / notes photo."""
    from sarvamai import SarvamAI
    client = SarvamAI(api_subscription_key=api_key)
    mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    job = client.doc_ai.digitise(file=[(filename, io.BytesIO(img_bytes), mime)],
                                 language=lang_code, output_format="md")
    terminal = {"completed", "partially_completed", "failed", "rejected"}
    status = None
    for _ in range(40):
        status = client.doc_ai.get_status(job_id=job.job_id)
        if status.status.lower() in terminal:
            break
        time.sleep(3)
    if status is None or status.status.lower() not in ("completed", "partially_completed"):
        raise RuntimeError(f"Sarvam Vision could not read {filename}")
    dl = client.doc_ai.get_download_url(job_id=job.job_id)
    z = zipfile.ZipFile(io.BytesIO(requests.get(dl.url, timeout=60).content))
    return "\n".join(z.read(n).decode("utf-8", "ignore") for n in z.namelist() if n.endswith(".md"))


def render_mermaid(code, height=480):
    code = re.sub(r"```(?:mermaid)?", "", code or "").strip()
    if not code:
        return
    js_code = json.dumps(code).replace("</", "<\\/")
    components.html(
        f"""<div id="out" style="font-family:sans-serif"></div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>
mermaid.initialize({{startOnLoad:false, theme:'neutral'}});
mermaid.render('g1', {js_code})
  .then(r => document.getElementById('out').innerHTML = r.svg)
  .catch(e => document.getElementById('out').innerText = 'The diagram could not be drawn. See the code below.');
</script>""",
        height=height, scrolling=True,
    )
    with st.expander("Diagram code"):
        st.code(code, language="text")


# ----------------------------------------------------------------------------- progress store
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"concepts": {}, "events": [], "sessions": []}


if "progress" not in st.session_state:
    st.session_state.progress = load_progress()
P = st.session_state.progress


def save_progress():
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(P, f, ensure_ascii=False, indent=2)


def norm_tag(t):
    t = re.sub(r"[^a-z0-9/\-]+", "-", (t or "").strip().lower())
    return re.sub(r"-+", "-", t).strip("-/")


def _new_concept(name, subject="", topic=""):
    return {"name": name, "subject": subject, "topic": topic, "mastery": 40, "tested": False,
            "attempts": 0, "weak_notes": [], "strengths": [], "updated": now(), "last_mode": ""}


def register_concepts(concepts, subject, topic):
    tags = []
    for c in concepts or []:
        tag = norm_tag(c.get("tag", ""))
        if tag:
            P["concepts"].setdefault(tag, _new_concept(c.get("name", tag), subject, topic))
            tags.append(tag)
    save_progress()
    return tags


def update_concept(tag, delta, mode, weak_note=None, good_note=None, name=None):
    tag = norm_tag(tag)
    if not tag:
        return
    c = P["concepts"].setdefault(tag, _new_concept(name or tag.split("/")[-1].replace("-", " ")))
    c["mastery"] = max(0, min(100, c["mastery"] + delta))
    c.update(tested=True, attempts=c["attempts"] + 1, updated=now(), last_mode=mode)
    if weak_note:
        c["weak_notes"] = (c["weak_notes"] + [weak_note])[-5:]
    if good_note:
        c["strengths"] = (c.get("strengths", []) + [good_note])[-5:]
    tested = [x["mastery"] for x in P["concepts"].values() if x["tested"]]
    P["events"].append({"time": now(), "mode": mode, "concept": tag, "delta": delta,
                        "avg": round(sum(tested) / len(tested), 1)})
    save_progress()


def log_session(mode, topic, summary):
    P["sessions"].append({"time": now(), "mode": mode, "topic": topic, "summary": summary})
    P["sessions"] = P["sessions"][-30:]
    save_progress()


def known_tags(limit=80):
    items = list(P["concepts"].items())[-limit:]
    return "\n".join(f"- {t}: {c['name']}" for t, c in items) or "none yet"


def weak_concepts(limit=12):
    items = [(t, c) for t, c in P["concepts"].items() if c["tested"] and (c["mastery"] < 60 or c["weak_notes"])]
    items.sort(key=lambda x: x[1]["mastery"])
    if not items:
        return "none yet"
    return "\n".join(
        f"- {t} ({c['name']}): mastery {c['mastery']}/100; past mistakes: {'; '.join(c['weak_notes'][-2:]) or 'n/a'}"
        for t, c in items[:limit]
    )


def related_weak(tags):
    prefixes = {"/".join(norm_tag(t).split("/")[:2]) for t in tags}
    return [(t, c) for t, c in P["concepts"].items()
            if c["tested"] and c["mastery"] < 60 and "/".join(t.split("/")[:2]) in prefixes]


def show_connection(tags):
    rel = related_weak(tags)
    if rel:
        names = ", ".join(f"{c['name']} ({c['mastery']}%)" for _, c in rel[:4])
        st.info(f"Connected to your past sessions. Extra focus on: {names}")


# ----------------------------------------------------------------------------- coach feedback
COACH_SCHEMA = """{
  "summary": "2-3 sentences spoken directly to the student",
  "strengths": [{"point": "...", "evidence": "what they actually did that shows it"}],
  "weak_points": [{"concept": "...", "what_went_wrong": "...", "why_it_matters": "..."}],
  "how_to_improve": [{"action": "specific, doable action", "where": "Samjho | Samjhao | Parkho | Self-study"}],
  "next_step": "the single most useful thing to do next"
}"""


def coach(context):
    fb = llm_json(
        "You are a supportive but honest learning coach giving feedback directly to a student. "
        "Be specific: name the exact concepts and mistakes. Never be vague.",
        f"{context}\n\n{RULES}\n\nReturn JSON in exactly this shape:\n{COACH_SCHEMA}",
    )
    return tl(fb)


def render_feedback(fb, holder):
    st.markdown("#### Your feedback")
    st.write(fb.get("summary", ""))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**What you're good at**")
        for s in fb.get("strengths", []):
            st.markdown(f"- {s.get('point', '')}  \n  _{s.get('evidence', '')}_")
    with c2:
        st.markdown("**Where to improve**")
        for w in fb.get("weak_points", []):
            st.markdown(f"- **{w.get('concept', '')}**: {w.get('what_went_wrong', '')}  \n  _{w.get('why_it_matters', '')}_")
    st.markdown("**How to get better**")
    for h in fb.get("how_to_improve", []):
        st.markdown(f"- {h.get('action', '')} ({h.get('where', '')})")
    if fb.get("next_step"):
        st.success(f"Next step: {fb['next_step']}")
    play(holder, "fb_played")


# ----------------------------------------------------------------------------- page
st.markdown('<div class="hero"><h1>📚 Saathi</h1><p>Samjho. Samjhao. Parkho. Your study buddy, in your language.</p></div>',
            unsafe_allow_html=True)


def fx(kind=None, msg=None):
    st.session_state.fx = (kind, msg)


_fx = st.session_state.pop("fx", None)
if _fx:
    if _fx[1]:
        st.toast(_fx[1])
    if _fx[0] == "balloons":
        st.balloons()
    elif _fx[0] == "snow":
        st.snow()
tab1, tab2, tab3, tab4 = st.tabs(["Samjho: understand", "Samjhao: revise by teaching", "Parkho: test yourself", "Meri Pragati: my progress"])

# ============================================================================= SAMJHO
SAMJHO_SCHEMA = """{
  "title": "short title of the lecture",
  "subject": "e.g. physics",
  "topic": "e.g. seasons",
  "concepts": [{"tag": "subject/topic/concept", "name": "readable concept name"}],
  "simple_explanation": "markdown; short paragraphs and bullet points; 150-350 words",
  "key_points": ["..."],
  "lost_deep_dives": [{"segment": 0, "explanation": "step-by-step, extra simple explanation of that part"}],
  "watch_out": ["only if one of the student's past weak concepts relates to this lecture; otherwise leave empty"],
  "mermaid": "Mermaid 'flowchart TD' code showing how the ideas connect; 5-12 nodes; every node label in double quotes, e.g. A[\\"Sunlight\\"] --> B[\\"Heat\\"]"
}"""

with tab1:
    st.subheader("Samjho: understand your lecture")
    st.caption("Record or upload the lecture. Add a photo of the board if the professor wrote things down.")
    c1, c2 = st.columns(2)
    with c1:
        rec = st.audio_input("Record the lecture", key="sm_rec")
        up = st.file_uploader("Or upload lecture audio", type=["wav", "mp3", "m4a", "ogg", "webm", "aac", "flac"], key="sm_up")
    with c2:
        boards = st.file_uploader("Board or notes photos (optional)", type=["png", "jpg", "jpeg"],
                                  accept_multiple_files=True, key="sm_board")
        board_lang = st.selectbox("Language on the board", list(LANGS), index=0, key="sm_board_lang")

    if st.button("Transcribe lecture", type="primary", key="sm_go"):
        src = up or rec
        if not src and not boards:
            st.warning("Record or upload a lecture, or add a board photo.")
        else:
            S = {"id": str(time.time()), "segments": [], "board": "", "result": None, "audio": []}
            try:
                if src:
                    name = getattr(src, "name", None) or "lecture.wav"
                    chunks, ext = split_audio(src.getvalue(), name)
                    bar = st.progress(0.0, text="Saaras is transcribing the lecture")
                    for i, ch in enumerate(chunks):
                        S["segments"].append(stt(ch, f"chunk.{ext}"))
                        bar.progress((i + 1) / len(chunks), text=f"Transcribed part {i + 1} of {len(chunks)}")
                for b in boards or []:
                    with st.spinner(f"Sarvam Vision is reading {b.name}"):
                        S["board"] += vision_ocr(b.getvalue(), b.name, LANGS[board_lang]) + "\n"
                st.session_state.samjho = S
            except Exception as e:
                st.error(str(e))

    S = st.session_state.get("samjho")
    if S:
        if S["segments"]:
            st.markdown("**Tick the parts where you got lost.** Those parts get a deeper explanation.")
            for i, seg in enumerate(S["segments"]):
                a, b = st.columns([1, 9])
                a.checkbox("Lost", key=f"lost_{S['id']}_{i}")
                start = i * SEG_SECONDS
                b.markdown(f"`{start // 60}:{start % 60:02d}` {seg or '_(silence)_'}")
        if S["board"].strip():
            with st.expander("Text read from the board"):
                st.markdown(S["board"])

        if st.button("Explain it simply", type="primary", key="sm_explain"):
            lost = [i for i in range(len(S["segments"])) if st.session_state.get(f"lost_{S['id']}_{i}")]
            seg_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(S["segments"])) or "(no audio)"
            prompt = (
                f"LECTURE TRANSCRIPT (numbered parts):\n{seg_text}\n\n"
                f"BLACKBOARD / NOTES TEXT:\n{S['board'] or '(none)'}\n\n"
                f"PARTS WHERE THE STUDENT GOT LOST: {lost or 'none'}  (give a lost_deep_dives entry for each)\n\n"
                f"STUDENT'S PAST WEAK CONCEPTS (may be unrelated):\n{weak_concepts()}\n\n"
                f"EXISTING CONCEPT TAGS:\n{known_tags()}\n\n{RULES}\n\n"
                f"Return JSON in exactly this shape:\n{SAMJHO_SCHEMA}"
            )
            with st.spinner("Sarvam-105B is simplifying the lecture"):
                try:
                    R = llm_json("You are Samjho, a patient teacher who turns a confusing lecture into a simple, "
                                 "clear explanation for Indian students.", prompt)
                    R["tags"] = register_concepts(R.get("concepts"), R.get("subject", ""), R.get("topic", ""))
                    R = tl(R)
                    S["result"], S["audio"], S["played"] = R, [], True
                    log_session("Samjho", R.get("topic", ""), R.get("title", ""))
                    fx(None, "✨ Lecture decoded! Scroll down.")
                except Exception as e:
                    st.error(str(e))

        R = S.get("result")
        if R:
            st.markdown(f"### {R.get('title', '')}")
            show_connection(R.get("tags", []))
            left, right = st.columns([3, 2])
            with left:
                st.markdown(R.get("simple_explanation", ""))
                if R.get("key_points"):
                    st.markdown("**Remember these**")
                    for k in R["key_points"]:
                        st.markdown(f"- {k}")
                for w in R.get("watch_out", []):
                    if w:
                        st.warning(w)
                for d in R.get("lost_deep_dives", []):
                    with st.expander(f"Where you got lost: part {d.get('segment')}", expanded=True):
                        st.markdown(d.get("explanation", ""))
            with right:
                render_mermaid(R.get("mermaid", ""))
            if not NEEDS_TRANSLATE and st.button("Listen to the explanation", key="sm_listen"):
                with st.spinner("Bulbul is speaking"):
                    S["audio"], S["played"] = tts(R.get("simple_explanation", ""), "ritu"), False
            play(S)

# ============================================================================= SAMJHAO
SJ_SETUP_SCHEMA = """{
  "subject": "...", "topic": "...",
  "concepts": [{"tag": "subject/topic/concept", "name": "..."}],
  "key_points": [{"id": "k1", "point": "an essential idea a good explanation must cover", "tag": "subject/topic/concept"}],
  "misconception": {"belief": "one realistic wrong belief students hold about this topic", "tag": "subject/topic/concept", "why_wrong": "..."},
  "opening_line": "the confused classmate's first message: casually asks the student to explain the topic and reveals the wrong belief naturally (1-3 sentences)",
  "curveballs": ["3 tricky, deep questions a topper would ask"]
}"""
SJ_JUDGE_SCHEMA = """{
  "newly_covered": ["ids of key points the latest explanation covered correctly"],
  "misconception_fixed": false,
  "errors": [{"tag": "subject/topic/concept", "student_said": "short quote", "correction": "the correct idea, explained clearly"}],
  "concept_judgements": [{"tag": "subject/topic/concept", "verdict": "good | partial | wrong", "note": "short reason"}],
  "next_speaker": "confused | topper",
  "probe": "what the next question should push on"
}"""
PERSONAS = {
    "student": ("You", "🧑‍🎓"), "confused": ("Chintu", "🤔"), "topper": ("Priya", "🎯"), "guru": ("Correction", "🧑‍🏫"),
}
VOICE = {"confused": "rohan", "topper": "priya"}


def sj_convo(msgs, last=10):
    return "\n".join(f"{PERSONAS[m['who']][0]}: {m['text']}" for m in msgs[-last:])


def sj_reply(SJ, who, probe):
    setup = SJ["setup"]
    if who == "confused":
        mc = setup.get("misconception", {})
        stance = ("The student has convinced you your belief was wrong: admit it happily, then ask one simple follow-up."
                  if SJ["fixed"] else
                  f"You believe: '{mc.get('belief', '')}'. Keep this belief unless the student clearly explains "
                  "why it is wrong. Ask naive 'why?' questions.")
        system = (f"You are Chintu, a friendly but confused classmate. Speak casually in {OUT_LANG} "
                  f"(English technical words are fine). {stance} Never lecture. 1-3 short sentences.")
    else:
        system = (f"You are Priya, the class topper: sharp and curious. Speak in {OUT_LANG}. Ask ONE tricky, deep "
                  f"question that tests real understanding. Ideas: {setup.get('curveballs', [])}. "
                  "Never give the answer. 1-2 sentences.")
    user = f"Topic: {setup.get('topic', '')}\nConversation so far:\n{sj_convo(SJ['msgs'])}\n\nFocus on: {probe}\nWrite only your next message."
    return llm([{"role": "system", "content": system}, {"role": "user", "content": user}],
               model="sarvam-105b-conversations", max_tokens=1500, temperature=0.7)


with tab2:
    st.subheader("Samjhao: revise by teaching")
    st.caption("Pick any topic and explain it out loud. Chintu gets confused easily, Priya asks the tough "
               "questions, and you're corrected whenever you say something wrong.")
    SJ = st.session_state.get("sj")

    if not SJ:
        topic = st.text_input("What do you want to revise?", placeholder="e.g. Why do we have seasons?", key="sj_topic")
        if st.button("Start revising", type="primary", key="sj_start") and topic.strip():
            with st.spinner("Setting up your classmates"):
                try:
                    setup = llm_json(
                        "You design a teach-back revision session where a student explains a topic to AI classmates.",
                        f"TOPIC THE STUDENT WANTS TO REVISE: {topic}\n\n"
                        f"STUDENT'S PAST WEAK CONCEPTS (use only if related; if one relates, base the misconception on it):\n{weak_concepts()}\n\n"
                        f"EXISTING CONCEPT TAGS:\n{known_tags()}\n\n{RULES}\n"
                        "Give 4-6 key points with ids k1, k2, ...\n\n"
                        f"Return JSON in exactly this shape:\n{SJ_SETUP_SCHEMA}",
                    )
                    setup["tags"] = register_concepts(setup.get("concepts"), setup.get("subject", ""), setup.get("topic", ""))
                    opening = tl(setup.get("opening_line", "Can you explain this to me?"))
                    SJ = {"setup": setup, "msgs": [{"who": "confused", "text": opening}], "covered": [],
                          "fixed": False, "turn": 0, "judgements": [], "errors": [],
                          "audio": tts(opening, VOICE["confused"]) if voice_on else [], "played": False}
                    st.session_state.sj = SJ
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    else:
        setup = SJ["setup"]
        kps = setup.get("key_points", [])
        meter = round(70 * len(SJ["covered"]) / max(1, len(kps)) + (30 if SJ["fixed"] else 0))
        st.markdown(f"**Revising:** {setup.get('topic', '')}")
        st.progress(meter / 100, text=f"How well your classmates understand: {meter}%")
        show_connection(setup.get("tags", []))

        for m in SJ["msgs"]:
            name, avatar = PERSONAS[m["who"]]
            with st.chat_message(name, avatar=avatar):
                st.markdown(f"**{name}:** {m['text']}")
        play(SJ)

        if not SJ.get("feedback"):
            t = SJ["turn"]
            aud = st.audio_input("Explain out loud", key=f"sj_audio_{t}")
            txt = st.text_area("Or type your explanation", key=f"sj_text_{t}", height=100)
            b1, b2 = st.columns(2)
            if b1.button("Send", type="primary", key=f"sj_send_{t}"):
                try:
                    answer = stt(aud.getvalue(), "answer.wav") if aud else txt.strip()
                    if not answer:
                        st.warning("Say or type your explanation first.")
                        st.stop()
                    with st.spinner("Your classmates are thinking"):
                        J = llm_json(
                            "You are a strict but kind examiner silently judging a student's explanation in a "
                            "teach-back session.",
                            f"TOPIC: {setup.get('topic', '')}\nKEY POINTS: {json.dumps(kps, ensure_ascii=False)}\n"
                            f"CLASSMATE'S HIDDEN MISCONCEPTION: {setup.get('misconception', {})}\n"
                            f"ALREADY COVERED: {SJ['covered']}\nMISCONCEPTION ALREADY FIXED: {SJ['fixed']}\n\n"
                            f"CONVERSATION SO FAR:\n{sj_convo(SJ['msgs'])}\n\nSTUDENT'S LATEST EXPLANATION:\n{answer}\n\n"
                            "Judge only the latest explanation. misconception_fixed is true only if the student "
                            "clearly explained why the belief is wrong. Choose next_speaker 'confused' while the "
                            "misconception is not fixed or basics are missing; 'topper' when basics are solid. "
                            f"Write corrections in {OUT_LANG}.\n\nReturn JSON in exactly this shape:\n{SJ_JUDGE_SCHEMA}",
                        )
                        SJ["msgs"].append({"who": "student", "text": answer})
                        SJ["covered"] = sorted(set(SJ["covered"]) | set(J.get("newly_covered", [])))
                        was_fixed = SJ["fixed"]
                        SJ["fixed"] = SJ["fixed"] or bool(J.get("misconception_fixed"))
                        if SJ["fixed"] and not was_fixed:
                            fx("balloons", "🎉 You cleared Chintu's confusion!")
                        elif J.get("newly_covered"):
                            fx(None, f"🧠 +{len(J['newly_covered'])} key idea(s) explained!")
                        elif J.get("errors"):
                            fx(None, "🧑‍🏫 Small slip, check the correction.")
                        for j in J.get("concept_judgements", []):
                            v = j.get("verdict", "partial")
                            update_concept(j.get("tag", ""), {"good": 8, "partial": 3, "wrong": -8}.get(v, 0), "Samjhao",
                                           weak_note=j.get("note") if v != "good" else None,
                                           good_note=j.get("note") if v == "good" else None)
                        SJ["judgements"] += J.get("concept_judgements", [])
                        errors = tl(J.get("errors", []))
                        SJ["errors"] += errors
                        if errors:
                            SJ["msgs"].append({"who": "guru", "text": "\n".join(f"- {e.get('correction', '')}" for e in errors)})
                        who = "topper" if J.get("next_speaker") == "topper" else "confused"
                        reply = tl(sj_reply(SJ, who, J.get("probe", "")))
                        SJ["msgs"].append({"who": who, "text": reply})
                        SJ["audio"], SJ["played"] = (tts(reply, VOICE[who]) if voice_on else []), False
                        SJ["turn"] += 1
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
            if b2.button("Finish and get feedback", key=f"sj_end_{t}"):
                with st.spinner("Preparing your feedback"):
                    try:
                        missing = [k["point"] for k in kps if k.get("id") not in SJ["covered"]]
                        ctx = (f"MODE: Samjhao (the student explained '{setup.get('topic', '')}' to AI classmates).\n"
                               f"Key points covered: {len(SJ['covered'])}/{len(kps)}. Missing: {missing}\n"
                               f"Classmate's misconception '{setup.get('misconception', {}).get('belief', '')}' fixed: {SJ['fixed']}\n"
                               f"Student's errors: {json.dumps(SJ['errors'], ensure_ascii=False)}\n"
                               f"Concept judgements: {json.dumps(SJ['judgements'], ensure_ascii=False)}\n"
                               f"Conversation:\n{sj_convo(SJ['msgs'], 14)}")
                        SJ["feedback"] = coach(ctx)
                        SJ["fb_audio"] = tts(SJ["feedback"].get("summary", ""), "ritu") if voice_on else []
                        log_session("Samjhao", setup.get("topic", ""), f"understanding {meter}%, errors: {len(SJ['errors'])}")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        if SJ.get("feedback"):
            render_feedback(SJ["feedback"], {"audio": SJ.get("fb_audio"), "fb_played": SJ.get("fb_played")})
            SJ["fb_played"] = True
        if st.button("Revise a new topic", key="sj_reset"):
            st.session_state.sj = None
            st.rerun()

# ============================================================================= PARKHO
Q_SCHEMA = '{"text": "the question", "type": "Conceptual | Numerical | Logical reasoning | Application", "level": 1, "tag": "subject/topic/concept", "expected_answer": "model answer / final value with units", "hint": "a small nudge"}'
PK_START_SCHEMA = '{"subject": "...", "topic": "...", "concepts": [{"tag": "subject/topic/concept", "name": "..."}], "question": ' + Q_SCHEMA + "}"
PK_EVAL_SCHEMA = ('{"evaluation": {"verdict": "correct | partial | wrong", "score": 0.0, "feedback": "what was right or wrong in their answer", '
                  '"correct_solution": "the right answer with short working", "misconception": "the underlying mistake, or empty", '
                  '"tag": "subject/topic/concept"}, "next_question": ' + Q_SCHEMA + " or null}")
LEVEL_GUIDE = ("Levels: 1 easy recall, 2 basic understanding, 3 application, 4 multi-step / analytical, "
               "5 very tough (interview-level, tricky edge cases).")

with tab3:
    st.subheader("Parkho: test yourself")
    st.caption("Questions get harder when you answer well and easier when you struggle. You get a full report at the end.")
    PK = st.session_state.get("pk")

    if not PK:
        topic = st.text_input("Topic to be tested on", placeholder="e.g. Projectile motion", key="pk_topic")
        c1, c2 = st.columns(2)
        types = c1.multiselect("Question types", ["Conceptual", "Numerical", "Logical reasoning", "Application"],
                               default=["Conceptual", "Numerical", "Logical reasoning"], key="pk_types")
        persona = c2.radio("Who asks the questions?", ["Teacher", "Interviewer"], horizontal=True, key="pk_persona")
        n_q = c1.slider("Number of questions", 3, 10, 5, key="pk_n")
        start = c2.select_slider("Start at", options=list(LEVELS), value=1, format_func=lambda x: LEVELS[x], key="pk_start")
        if st.button("Start the test", type="primary", key="pk_go") and topic.strip() and types:
            with st.spinner("Preparing your first question"):
                try:
                    setup = llm_json(
                        f"You are an adaptive examiner acting as a {persona.lower()}.",
                        f"TOPIC: {topic}\nALLOWED QUESTION TYPES: {types}\nFIRST QUESTION LEVEL: {start}\n{LEVEL_GUIDE}\n"
                        f"STUDENT'S PAST WEAK CONCEPTS (if related, test them):\n{weak_concepts()}\n\n"
                        f"EXISTING CONCEPT TAGS:\n{known_tags()}\n\n{RULES}\n"
                        "Numerical questions must give all values and ask for a final answer with units.\n\n"
                        f"Return JSON in exactly this shape:\n{PK_START_SCHEMA}",
                    )
                    setup["tags"] = register_concepts(setup.get("concepts"), setup.get("subject", ""), setup.get("topic", ""))
                    q = setup.get("question", {})
                    q["level"] = int(q.get("level") or start)
                    q = tl(q)
                    st.session_state.pk = {"setup": setup, "types": types, "persona": persona, "n": n_q, "q": q,
                                           "records": [], "done": False, "audio": tts(q.get("text", ""), "aditya") if voice_on else [],
                                           "played": False}
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    else:
        setup, recs = PK["setup"], PK["records"]
        st.markdown(f"**Topic:** {setup.get('topic', '')}  |  **Question {min(len(recs) + 1, PK['n'])} of {PK['n']}**")
        st.progress(len(recs) / PK["n"])
        show_connection(setup.get("tags", []))

        for i, r in enumerate(recs):
            icon = {"correct": "✅", "partial": "🟡", "wrong": "❌"}.get(r["verdict"], "•")
            with st.expander(f"{icon} Q{i + 1} ({r['type']}, {LEVELS.get(r['level'], r['level'])}): {r['question'][:80]}"):
                st.markdown(f"**Your answer:** {r['answer']}")
                st.markdown(f"**Feedback:** {r['feedback']}")
                st.markdown(f"**Correct solution:** {r['solution']}")

        if not PK["done"]:
            q = PK["q"]
            st.markdown(f"<span class='chip'>{q.get('type', '')}</span><span class='chip'>Level {q.get('level')}: {LEVELS.get(q.get('level'), '')}</span>", unsafe_allow_html=True)
            st.markdown(q.get("text", ""))
            with st.expander("Need a hint?"):
                st.write(q.get("hint", ""))
            play(PK)
            k = len(recs)
            aud = st.audio_input("Answer out loud", key=f"pk_audio_{k}")
            txt = st.text_area("Or type your answer (best for numericals)", key=f"pk_text_{k}", height=110)
            b1, b2, b3 = st.columns(3)
            submit = b1.button("Submit answer", type="primary", key=f"pk_sub_{k}")
            skip = b2.button("I don't know", key=f"pk_skip_{k}")
            stop = b3.button("End test now", key=f"pk_stop_{k}")
            if submit or skip:
                try:
                    answer = "(student skipped: does not know)" if skip else (stt(aud.getvalue(), "answer.wav") if aud else txt.strip())
                    if not answer:
                        st.warning("Say or type your answer first.")
                        st.stop()
                    is_last = len(recs) + 1 >= PK["n"]
                    history = "\n".join(f"- L{r['level']} {r['type']}: {r['question'][:100]} -> {r['verdict']}" for r in recs) or "none"
                    with st.spinner("Checking your answer"):
                        E = llm_json(
                            f"You are an adaptive examiner acting as a {PK['persona'].lower()}. Grade fairly: accept "
                            "correct answers in any language or wording; give partial credit for right method with small slips.",
                            f"TOPIC: {setup.get('topic', '')}\nALLOWED TYPES: {PK['types']}\n{LEVEL_GUIDE}\n"
                            f"PREVIOUS QUESTIONS:\n{history}\n\nCURRENT QUESTION: {json.dumps(q, ensure_ascii=False)}\n"
                            f"STUDENT'S ANSWER: {answer}\n\n"
                            + ("This was the LAST question: next_question must be null.\n" if is_last else
                               "Next question level: +1 if correct, same if partial, -1 if wrong (between 1 and 5). "
                               "Vary the type, never repeat a question, and prefer concepts the student got wrong.\n")
                            + f"STUDENT'S PAST WEAK CONCEPTS:\n{weak_concepts()}\n\n{RULES}\n\n"
                            f"Return JSON in exactly this shape:\n{PK_EVAL_SCHEMA}",
                        )
                    ev = E.get("evaluation", {})
                    verdict = ev.get("verdict", "wrong") if not skip else "wrong"
                    lvl = int(q.get("level") or 1)
                    delta = {"correct": 6 + 2 * lvl, "partial": 2}.get(verdict, -10 + lvl)
                    update_concept(ev.get("tag") or q.get("tag", ""), delta, "Parkho",
                                   weak_note=(ev.get("misconception") or f"missed a level-{lvl} {q.get('type', '')} question") if verdict != "correct" else None,
                                   good_note=f"solved a level-{lvl} {q.get('type', '')} question" if verdict == "correct" else None)
                    ev = tl(ev)
                    recs.append({"question": q.get("text", ""), "type": q.get("type", ""), "level": lvl,
                                 "tag": ev.get("tag") or q.get("tag", ""), "answer": answer, "verdict": verdict,
                                 "score": float(ev.get("score") or {"correct": 1, "partial": 0.5}.get(verdict, 0)),
                                 "feedback": ev.get("feedback", ""), "solution": ev.get("correct_solution", ""),
                                 "misconception": ev.get("misconception", "")})
                    streak = 0
                    for r in reversed(recs):
                        if r["verdict"] != "correct":
                            break
                        streak += 1
                    fx(None, f"🔥 Streak x{streak}! Level up." if streak >= 2 else "✅ Correct!" if verdict == "correct" else "🟡 Almost there!" if verdict == "partial" else "💪 Learn it and bounce back.")
                    nq = E.get("next_question")
                    if is_last or not nq:
                        PK["done"] = True
                    else:
                        nq["level"] = max(1, min(5, int(nq.get("level") or lvl)))
                        PK["q"] = tl(nq)
                        PK["audio"], PK["played"] = (tts(PK["q"].get("text", ""), "aditya") if voice_on else []), False
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
            if stop:
                PK["done"] = True
                st.rerun()

        if PK["done"] and recs:
            st.markdown("### Your test report")
            df = pd.DataFrame(recs)
            correct_lvls = df.loc[df.verdict == "correct", "level"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Score", f"{round(100 * df.score.mean())}%")
            m2.metric("Correct answers", f"{(df.verdict == 'correct').sum()} of {len(df)}")
            m3.metric("Hardest level solved", LEVELS.get(int(correct_lvls.max()), "None") if len(correct_lvls) else "None")
            by_type = df.groupby("type").score.mean().mul(100).round().astype(int).rename("Score %")
            by_tag = df.groupby("tag").score.mean().mul(100).round().astype(int).rename("Score %")
            c1, c2 = st.columns(2)
            c1.markdown("**By question type**")
            c1.dataframe(by_type, use_container_width=True)
            c2.markdown("**By concept**")
            c2.dataframe(by_tag, use_container_width=True)
            if not PK.get("feedback"):
                with st.spinner("Writing your feedback"):
                    try:
                        ctx = (f"MODE: Parkho test on '{setup.get('topic', '')}'.\n"
                               + "\n".join(f"- Q (level {r['level']}, {r['type']}, {r['tag']}): {r['question']}\n"
                                           f"  answer: {r['answer']}\n  verdict: {r['verdict']}; mistake: {r['misconception']}"
                                           for r in recs))
                        PK["feedback"] = coach(ctx)
                        PK["fb_audio"] = tts(PK["feedback"].get("summary", ""), "ritu") if voice_on else []
                        log_session("Parkho", setup.get("topic", ""), f"score {round(100 * df.score.mean())}%")
                        if df.score.mean() >= 0.7:
                            st.balloons()
                        else:
                            st.snow()
                    except Exception as e:
                        st.error(str(e))
            if PK.get("feedback"):
                render_feedback(PK["feedback"], {"audio": PK.get("fb_audio"), "fb_played": PK.get("fb_played")})
                PK["fb_played"] = True
        if st.button("Take a new test", key="pk_reset"):
            st.session_state.pk = None
            st.rerun()

# ============================================================================= MERI PRAGATI
with tab4:
    st.subheader("Meri Pragati: my progress")
    concepts = P["concepts"]
    if not concepts:
        st.info("Your progress appears here after your first session in any mode.")
    else:
        tested = {t: c for t, c in concepts.items() if c["tested"]}
        weak = sorted([(t, c) for t, c in tested.items() if c["mastery"] < 60], key=lambda x: x[1]["mastery"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Concepts practised", len(tested))
        m2.metric("Average mastery", f"{round(sum(c['mastery'] for c in tested.values()) / len(tested))}%" if tested else "Not tested yet")
        m3.metric("Need more work", len(weak))

        rows = [{"Concept": c["name"], "Subject": c["subject"], "Topic": c["topic"],
                 "Mastery": c["mastery"] if c["tested"] else None,
                 "Status": ("Strong" if c["mastery"] >= 75 else "Getting there" if c["mastery"] >= 60 else "Needs work")
                 if c["tested"] else "Not tested yet",
                 "Attempts": c["attempts"], "Last practised": c["updated"]} for c in concepts.values()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"Mastery": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d")})

        if len(P["events"]) > 1:
            st.markdown("**Average mastery over time**")
            st.line_chart(pd.DataFrame(P["events"])["avg"])

        if weak:
            st.markdown("**Your weak points**")
            for t, c in weak[:8]:
                notes = "; ".join(c["weak_notes"][-2:]) or "not enough correct answers yet"
                st.markdown(f"- **{c['name']}** ({c['mastery']}%): {notes}")

        if st.button("Get my personal feedback", type="primary", key="pg_fb"):
            with st.spinner("Looking at all your sessions"):
                try:
                    ctx = ("MODE: overall progress across all sessions.\nCONCEPTS:\n"
                           + "\n".join(f"- {c['name']} ({t}): mastery {c['mastery']}, tested {c['tested']}, "
                                       f"attempts {c['attempts']}, mistakes: {c['weak_notes'][-3:]}, strengths: {c.get('strengths', [])[-2:]}"
                                       for t, c in concepts.items())
                           + "\nRECENT SESSIONS:\n" + "\n".join(f"- {s['time']} {s['mode']}: {s['topic']} ({s['summary']})" for s in P["sessions"][-10:]))
                    fb = coach(ctx)
                    st.session_state.pg = {"feedback": fb, "audio": tts(fb.get("summary", ""), "ritu") if voice_on else []}
                except Exception as e:
                    st.error(str(e))
        PG = st.session_state.get("pg")
        if PG:
            render_feedback(PG["feedback"], PG)

    with st.expander("Reset my progress"):
        if st.button("Delete all progress", key="pg_reset"):
            st.session_state.progress = {"concepts": {}, "events": [], "sessions": []}
            P = st.session_state.progress
            save_progress()
            for k in ("samjho", "sj", "pk", "pg"):
                st.session_state.pop(k, None)
            st.rerun()
