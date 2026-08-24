"""
AI Language Translation Tool
-----------------------------
A modern, portfolio-worthy language translator built with Streamlit.

Author : Fatima
Project: CodeAlpha Artificial Intelligence Internship - Task 1
Stack  : Python, Streamlit, deep-translator (Google Translate engine),
         gTTS (Text-to-Speech), pyperclip (clipboard)

This file is intentionally organized into small, readable functions so
that anyone reading the code (or reviewing it for the internship) can
follow the logic step by step.
"""

import base64
import io
from datetime import datetime

import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
from gtts.lang import tts_langs

# pyperclip needs a system clipboard (xclip/xsel) to work locally.
# On some cloud environments (like Streamlit Cloud) it can fail silently,
# so we wrap the import and every call in a try/except and fall back
# to a "click to copy" HTML/JS button instead. See copy_to_clipboard_button().
try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except Exception:
    PYPERCLIP_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Language Translation Tool",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    """Set up every piece of state the app needs to remember between reruns."""
    defaults = {
        "theme": "dark",                # "dark" or "light"
        "history": [],                  # list of dicts -> last 10 translations
        "favourites": [],                # list of dicts -> starred translations
        "translated_text": "",
        "source_text": "",
        "src_lang_name": "auto",
        "tgt_lang_name": "urdu",
        "last_translation_time": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================
# LANGUAGE DATA
# ============================================================
def get_supported_languages():
    """
    Returns a dict of {language_name: language_code} using deep-translator's
    own Google Translate language list, so we automatically support every
    language Google Translate supports (100+) without hardcoding a list
    that could go stale.
    """
    langs = GoogleTranslator().get_supported_languages(as_dict=True)
    return langs


LANGUAGES = get_supported_languages()          # e.g. {"english": "en", ...}
LANGUAGE_NAMES = sorted(LANGUAGES.keys())      # alphabetical dropdown list
TTS_SUPPORTED_CODES = set(tts_langs().keys())  # gTTS doesn't support every language


# ============================================================
# CORE LOGIC FUNCTIONS
# ============================================================
def translate_text(text: str, source: str, target: str):
    """
    Translate `text` from `source` language to `target` language.
    Returns (translated_text, error_message). One of them will be None.
    """
    if not text or not text.strip():
        return None, "Please enter some text to translate."

    try:
        src_code = "auto" if source == "auto" else LANGUAGES[source]
        tgt_code = LANGUAGES[target]
        result = GoogleTranslator(source=src_code, target=tgt_code).translate(text)
        if not result:
            return None, "Translation returned empty. Please try again."
        return result, None
    except Exception as exc:
        # Friendly, human-readable error instead of a raw stack trace
        return None, f"Translation failed due to a network/API issue. ({exc})"


def text_to_speech_bytes(text: str, lang_code: str):
    """
    Converts text to speech using gTTS and returns raw audio bytes (mp3),
    or None + error message if it can't be generated.
    """
    if not text or not text.strip():
        return None, "No text available to read aloud."
    if lang_code not in TTS_SUPPORTED_CODES:
        return None, "Voice is not available for this language yet."
    try:
        tts = gTTS(text=text, lang=lang_code)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read(), None
    except Exception as exc:
        return None, f"Couldn't generate audio right now. ({exc})"


def add_to_history(src_text, tgt_text, src_lang, tgt_lang):
    """Push a new translation into history, keeping only the latest 10."""
    entry = {
        "source_text": src_text,
        "translated_text": tgt_text,
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "time": datetime.now().strftime("%d %b, %I:%M %p"),
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:10]


def is_favourited(entry):
    return any(
        f["source_text"] == entry["source_text"]
        and f["translated_text"] == entry["translated_text"]
        for f in st.session_state.favourites
    )


def copy_to_clipboard_button(text: str, key: str, label: str = "📋 Copy"):
    """
    Renders a copy button. Uses pyperclip when a real system clipboard is
    available (local machine); otherwise falls back to a tiny JS snippet
    that copies via the browser clipboard API - this keeps the feature
    working both locally and when deployed to the cloud.
    """
    if PYPERCLIP_AVAILABLE:
        if st.button(label, key=key):
            try:
                pyperclip.copy(text)
                st.toast("Copied to clipboard ✅")
            except Exception:
                st.toast("Couldn't access system clipboard.", icon="⚠️")
    else:
        safe_text = text.replace("`", "\\`").replace("\\", "\\\\")
        components_html = f"""
        <button onclick="navigator.clipboard.writeText(`{safe_text}`)"
            style="
                background: var(--accent-gradient);
                color: white; border: none; padding: 0.5rem 1rem;
                border-radius: 10px; cursor: pointer; font-weight: 600;
                box-shadow: 0 4px 14px rgba(108, 99, 255, 0.35);">
            {label}
        </button>
        """
        st.markdown(components_html, unsafe_allow_html=True)


# ============================================================
# CUSTOM CSS - GLASSMORPHISM + GRADIENT THEME
# ============================================================
def load_css(theme: str):
    """Injects handcrafted CSS. Two palettes: dark (default) and light."""

    if theme == "dark":
        bg_gradient = "linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)"
        card_bg = "rgba(255, 255, 255, 0.06)"
        card_border = "rgba(255, 255, 255, 0.14)"
        text_color = "#f5f5fa"
        subtext_color = "#c8c6d8"
        input_bg = "rgba(255, 255, 255, 0.07)"
    else:
        bg_gradient = "linear-gradient(135deg, #e0eafc 0%, #cfdef3 50%, #e8d9f3 100%)"
        card_bg = "rgba(255, 255, 255, 0.55)"
        card_border = "rgba(255, 255, 255, 0.7)"
        text_color = "#1f1c33"
        subtext_color = "#4a4660"
        input_bg = "rgba(255, 255, 255, 0.65)"

    st.markdown(
        f"""
        <style>
        :root {{
            --accent-gradient: linear-gradient(135deg, #6c63ff 0%, #8e54e9 50%, #4776e6 100%);
            --card-bg: {card_bg};
            --card-border: {card_border};
            --text-color: {text_color};
            --subtext-color: {subtext_color};
        }}

        /* Overall app background */
        .stApp {{
            background: {bg_gradient};
            background-attachment: fixed;
        }}

        html, body, [class*="css"] {{
            font-family: 'Segoe UI', 'Poppins', sans-serif;
            color: {text_color};
        }}

        /* Hide default streamlit chrome for a cleaner, custom look */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        /* Hero title */
        .app-title {{
            font-size: 2.6rem;
            font-weight: 800;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            letter-spacing: -0.5px;
        }}
        .app-subtitle {{
            color: {subtext_color};
            font-size: 1.05rem;
            margin-top: 0.2rem;
            margin-bottom: 1.6rem;
        }}

        /* Glassmorphism card */
        .glass-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.5rem 1.6rem;
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.18);
            margin-bottom: 1.2rem;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(31, 38, 135, 0.28);
        }}

        .section-label {{
            font-weight: 700;
            font-size: 0.95rem;
            color: {text_color};
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        /* Text areas & inputs */
        .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {{
            background: {input_bg} !important;
            border-radius: 14px !important;
            color: {text_color} !important;
            border: 1px solid {card_border} !important;
        }}

        /* Primary buttons */
        div.stButton > button {{
            background: var(--accent-gradient);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.55rem 1.4rem;
            font-weight: 600;
            box-shadow: 0 4px 14px rgba(108, 99, 255, 0.35);
            transition: all 0.2s ease-in-out;
        }}
        div.stButton > button:hover {{
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 22px rgba(108, 99, 255, 0.5);
            color: white;
        }}
        div.stButton > button:active {{
            transform: translateY(0px) scale(0.98);
        }}

        /* Character counter chip */
        .char-counter {{
            display: inline-block;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 0.15rem 0.8rem;
            font-size: 0.8rem;
            color: {subtext_color};
            margin-top: 0.3rem;
        }}

        /* History item */
        .history-item {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
            font-size: 0.9rem;
        }}
        .history-meta {{
            font-size: 0.75rem;
            color: {subtext_color};
        }}

        /* Footer */
        .app-footer {{
            text-align: center;
            padding: 1.6rem 0 0.8rem 0;
            color: {subtext_color};
            font-size: 0.85rem;
            border-top: 1px solid {card_border};
            margin-top: 2rem;
        }}
        .app-footer a {{
            color: #8e54e9;
            text-decoration: none;
            font-weight: 600;
        }}

        /* Badge */
        .badge {{
            display: inline-block;
            background: var(--accent-gradient);
            color: white;
            padding: 0.15rem 0.7rem;
            border-radius: 20px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.4px;
        }}

        /* Responsive tweaks for small screens */
        @media (max-width: 768px) {{
            .app-title {{ font-size: 2rem; }}
            .glass-card {{ padding: 1.1rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        theme_choice = st.toggle(
            "🌙 Dark Mode",
            value=(st.session_state.theme == "dark"),
            help="Switch between dark and light themes",
        )
        st.session_state.theme = "dark" if theme_choice else "light"

        st.markdown("---")
        st.markdown("### ⭐ Favourites")
        if not st.session_state.favourites:
            st.caption("No favourites yet. Star a translation to save it here.")
        else:
            for i, fav in enumerate(reversed(st.session_state.favourites)):
                with st.expander(f"{fav['source_lang']} → {fav['target_lang']}"):
                    st.write(f"**Original:** {fav['source_text']}")
                    st.write(f"**Translated:** {fav['translated_text']}")
                    if st.button("🗑️ Remove", key=f"remove_fav_{i}"):
                        st.session_state.favourites.remove(fav)
                        st.rerun()

        st.markdown("---")
        st.markdown("### 🕘 Recent History")
        if not st.session_state.history:
            st.caption("Your last 10 translations will show up here.")
        else:
            for i, item in enumerate(st.session_state.history):
                st.markdown(
                    f"""
                    <div class="history-item">
                        <b>{item['source_lang']} → {item['target_lang']}</b><br>
                        {item['translated_text'][:60]}{'...' if len(item['translated_text']) > 60 else ''}
                        <div class="history-meta">{item['time']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# MAIN APP LAYOUT
# ============================================================
def render_header():
    st.markdown('<div class="app-title">🌐 AI Language Translation Tool</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Translate text instantly across 100+ languages, '
        'listen to it out loud, and keep track of everything you translate. '
        '<span class="badge">CodeAlpha AI Internship</span></div>',
        unsafe_allow_html=True,
    )


def render_translator():
    col1, col2 = st.columns(2, gap="medium")

    # ---------- Language selectors + swap button ----------
    lang_col1, swap_col, lang_col2 = st.columns([5, 1, 5])
    with lang_col1:
        source_options = ["auto"] + LANGUAGE_NAMES
        default_src_index = source_options.index(st.session_state.src_lang_name) \
            if st.session_state.src_lang_name in source_options else 0
        src_lang = st.selectbox(
            "🔤 Source Language",
            options=source_options,
            index=default_src_index,
            format_func=lambda x: "🌍 Auto-Detect" if x == "auto" else x.title(),
        )

    with swap_col:
        st.markdown("<div style='margin-top: 1.9rem;'></div>", unsafe_allow_html=True)
        swap_clicked = st.button("⇄", help="Swap languages", key="swap_btn")

    with lang_col2:
        default_tgt_index = LANGUAGE_NAMES.index(st.session_state.tgt_lang_name) \
            if st.session_state.tgt_lang_name in LANGUAGE_NAMES else 0
        tgt_lang = st.selectbox(
            "🎯 Target Language",
            options=LANGUAGE_NAMES,
            index=default_tgt_index,
            format_func=lambda x: x.title(),
        )

    if swap_clicked and src_lang != "auto":
        st.session_state.src_lang_name, st.session_state.tgt_lang_name = tgt_lang, src_lang
        st.session_state.source_text, st.session_state.translated_text = (
            st.session_state.translated_text,
            st.session_state.source_text,
        )
        st.rerun()
    else:
        st.session_state.src_lang_name = src_lang
        st.session_state.tgt_lang_name = tgt_lang

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- Text areas ----------
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">✍️ Original Text</div>', unsafe_allow_html=True)
        source_text = st.text_area(
            "Original text input",
            value=st.session_state.source_text,
            height=200,
            placeholder="Type or paste text here...",
            label_visibility="collapsed",
            key="source_text_area",
        )
        st.markdown(
            f'<span class="char-counter">{len(source_text)} characters</span>',
            unsafe_allow_html=True,
        )

        btn_row1, btn_row2, btn_row3 = st.columns(3)
        with btn_row1:
            translate_clicked = st.button("🚀 Translate", use_container_width=True)
        with btn_row2:
            if st.button("🧹 Clear", use_container_width=True):
                st.session_state.source_text = ""
                st.session_state.translated_text = ""
                st.rerun()
        with btn_row3:
            if st.button("🔊 Listen", use_container_width=True, key="listen_source"):
                src_code = "en" if src_lang == "auto" else LANGUAGES[src_lang]
                audio, err = text_to_speech_bytes(source_text, src_code)
                if err:
                    st.warning(err)
                else:
                    st.audio(audio, format="audio/mp3")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🌟 Translated Text</div>', unsafe_allow_html=True)
        st.text_area(
            "Translated text output",
            value=st.session_state.translated_text,
            height=200,
            placeholder="Your translation will appear here...",
            label_visibility="collapsed",
            disabled=True,
            key="translated_text_area",
        )
        st.markdown(
            f'<span class="char-counter">{len(st.session_state.translated_text)} characters</span>',
            unsafe_allow_html=True,
        )

        btn_row1, btn_row2, btn_row3, btn_row4 = st.columns(4)
        with btn_row1:
            copy_to_clipboard_button(st.session_state.translated_text, key="copy_btn")
        with btn_row2:
            if st.button("🔊 Listen", use_container_width=True, key="listen_target"):
                tgt_code = LANGUAGES[tgt_lang]
                audio, err = text_to_speech_bytes(st.session_state.translated_text, tgt_code)
                if err:
                    st.warning(err)
                else:
                    st.audio(audio, format="audio/mp3")
        with btn_row3:
            if st.button("⭐ Save", use_container_width=True, key="fav_btn"):
                if st.session_state.translated_text:
                    entry = {
                        "source_text": st.session_state.source_text,
                        "translated_text": st.session_state.translated_text,
                        "source_lang": src_lang,
                        "target_lang": tgt_lang,
                    }
                    if not is_favourited(entry):
                        st.session_state.favourites.append(entry)
                        st.toast("Added to favourites ⭐")
                    else:
                        st.toast("Already in favourites.")
                else:
                    st.toast("Translate something first!", icon="⚠️")
        with btn_row4:
            if st.session_state.translated_text:
                st.download_button(
                    "⬇️ TXT",
                    data=st.session_state.translated_text,
                    file_name="translation.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.source_text = source_text

    # ---------- Translate action ----------
    if translate_clicked:
        with st.spinner("Translating with AI... 🌐"):
            result, error = translate_text(source_text, src_lang, tgt_lang)
        if error:
            st.error(f"⚠️ {error}")
        else:
            st.session_state.translated_text = result
            add_to_history(source_text, result, src_lang, tgt_lang)
            st.success("Translation complete! ✅")
            st.rerun()


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            Built with ❤️ by <b>Fatima</b> &nbsp;|&nbsp;
            CodeAlpha Artificial Intelligence Internship - Task 1 &nbsp;|&nbsp;
            Powered by Streamlit &amp; Google Translate
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# APP ENTRY POINT
# ============================================================
def main():
    load_css(st.session_state.theme)
    render_sidebar()
    render_header()
    render_translator()
    render_footer()


if __name__ == "__main__":
    main()
