"""
app.py
------
AI Student Helpdesk - FAQ Chatbot for Nova University (fictional demo institution)

CodeAlpha Artificial Intelligence Internship - Task 2

This file contains ONLY the UI layer (Streamlit). All NLP logic lives in
preprocessing.py and chatbot.py, and the FAQ knowledge base lives in
faq_data.json, keeping the project cleanly modular.

Author: Fatima
"""

import streamlit as st

from chatbot import FAQChatbot


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Nova University | AI Student Helpdesk",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUESTIONS = [
    "What are the admission requirements?",
    "How can I pay my semester fee?",
    "What is the attendance requirement?",
    "When are exams held?",
    "How can I access the student portal?",
]

# Demo questions kept separately for suggested-question chips in the sidebar
SUGGESTED_QUESTIONS = [
    "How much attendance do I need?",
    "What scholarships are available?",
    "How do I submit assignments?",
    "What are the library timings?",
    "Is there a fee installment plan?",
    "Does the university offer online learning?",
]


# ============================================================
# CACHED CHATBOT ENGINE
# ============================================================
@st.cache_resource(show_spinner=False)
def load_chatbot():
    """
    Loads the FAQ dataset and builds the TF-IDF index once per app session
    (not on every rerun), since building the index is the "expensive" part.
    """
    return FAQChatbot(dataset_path="faq_data.json")


bot = load_chatbot()


# ============================================================
# SESSION STATE
# ============================================================
def init_session_state():
    defaults = {
        "theme": "light",
        "messages": [],              # list of message dicts
        "questions_asked": 0,
        "confidence_sum": 0.0,
        "explorer_category": "All",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================
# CORE MESSAGE PROCESSING
# ============================================================
def process_user_message(user_text: str):
    """Adds the user's message + the bot's response to the chat history."""
    user_text = user_text.strip()
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})

    result = bot.get_response(user_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "category": result.get("category"),
        "matched_question": result.get("matched_question"),
        "confidence": result.get("confidence", 0.0),
        "confidence_level": result.get("confidence_level", "low"),
        "top_matches": result.get("top_matches", []),
        "feedback": None,
    })

    st.session_state.questions_asked += 1
    st.session_state.confidence_sum += result.get("confidence", 0.0)


# ============================================================
# CUSTOM CSS
# ============================================================
def load_css(theme: str):
    if theme == "dark":
        bg_gradient = "linear-gradient(135deg, #0f1224 0%, #1b1f3b 50%, #241a3d 100%)"
        card_bg = "rgba(255, 255, 255, 0.06)"
        card_border = "rgba(255, 255, 255, 0.13)"
        text_color = "#f1f1f7"
        subtext_color = "#b9b7cc"
        bot_bubble_bg = "rgba(255, 255, 255, 0.08)"
        user_bubble_bg = "linear-gradient(135deg, #5b6ee8 0%, #7c5cf0 100%)"
    else:
        bg_gradient = "linear-gradient(135deg, #f4f7ff 0%, #eef1fb 50%, #f3edfb 100%)"
        card_bg = "rgba(255, 255, 255, 0.75)"
        card_border = "rgba(180, 180, 210, 0.35)"
        text_color = "#232244"
        subtext_color = "#5c5a78"
        bot_bubble_bg = "rgba(255, 255, 255, 0.9)"
        user_bubble_bg = "linear-gradient(135deg, #4f6ef7 0%, #7c5cf0 100%)"

    st.markdown(
        f"""
        <style>
        :root {{
            --accent-gradient: linear-gradient(135deg, #4f6ef7 0%, #7c5cf0 60%, #a15ce0 100%);
            --card-bg: {card_bg};
            --card-border: {card_border};
            --text-color: {text_color};
            --subtext-color: {subtext_color};
        }}

        .stApp {{
            background: {bg_gradient};
            background-attachment: fixed;
        }}
        html, body, [class*="css"] {{
            font-family: 'Segoe UI', 'Poppins', sans-serif;
            color: {text_color};
        }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        /* ---------- Header ---------- */
        .helpdesk-header {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.4rem 1.8rem;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 8px 28px rgba(79, 110, 247, 0.12);
            margin-bottom: 1.3rem;
        }}
        .uni-name {{
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #7c5cf0;
            margin-bottom: 0.1rem;
        }}
        .app-title {{
            font-size: 2.2rem;
            font-weight: 800;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            letter-spacing: -0.5px;
        }}
        .app-subtitle {{
            color: {subtext_color};
            font-size: 0.98rem;
            margin-top: 0.35rem;
        }}
        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(46, 204, 113, 0.12);
            color: #1fa460;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            margin-top: 0.7rem;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #2ecc71;
            box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.6);
            animation: pulse 1.8s infinite;
        }}
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.5); }}
            70% {{ box-shadow: 0 0 0 7px rgba(46, 204, 113, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }}
        }}

        /* ---------- Chat bubbles ---------- */
        .chat-row {{ display: flex; margin-bottom: 0.9rem; }}
        .chat-row.user {{ justify-content: flex-end; }}
        .chat-row.bot {{ justify-content: flex-start; }}

        .bubble {{
            max-width: 72%;
            padding: 0.85rem 1.1rem;
            border-radius: 18px;
            font-size: 0.95rem;
            line-height: 1.5;
            box-shadow: 0 4px 16px rgba(31, 38, 135, 0.08);
        }}
        .bubble.user {{
            background: {user_bubble_bg};
            color: white;
            border-bottom-right-radius: 4px;
        }}
        .bubble.bot {{
            background: {bot_bubble_bg};
            border: 1px solid var(--card-border);
            color: {text_color};
            border-bottom-left-radius: 4px;
        }}

        .meta-row {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
            margin-top: 0.5rem;
            flex-wrap: wrap;
        }}
        .category-tag {{
            background: rgba(124, 92, 240, 0.12);
            color: #7c5cf0;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.15rem 0.65rem;
            border-radius: 20px;
        }}
        .confidence-tag {{
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.15rem 0.65rem;
            border-radius: 20px;
        }}
        .confidence-high {{ background: rgba(46, 204, 113, 0.15); color: #1fa460; }}
        .confidence-medium {{ background: rgba(241, 196, 15, 0.18); color: #b8860b; }}
        .confidence-low {{ background: rgba(231, 76, 60, 0.13); color: #c0392b; }}

        .matched-faq {{
            font-size: 0.78rem;
            color: {subtext_color};
            margin-top: 0.4rem;
            font-style: italic;
        }}

        /* ---------- Glass card (sidebar sections, welcome cards) ---------- */
        .glass-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 6px 22px rgba(31, 38, 135, 0.1);
            margin-bottom: 1rem;
        }}

        .how-it-works-step {{
            display: inline-block;
            background: var(--accent-gradient);
            color: white;
            padding: 0.35rem 0.9rem;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
            margin: 0.15rem;
        }}

        /* ---------- Buttons ---------- */
        div.stButton > button {{
            background: var(--accent-gradient);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.5rem 1.1rem;
            font-weight: 600;
            font-size: 0.85rem;
            box-shadow: 0 4px 14px rgba(124, 92, 240, 0.28);
            transition: all 0.2s ease-in-out;
        }}
        div.stButton > button:hover {{
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 22px rgba(124, 92, 240, 0.4);
            color: white;
        }}

        .app-footer {{
            text-align: center;
            padding: 1.4rem 0 0.6rem 0;
            color: {subtext_color};
            font-size: 0.82rem;
            border-top: 1px solid var(--card-border);
            margin-top: 1.6rem;
        }}

        .demo-note {{
            font-size: 0.75rem;
            color: {subtext_color};
            text-align: center;
            margin-top: 0.3rem;
        }}

        @media (max-width: 768px) {{
            .app-title {{ font-size: 1.7rem; }}
            .bubble {{ max-width: 88%; }}
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
        st.markdown("### 🎓 Nova Helpdesk")

        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.questions_asked = 0
            st.session_state.confidence_sum = 0.0
            st.rerun()

        theme_choice = st.toggle(
            "🌙 Dark Mode", value=(st.session_state.theme == "dark")
        )
        st.session_state.theme = "dark" if theme_choice else "light"

        st.markdown("---")
        st.markdown("#### 💡 Suggested Questions")
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if st.button(q, key=f"suggested_{i}", use_container_width=True):
                process_user_message(q)
                st.rerun()

        st.markdown("---")
        st.markdown("#### 📚 FAQ Categories")
        categories = ["All"] + bot.get_categories()
        st.session_state.explorer_category = st.selectbox(
            "Filter FAQ Explorer by category",
            options=categories,
            index=categories.index(st.session_state.explorer_category)
            if st.session_state.explorer_category in categories else 0,
            label_visibility="collapsed",
        )

        st.markdown("#### 🔍 FAQ Explorer")
        if bot.load_error:
            st.caption("FAQ data unavailable.")
        else:
            faqs_to_show = (
                bot.faqs if st.session_state.explorer_category == "All"
                else bot.get_faqs_by_category(st.session_state.explorer_category)
            )
            with st.expander(f"Browse {len(faqs_to_show)} FAQ(s)"):
                for faq in faqs_to_show:
                    st.markdown(f"**Q: {faq['question']}**")
                    st.caption(faq["answer"])
                    st.markdown("—")

        st.markdown("---")
        st.markdown("#### 📊 Chat Statistics")
        avg_conf = (
            st.session_state.confidence_sum / st.session_state.questions_asked
            if st.session_state.questions_asked > 0 else 0
        )
        stat_col1, stat_col2 = st.columns(2)
        stat_col1.metric("Total FAQs", bot.total_faqs())
        stat_col2.metric("Asked", st.session_state.questions_asked)
        st.metric("Avg. Confidence", f"{avg_conf * 100:.0f}%")

        st.markdown("---")
        with st.expander("ℹ️ About this Project"):
            st.caption(
                "This is a demonstration AI Student Helpdesk chatbot built for the "
                "CodeAlpha Artificial Intelligence Internship (Task 2). It uses "
                "TF-IDF vectorization and cosine similarity - not a hard-coded "
                "if/else system, and not an external LLM API - to match your "
                "question against a fictional FAQ dataset for 'Nova University'."
            )


# ============================================================
# HEADER
# ============================================================
def render_header():
    st.markdown(
        f"""
        <div class="helpdesk-header">
            <div class="uni-name">Nova University</div>
            <div class="app-title">🎓 AI Student Helpdesk</div>
            <div class="app-subtitle">
                Ask questions about admissions, academics, fees, exams, and student services.
            </div>
            <div class="status-pill"><span class="status-dot"></span> AI Assistant Online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# WELCOME SCREEN ("How it works" + example questions)
# ============================================================
def render_welcome():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 👋 Welcome! Here's how this chatbot works")
    st.markdown(
        """
        <span class="how-it-works-step">Question</span> →
        <span class="how-it-works-step">NLP Preprocessing</span> →
        <span class="how-it-works-step">TF-IDF</span> →
        <span class="how-it-works-step">Cosine Similarity</span> →
        <span class="how-it-works-step">Best Match</span> →
        <span class="how-it-works-step">Answer</span>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Every question you type is cleaned and vectorized, then compared "
        "against 60 sample FAQs using cosine similarity to find the closest match."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 💬 Try asking one of these:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(question, key=f"example_{question}", use_container_width=True):
                process_user_message(question)
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="demo-note">📌 All university, fee, and policy information '
        'shown by this chatbot is fictional sample data for demonstration purposes only.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT RENDERING
# ============================================================
def render_confidence_tag(level: str, confidence: float):
    css_class = f"confidence-{level}"
    label = {"high": "High confidence", "medium": "Medium confidence", "low": "Low confidence"}.get(level, "")
    return f'<span class="confidence-tag {css_class}">{label} ({confidence * 100:.0f}%)</span>'


def render_chat():
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-row user">
                    <div class="bubble user">{msg['content']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            category_html = (
                f'<span class="category-tag">{msg["category"]}</span>' if msg.get("category") else ""
            )
            confidence_html = render_confidence_tag(
                msg.get("confidence_level", "low"), msg.get("confidence", 0.0)
            )
            matched_html = (
                f'<div class="matched-faq">Matched FAQ: "{msg["matched_question"]}"</div>'
                if msg.get("matched_question") else ""
            )

            st.markdown(
                f"""
                <div class="chat-row bot">
                    <div class="bubble bot">
                        {msg['content']}
                        <div class="meta-row">{category_html}{confidence_html}</div>
                        {matched_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Top matches (debug/demo transparency section)
            if msg.get("top_matches"):
                with st.expander("🔬 View top 3 matching FAQs (NLP debug view)"):
                    for m in msg["top_matches"]:
                        st.progress(min(m["confidence"], 1.0))
                        st.caption(f"{m['question']}  —  {m['category']}  —  {m['confidence'] * 100:.1f}% similarity")

            # Feedback buttons
            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])
            if msg.get("feedback") is None:
                with fb_col1:
                    if st.button("👍", key=f"up_{idx}"):
                        st.session_state.messages[idx]["feedback"] = "up"
                        st.rerun()
                with fb_col2:
                    if st.button("👎", key=f"down_{idx}"):
                        st.session_state.messages[idx]["feedback"] = "down"
                        st.rerun()
            else:
                st.caption("🙏 Thanks for your feedback!")


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            Built with ❤️ by <b>Fatima</b> for the CodeAlpha AI Internship - Task 2 &nbsp;|&nbsp;
            Powered by scikit-learn TF-IDF &amp; Cosine Similarity (no external LLM API)
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN
# ============================================================
def main():
    load_css(st.session_state.theme)
    render_sidebar()
    render_header()

    if bot.load_error:
        st.error(f"⚠️ {bot.load_error}")
        return

    if not st.session_state.messages:
        render_welcome()
    else:
        render_chat()

    user_input = st.chat_input("Type your question here... e.g. 'What is the attendance requirement?'")
    if user_input:
        process_user_message(user_input)
        st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
