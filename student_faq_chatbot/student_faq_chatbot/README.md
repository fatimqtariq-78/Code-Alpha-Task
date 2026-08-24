# 🎓 AI Student Helpdesk — FAQ Chatbot for Nova University

> **CodeAlpha Artificial Intelligence Internship — Task 2: Chatbot for FAQs**

A genuine NLP-powered FAQ chatbot that answers student questions using **TF-IDF
vectorization** and **cosine similarity** — not hard-coded if/else logic, and
not an external LLM API.

---

## 1. Project Overview

The **AI Student Helpdesk** is a Streamlit web application that simulates a
university support chatbot for a fictional institution called **Nova
University**. Students can ask natural-language questions about admissions,
fees, exams, attendance, the library, and more, and the chatbot finds the most
relevant answer from a structured FAQ knowledge base using classic NLP
techniques.

> ⚠️ **Nova University is entirely fictional.** All FAQ content, policies,
> fees, and dates are sample/demo data created for this internship project
> and do not represent any real institution.

---

## 2. Problem Statement

Students frequently ask the same repetitive questions to university staff —
about deadlines, fees, attendance rules, and so on. Answering these manually
is time-consuming. An automated FAQ chatbot that can understand differently
phrased versions of the same question and respond instantly reduces staff
workload and gives students immediate answers.

---

## 3. Objectives

- Build a chatbot that can understand a user's question even when it is
  phrased differently from the original FAQ.
- Use real NLP preprocessing and similarity techniques, not keyword-only
  matching or hard-coded rules.
- Provide a transparent, evaluator-friendly interface that visibly
  demonstrates *how* the AI reached its answer.
- Handle unknown, ambiguous, and invalid input safely, without crashing.

---

## 4. Features

- 💬 Modern conversational chat interface with distinct user/bot message bubbles
- 🎯 Confidence score shown for every bot response (High / Medium / Low)
- 🔬 "Top 3 matches" debug panel showing the actual cosine-similarity scores
- 🗂️ Matched FAQ question displayed under every answer for transparency
- 💡 Clickable suggested questions and example questions
- 📚 FAQ Explorer with category filtering in the sidebar
- 📊 Live chat statistics (total FAQs, questions asked, average confidence)
- 👍👎 Feedback system with a thank-you message after rating an answer
- 🆕 "New Chat" button to clear the conversation and start fresh
- 🌗 Light/Dark mode toggle
- 🎨 Polished, glassmorphism-based UI with a blue/purple/neutral theme
- 🛡️ Safe error handling for empty, very short, or unknown questions
- 🔒 No login, no personal data collection — fully anonymous demo use

---

## 5. How the Chatbot Works

```
User Question
      ↓
Text Preprocessing   (lowercase, remove punctuation)
      ↓
Tokenization          (split into individual words)
      ↓
Stop-word Removal      (remove words like "is", "the", "a")
      ↓
Lemmatization           (reduce words to base form, e.g. "paying" → "pay")
      ↓
TF-IDF Vectorization     (convert text into a weighted numeric vector)
      ↓
Cosine Similarity Comparison   (compare against all 60 FAQ vectors)
      ↓
Best Match + Confidence Score
      ↓
Confidence Threshold Decision   (High / Medium / Low)
      ↓
Final Answer Returned to User
```

This pipeline is implemented in two dedicated modules:
- `preprocessing.py` handles steps 1–4 (using NLTK)
- `chatbot.py` handles steps 5–8 (using scikit-learn)

---

## 6. NLP Methodology

### 6.1 Text Preprocessing (NLTK)
Every question — both the FAQ questions in the dataset and the user's typed
question — goes through the same cleaning pipeline:
1. Convert to lowercase and strip punctuation
2. Tokenize into individual words (`nltk.word_tokenize`)
3. Remove English stopwords (`nltk.corpus.stopwords`)
4. Lemmatize each remaining word to its dictionary form (`WordNetLemmatizer`)

This ensures that "How much attendance do I need?" and "What is the
attendance requirement?" both reduce to a similar bag of meaningful words
(e.g. `attendance`, `requirement`, `need`), even though they're phrased
completely differently.

If NLTK's data packages cannot be downloaded (e.g. no internet on first
run), the app automatically falls back to a simple built-in tokenizer and
stopword list so it never crashes.

### 6.2 TF-IDF Vectorization
**TF-IDF (Term Frequency–Inverse Document Frequency)** converts cleaned text
into a numeric vector where:
- **Term Frequency** measures how often a word appears in a given question
- **Inverse Document Frequency** reduces the weight of words that appear in
  *many* FAQs (common, less distinctive words) and increases the weight of
  words that are rarer and more specific (e.g. "scholarship", "attendance")

We use `sklearn.feature_extraction.text.TfidfVectorizer`, fit once on the
combined text of all 60 FAQ questions + their keyword lists when the app
starts.

### 6.3 Cosine Similarity
**Cosine similarity** measures the angle between two TF-IDF vectors,
producing a score between 0 (completely unrelated) and 1 (identical
meaning/wording). For every user question, we compute its similarity against
all 60 FAQ vectors and pick the highest-scoring match using
`sklearn.metrics.pairwise.cosine_similarity`.

### 6.4 Confidence Threshold
The best similarity score is compared against two configurable thresholds
(set in `chatbot.py`):

| Score Range | Confidence Level | Behavior |
|---|---|---|
| ≥ 0.45 | **High** | Answer returned normally |
| 0.20 – 0.45 | **Medium** | Answer returned with a gentle clarification note |
| < 0.20 | **Low** | Bot declines to guess and asks the user to rephrase or pick a known category |

This prevents the chatbot from confidently answering questions it doesn't
actually understand.

---

## 7. Dataset Description

- **File:** `faq_data.json`
- **Total entries:** 60 FAQs
- **Categories (12):** Admissions, Courses & Departments, Fees & Payments,
  Exams, Assignments, Attendance, Library, Timetable, Scholarships, Campus
  Facilities, IT / Student Portal, General University Information
- **Fields per entry:**
  - `id` — unique identifier
  - `category` — one of the 12 categories above
  - `question` — the canonical FAQ question
  - `answer` — the fictional demo answer
  - `keywords` — extra related terms/phrases to widen the TF-IDF vocabulary
    for that topic (helps match differently worded user questions)

All content is explicitly fictional/demo data for **Nova University**.

---

## 8. Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core language |
| **Streamlit** | Web UI framework |
| **NLTK** | Tokenization, stopword removal, lemmatization |
| **scikit-learn** | TF-IDF vectorization + cosine similarity |
| **JSON** | Structured FAQ dataset storage |

No paid or external LLM APIs are used anywhere in this project.

---

## 9. Project Structure

```
student_faq_chatbot/
│
├── app.py                # Streamlit UI layer (chat, sidebar, styling)
├── chatbot.py             # NLP engine: TF-IDF + cosine similarity matching
├── preprocessing.py        # Text cleaning: tokenize, stopwords, lemmatize
├── faq_data.json            # 60-entry fictional FAQ dataset for Nova University
├── requirements.txt          # Python dependencies
├── README.md                  # Project documentation (this file)
├── assets/                     # Static assets
└── screenshots/                 # App screenshots for submission
```

---

## 10. Installation

```bash
# 1. Navigate into the project folder
cd student_faq_chatbot

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

The first time the app runs, it will automatically download a few small
NLTK data packages (punkt, stopwords, wordnet) if they aren't already
present on your machine — this requires an internet connection just once.

---

## 11. How to Run

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 12. Example Questions to Try

- "What are the admission requirements?"
- "How much attendance do I need?"
- "What percentage attendance is required?" *(differently phrased version of the same question)*
- "How can I pay my semester fee?"
- "When are exams held?"
- "What scholarships are available?"

---

## 13. Screenshots

> Add screenshots of the running app here after testing, for example:

```
screenshots/
├── welcome-screen.png
├── chat-high-confidence.png
├── chat-low-confidence.png
├── faq-explorer.png
└── dark-mode.png
```

---

## 14. Limitations

- The FAQ dataset is fixed/sample data — it does not connect to any real
  university system or live database.
- TF-IDF + cosine similarity is a strong classical NLP technique but does
  not have true language understanding like a large language model — very
  unusual phrasing may still fall into the "low confidence" category.
- Conversation history and feedback are stored only in the current browser
  session (`st.session_state`) and are not saved permanently.

---

## 15. Future Improvements

- Upgrade matching with sentence embeddings (e.g. Sentence-BERT) for deeper
  semantic understanding beyond keyword overlap
- Add a real admin dashboard to review low-confidence questions and expand
  the FAQ dataset over time
- Persist chat history and feedback to a database
- Add multi-language support for non-English student queries
- Connect to a real (permissioned) university information system

---

## 16. Internship Task Mapping (CodeAlpha Task 2)

| Requirement | How it's fulfilled |
|---|---|
| Collect FAQs related to a topic/product | 60 FAQs across 12 categories in `faq_data.json`, for the fictional Nova University |
| Preprocess text using NLP libraries (NLTK/spaCy) | `preprocessing.py` uses NLTK for tokenization, stopword removal, and lemmatization |
| Match user's question to the most similar FAQ (cosine similarity/intent matching) | `chatbot.py` uses TF-IDF + `cosine_similarity` from scikit-learn |
| Display the best matching answer as the chatbot response | Rendered in the chat bubble UI in `app.py`, including category and confidence |
| (Optional) Simple chat UI | Full Streamlit chat interface with bubbles, sidebar, stats, and feedback — well beyond the minimum requirement |

---

## 17. Demo Scenarios (for presenting to an evaluator)

| # | Question to type | Type | Expected Result |
|---|---|---|---|
| 1 | "What is the attendance requirement?" | Exact FAQ wording | High confidence, direct answer, matched FAQ shown |
| 2 | "How much attendance do I need?" | Reworded version of #1 | High/medium confidence, same answer as #1 — demonstrates NLP understanding beyond exact text |
| 3 | "What percentage attendance is required?" | Another reworded version of #1 | Same FAQ matched again — shows consistency across phrasings |
| 4 | "How can I pay my semester fee?" | Different category (Fees) | High confidence, Fees & Payments category tag |
| 5 | "When are exams held?" | Different category (Exams) | High confidence, Exams category tag |
| 6 | "What scholarships are available?" | Different category (Scholarships) | High confidence answer |
| 7 | "library" | Single-word, minimal query | Still matches Library FAQs reasonably (demonstrates TF-IDF keyword weighting) |
| 8 | "asdkjaskdjaskjd random gibberish" | Out-of-scope / nonsense | Low confidence response — bot declines to guess, suggests topics |
| 9 | "fee" | Ambiguous / very short | Bot returns its best guess with a lower confidence tag — show the "Top 3 matches" panel to explain why |
| 10 | (empty input / just spaces) | Invalid input | Friendly message asking the user to type a full question — no crash |

**Tip for the demo:** After asking question #1 and #2, open the "🔬 View top
3 matching FAQs" expander to show the evaluator the actual cosine similarity
scores — this is the clearest way to prove the NLP pipeline is real and
working, not hard-coded.

---

## 18. Author

**Fatima**
Bachelor of Computer Science Student, Capital University of Science and Technology (CUST)
Built for the **CodeAlpha Artificial Intelligence Internship — Task 2**
