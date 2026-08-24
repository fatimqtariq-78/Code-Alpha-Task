# 🌐 AI Language Translation Tool

> **CodeAlpha Artificial Intelligence Internship — Task 1**
> A modern, portfolio-worthy language translator built with Python and Streamlit.

---

## 📖 Project Overview

The **AI Language Translation Tool** is a web application that lets users translate
text between **100+ languages** in real time, listen to both the original and
translated text via text-to-speech, and keep track of their recent and favourite
translations — all wrapped in a clean, glassmorphism-style UI with a blue-purple
gradient theme and a dark/light mode toggle.

This project was built as part of Task 1 of the CodeAlpha AI Internship, with a
focus on writing clean, modular, beginner-friendly Python code while still
delivering a UI that feels like a real, polished SaaS product rather than a
basic tutorial script.

---

## ✨ Features

### Core Features
- 📝 Text input box for the source text
- 🔤 Source language dropdown (with **Auto-Detect**)
- 🎯 Target language dropdown
- 🚀 Translate button powered by Google Translate (via `deep-translator`)
- 🌟 Clean display of the translated text

### Premium / Extra Features
- 🎨 Beautiful, modern, glassmorphism-based UI
- 🌌 Blue + purple gradient background
- 🌗 Dark Mode / Light Mode toggle
- 🌍 100+ supported languages (pulled directly from Google Translate's language list)
- ⇄ Swap Languages button
- 📋 Copy translated text to clipboard
- 🧹 Clear button to reset the text areas
- 🔢 Live character counter
- 🕘 Translation history (automatically keeps the last 10 translations)
- ⭐ Favourite / save translations for later
- ⬇️ Download translation as a `.txt` file
- 🔊 Text-to-Speech playback for **both** original and translated text
- ⏳ Loading spinner while a translation is in progress
- ⚠️ Friendly error messages for empty input, network issues, or API failures
- ✅ Success toast/notification after a translation completes
- 📱 Responsive layout that adapts to smaller screens
- 🦶 Professional footer with author and project credit

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core programming language |
| **Streamlit** | Web app framework / UI rendering |
| **deep-translator** | Free wrapper around Google Translate for text translation |
| **gTTS (Google Text-to-Speech)** | Converts text to spoken audio |
| **pyperclip** | Copies translated text to the system clipboard |
| **HTML + CSS** | Custom styling injected into Streamlit for the glassmorphism/gradient theme |

No paid APIs or API keys are required — everything runs on free, open-source
libraries.

---

## ⚙️ Installation

1. **Clone or download** this project folder.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

From inside the `LanguageTranslator` folder, run:

```bash
streamlit run app.py
```

Streamlit will start a local server and automatically open the app in your
default browser (usually at `http://localhost:8501`).

> **Note:** An active internet connection is required, since translation and
> text-to-speech both rely on free online services.

---

## 📸 Screenshots

> Add your own screenshots of the running app to the `screenshots/` folder and
> reference them here, for example:

```
screenshots/
├── home-dark-mode.png
├── home-light-mode.png
└── translation-history.png
```

```markdown
![Dark Mode](screenshots/home-dark-mode.png)
![Light Mode](screenshots/home-light-mode.png)
```

---

## 🚀 Future Improvements

- Add support for translating entire uploaded files (`.txt`, `.docx`, `.pdf`)
- Add a "detect language" badge that shows what language was auto-detected
- Add user accounts so history/favourites persist across sessions (currently
  session-based only)
- Add a browser extension / API endpoint version of the tool
- Support voice input (speech-to-text) in addition to text input
- Add batch translation for multiple sentences/paragraphs at once

---

## 📂 Project Structure

```
LanguageTranslator/
│
├── app.py               # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md              # Project documentation (this file)
├── assets/                # Static assets (icons, images, etc.)
└── screenshots/            # App screenshots for documentation
```

---

## 🙋‍♀️ Author

**Fatima**
Bachelor of Computer Science Student, Capital University of Science and Technology (CUST)
Built for the **CodeAlpha Artificial Intelligence Internship — Task 1**
