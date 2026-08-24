"""
preprocessing.py
-----------------
Handles all NLP text preprocessing for the FAQ chatbot:

    raw text -> lowercase -> tokenize -> remove punctuation/stopwords -> lemmatize -> clean tokens

We use NLTK for tokenization, stopword removal, and lemmatization. Because NLTK
needs small data packages (punkt, stopwords, wordnet) downloaded the first time
it runs, this module automatically checks for them and downloads them quietly
if missing. If, for any reason (no internet, blocked download), NLTK data is
not available, we fall back to a simple regex-based tokenizer and a built-in
stopword list so the chatbot never crashes.
"""

import re
import string

# ------------------------------------------------------------------
# Try to set up NLTK. Fall back gracefully if it isn't available.
# ------------------------------------------------------------------
NLTK_READY = False
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    def _ensure_nltk_data():
        """Download the small NLTK packages we need, only if missing."""
        packages = {
            "tokenizers/punkt": "punkt",
            "tokenizers/punkt_tab": "punkt_tab",
            "corpora/stopwords": "stopwords",
            "corpora/wordnet": "wordnet",
        }
        for path, package_name in packages.items():
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(package_name, quiet=True)
                except Exception:
                    # If download fails (e.g. no internet), we'll fall back below.
                    pass

    _ensure_nltk_data()

    # Quick functional test - if this fails, we drop to the fallback path.
    word_tokenize("test sentence")
    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
    NLTK_READY = True
except Exception:
    NLTK_READY = False


# ------------------------------------------------------------------
# Fallback resources (used only if NLTK setup above did not succeed)
# ------------------------------------------------------------------
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "having", "i", "me",
    "my", "we", "our", "you", "your", "he", "she", "it", "they", "them",
    "this", "that", "these", "those", "to", "of", "in", "on", "at", "for",
    "with", "about", "as", "by", "from", "and", "or", "but", "if", "so",
    "than", "too", "very", "can", "will", "just", "should", "now", "what",
    "which", "who", "whom", "how", "when", "where", "why", "there", "here",
}


def _fallback_tokenize(text: str):
    """Simple regex tokenizer used only when NLTK isn't available."""
    return re.findall(r"[a-zA-Z]+", text.lower())


# ------------------------------------------------------------------
# Public preprocessing function
# ------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """
    Cleans and normalizes a piece of text for NLP comparison.

    Steps:
        1. Lowercase the text
        2. Remove punctuation and non-alphabetic tokens
        3. Tokenize into words
        4. Remove stopwords (common words that add little meaning)
        5. Lemmatize each remaining word to its base/dictionary form
           (e.g. "requirements" -> "requirement", "paying" -> "pay")

    Returns a single cleaned string, ready to be fed into TF-IDF.
    """
    if not text:
        return ""

    text = text.lower().strip()
    # Remove punctuation early so tokenization is cleaner
    text = text.translate(str.maketrans("", "", string.punctuation))

    if NLTK_READY:
        tokens = word_tokenize(text)
        cleaned_tokens = [
            _LEMMATIZER.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in _STOPWORDS
        ]
    else:
        tokens = _fallback_tokenize(text)
        cleaned_tokens = [t for t in tokens if t not in _FALLBACK_STOPWORDS]

    return " ".join(cleaned_tokens)


def is_valid_question(text: str, min_length: int = 2) -> bool:
    """
    Basic validation to catch empty or extremely short/meaningless input
    before it ever reaches the similarity engine.
    """
    if not text or not text.strip():
        return False
    cleaned = preprocess_text(text)
    return len(cleaned.split()) >= 1 and len(text.strip()) >= min_length
