"""
chatbot.py
----------
The core NLP engine of the FAQ chatbot.

Pipeline for every user question:

    User Question
        -> preprocess_text()               (preprocessing.py)
        -> TF-IDF vector (using vocabulary learned from the FAQ dataset)
        -> cosine similarity against every FAQ's TF-IDF vector
        -> best match + confidence score
        -> confidence-threshold decision (HIGH / MEDIUM / LOW)
        -> final response returned to the UI (app.py)

This is a genuine, from-scratch implementation using scikit-learn's
TfidfVectorizer and cosine_similarity - there is no hard-coded if/else
question matching and no external LLM API involved.
"""

import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess_text, is_valid_question


# Confidence thresholds - tune these to make the bot stricter or looser.
HIGH_CONFIDENCE_THRESHOLD = 0.45
MEDIUM_CONFIDENCE_THRESHOLD = 0.20


class FAQChatbot:
    """
    Loads the FAQ dataset, builds a TF-IDF index over it, and answers
    user questions using cosine similarity matching.
    """

    def __init__(self, dataset_path: str = "faq_data.json"):
        self.dataset_path = dataset_path
        self.faqs = []
        self.university_name = "Nova University"
        self.vectorizer = None
        self.faq_matrix = None
        self.load_error = None

        self._load_dataset()
        if self.faqs:
            self._build_index()

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def _load_dataset(self):
        """Reads faq_data.json. Sets self.load_error if anything goes wrong."""
        try:
            if not os.path.exists(self.dataset_path):
                self.load_error = (
                    f"FAQ dataset not found at '{self.dataset_path}'. "
                    "Please make sure faq_data.json is in the project folder."
                )
                return

            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.university_name = data.get("university", "Nova University")
            self.faqs = data.get("faqs", [])

            if not self.faqs:
                self.load_error = "The FAQ dataset was loaded but contains no entries."
        except json.JSONDecodeError:
            self.load_error = "The FAQ dataset file is not valid JSON."
        except Exception as exc:
            self.load_error = f"Unexpected error while loading the FAQ dataset: {exc}"

    # ------------------------------------------------------------------
    # TF-IDF index building
    # ------------------------------------------------------------------
    def _build_index(self):
        """
        Builds one combined text per FAQ (question + keywords, repeated
        for weight) and fits a TF-IDF vectorizer over the whole dataset.
        """
        corpus = []
        for faq in self.faqs:
            keyword_text = " ".join(faq.get("keywords", []))
            # Repeat the question text so it carries more weight than the
            # keyword list alone, then combine with keywords for extra coverage.
            combined = f"{faq['question']} {faq['question']} {keyword_text}"
            corpus.append(preprocess_text(combined))

        self.vectorizer = TfidfVectorizer()
        self.faq_matrix = self.vectorizer.fit_transform(corpus)

    # ------------------------------------------------------------------
    # Core matching logic
    # ------------------------------------------------------------------
    def get_response(self, user_question: str) -> dict:
        """
        Given a raw user question, returns a dictionary describing the
        chatbot's response:

            {
                "status": "ok" | "error" | "no_match" | "low_confidence",
                "answer": str,
                "category": str or None,
                "matched_question": str or None,
                "confidence": float (0-1),
                "confidence_level": "high" | "medium" | "low",
                "top_matches": [ {question, category, confidence}, ... ]
            }
        """
        # --- Guard: dataset failed to load ---
        if self.load_error:
            return {
                "status": "error",
                "answer": (
                    "Sorry, the FAQ knowledge base could not be loaded right now. "
                    "Please try again later."
                ),
                "category": None,
                "matched_question": None,
                "confidence": 0.0,
                "confidence_level": "low",
                "top_matches": [],
            }

        # --- Guard: empty / too short input ---
        if not is_valid_question(user_question):
            return {
                "status": "error",
                "answer": (
                    "Please type a full question so I can help you - "
                    "for example, 'What is the attendance requirement?'"
                ),
                "category": None,
                "matched_question": None,
                "confidence": 0.0,
                "confidence_level": "low",
                "top_matches": [],
            }

        try:
            cleaned_query = preprocess_text(user_question)
            if not cleaned_query.strip():
                raise ValueError("Question became empty after preprocessing.")

            query_vector = self.vectorizer.transform([cleaned_query])
            similarities = cosine_similarity(query_vector, self.faq_matrix)[0]

            # Get indices of the top 3 matches, best first.
            top_indices = similarities.argsort()[::-1][:3]

            top_matches = [
                {
                    "question": self.faqs[i]["question"],
                    "category": self.faqs[i]["category"],
                    "confidence": round(float(similarities[i]), 3),
                }
                for i in top_indices
            ]

            best_index = int(top_indices[0])
            best_score = float(similarities[best_index])
            best_faq = self.faqs[best_index]

            # --- Confidence-threshold decision ---
            if best_score >= HIGH_CONFIDENCE_THRESHOLD:
                return {
                    "status": "ok",
                    "answer": best_faq["answer"],
                    "category": best_faq["category"],
                    "matched_question": best_faq["question"],
                    "confidence": round(best_score, 3),
                    "confidence_level": "high",
                    "top_matches": top_matches,
                }

            elif best_score >= MEDIUM_CONFIDENCE_THRESHOLD:
                return {
                    "status": "ok",
                    "answer": (
                        f"{best_faq['answer']}\n\n"
                        "_(I'm not fully certain this is exactly what you meant - "
                        "let me know if you were asking something else.)_"
                    ),
                    "category": best_faq["category"],
                    "matched_question": best_faq["question"],
                    "confidence": round(best_score, 3),
                    "confidence_level": "medium",
                    "top_matches": top_matches,
                }

            else:
                return {
                    "status": "low_confidence",
                    "answer": (
                        "I'm not confident that I found the right answer in my "
                        "knowledge base. Try asking about admissions, fees, exams, "
                        "attendance, library, timetable, scholarships, campus "
                        "facilities, or the student portal."
                    ),
                    "category": None,
                    "matched_question": None,
                    "confidence": round(best_score, 3),
                    "confidence_level": "low",
                    "top_matches": top_matches,
                }

        except Exception as exc:
            return {
                "status": "error",
                "answer": f"Something went wrong while processing your question. ({exc})",
                "category": None,
                "matched_question": None,
                "confidence": 0.0,
                "confidence_level": "low",
                "top_matches": [],
            }

    # ------------------------------------------------------------------
    # Helper accessors used by the UI
    # ------------------------------------------------------------------
    def get_categories(self):
        """Returns a sorted list of unique FAQ categories."""
        return sorted({faq["category"] for faq in self.faqs})

    def get_faqs_by_category(self, category: str):
        """Returns all FAQ entries belonging to a given category."""
        return [faq for faq in self.faqs if faq["category"] == category]

    def total_faqs(self):
        return len(self.faqs)
