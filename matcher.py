"""
CV-JD Matching

Scoring logic:
  50% — Skill match        (most meaningful — directly compares extracted skills)
  25% — Keyword overlap    (direct token match after stopword removal)
  15% — sklearn TF-IDF cosine  (optimized vectorization)
  10% — TF-IDF from scratch    (kept for prof — shows the math)
"""

import math
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


STOPWORDS = {
    "i","me","my","we","our","you","your","he","she","it","they","them",
    "this","that","a","an","the","and","or","but","in","on","at","to",
    "for","of","with","is","are","was","were","be","been","have","has",
    "had","do","does","did","will","would","can","could","should","may",
    "might","not","from","by","as","so","if","about","into","through",
    "looking","seeking","candidate","must","able","experience","work",
    "working","role","position","job","team","good","strong","required",
    "excellent","proficient","knowledge","understanding","ability","years",
}


def preprocess(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]


# ── TF-IDF FROM SCRATCH (kept for professor) ──────────────

def compute_tf(tokens: list) -> dict:
    count = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {term: freq / total for term, freq in count.items()}


def compute_idf(documents: list) -> dict:
    N = len(documents)
    idf = {}
    all_terms = set(t for doc in documents for t in doc)
    for term in all_terms:
        df = sum(1 for doc in documents if term in doc)
        idf[term] = math.log((N + 1) / (df + 1)) + 1
    return idf


def compute_tfidf_vector(tf: dict, idf: dict) -> dict:
    return {term: tf_val * idf.get(term, 0) for term, tf_val in tf.items()}


def cosine_similarity_scratch(vec_a: dict, vec_b: dict) -> float:
    common = set(vec_a.keys()) & set(vec_b.keys())
    dot    = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a  = math.sqrt(sum(v**2 for v in vec_a.values()))
    mag_b  = math.sqrt(sum(v**2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def tfidf_cosine_scratch(cv_text: str, jd_text: str) -> float:
    cv_tok = preprocess(cv_text)
    jd_tok = preprocess(jd_text)
    idf    = compute_idf([cv_tok, jd_tok])
    cv_vec = compute_tfidf_vector(compute_tf(cv_tok), idf)
    jd_vec = compute_tfidf_vector(compute_tf(jd_tok), idf)
    return cosine_similarity_scratch(cv_vec, jd_vec)


# ── sklearn TF-IDF + cosine ────────────────────────────────

def tfidf_cosine_sklearn(cv_text: str, jd_text: str) -> float:
    tfidf  = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = tfidf.fit_transform([cv_text, jd_text])
    return float(sklearn_cosine(matrix[0], matrix[1])[0][0])


# ── Keyword overlap ────────────────────────────────────────

def keyword_overlap(cv_text: str, jd_text: str) -> float:
    cv_set = set(preprocess(cv_text))
    jd_set = set(preprocess(jd_text))
    if not jd_set:
        return 0.0
    return len(cv_set & jd_set) / len(jd_set)


# ── PUBLIC API ─────────────────────────────────────────────

def match_cv_to_jd(cv_text: str, jd_text: str,
                   skill_match_pct: float = 0.0) -> dict:
    """
    skill_match_pct: pass in gap['match_percentage'] from compute_skill_gap.
    This is the most reliable signal so it gets highest weight.

    Weights:
      50% skill match  (extracted skills vs JD skills)
      25% keyword overlap
      15% sklearn cosine
      10% scratch cosine
    """
    scratch  = tfidf_cosine_scratch(cv_text, jd_text)
    skl      = tfidf_cosine_sklearn(cv_text, jd_text)
    overlap  = keyword_overlap(cv_text, jd_text)
    skill_f  = skill_match_pct / 100.0   # convert % to 0-1

    blended = (0.50 * skill_f) + (0.25 * overlap) + (0.15 * skl) + (0.10 * scratch)
    final   = min(round(blended * 100, 2), 99.0)

    # top terms for display
    cv_tok = preprocess(cv_text)
    jd_tok = preprocess(jd_text)
    idf    = compute_idf([cv_tok, jd_tok])
    cv_vec = compute_tfidf_vector(compute_tf(cv_tok), idf)
    jd_vec = compute_tfidf_vector(compute_tf(jd_tok), idf)

    return {
        "match_score":   final,
        "tfidf_scratch": round(scratch * 100, 2),
        "tfidf_sklearn": round(skl * 100, 2),
        "overlap_score": round(overlap * 100, 2),
        "skill_match":   round(skill_match_pct, 2),
        "cv_top_terms":  sorted(cv_vec, key=cv_vec.get, reverse=True)[:10],
        "jd_top_terms":  sorted(jd_vec, key=jd_vec.get, reverse=True)[:10],
    }


def compute_skill_gap(cv_skills: list, jd_text: str) -> dict:
    from extractor import SKILLS_VOCAB
    jd_lower    = jd_text.lower()
    jd_required = [s for s in SKILLS_VOCAB if s in jd_lower]
    cv_lower    = [s.lower() for s in cv_skills]
    matched     = [s for s in jd_required if s in cv_lower]
    missing     = [s for s in jd_required if s not in cv_lower]
    return {
        "required_skills":  jd_required,
        "matched_skills":   matched,
        "missing_skills":   missing,
        "match_percentage": round(len(matched) / len(jd_required) * 100, 2)
                            if jd_required else 0.0,
    }