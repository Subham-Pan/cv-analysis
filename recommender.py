"""
Skill Gap Recommendations
- Rule-based recommendations always work (no API needed)
- Claude API enriches recommendations if ANTHROPIC_API_KEY is set
"""

import os
import json


# ─────────────────────────────────────────────
#  RULE-BASED RECOMMENDATIONS  (always available)
# ─────────────────────────────────────────────

RECOMMENDATIONS_DB = {
    "python":            ["Python for Everybody – Coursera", "Automate the Boring Stuff (free book)"],
    "machine learning":  ["Andrew Ng ML Course – Coursera", "Hands-On ML with Scikit-Learn (book)"],
    "deep learning":     ["fast.ai (free)", "Deep Learning Specialization – Coursera"],
    "sql":               ["SQLZoo (free)", "Mode SQL Tutorial (free)"],
    "tensorflow":        ["TensorFlow Developer Certificate – Coursera"],
    "pytorch":           ["PyTorch official tutorials (free)"],
    "docker":            ["Docker Getting Started (official docs)", "TechWorld with Nana – YouTube"],
    "aws":               ["AWS Cloud Practitioner – free tier", "A Cloud Guru"],
    "react":             ["React official docs", "Full Stack Open – University of Helsinki (free)"],
    "data analysis":     ["Google Data Analytics Certificate – Coursera"],
    "nlp":               ["HuggingFace NLP Course (free)", "Stanford CS224N (free lectures)"],
    "computer vision":   ["CS231n Stanford (free)", "PyImageSearch blog"],
    "git":               ["Pro Git Book (free)", "GitHub Learning Lab"],
    "communication":     ["Toastmasters", "Coursera: Successful Negotiation"],
    "agile":             ["Scrum.org free resources", "Agile Manifesto + Google Coursera cert"],
}

DEFAULT_REC = ["Search '{skill}' on Coursera / edX / YouTube for free resources"]


def rule_based_recommendations(missing_skills: list) -> dict:
    recs = {}
    for skill in missing_skills:
        recs[skill] = RECOMMENDATIONS_DB.get(
            skill.lower(),
            [r.format(skill=skill) for r in DEFAULT_REC]
        )
    return recs


# ─────────────────────────────────────────────
#  API-BASED RECOMMENDATIONS  (optional enrichment)
# ─────────────────────────────────────────────

def api_recommendations(missing_skills: list, job_category: str) -> str:
    """
    Call Claude API to get richer, personalized recommendations.
    Returns empty string if API key is not set or call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        prompt = (
            f"A candidate is applying for a {job_category} role. "
            f"They are missing these skills: {', '.join(missing_skills)}. "
            f"For each missing skill, suggest 1-2 specific free online resources "
            f"or actionable steps to learn it quickly. Be concise."
        )

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    except Exception as e:
        print(f"[API] Recommendation API unavailable: {e}")
        return ""


# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────

def get_recommendations(missing_skills: list, job_category: str = "") -> dict:
    """
    Always returns rule-based recommendations.
    Appends API-enriched text if API key is available.
    """
    result = {
        "rule_based": rule_based_recommendations(missing_skills),
        "api_enriched": ""
    }

    if missing_skills:
        api_text = api_recommendations(missing_skills, job_category)
        if api_text:
            result["api_enriched"] = api_text

    return result
