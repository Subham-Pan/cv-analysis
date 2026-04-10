"""
CV Analysis Tool — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import os, tempfile
import pandas as pd

from extractor import parse_cv
from matcher import match_cv_to_jd, compute_skill_gap
from classifier import predict_category
from recommender import get_recommendations

# ── Page config ──────────────────────────────
st.set_page_config(page_title="CV Analyser", page_icon="📄", layout="wide")
st.title("📄 CV Analysis System")
st.caption("NLP + SVM Pipeline | IIT Kharagpur AI/ML Project")

# ── Load JD dataset (resume_data.csv) ────────
@st.cache_data
def load_jd_data(path="data/resume_data.csv"):
    """Load job titles and their required skills from resume_data.csv."""
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
        # find label col
        label_col = next((c for c in df.columns
                          if "job_position" in c.lstrip("\ufeff").lower()), None)
        if label_col is None:
            return None, None
        # clean label
        df[label_col] = df[label_col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        # build job → skills_required mapping (take first row per job)
        jd_map = {}
        for job in df[label_col].unique():
            rows = df[df[label_col] == job]
            # combine skills_required + responsibilities as JD text
            parts = []
            for col in ["skills", "skills_required", "responsibilities.1",
                        "educationaL_requirements", "experiencere_requirement",
                        "related_skils_in_job"]:
                if col in df.columns:
                    val = str(rows[col].iloc[0])
                    if val.strip() not in ("nan", ""):
                        parts.append(val)
            jd_map[job] = " ".join(parts)
        return df, jd_map
    except Exception as e:
        st.warning(f"Could not load resume_data.csv: {e}")
        return None, None


df_jd, jd_map = load_jd_data()

# ── Sidebar ───────────────────────────────────
st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("Anthropic API Key (optional)",
                                 type="password",
                                 help="Leave blank — rule-based recommendations work without it")
if api_key:
    os.environ["ANTHROPIC_API_KEY"] = api_key

st.sidebar.markdown("---")
st.sidebar.markdown("**How to use:**")
st.sidebar.markdown("1. Upload your CV (PDF or TXT)")
st.sidebar.markdown("2. Select a job title OR paste a custom JD")
st.sidebar.markdown("3. Click Analyse")

# ── Main inputs ───────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📎 Upload CV")
    uploaded_file = st.file_uploader("PDF or TXT", type=["pdf", "txt"])

with col2:
    st.subheader("💼 Job Description")

    jd_mode = st.radio("Choose JD source:", ["Select Job Title", "Paste Custom JD"],
                        horizontal=True)

    jd_text = ""
    selected_job = None

    if jd_mode == "Select Job Title" and jd_map:
        job_titles = sorted(jd_map.keys())
        selected_job = st.selectbox("Select a job position:", job_titles)
        jd_text = jd_map.get(selected_job, "")
        if jd_text:
            with st.expander("📋 View loaded JD requirements"):
                st.write(jd_text[:800] + ("..." if len(jd_text) > 800 else ""))
    else:
        jd_text = st.text_area("Paste the job description here", height=180,
                                placeholder="e.g. Looking for a Software Engineer with Python, SQL, machine learning...")

analyse_btn = st.button("🔍 Analyse CV", use_container_width=True, type="primary")

# ── Analysis ─────────────────────────────────
if analyse_btn:
    if not uploaded_file:
        st.error("Please upload a CV first.")
    elif not jd_text.strip():
        st.error("Please select a job title or paste a job description.")
    else:
        with st.spinner("Analysing your CV..."):
            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            cv_data  = parse_cv(tmp_path)
            gap      = compute_skill_gap(cv_data["skills"], jd_text)
            match    = match_cv_to_jd(cv_data["raw_text"], jd_text,
                                      skill_match_pct=gap["match_percentage"])

            try:
                category = predict_category(cv_data["raw_text"])
            except FileNotFoundError:
                category = "⚠️ Run python train.py first"

            recs = get_recommendations(gap["missing_skills"], category)
            os.unlink(tmp_path)

        # ── Results ───────────────────────────
        st.markdown("---")
        st.subheader("📊 Results")

        info_col, score_col = st.columns([2, 1])

        with info_col:
            st.markdown("### 👤 Candidate Info")
            st.write(f"**Name:** {cv_data['name'] or 'Not detected'}")
            st.write(f"**Email:** {cv_data['email'] or 'Not detected'}")
            st.write(f"**Phone:** {cv_data['phone'] or 'Not detected'}")
            st.write(f"**Experience:** {cv_data['experience_years']} years")
            st.write(f"**Predicted Category (SVM):** `{category}`")
            if selected_job:
                st.write(f"**Selected Job:** `{selected_job}`")

        with score_col:
            st.markdown("### 🎯 Match Score")
            score = match["match_score"]
            color = "green" if score >= 60 else ("orange" if score >= 40 else "red")
            st.markdown(
                f"<h1 style='color:{color}; text-align:center'>{score}%</h1>",
                unsafe_allow_html=True
            )
            st.caption("Score breakdown:")
            st.write(f"🎯 Skill match:      `{match['skill_match']}%`")
            st.write(f"🔑 Keyword overlap:  `{match['overlap_score']}%`")
            st.write(f"⚙️ TF-IDF (sklearn): `{match['tfidf_sklearn']}%`")
            st.write(f"📐 TF-IDF (scratch): `{match['tfidf_scratch']}%`")

        st.markdown("---")

        sk1, sk2, sk3 = st.columns(3)
        with sk1:
            st.markdown("### ✅ CV Skills Found")
            if cv_data["skills"]:
                for s in cv_data["skills"]:
                    st.success(s)
            else:
                st.info("No skills detected")

        with sk2:
            st.markdown("### ✔️ Matched Skills")
            if gap["matched_skills"]:
                for s in gap["matched_skills"]:
                    st.success(s)
            else:
                st.info("No matched skills")

        with sk3:
            st.markdown("### ❌ Missing Skills")
            if gap["missing_skills"]:
                for s in gap["missing_skills"]:
                    st.error(s)
            else:
                st.success("No skill gaps!")

        st.markdown("---")

        if cv_data["education"]:
            st.markdown("### 🎓 Education")
            for edu in cv_data["education"]:
                st.write(f"• {edu}")
            st.markdown("---")

        st.markdown("### 📚 Learning Recommendations")
        if gap["missing_skills"]:
            if recs.get("api_enriched"):
                st.info("✨ AI-enriched recommendations:")
                st.write(recs["api_enriched"])
                st.markdown("---")
            for skill, resources in recs["rule_based"].items():
                with st.expander(f"📌 {skill}"):
                    for r in resources:
                        st.write(f"• {r}")
        else:
            st.success("No skill gaps to address for this job!")

        # ── Top TF-IDF terms ──────────────────
        with st.expander("🔬 TF-IDF Analysis (top weighted terms)"):
            t1, t2 = st.columns(2)
            with t1:
                st.markdown("**CV top terms:**")
                for term in match["cv_top_terms"]:
                    st.write(f"• {term}")
            with t2:
                st.markdown("**JD top terms:**")
                for term in match["jd_top_terms"]:
                    st.write(f"• {term}")