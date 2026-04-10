# CV Analysis System
### IIT Kharagpur — AI/ML Term Project
**Course:** Artificial Intelligence & Machine Learning
**Professors:** B. Roy & B. Banerjee



## Project Overview

An end-to-end NLP + ML pipeline that:
- Parses a CV (PDF or TXT) and extracts structured information
- Matches the CV against a Job Description using TF-IDF + Cosine Similarity
- Classifies the CV into a job category using a trained SVM model
- Identifies skill gaps and recommends learning resources
- Optionally enriches recommendations using the Claude API


## Project Structure

```
cv_analysis/
│
├── extractor.py      ← PDF/TXT parsing, regex-based info extraction (name, email, skills, education)
├── matcher.py        ← TF-IDF from scratch + sklearn TF-IDF + cosine similarity (blended score)
├── classifier.py     ← SVM classifier with 5-fold CV, trained on UpdatedResumeDataSet.csv
├── recommender.py    ← Skill gap recommendations (rule-based always + optional Claude API)
├── app.py            ← Streamlit UI
├── train.py          ← Run once to train and save the model
├── requirements.txt
│
├── data/
│   ├── UpdatedResumeDataSet.csv   ← Kaggle dataset: used for classifier training
│   └── resume_data.csv            ← 35-column dataset: used for job title dropdown + JD text
│
└── saved_model/                   ← Auto-created after running train.py
    ├── svm_model.pkl
    ├── tfidf_vectorizer.pkl
    ├── label_encoder.pkl
    ├── classification_report.txt
    ├── confusion_matrix.png
    └── model_comparison.png
```

---

## Dataset Usage

| Dataset | Rows | Used For |
|---|---|---|
| `UpdatedResumeDataSet.csv` | 962 real resumes, 25 categories | **Classifier training only** |
| `resume_data.csv` | 9544 rows, 35 columns | **Job title dropdown + JD text in app** |

**Why two datasets?**
- `UpdatedResumeDataSet.csv` contains authentic resumes with natural language — suitable for training a generalizable classifier.
- `resume_data.csv` contains structured job requirement data (skills, responsibilities, educational requirements) per job title — used to auto-populate the Job Description when a user selects a job title in the app.

---

## Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Get the Datasets

**Dataset 1 — Classifier training (required):**
1. Go to: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
2. Download `UpdatedResumeDataSet.csv`
3. Place it in the `data/` folder

**Dataset 2 — Job description source (required for dropdown):**
- Place your `resume_data.csv` (35-column structured dataset) in the `data/` folder

---

## Step 3 — Train the Model

```bash
python train.py
```

This will:
- Load `UpdatedResumeDataSet.csv` (962 resumes, 25 job categories)
- Split into 80% train (769) / 20% test (193)
- Run **5-fold cross-validation** on 4 models
- Save the best model (SVM) to `saved_model/`
- Save confusion matrix and model comparison chart

**Actual training output:**

```
─── 5-Fold CV on Training Set ────────────────────────
  Model                      CV Mean   CV Std  Final Test
  ──────────────────────────────────────────────────────
  Naive Bayes                 96.75%  ± 1.59%      97.93%
  Logistic Regression         99.22%  ± 0.49%      98.96%
  SVM (Linear)                99.48%  ± 0.26%      99.48%
  Random Forest               98.31%  ± 1.27%      99.48%
  ──────────────────────────────────────────────────────
  Best: SVM (Linear) (CV: 99.48% ± 0.26%, Test: 99.48%)
```

**Why is accuracy ~99%?**
Resume classification is a naturally high-accuracy NLP task because job domains have near-zero vocabulary overlap. A Java Developer resume contains `spring`, `hibernate`, `maven` — a Data Science resume contains `pandas`, `tensorflow`, `scikit-learn`. TF-IDF captures this domain-specific vocabulary perfectly. The low CV standard deviation (±0.26%) confirms the model is not overfitting — it generalizes consistently across all 5 folds.

---

## Step 4 — Run the App

```bash
streamlit run app.py
```

Opens at: https://cv-analysis-kpcukjabx4v3rzd5l7v4bm.streamlit.app

---

## Step 5 — Use the App

1. Upload a CV (PDF or TXT)
2. Choose JD source:
   - **Select Job Title** — picks from `resume_data.csv` job titles, auto-loads requirements
   - **Paste Custom JD** — paste any job description manually
3. Click **Analyse CV**

**Results shown:**
- Candidate name, email, phone, experience years
- Predicted job category (SVM classifier)
- Match score (blended: skill match + keyword overlap + TF-IDF cosine)
- Score breakdown showing all 4 signals
- Matched skills / missing skills
- Education entries
- Learning recommendations per missing skill
- TF-IDF top terms (expandable)

---

## Match Score — How It Works

The match score is a **weighted blend of 4 signals:**

| Signal | Weight | Method |
|---|---|---|
| Skill match | 50% | Extracted CV skills vs skills in JD (most reliable) |
| Keyword overlap | 25% | Token-level overlap after stopword removal |
| TF-IDF cosine (sklearn) | 15% | Optimized sparse matrix cosine similarity |
| TF-IDF cosine (scratch) | 10% | From-scratch implementation (shows the math) |

TF-IDF is implemented **twice** — once from scratch using only `math` and `collections` (in `matcher.py`) to demonstrate the algorithm, and once using sklearn's optimized sparse matrix version for better numerical performance.

---

## NLP/ML Concepts Used

| Component | Technique | File |
|---|---|---|
| Text extraction | pdfplumber | `extractor.py` |
| Preprocessing | tokenization, stopword removal, lowercasing, regex | `extractor.py`, `matcher.py` |
| Information extraction | regex (email, phone, name), vocabulary matching (skills) | `extractor.py` |
| Text representation | TF-IDF from scratch + sklearn TF-IDF | `matcher.py`, `classifier.py` |
| CV-JD matching | Cosine similarity (scratch + sklearn), keyword overlap | `matcher.py` |
| Job classification | SVM with linear kernel, 5-fold cross-validation | `classifier.py` |
| Model comparison | Naive Bayes, Logistic Regression, SVM, Random Forest | `classifier.py` |
| Skill gap detection | Set difference of extracted vs required skills | `matcher.py` |
| Recommendations | Rule-based resource dictionary + optional Claude API | `recommender.py` |

---

## Optional — Claude API Recommendations

If you have an Anthropic API key:
1. Paste it in the sidebar under **Settings → API Key**
2. Missing skill recommendations will be enriched by Claude AI

The app works fully **without** an API key — rule-based recommendations are always available.

**API cost:** negligible (~$0.001 per call). Free $5 credits available at console.anthropic.com.

---

## Saved Outputs (after training)

| File | Description |
|---|---|
| `saved_model/svm_model.pkl` | Trained SVM classifier |
| `saved_model/tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `saved_model/label_encoder.pkl` | Label encoder for job categories |
| `saved_model/classification_report.txt` | Full precision/recall/F1 per category |
| `saved_model/confusion_matrix.png` | SVM confusion matrix heatmap |
| `saved_model/model_comparison.png` | CV accuracy bar chart for all 4 models |

---

## Dataset Citations

- **Kaggle Resume Dataset** by Gaurav Dutta: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
- **resume_data.csv**: Structured job requirement dataset (35 columns, 9544 rows)
