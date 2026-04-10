"""
CV Job Category Classifier
- Uses UpdatedResumeDataSet.csv (real resumes)
- Cross-validation to show honest accuracy (not just one lucky split)
- SVM vs others with train/test gap shown
"""

import os, pickle, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

MODEL_PATH = "saved_model/svm_model.pkl"
TFIDF_PATH = "saved_model/tfidf_vectorizer.pkl"
LABEL_PATH = "saved_model/label_encoder.pkl"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def plot_confusion_matrix(y_test, preds, labels,
                          save_path="saved_model/confusion_matrix.png"):
    cm = confusion_matrix(y_test, preds)
    if len(labels) > 15:
        from collections import Counter
        top_ids = sorted([i for i, _ in Counter(list(y_test)).most_common(15)])
        cm = cm[np.ix_(top_ids, top_ids)]
        labels = [labels[i] for i in top_ids]
        print("  (Showing top 15 categories)")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("SVM Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs("saved_model", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_model_comparison(results: dict,
                          save_path="saved_model/model_comparison.png"):
    """
    Bar chart showing CV mean accuracy ± std for each model.
    This is more honest than a single train/test split.
    """
    names  = list(results.keys())
    means  = [results[n]["cv_mean"] for n in names]
    stds   = [results[n]["cv_std"]  for n in names]
    colors = ["#4CAF50" if m == max(means) else "#2196F3" for m in means]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, means, xerr=stds, color=colors,
                   error_kw=dict(ecolor="black", capsize=5))
    ax.set_xlabel("5-Fold CV Accuracy (%)")
    ax.set_title("Model Comparison — 5-Fold Cross Validation")
    ax.set_xlim(0, 110)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_width() + std + 1,
                bar.get_y() + bar.get_height()/2,
                f"{mean:.1f}±{std:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def train(csv_path: str = "data/UpdatedResumeDataSet.csv"):
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    df = df[["Category", "Resume"]].dropna()
    df["Category"] = df["Category"].astype(str).str.strip()
    df["clean"]    = df["Resume"].apply(clean_text)

    # keep only categories with enough samples for 5-fold CV (>=10)
    counts = df["Category"].value_counts()
    valid  = counts[counts >= 10].index
    df     = df[df["Category"].isin(valid)]
    print(f"  {len(df)} samples across {df['Category'].nunique()} categories")

    le = LabelEncoder()
    y  = le.fit_transform(df["Category"])
    X  = np.array(df["clean"].tolist())

    # hold out a final test set — never seen during CV
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  |  Final test: {len(X_test)}")

    tfidf = TfidfVectorizer(
        max_features=15000, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, strip_accents="unicode"
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    # ── 5-Fold Cross Validation on training set ───────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    raw_models = {
        "Naive Bayes":         MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "SVM (Linear)":        LinearSVC(C=0.5, max_iter=3000),
        "Random Forest":       RandomForestClassifier(n_estimators=100,
                                                      max_depth=15,
                                                      random_state=42),
    }

    print("\n─── 5-Fold CV on Training Set ────────────────────────")
    print(f"  {'Model':<25} {'CV Mean':>8}  {'CV Std':>7}  {'Final Test':>10}")
    print("  " + "─"*58)

    results = {}
    trained_models = {}
    for name, model in raw_models.items():
        cv_scores = cross_val_score(model, X_train_vec, y_train,
                                    cv=cv, scoring="accuracy", n_jobs=-1)
        model.fit(X_train_vec, y_train)
        test_preds = model.predict(X_test_vec)
        test_acc   = accuracy_score(y_test, test_preds)

        results[name] = {
            "cv_mean":  round(cv_scores.mean() * 100, 2),
            "cv_std":   round(cv_scores.std()  * 100, 2),
            "test_acc": round(test_acc * 100, 2),
        }
        trained_models[name] = (model, test_preds)

        print(f"  {name:<25} "
              f"{cv_scores.mean()*100:7.2f}%  "
              f"±{cv_scores.std()*100:5.2f}%  "
              f"{test_acc*100:9.2f}%")

    print("─────────────────────────────────────────────────────")
    best = max(results, key=lambda n: results[n]["cv_mean"])
    print(f"  Best: {best}  "
          f"(CV: {results[best]['cv_mean']}% ± {results[best]['cv_std']}%,  "
          f"Test: {results[best]['test_acc']}%)\n")

    svm_model, svm_preds = trained_models["SVM (Linear)"]

    print("─── SVM Classification Report ────────────────────────")
    report = classification_report(y_test, svm_preds,
                                   target_names=le.classes_, zero_division=0)
    print(report)

    os.makedirs("saved_model", exist_ok=True)
    with open("saved_model/classification_report.txt", "w") as f:
        f.write("SVM Classification Report\n" + "="*60 + "\n" + report)
        f.write("\n5-Fold CV Results\n" + "="*60 + "\n")
        f.write(f"{'Model':<25} {'CV Mean':>8}  {'CV Std':>7}  {'Test':>7}\n")
        for name in results:
            r = results[name]
            f.write(f"{name:<25} {r['cv_mean']:7.2f}%  ±{r['cv_std']:5.2f}%  {r['test_acc']:6.2f}%\n")
    print("  Saved → saved_model/classification_report.txt")

    plot_confusion_matrix(y_test, svm_preds, list(le.classes_))
    plot_model_comparison(results)

    with open(MODEL_PATH, "wb") as f: pickle.dump(svm_model, f)
    with open(TFIDF_PATH, "wb") as f: pickle.dump(tfidf, f)
    with open(LABEL_PATH, "wb") as f: pickle.dump(le, f)
    print("Model saved.\n")

    return {n: results[n]["cv_mean"] for n in results}


def load_model():
    with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f: tfidf = pickle.load(f)
    with open(LABEL_PATH, "rb") as f: le    = pickle.load(f)
    return model, tfidf, le


def predict_category(cv_text: str) -> str:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run: python train.py")
    model, tfidf, le = load_model()
    vec = tfidf.transform([clean_text(cv_text)])
    return le.inverse_transform([model.predict(vec)[0]])[0]