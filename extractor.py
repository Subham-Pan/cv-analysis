import re
import pdfplumber

# ─────────────────────────────────────────────
#  SKILLS VOCABULARY  (extend this list freely)
# ─────────────────────────────────────────────
SKILLS_VOCAB = [
    # Programming
    "python", "java", "c++", "c", "javascript", "typescript", "r", "scala",
    "golang", "ruby", "php", "swift", "kotlin", "matlab",
    # ML / AI
    "machine learning", "deep learning", "neural network", "nlp",
    "computer vision", "reinforcement learning", "tensorflow", "pytorch",
    "keras", "scikit-learn", "xgboost", "opencv",
    # Data
    "sql", "mysql", "postgresql", "mongodb", "pandas", "numpy",
    "data analysis", "data visualization", "tableau", "power bi",
    "big data", "hadoop", "spark", "etl",
    # Web
    "html", "css", "react", "angular", "vue", "node.js", "django",
    "flask", "fastapi", "rest api", "graphql",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "git",
    "linux", "bash",
    # Soft skills
    "communication", "leadership", "teamwork", "problem solving",
    "project management", "agile", "scrum",
]


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_email(text: str) -> str:
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = re.search(r"(\+?\d[\d\s\-]{8,}\d)", text)
    return match.group(0).strip() if match else ""


def extract_name(text: str) -> str:
    """
    Extracts candidate name from CV text.

    Problem: Two-column PDFs merge the first line into one long string like:
    'Saptarshi Das SAPTARSHI DAS | 25MA60R07 Contact No.: +91...'

    Strategy 1: Match title-case name at very start of text (handles merged lines)
    Strategy 2: Scan lines, strip pipe-separated roll numbers, check for clean name
    """
    BAD_WORDS = {
        "education", "skills", "experience", "summary", "objective",
        "institute", "university", "college", "school", "tech",
        "computer", "science", "engineering", "management", "contact",
        "processing", "information", "technology", "data", "mtech",
    }

    # Strategy 1: first 300 chars, find title-case name before any noise
    head = text[:300].replace("\n", " ").strip()
    m = re.match(r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})", head)
    if m:
        candidate = m.group(1).strip()
        if not any(w in candidate.lower() for w in BAD_WORDS):
            return candidate

    # Strategy 2: scan lines
    for line in text.split("\n"):
        line = line.strip()
        # strip pipe-separated suffixes e.g. "SAPTARSHI DAS | 25MA60R07"
        line = re.split(r"\|", line)[0].strip()
        # convert ALL CAPS name to Title Case
        if line.isupper() and len(line.split()) <= 4:
            line = line.title()
        words = line.split()
        if 2 <= len(words) <= 4:
            if not re.search(r"[@/\\.\\d:]", line):
                if not any(w in line.lower() for w in BAD_WORDS):
                    if any(w[0].isupper() for w in words if w):
                        return line
    return ""


def extract_skills(text: str) -> list:
    text_lower = text.lower()
    found = []
    for skill in SKILLS_VOCAB:
        if skill in text_lower:
            found.append(skill)
    return found


def extract_education(text: str) -> list:
    degree_keywords = ["b.tech", "b.e.", "m.tech", "mba", "b.sc", "m.sc",
                       "bachelor", "master", "phd", "diploma", "b.com",
                       "isc", "icse", "cbse", "hsc", "ssc", "10th", "12th"]
    institute_keywords = ["iit", "nit", "university", "college", "institute"]
    skip_patterns = ["year", "degree/exam", "cgpa/marks", "class representative",
                     "represented", "conducted by", "scholarship", "medalist",
                     "certificate", "secured", "recipient", "analytics", "algebra"]
    lines = text.split("\n")
    edu = []
    for line in lines:
        line_clean = line.strip()
        line_lower = line_clean.lower()
        # must contain a degree keyword OR (institute keyword + a year)
        has_degree    = any(kw in line_lower for kw in degree_keywords)
        has_institute = any(kw in line_lower for kw in institute_keywords)
        has_year      = bool(re.search(r"\b(19|20)\d{2}\b", line_clean))
        if (has_degree or (has_institute and has_year)):
            if not any(sk in line_lower for sk in skip_patterns):
                if line_clean:
                    edu.append(line_clean)
    return edu


def extract_experience_years(text: str) -> float:
    """Try to pull an explicit years-of-experience number from the text."""
    match = re.search(r"(\d+\.?\d*)\s*(years?|yrs?)\s*(of)?\s*(experience)?",
                      text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def parse_cv(path: str) -> dict:
    """Master function: given a path to a PDF or TXT, return structured info."""
    if path.endswith(".pdf"):
        text = extract_text_from_pdf(path)
    else:
        text = extract_text_from_txt(path)

    return {
        "raw_text": text,
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "education": extract_education(text),
        "experience_years": extract_experience_years(text),
    }