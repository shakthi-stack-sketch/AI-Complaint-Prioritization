# AI Complaint Classification & Priority Prediction System

> **Academic Prototype Notice**: This system is designed as an academic **Text Analysis + Machine Learning & NLP** research project. It uses supervised machine learning models trained via `scikit-learn` for complaint category classification and priority prediction. It operates **100% locally** without using any external generative AI APIs (e.g. OpenAI, Gemini, Claude, RAG, or LangChain).

---

## 📌 Problem Statement

In large educational campuses and municipal facilities, handling natural language complaints manually leads to routing delays, improper prioritization, and overlooked high-hazard emergencies. Traditional keyword-matching rules fail to handle complex sentence structures, leading to operational bottlenecks.

This project solves this challenge by implementing an automated **Complaint Operations Center** powered by supervised Machine Learning (TF-IDF + Logistic Regression & Random Forest) and lightweight Natural Language Processing (NLP).

---

## 🎯 Objectives

1. **Automated Category Classification**: Classify natural language complaints into 10 operational domains (Electrical, Plumbing, Infrastructure, IT/Technical, Security, Cleaning, Transportation, Academic, Hostel, Other).
2. **Priority & Severity Prediction**: Predict ticket urgency (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) using supervised Machine Learning supplemented with hazard safety protocols.
3. **Information & Entity Extraction**: Automatically extract location, target equipment, identified problem phrase, and time references using lightweight NLP regex patterns.
4. **Smart Department Routing**: Dynamically route complaints to the appropriate facility division with an automated recommended work order action.
5. **Modern Operations Center UI**: Provide a non-chatbot work order management portal with real-time analytics, filtering, queue controls, and voice speech-to-text input.

---

## 🚀 System Architecture & Methodology

```text
Natural Language Text Input
       │
       ▼
Text Preprocessing (Lowercasing, Regex Cleaning, Tokenization)
       │
       ▼
TF-IDF Vectorization (Unigrams & Bigrams, 2,500 N-Grams)
       │
  ┌────┴──────────────────────────┐
  ▼                               ▼
Category Classifier              Priority Predictor
(Logistic Regression, C=2.0)     (Random Forest Classifier)
  │                               │
  ▼                               ▼
Predicted Category (84.4% Acc)   Priority & Severity (LOW..CRITICAL)
  │                               │
  └────┬──────────────────────────┘
       ▼
NLP Entity Extractor (Location, Equipment, Problem, Time)
       │
       ▼
Automated Work Order & Department Routing Engine
```

### 1. Machine Learning Pipeline (`train_model.py`)
- **Dataset**: Built from 383 facility and academic complaints spanning 10 distinct categories.
- **Feature Extraction**: `TfidfVectorizer` with `sublinear_tf=True`, n-gram range `(1, 2)`, and max features of `2,500`.
- **Category Model**: `LogisticRegression` tuned for multi-class prediction with probabilistic confidence calculation (`cat_model.predict_proba`).
- **Priority Model**: `RandomForestClassifier` with 100 decision trees to evaluate priority feature combinations.
- **Empirical Accuracy**:
  - Category Classifier Accuracy: **84.42%**
  - Priority Predictor Accuracy: **72.73%**

### 2. Information Extraction (NLP)
- **Location Extraction**: Uses regex patterns to identify block names ("Block B"), room numbers ("Room 302", "Classroom 204"), floors ("3rd Floor"), and outdoor landmarks ("Auditorium", "Library").
- **Equipment & Problem Identification**: Maps key facility objects (fans, projectors, geysers, routers) and problem states ("continuous leak", "short circuit", "broken lock").
- **Reasoning Signals**: Extracts top TF-IDF n-gram tokens for clear explainability.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.9+, Flask, Flask-Cors
- **Machine Learning**: `scikit-learn`, `pandas`, `numpy`, `joblib`
- **Frontend**: HTML5, CSS3 (Modern Operations Theme), JavaScript (ES6+)
- **Audio Input**: Browser Web Speech API (`webkitSpeechRecognition`)

---

## 📂 Project Structure

```text
AI-Complaint-Prioritization/
│
├── app.py                   # Flask server providing Web UI and REST API
├── train_model.py           # Machine learning training and evaluation script
├── analyzer.py              # ComplaintAnalyzer class (preprocessing, NLP, ML inference)
├── requirements.txt         # Python package dependencies
├── README.md                # Project documentation
│
├── data/
│   └── complaints.csv       # Synthetic multi-category complaint dataset (383 rows)
│
├── models/
│   ├── category_model.pkl   # Trained scikit-learn Category Logistic Regression model
│   ├── priority_model.pkl   # Trained scikit-learn Priority Random Forest model
│   ├── tfidf_vectorizer.pkl # Fitted TF-IDF Vectorizer
│   └── metrics.json         # Evaluated metrics, confusion stats, and top TF-IDF n-grams
│
├── templates/
│   └── index.html           # Operations Center dashboard template
│
└── static/
    ├── style.css            # Deep Navy & Electric Violet Operations CSS styling
    └── script.js            # Interactive controller, API handlers, queue state & speech
```

---

## 💻 Installation & Usage

### 1. Prerequisites
Ensure Python 3.9+ is installed.

### 2. Clone repository & install dependencies
```bash
git clone https://github.com/your-username/AI-Complaint-Prioritization.git
cd AI-Complaint-Prioritization
pip install -r requirements.txt
```

### 3. Train the Machine Learning Models
Execute `train_model.py` to generate the dataset, train models, calculate accuracy metrics, and save model artifacts:

```bash
python train_model.py
```

*Output:*
```text
Dataset generated and saved to data\complaints.csv with 383 records.
--- Category Classifier Accuracy: 84.42% ---
--- Priority Classifier Accuracy: 72.73% ---
Models and metrics successfully saved to 'models/' folder.
```

### 4. Run the Web Application
Start the Flask web server:

```bash
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🧪 Test Case Examples

| Complaint Input Text | Predicted Category | Priority | Extracted Location | Target Department |
| :--- | :--- | :--- | :--- | :--- |
| *"Water is leaking continuously from the ceiling of Block B classroom 204."* | Plumbing | **HIGH** | Block B Classroom 204 | Maintenance & Plumbing |
| *"Sparks flying from the main switchboard near Room 301, major electrical hazard!"* | Electrical | **CRITICAL** | Room 301 | Electrical Works |
| *"The classroom projector is not working in Room 302."* | IT / Technical | **MEDIUM** | Room 302 | IT Support Desk |
| *"Wi-Fi network signal dropping constantly in Hostel 3 wing B since morning."* | IT / Technical | **HIGH** | Hostel 3 Wing B | IT Support Desk |
| *"Hostel room 304 door lock jammed, student trapped inside room!"* | Hostel | **CRITICAL** | Hostel Room 304 | Hostel Management |
| *"Inquiry regarding lost student wallet containing ID cards found near main quad."* | Other | **LOW** | Unspecified | Administrative Services |

---

## ⚠️ Limitations & Future Enhancements

- **Current Limitations**:
  - The dataset currently contains 383 training instances tailored for educational/facility environments.
  - Entity extraction relies on rule-based regular expressions and keyword parsing rather than full Named Entity Recognition (NER) models (such as spaCy).
- **Future Enhancements**:
  - Fine-tuning a lightweight domain-specific NER model for advanced location parsing.
  - Adding multi-lingual speech recognition support for regional languages.
  - Implementing persistent SQLite/PostgreSQL storage for historical work order tracking.

---

## 📜 Academic License
Developed for academic code evaluation and machine learning demonstration purposes.
