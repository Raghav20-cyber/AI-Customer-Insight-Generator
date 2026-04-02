# AI-Powered Customer Complaint Intelligence System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![Ollama](https://img.shields.io/badge/LLM-Llama3%20via%20Ollama-black?style=flat-square)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange?style=flat-square&logo=scikit-learn)

An end-to-end system that uses Machine Learning and Generative AI to automatically classify, summarize, and extract insights from customer support complaints — reducing manual effort and helping teams act faster on support data.

---

## Problem

Customer support teams receive thousands of unstructured complaints daily. Manually reviewing them is slow, inconsistent, and makes it hard to spot recurring issues before they escalate.

Key pain points:
- No easy way to identify patterns across large volumes of tickets
- Repetitive issues go unnoticed without structured categorization
- Decision-making is delayed because insights aren't surfaced automatically

---

## Solution

This system combines traditional NLP with a local LLM to handle the full pipeline — from raw complaint text to actionable insight — with a simple web UI anyone on the team can use.

**What it does:**
- Classifies each complaint into a category (billing, technical, refund, etc.)
- Generates a one-line AI summary of each complaint
- Visualizes the distribution of complaint types across a dataset
- Produces dataset-level insights powered by Llama3

---

## Approach

### 1. Text Preprocessing (NLTK)
Raw complaint text is cleaned before any model sees it:
- Lowercasing and punctuation removal
- Stopword removal
- Lemmatization to normalize word forms

### 2. Classification (TF-IDF + Logistic Regression)
- Complaint text is vectorized using **TF-IDF**
- A **Logistic Regression** classifier predicts the complaint category
- Rule-based overrides handle edge cases and improve accuracy on common patterns

### 3. AI Summarization & Insights (Llama3 via Ollama)
- Each complaint is passed to Llama3 for a concise one-line summary
- For dataset-level analysis, the model generates structured insights across all tickets

### 4. UI (Streamlit)
A lightweight web app lets users:
- Input a single complaint and see classification + summary instantly
- Upload a dataset and view visualizations and AI-generated insights

---

## Features

| Feature | Description |
|---|---|
| Complaint Classification | Categorizes tickets into billing, technical, refund, etc. |
| AI Summary | One-line LLM-generated summary per complaint |
| Dataset Visualization | Bar chart of ticket type distribution |
| AI Insights | Dataset-level patterns surfaced by Llama3 |
| Simple UI | Streamlit app — no setup needed for end users |

---

## Example

**Input complaint:**
> "I have been charged twice for the same subscription and nobody from support has responded in 5 days."

**Output:**
- **Category:** Billing
- **Summary:** Customer was double-charged for a subscription and has not received a support response after 5 days.
- **Insight (dataset level):** Billing complaints account for 34% of tickets this week, with response delay being the most common sub-issue.

---

## Tech Stack

- **Language:** Python 3.10+
- **NLP:** NLTK (tokenization, stopwords, lemmatization)
- **ML:** scikit-learn (TF-IDF, Logistic Regression)
- **LLM:** Llama3 via [Ollama](https://ollama.com/)
- **UI:** Streamlit
- **Visualization:** Matplotlib / Seaborn

---

## Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

### 1. Clone the repository
```bash
git clone https://github.com/Raghav20-cyber/AI-Customer-Complaint-Intelligence.git
cd AI-Customer-Complaint-Intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull the Llama3 model
```bash
ollama pull llama3
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## Project Structure
```
├── app.py                  # Streamlit UI
├── classifier.py           # TF-IDF + Logistic Regression model
├── summarizer.py           # Llama3 summarization via Ollama
├── preprocessor.py         # NLTK text cleaning pipeline
├── insights.py             # Dataset-level insight generation
├── data/
│   └── complaints.csv      # Sample dataset
├── requirements.txt
└── README.md
```

---

## Limitations & Future Work

- Currently runs Llama3 locally — cloud LLM support (OpenAI, Gemini) can be added
- Classification accuracy depends on training data quality and size
- Planned: confidence scores per classification, multi-label support, export to CSV
