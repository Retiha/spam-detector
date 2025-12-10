# Spam Email Detector

## Project Overview
The Spam Email Detector is a Python machine learning project that classifies emails as **spam** or **ham (legitimate email)**. It uses a **Naive Bayes classifier** along with **TF-IDF vectorization** to analyze email content and make predictions. The project also includes an **interactive terminal interface** for real-time email classification.

---

## Features
- Preprocessing of text data: lowercase conversion, punctuation removal.
- TF-IDF vectorization with n-grams (1,2) for better feature extraction.
- Train-test split and evaluation using accuracy metrics.
- Interactive terminal interface to test custom emails.
- Works with realistic email messages.

---

## Dataset
- The dataset is stored in `spam.csv` and contains emails labeled as `spam` or `ham`.
- Example rows:

| label | message |
|-------|---------|
| ham   | Hey how are you |
| spam  | Congratulations! You won a prize |
| ham   | Are we meeting tomorrow |
| spam  | Claim your free gift now |

---

## How to Run
1. Make sure you have Python installed (Python 3.8+ recommended).
2. Install dependencies:

```bash
pip install pandas scikit-learn
