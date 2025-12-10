import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string

# Load dataset
data = pd.read_csv("spam.csv", header=0)
data = data.dropna(subset=['message'])

# Preprocessing function
def preprocess(text):
    text = text.lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text

# Apply preprocessing
data['message'] = data['message'].apply(preprocess)

# Prepare features and labels
X = data['message']
y = data['label']

# Vectorize text with n-grams (1,2)
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))

# Function to predict new emails
def predict_email(text):
    text = preprocess(text)
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

# Interactive loop
print("\n=== Spam Email Detector ===")
while True:
    email = input("Enter an email (or type 'exit' to quit): ")
    if email.lower() == "exit":
        print("Exiting Spam Detector.")
        break
    result = predict_email(email)
    print("Prediction:", result, "\n")
