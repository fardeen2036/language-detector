import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load data
data = pd.read_csv("dataset.csv", encoding="latin1")
data.columns = data.columns.str.strip()

# Rename if needed
if "sentence" in data.columns:
    data = data.rename(columns={"sentence": "Text"})

# Drop missing
data = data.dropna(subset=["Text", "language"])

# Sample 50% for speed
data = data.sample(frac=0.5, random_state=42)

# Preprocess (unicode safe)
def preprocess_text(text):
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", str(text).lower())
    return text

data["Text"] = data["Text"].apply(preprocess_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data["Text"], data["language"], test_size=0.2, random_state=42
)

# Vectorize (char ngrams)
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,5), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train_enc)

# Save artifacts
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("Training complete.")
