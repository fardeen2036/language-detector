from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

def preprocess(text):
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text.lower())
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        text = request.form["text"]
        vec = vectorizer.transform([preprocess(text)])
        pred = model.predict(vec)
        code = le.inverse_transform(pred)[0]
        LANG_MAP = {
            "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek",
            "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
            "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
            "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai",
            "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese"
        }

        prediction = LANG_MAP.get(code, code)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
