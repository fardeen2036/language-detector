from flask import Flask, render_template, request
import fasttext

app = Flask(__name__)

# Load fastText model
model = fasttext.load_model("lid.176.bin")

LANG_MAP = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan",
    "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian",
    "id": "Indonesian", "it": "Italian", "ja": "Japanese", "kn": "Kannada", "ko": "Korean",
    "lt": "Lithuanian", "lv": "Latvian", "ml": "Malayalam", "mr": "Marathi", "ms": "Malay",
    "nl": "Dutch", "no": "Norwegian", "pa": "Punjabi", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese"
}

def detect_language(text):
    label, prob = model.predict(text.replace("\n", " "), k=1)
    code = label[0].replace("__label__", "")
    return LANG_MAP.get(code, code), prob[0]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    if request.method == "POST":
        text = request.form["text"]
        prediction, confidence = detect_language(text)
        confidence = round(confidence * 100, 2)
    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
