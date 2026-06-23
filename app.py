
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    confidence = ""

    if request.method == "POST":

        review = request.form["review"]

        transformed_review = vectorizer.transform([review])

        sentiment = model.predict(transformed_review)[0]

        probability = max(
            model.predict_proba(transformed_review)[0]
        ) * 100

        if sentiment == "Positive":
            prediction = "😊 Positive"

        elif sentiment == "Negative":
            prediction = "😞 Negative"

        else:
            prediction = "😐 Neutral"

        confidence = f"{probability:.2f}%"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)

