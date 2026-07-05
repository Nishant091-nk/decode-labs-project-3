from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

df = pd.read_csv("dataset.csv")

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(df["Description"])

with open("index.html", "r", encoding="utf-8") as file:
    html = file.read()


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = ""

    if request.method == "POST":

        interest = request.form["interest"]

        user_vector = vectorizer.transform([interest])

        similarity = cosine_similarity(user_vector, tfidf)

        scores = similarity.flatten()

        top = scores.argsort()[::-1][:5]

        table = """
        <table>
        <tr>
        <th>Title</th>
        <th>Category</th>
        <th>Match %</th>
        </tr>
        """

        for i in top:
            table += f"""
            <tr>
            <td>{df.iloc[i]['Title']}</td>
            <td>{df.iloc[i]['Category']}</td>
            <td>{scores[i]*100:.2f}%</td>
            </tr>
            """

        table += "</table>"

        recommendations = table

    return render_template_string(
        html,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)