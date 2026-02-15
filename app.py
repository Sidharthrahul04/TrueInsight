from flask import Flask, render_template, request, redirect, session
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from collections import defaultdict
import joblib
from textblob import TextBlob

# =========================================================
# LOAD ML MODEL (LOGISTIC REGRESSION OR RANDOM FOREST)
# =========================================================
# 👉 Use ONE of the following lines

# Logistic Regression
bundle = joblib.load("model/review_model_lr.pkl")

# Random Forest (if you want to switch back)
# bundle = joblib.load("model/review_model.pkl")

model = bundle["model"]

# =========================================================
# APP SETUP
# =========================================================
app = Flask(__name__)
app.secret_key = "dev-secret"

# =========================================================
# MYSQL CONFIG
# =========================================================
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "password"
app.config["MYSQL_DB"] = "trueinsight"

mysql = MySQL(app)

# =========================================================
# CATEGORY-AWARE RELEVANCE CHECK
# =========================================================
def is_review_relevant(review_text, product_category):
    text = review_text.lower()
    category = product_category.lower()

    INVALID_KEYWORDS = {
        "phone": ["spf", "skin", "rash", "irritation", "greasy", "sole", "grip", "running", "walking"],
        "laptop": ["spf", "skin", "rash", "irritation", "greasy", "sole", "grip", "running", "walking"],
        "sunscreen": ["camera", "battery", "display", "processor", "performance", "heating", "speaker", "charging"],
        "shoes": ["camera", "battery", "display", "processor", "performance", "heating", "speaker", "charging"]
    }

    for word in INVALID_KEYWORDS.get(category, []):
        if word in text:
            return False

    return True

# =========================================================
# ML REVIEW ANALYSIS
# =========================================================
def analyze_reviews(reviews, product_category):
    if not reviews:
        return []

    texts = [r["text"].lower().strip() for r in reviews]
    analyzed = []

    # ---- BURST DETECTION ----
    time_groups = defaultdict(list)
    for i, r in enumerate(reviews):
        minute_key = r["created_at"].replace(second=0, microsecond=0)
        time_groups[minute_key].append(i)

    burst_indices = set()
    for group in time_groups.values():
        if len(group) >= 3:
            burst_indices.update(group)

    cur = mysql.connection.cursor()

    for i, r in enumerate(reviews):
        text = r["text"]
        rating = r["rating"]
        created_at = r["created_at"]
        user_id = r["user_id"]

        review_length = len(text)
        word_count = len(text.split())
        sentiment = TextBlob(text).sentiment.polarity

        cur.execute("SELECT COUNT(*) FROM reviews WHERE user_id=%s", (user_id,))
        user_review_count = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM reviews
            WHERE user_id=%s AND DATE(created_at)=DATE(%s)
        """, (user_id, created_at))
        daily_review_count = cur.fetchone()[0]

        duplicate_flag = 1 if texts.count(text.lower().strip()) > 1 else 0
        burst_flag = 1 if i in burst_indices else 0

        features = [[
            review_length,
            word_count,
            sentiment,
            rating,
            user_review_count,
            daily_review_count,
            duplicate_flag,
            burst_flag
        ]]

        # Probability-based decision
        prob_fake = model.predict_proba(features)[0][1]

        if product_category.lower() == "sunscreen":
            suspicious = prob_fake >= 0.75
        else:
            suspicious = prob_fake >= 0.5

        reasons = []
        if duplicate_flag:
            reasons.append("Duplicate review text")
        if burst_flag:
            reasons.append("Burst posting detected")
        if user_review_count <= 1:
            reasons.append("Low reviewer activity")

        analyzed.append({
            "rating": rating,
            "text": text,
            "created_at": created_at,
            "suspicious": suspicious,
            "reasons": ", ".join(reasons) if reasons else "Predicted by ML model"
        })

    cur.close()
    return analyzed

# =========================================================
# AUTH
# =========================================================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute("SELECT id, password_hash FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[1], password):
            session["user_id"] = user[0]
            return redirect("/home")

        return "Invalid credentials"

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (%s,%s,%s)",
            (request.form["email"], generate_password_hash(request.form["password"]), datetime.now())
        )
        mysql.connection.commit()
        cur.close()
        return redirect("/")

    return render_template("register.html")

# =========================================================
# HOME
# =========================================================
@app.route("/home")
def home():
    if "user_id" not in session:
        return redirect("/")

    cur = mysql.connection.cursor()
    cur.execute("SELECT id,name,price,raw_rating,image_url FROM products LIMIT 12")
    products = cur.fetchall()
    cur.close()

    return render_template("home.html", products=products)

# =========================================================
# PRODUCT DETAIL
# =========================================================
@app.route("/product/<int:product_id>")
def product_detail(product_id):
    if "user_id" not in session:
        return redirect("/")

    cur = mysql.connection.cursor()

    cur.execute("""
        SELECT name, category, description, price, raw_rating, image_url
        FROM products WHERE id=%s
    """, (product_id,))
    product = cur.fetchone()

    if not product:
        cur.close()
        return "Product not found", 404

    product_category = product[1]

    cur.execute("""
        SELECT user_id, rating, review_text, created_at
        FROM reviews
        WHERE product_id=%s
        ORDER BY created_at DESC
    """, (product_id,))
    raw_reviews = cur.fetchall()
    cur.close()

    relevant_reviews = []
    final_reviews = []

    for r in raw_reviews:
        review = {
            "user_id": r[0],
            "rating": r[1],
            "text": r[2],
            "created_at": r[3]
        }

        if not is_review_relevant(r[2], product_category):
            final_reviews.append({
                "rating": r[1],
                "text": r[2],
                "created_at": r[3],
                "suspicious": True,
                "reasons": "Irrelevant to product specifications"
            })
        else:
            relevant_reviews.append(review)

    analyzed = analyze_reviews(relevant_reviews, product_category)
    final_reviews.extend(analyzed)

    total = len(final_reviews)
    genuine = [r for r in final_reviews if not r["suspicious"]]

    raw_rating = round(sum(r["rating"] for r in final_reviews) / total, 2) if total else 0
    filtered_rating = round(sum(r["rating"] for r in genuine) / len(genuine), 2) if genuine else 0

    integrity = {
        "total": total,
        "genuine": len(genuine),
        "suspicious": total - len(genuine)
    }

    return render_template(
        "product.html",
        product=product,
        reviews=final_reviews,
        raw_rating=raw_rating,
        filtered_rating=filtered_rating,
        integrity=integrity
    )

# =========================================================
# LOGOUT
# =========================================================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
