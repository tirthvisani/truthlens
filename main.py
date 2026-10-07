from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
import ipaddress
import os
import socket

import joblib
import requests
from bs4 import BeautifulSoup


app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-secret")
# ---------------------------------------------------------
# Flask configuration
# ---------------------------------------------------------



# MySQL database
# Database name: truthlens_ai
app.config['SQLALCHEMY_DATABASE_URI'] = \
    'mysql+pymysql://root:@localhost/truthlens_ai'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------------------------------------------------------
# TruthLens ML Model
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "training" / "truthlens_model.pkl"
VECTORIZER_PATH = BASE_DIR / "training" / "truthlens_tfidf_vectorizer.pkl"

truthlens_model = None
truthlens_vectorizer = None
model_load_error = None

try:
    truthlens_model = joblib.load(MODEL_PATH)
    truthlens_vectorizer = joblib.load(VECTORIZER_PATH)
    print("TruthLens ML model loaded successfully.")
except Exception as exc:
    model_load_error = str(exc)
    print(f"Warning: TruthLens ML model could not be loaded: {exc}")


# ---------------------------------------------------------
# User Model
# ---------------------------------------------------------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), nullable=False)

    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    date_created = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )
    role = db.Column(db.String(20), default='user')

    def __repr__(self) -> str:
        return f"{self.id} - {self.email}"

    analyses = db.relationship(
        'AnalysisHistory',
        backref='user',
        lazy=True,
        cascade='all, delete-orphan'
    )


# ---------------------------------------------------------
# Analysis History Model
# ---------------------------------------------------------

class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    content = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
    label = db.Column(db.Integer, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    date_created = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    def __repr__(self) -> str:
        return f"{self.id} - user:{self.user_id} - {self.prediction}"

# with app.app_context():

#     admin = User.query.filter_by(
#         email='admin@truthlens.com'
#     ).first()

#     if not admin:
#         admin = User(
#             name='Admin',
#             email='admin@truthlens.com',
#             password=generate_password_hash('Admin@123'),
#             role='admin'
#         )

#         db.session.add(admin)
#         db.session.commit()

#         print("Admin user created successfully!")

# ---------------------------------------------------------
# Login Required Decorator
# ---------------------------------------------------------

def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if 'user_id' not in session:
            return redirect(url_for('login'))

        return function(*args, **kwargs)

    return wrapper

# ---------------------------------------------------------
# Admin Required Decorator
# ---------------------------------------------------------

def admin_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if 'user_id' not in session:
            return redirect(url_for('login'))

        user = User.query.get(session['user_id'])

        if not user or user.role != 'admin':
            flash("Access denied. Admin privileges required.", "danger")
            return redirect(url_for('home'))

        return function(*args, **kwargs)

    return wrapper
# ---------------------------------------------------------
# Landing Page
# ---------------------------------------------------------

@app.route("/")
def index():

    if 'user_id' in session:
        return redirect(url_for('home'))

    return render_template('index.html')


# ---------------------------------------------------------
# Register
# ---------------------------------------------------------

@app.route("/register", methods=['GET', 'POST'])
def register():

    if 'user_id' in session:
        return redirect(url_for('home'))

    if request.method == "POST":

        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Check empty fields
        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template('register.html')

        # Check password
        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template('register.html')

        # Minimum password length
        if len(password) < 6:
            flash(
                "Password must contain at least 6 characters.",
                "danger"
            )
            return render_template('register.html')

        # Check existing email
        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:
            flash(
                "An account with this email already exists.",
                "danger"
            )
            return render_template('register.html')

        # Hash password
        hashed_password = generate_password_hash(password)

        # Create user
        new_user = User(
            name=name,
            email=email,
            password=hashed_password
        )

        db.session.add(new_user)
        db.session.commit()

        flash(
            "Registration successful. Please login.",
            "success"
        )

        return redirect(url_for('login'))

    return render_template('register.html')


# ---------------------------------------------------------
# Login
# ---------------------------------------------------------

@app.route("/login", methods=['GET', 'POST'])
def login():

    if 'user_id' in session:
        return redirect(url_for('home'))

    if request.method == "POST":

        email = request.form.get(
            'email',
            ''
        ).strip().lower()

        password = request.form.get(
            'password',
            ''
        )

        if not email or not password:

            flash(
                "Email and password are required.",
                "danger"
            )

            return render_template('login.html')

        # Find user
        user = User.query.filter_by(
            email=email
        ).first()

        # Verify password
        if user and check_password_hash(
            user.password,
            password
        ):

            # Clear old session
            session.clear()

            # Store logged-in user information
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_email'] = user.email
            session['user_role'] = user.role

           

            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))

            return redirect(url_for('home'))

        # Login failed
        flash(
            "Invalid email or password.",
            "danger"
        )

    return render_template('login.html')


# ---------------------------------------------------------
# TruthLensAI Home Page
# ---------------------------------------------------------

@app.route("/home")
@login_required
def home():

    return render_template(
        'home.html',
        user_name=session.get('user_name'),
        user_email=session.get('user_email')
    )


# ---------------------------------------------------------
# Analysis History
# ---------------------------------------------------------

@app.route("/history")
@login_required
def history():
    analyses = AnalysisHistory.query.filter_by(
        user_id=session['user_id']
    ).order_by(
        AnalysisHistory.date_created.desc()
    ).all()

    return render_template(
        'history.html',
        analyses=analyses,
        user_name=session.get('user_name')
    )


@app.route("/history/delete/<int:analysis_id>", methods=['POST'])
@login_required
def delete_history(analysis_id):
    analysis = AnalysisHistory.query.filter_by(
        id=analysis_id,
        user_id=session['user_id']
    ).first_or_404()

    db.session.delete(analysis)
    db.session.commit()

    flash("History entry deleted.", "success")
    return redirect(url_for('history'))


@app.route("/history/clear", methods=['POST'])
@login_required
def clear_history():
    AnalysisHistory.query.filter_by(
        user_id=session['user_id']
    ).delete(synchronize_session=False)

    db.session.commit()

    flash("Analysis history cleared.", "success")
    return redirect(url_for('history'))


# ---------------------------------------------------------
# TruthLens Analysis Helpers
# ---------------------------------------------------------

def predict_and_save(content, history_content=None):
    features = truthlens_vectorizer.transform([content])
    predicted_label = int(truthlens_model.predict(features)[0])

    probabilities = truthlens_model.predict_proba(features)[0]
    classes = list(truthlens_model.classes_)
    predicted_index = classes.index(predicted_label)
    confidence = float(probabilities[predicted_index])

    verdict = "Fake" if predicted_label == 1 else "Real"
    confidence_percent = round(confidence * 100, 2)

    history_saved = False
    history_id = None

    try:
        analysis = AnalysisHistory(
            user_id=session['user_id'],
            content=history_content or content,
            prediction=verdict,
            label=predicted_label,
            confidence=confidence_percent
        )
        db.session.add(analysis)
        db.session.commit()
        history_saved = True
        history_id = analysis.id
    except Exception:
        db.session.rollback()
        app.logger.exception("TruthLens history save failed")

    return {
        "success": True,
        "prediction": verdict,
        "label": predicted_label,
        "confidence": confidence_percent,
        "history_saved": history_saved,
        "history_id": history_id,
        "message": (
            "The model found text patterns associated with potentially fake news."
            if predicted_label == 1
            else
            "The model found text patterns associated with real news."
        ),
        "disclaimer": (
            "This is a machine-learning prediction based on language patterns, "
            "not a definitive fact-check. Verify important claims with reliable sources."
        )
    }


def validate_public_url(url):
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Please enter a valid http:// or https:// news URL.")

    hostname = parsed.hostname.lower()

    if hostname in {"localhost", "localhost.localdomain"}:
        raise ValueError("Local or private URLs are not allowed.")

    try:
        addresses = socket.getaddrinfo(hostname, parsed.port or 443)
    except socket.gaierror as exc:
        raise ValueError("The website address could not be resolved.") from exc

    for address in addresses:
        ip_text = address[4][0].split("%")[0]
        ip = ipaddress.ip_address(ip_text)

        if not ip.is_global:
            raise ValueError("Local or private network URLs are not allowed.")

    return parsed


def extract_article_from_url(url):
    current_url = url
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; TruthLensAI/1.0; "
            "+https://github.com/tirthvisani/truthlens)"
        )
    }

    for _ in range(4):
        validate_public_url(current_url)

        response = requests.get(
            current_url,
            headers=headers,
            timeout=(5, 10),
            allow_redirects=False,
            stream=True
        )

        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")

            if not location:
                raise ValueError("The news website returned an invalid redirect.")

            current_url = urljoin(current_url, location)
            continue

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" not in content_type:
            raise ValueError("The URL does not appear to contain a readable HTML article.")

        max_bytes = 2 * 1024 * 1024
        chunks = []
        total_bytes = 0

        for chunk in response.iter_content(chunk_size=16384):
            if not chunk:
                continue

            total_bytes += len(chunk)

            if total_bytes > max_bytes:
                raise ValueError("The webpage is too large to analyze safely.")

            chunks.append(chunk)

        html = b"".join(chunks)
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup([
            "script", "style", "noscript", "nav", "footer",
            "header", "aside", "form", "svg", "iframe"
        ]):
            tag.decompose()

        article = soup.find("article") or soup.find("main") or soup
        paragraphs = []

        for paragraph in article.find_all("p"):
            text = " ".join(paragraph.get_text(" ", strip=True).split())

            if len(text) >= 40:
                paragraphs.append(text)

        extracted_text = "\n\n".join(paragraphs).strip()

        if len(extracted_text) < 200:
            raise ValueError(
                "TruthLens could not extract enough article text from this page. "
                "Try pasting the article text instead."
            )

        return extracted_text[:50000], current_url

    raise ValueError("The news URL redirected too many times.")


# ---------------------------------------------------------
# TruthLens AI - Prediction APIs
# ---------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
@login_required
def analyze_news():

    if truthlens_model is None or truthlens_vectorizer is None:
        return jsonify({
            "success": False,
            "error": "The TruthLens ML model is not available.",
            "details": model_load_error
        }), 503

    data = request.get_json(silent=True) or {}
    content = str(data.get("content", "")).strip()

    if not content:
        return jsonify({
            "success": False,
            "error": "Please enter a news article, headline, or claim."
        }), 400

    if len(content) < 20:
        return jsonify({
            "success": False,
            "error": "Please enter at least 20 characters for a more meaningful analysis."
        }), 400

    try:
        return jsonify(predict_and_save(content))
    except Exception:
        app.logger.exception("TruthLens prediction failed")
        return jsonify({
            "success": False,
            "error": "The analysis could not be completed."
        }), 500


@app.route("/api/analyze-url", methods=["POST"])
@login_required
def analyze_news_url():

    if truthlens_model is None or truthlens_vectorizer is None:
        return jsonify({
            "success": False,
            "error": "The TruthLens ML model is not available.",
            "details": model_load_error
        }), 503

    data = request.get_json(silent=True) or {}
    news_url = str(data.get("url", "")).strip()

    if not news_url:
        return jsonify({
            "success": False,
            "error": "Please enter a news article URL."
        }), 400

    try:
        article_text, final_url = extract_article_from_url(news_url)

        result = predict_and_save(
            article_text,
            history_content=f"Source URL: {final_url}\n\n{article_text}"
        )
        result["source_url"] = final_url
        result["extracted_characters"] = len(article_text)

        return jsonify(result)

    except ValueError as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400
    except requests.RequestException:
        app.logger.exception("TruthLens URL request failed")
        return jsonify({
            "success": False,
            "error": "TruthLens could not retrieve that webpage. Try another URL or paste the article text."
        }), 502
    except Exception:
        app.logger.exception("TruthLens URL analysis failed")
        return jsonify({
            "success": False,
            "error": "The URL analysis could not be completed."
        }), 500


# ---------------------------------------------------------
# Admin Dashboard
# ---------------------------------------------------------

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():

    total_users = User.query.count()
    total_admins = User.query.filter_by(role='admin').count()
    total_normal_users = User.query.filter_by(role='user').count()

    total_analyses = AnalysisHistory.query.count()
    total_real = AnalysisHistory.query.filter_by(prediction='Real').count()
    total_fake = AnalysisHistory.query.filter_by(prediction='Fake').count()

    avg_confidence_value = db.session.query(db.func.avg(AnalysisHistory.confidence)).scalar()
    avg_confidence = round(float(avg_confidence_value or 0), 2)

    recent_analyses = AnalysisHistory.query.order_by(
        AnalysisHistory.date_created.desc()
    ).limit(10).all()

    activity_labels = []
    activity_total = []
    activity_real = []
    activity_fake = []
    today_date = datetime.utcnow().date()

    for days_ago in range(6, -1, -1):
        day = today_date - timedelta(days=days_ago)
        day_start = datetime.combine(day, datetime.min.time())
        day_end = day_start + timedelta(days=1)

        daily_query = AnalysisHistory.query.filter(
            AnalysisHistory.date_created >= day_start,
            AnalysisHistory.date_created < day_end
        )

        activity_labels.append(day.strftime('%d %b'))
        activity_total.append(daily_query.count())
        activity_real.append(daily_query.filter(AnalysisHistory.prediction == 'Real').count())
        activity_fake.append(daily_query.filter(AnalysisHistory.prediction == 'Fake').count())

    users = User.query.order_by(
        User.date_created.desc()
    ).all()

    # ---------------------------------------------------------
    # User Report Data
    # ---------------------------------------------------------
    # Monthly registration count for the last 6 months.
    # in the User table.
    today = datetime.utcnow()
    registration_labels = []
    registration_counts = []

    # Calculate the six month window without requiring an extra package.
    months = []
    year = today.year
    month = today.month

    for _ in range(6):
        months.append((year, month))
        month -= 1
        if month == 0:
            month = 12
            year -= 1

    months.reverse()

    for year, month in months:
        month_start = datetime(year, month, 1)

        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)

        count = User.query.filter(
            User.date_created >= month_start,
            User.date_created < next_month
        ).count()

        registration_labels.append(month_start.strftime('%b %Y'))
        registration_counts.append(count)

    return render_template(
        'admin.html',
        total_users=total_users,
        total_admins=total_admins,
        total_normal_users=total_normal_users,
        total_analyses=total_analyses,
        total_real=total_real,
        total_fake=total_fake,
        avg_confidence=avg_confidence,
        recent_analyses=recent_analyses,
        activity_labels=activity_labels,
        activity_total=activity_total,
        activity_real=activity_real,
        activity_fake=activity_fake,
        users=users,
        registration_labels=registration_labels,
        registration_counts=registration_counts
    )

# ---------------------------------------------------------
# Delete User
# ---------------------------------------------------------

@app.route("/admin/users/delete/<int:user_id>", methods=['POST'])
@admin_required
def delete_user(user_id):

    user = User.query.get_or_404(user_id)

    # Prevent admin from deleting themselves
    if user.id == session.get('user_id'):
        flash("You cannot delete your own admin account.", "danger")
        return redirect(url_for('admin_dashboard'))

    db.session.delete(user)
    db.session.commit()

    flash("User deleted successfully.", "success")

    return redirect(url_for('admin_dashboard'))
# ---------------------------------------------------------
# Logout
# ---------------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "info"
    )

    return redirect(url_for('login'))


# ---------------------------------------------------------
# Create Database Tables
# ---------------------------------------------------------

with app.app_context():
    db.create_all()


# ---------------------------------------------------------
# Run Application
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)