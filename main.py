from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
    Blueprint,
    abort,
    g,
    send_file,
    Response,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
import csv
import io
import ipaddress
import json
import re
import os
import socket
import secrets

import click
import joblib
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)

IS_PRODUCTION = os.environ.get("FLASK_ENV", "development").lower() == "production"

secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    if IS_PRODUCTION:
        raise RuntimeError("SECRET_KEY must be set when FLASK_ENV=production.")
    secret_key = "dev-only-secret"

app.config.update(
    SECRET_KEY=secret_key,
    MAX_CONTENT_LENGTH=1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
    # MySQL database (override with DATABASE_URL).
    # Database name: truthlens_ai
    SQLALCHEMY_DATABASE_URI=os.environ.get(
        "DATABASE_URL",
        "mysql+pymysql://root:@localhost/truthlens_ai",
    ),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)

MAX_ANALYSIS_CHARS = 50000
MAX_URL_CHARS = 2048

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

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

    is_active = db.Column(db.Boolean, default=True, nullable=False)
    last_login_at = db.Column(db.DateTime)
    last_login_ip = db.Column(db.String(45))
    failed_logins = db.Column(db.Integer, default=0, nullable=False)
    locked_until = db.Column(db.DateTime)

    def __repr__(self) -> str:
        return f"{self.id} - {self.email}"

    analyses = db.relationship(
        'AnalysisHistory',
        backref='user',
        lazy=True,
        cascade='all, delete-orphan',
        foreign_keys='AnalysisHistory.user_id'
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

    # Provenance of the analysed content.
    source_type = db.Column(db.String(10), default='text', nullable=False)
    source_url = db.Column(db.Text)
    model_version = db.Column(db.String(20), default='v1', nullable=False)

    # Human-corrected label (feeds metrics and the retraining export).
    # NULL means no human label has been supplied yet.
    corrected_label = db.Column(db.String(10), nullable=True, index=True)
    corrected_by = db.Column(
        db.Integer,
        db.ForeignKey('user.id', ondelete='SET NULL'),
        nullable=True
    )
    corrected_at = db.Column(db.DateTime)

    corrector = db.relationship('User', foreign_keys=[corrected_by])

    def __repr__(self) -> str:
        return f"{self.id} - user:{self.user_id} - {self.prediction}"


# ---------------------------------------------------------
# Admin Audit Log
# ---------------------------------------------------------

class AdminAuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id', ondelete='SET NULL'),
        nullable=True,
        index=True
    )
    action = db.Column(db.String(50), nullable=False, index=True)
    target_type = db.Column(db.String(30))
    target_id = db.Column(db.Integer)
    detail = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    admin = db.relationship('User', foreign_keys=[admin_id])

    def __repr__(self) -> str:
        return f"{self.id} - {self.action}"


# ---------------------------------------------------------
# System Settings (single source for tunable values)
# ---------------------------------------------------------

class SystemSetting(db.Model):
    key = db.Column(db.String(50), primary_key=True)
    value = db.Column(db.String(255), nullable=False)

    @staticmethod
    def get(key, default=None):
        row = db.session.get(SystemSetting, key)
        return row.value if row else default

    @staticmethod
    def set(key, value):
        row = db.session.get(SystemSetting, key)
        if row is None:
            db.session.add(SystemSetting(key=key, value=str(value)))
        else:
            row.value = str(value)


def get_fake_threshold():
    """Probability (0-1) at or above which a claim is labelled Fake."""
    try:
        value = float(SystemSetting.get('fake_threshold', '0.5'))
    except (TypeError, ValueError):
        return 0.5
    return min(0.95, max(0.05, value))

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

        user = db.session.get(User, session['user_id'])

        # Deleting a suspended account, or suspending it elsewhere, must
        # revoke its existing login session.
        if user is None or not user.is_active:
            session.clear()
            flash(
                "This account is no longer available. Please log in.",
                "warning"
            )
            return redirect(url_for('login'))

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

        if len(name) > 100:
            flash("Name must be 100 characters or fewer.", "danger")
            return render_template('register.html')

        if len(email) > 150 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            flash("Please enter a valid email address.", "danger")
            return render_template('register.html')

        if len(password) > 128:
            flash("Password must be 128 characters or fewer.", "danger")
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

        now = datetime.utcnow()

        # Find user
        user = User.query.filter_by(
            email=email
        ).first()

        # Temporary lockout after repeated failures.
        if user and user.locked_until and user.locked_until > now:
            flash(
                "Too many failed login attempts. Please try again later.",
                "danger"
            )
            return render_template('login.html')

        # Verify password
        if user and check_password_hash(
            user.password,
            password
        ):

            if not user.is_active:
                flash(
                    "This account has been suspended. "
                    "Please contact an administrator.",
                    "danger"
                )
                return render_template('login.html')

            # Record a successful login and clear failure counters.
            user.failed_logins = 0
            user.locked_until = None
            user.last_login_at = now
            user.last_login_ip = request.remote_addr
            db.session.commit()

            # Clear old session
            session.clear()
            session.permanent = True

            # Store logged-in user information
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['user_email'] = user.email
            session['user_role'] = user.role

            if user.role == 'admin':
                return redirect(url_for('admin.overview'))

            return redirect(url_for('home'))

        # Login failed — count the attempt without revealing whether
        # the email exists.
        if user:
            user.failed_logins = (user.failed_logins or 0) + 1

            if user.failed_logins >= MAX_LOGIN_ATTEMPTS:
                user.locked_until = now + timedelta(minutes=LOCKOUT_MINUTES)
                user.failed_logins = 0

            db.session.commit()

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

def fake_class_coefficients():
    """Return the model coefficients that push a prediction toward Fake."""
    coefficients = getattr(truthlens_model, 'coef_', None)

    if coefficients is None:
        return None

    classes = list(getattr(truthlens_model, 'classes_', []))

    if len(classes) > 2 and 1 in classes:
        return coefficients[classes.index(1)]

    if getattr(coefficients, 'ndim', 1) == 2:
        return coefficients[0]

    return coefficients


def explain_prediction(content, top_n=8):
    """Rank the terms that most influenced the Fake-vs-Real decision.

    Each TF-IDF feature value is multiplied by its Logistic Regression
    coefficient. Positive contributions push the verdict toward Fake;
    negative contributions push it toward Real.
    """
    if truthlens_model is None or truthlens_vectorizer is None or not content:
        return []

    try:
        feature_names = truthlens_vectorizer.get_feature_names_out()
        row = truthlens_vectorizer.transform([content]).tocsr()
        coefficients = fake_class_coefficients()

        if coefficients is None or row.nnz == 0:
            return []

        contributions = []

        for index, value in zip(row.indices, row.data):
            weight = float(value) * float(coefficients[index])

            if weight:
                contributions.append((str(feature_names[index]), weight))

        contributions.sort(key=lambda item: abs(item[1]), reverse=True)
        top = contributions[:max(1, top_n)]
        largest = max((abs(weight) for _, weight in top), default=0) or 1

        return [
            {
                'term': term,
                'weight': round(weight, 5),
                'direction': 'Fake' if weight > 0 else 'Real',
                'strength': round(abs(weight) / largest * 100, 1),
            }
            for term, weight in top
        ]
    except Exception:
        app.logger.exception("TruthLens explanation failed")
        return []


def predict_and_save(content, history_content=None, source_type='text',
                     source_url=None):
    features = truthlens_vectorizer.transform([content])
    probabilities = truthlens_model.predict_proba(features)[0]
    classes = list(truthlens_model.classes_)

    # Use the admin-configurable decision threshold instead of the raw
    # argmax so the dashboard can tune strictness.
    fake_probability = float(probabilities[classes.index(1)])
    threshold = get_fake_threshold()
    predicted_label = 1 if fake_probability >= threshold else 0

    if predicted_label == 1:
        confidence = fake_probability
    else:
        confidence = 1.0 - fake_probability

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
            confidence=confidence_percent,
            source_type=source_type,
            source_url=source_url,
            model_version=SystemSetting.get('model_version', 'v1')
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
        "explanation": explain_prediction(content),
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

    if len(url) > MAX_URL_CHARS:
        raise ValueError("The news URL is too long.")

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Please enter a valid http:// or https:// news URL.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing login credentials are not allowed.")

    hostname = parsed.hostname.lower()

    if hostname in {"localhost", "localhost.localdomain"}:
        raise ValueError("Local or private URLs are not allowed.")

    try:
        default_port = 443 if parsed.scheme == "https" else 80
        addresses = socket.getaddrinfo(hostname, parsed.port or default_port)
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

    if len(content) > MAX_ANALYSIS_CHARS:
        return jsonify({
            "success": False,
            "error": f"Please keep the submitted text under {MAX_ANALYSIS_CHARS:,} characters."
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
            history_content=article_text,
            source_type='url',
            source_url=final_url
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
# Admin Area
# ---------------------------------------------------------

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


@admin_bp.before_request
def protect_admin_area():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.session.get(User, session['user_id'])

    if user is None or not user.is_active:
        session.clear()
        flash("Your session is no longer valid. Please log in.", "warning")
        return redirect(url_for('login'))

    if user.role != 'admin':
        abort(403)

    g.admin = user

    if 'admin_csrf_token' not in session:
        session['admin_csrf_token'] = secrets.token_urlsafe(32)

    # Every mutating admin request must carry the session-bound token.
    if request.method == 'POST':
        token = request.form.get('csrf_token', '')
        expected = session['admin_csrf_token']

        if (
            not token
            or not token.isascii()
            or not secrets.compare_digest(expected, token)
        ):
            abort(400, description="Invalid or missing CSRF token.")


@admin_bp.after_request
def harden_admin_response(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@admin_bp.context_processor
def inject_admin_helpers():
    def page_url(page):
        args = request.args.to_dict()
        args['page'] = page
        return url_for(request.endpoint, **args)

    return {
        'csrf_token': session.get('admin_csrf_token'),
        'current_admin': g.get('admin'),
        'page_url': page_url,
    }


def log_admin_action(action, target_type=None, target_id=None, detail=None):
    db.session.add(AdminAuditLog(
        admin_id=g.admin.id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        detail=detail,
        ip_address=request.remote_addr,
    ))


def csv_safe(value):
    """Neutralise spreadsheet formulas in user-controlled CSV cells."""
    if value is None:
        return ''

    text_value = str(value)

    if text_value.startswith(('=', '+', '-', '@', '\t', '\r')):
        return f"'{text_value}"

    return text_value


def user_analysis_query():
    """Analyses created by normal users (excludes admin test runs)."""
    return AnalysisHistory.query.join(AnalysisHistory.user).filter(
        User.role == 'user'
    )


def parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, '%Y-%m-%d')
    except ValueError:
        return None


def file_info(path):
    try:
        stat = path.stat()
        return {
            'exists': True,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'modified': datetime.fromtimestamp(stat.st_mtime),
        }
    except OSError:
        return {'exists': False, 'size_mb': 0, 'modified': None}


def filtered_user_query():
    q = User.query
    search = request.args.get('q', '').strip()
    role = request.args.get('role', 'all')
    status = request.args.get('status', 'all')

    if search:
        like = f'%{search}%'
        q = q.filter(db.or_(User.name.ilike(like), User.email.ilike(like)))
    if role in ('user', 'admin'):
        q = q.filter(User.role == role)
    if status == 'active':
        q = q.filter(User.is_active.is_(True))
    elif status == 'suspended':
        q = q.filter(User.is_active.is_(False))

    return q.order_by(User.date_created.desc())


def filtered_analysis_query():
    q = AnalysisHistory.query
    verdict = request.args.get('verdict', 'all')
    user_id = request.args.get('user_id', type=int)
    min_conf = request.args.get('min_confidence', type=float)
    max_conf = request.args.get('max_confidence', type=float)
    start = parse_date(request.args.get('start'))
    end = parse_date(request.args.get('end'))

    if verdict in ('Real', 'Fake'):
        q = q.filter(AnalysisHistory.prediction == verdict)
    if user_id:
        q = q.filter(AnalysisHistory.user_id == user_id)
    if min_conf is not None:
        q = q.filter(AnalysisHistory.confidence >= min_conf)
    if max_conf is not None:
        q = q.filter(AnalysisHistory.confidence <= max_conf)
    if start:
        q = q.filter(AnalysisHistory.date_created >= start)
    if end:
        q = q.filter(
            AnalysisHistory.date_created < end + timedelta(days=1)
        )

    return q.order_by(AnalysisHistory.date_created.desc())


def corrected_label_metrics(query=None):
    """Compute human-label accuracy and Fake-class precision/recall/F1."""
    base = query if query is not None else AnalysisHistory.query
    rows = base.filter(
        AnalysisHistory.corrected_label.isnot(None)
    ).with_entities(
        AnalysisHistory.prediction,
        AnalysisHistory.corrected_label,
    ).all()

    total = len(rows)
    correct = sum(
        1 for prediction, corrected in rows if prediction == corrected
    )

    tp = sum(
        1 for prediction, corrected in rows
        if corrected == 'Fake' and prediction == 'Fake'
    )
    fp = sum(
        1 for prediction, corrected in rows
        if corrected == 'Real' and prediction == 'Fake'
    )
    fn = sum(
        1 for prediction, corrected in rows
        if corrected == 'Fake' and prediction == 'Real'
    )
    tn = sum(
        1 for prediction, corrected in rows
        if corrected == 'Real' and prediction == 'Real'
    )

    def ratio(numerator, denominator):
        if not denominator:
            return None
        return round(numerator / denominator * 100, 2)

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)

    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = round(2 * precision * recall / (precision + recall), 2)

    return {
        'total': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': ratio(correct, total),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }


@admin_bp.get('/')
@admin_bp.get('/dashboard')
def overview():
    stats_query = user_analysis_query()

    total_users = User.query.count()
    total_admins = User.query.filter_by(role='admin').count()
    total_normal_users = User.query.filter_by(role='user').count()
    active_users = User.query.filter_by(role='user', is_active=True).count()
    suspended_users = User.query.filter_by(
        role='user', is_active=False
    ).count()

    total_analyses = stats_query.count()
    total_real = stats_query.filter(
        AnalysisHistory.prediction == 'Real'
    ).count()
    total_fake = stats_query.filter(
        AnalysisHistory.prediction == 'Fake'
    ).count()

    avg_confidence_value = db.session.query(
        db.func.avg(AnalysisHistory.confidence)
    ).join(AnalysisHistory.user).filter(User.role == 'user').scalar()
    avg_confidence = round(float(avg_confidence_value or 0), 2)

    corrected_count = stats_query.filter(
        AnalysisHistory.corrected_label.isnot(None)
    ).count()
    low_confidence = stats_query.filter(
        AnalysisHistory.confidence < 60
    ).count()
    human_metrics = corrected_label_metrics(stats_query)

    recent_analyses = stats_query.order_by(
        AnalysisHistory.date_created.desc()
    ).limit(10).all()

    low_confidence_items = stats_query.filter(
        AnalysisHistory.confidence < 60
    ).order_by(
        AnalysisHistory.date_created.desc()
    ).limit(5).all()

    activity_labels = []
    activity_total = []
    activity_real = []
    activity_fake = []
    today_date = datetime.utcnow().date()

    for days_ago in range(6, -1, -1):
        day = today_date - timedelta(days=days_ago)
        day_start = datetime.combine(day, datetime.min.time())
        day_end = day_start + timedelta(days=1)

        daily_query = stats_query.filter(
            AnalysisHistory.date_created >= day_start,
            AnalysisHistory.date_created < day_end
        )

        activity_labels.append(day.strftime('%d %b'))
        activity_total.append(daily_query.count())
        activity_real.append(
            daily_query.filter(AnalysisHistory.prediction == 'Real').count()
        )
        activity_fake.append(
            daily_query.filter(AnalysisHistory.prediction == 'Fake').count()
        )

    today = datetime.utcnow()
    registration_labels = []
    registration_counts = []

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

        registration_labels.append(month_start.strftime('%b %Y'))
        registration_counts.append(User.query.filter(
            User.date_created >= month_start,
            User.date_created < next_month
        ).count())

    recent_admin_actions = AdminAuditLog.query.order_by(
        AdminAuditLog.created_at.desc()
    ).limit(6).all()

    return render_template(
        'admin/overview.html',
        total_users=total_users,
        total_admins=total_admins,
        total_normal_users=total_normal_users,
        active_users=active_users,
        suspended_users=suspended_users,
        total_analyses=total_analyses,
        total_real=total_real,
        total_fake=total_fake,
        avg_confidence=avg_confidence,
        corrected_count=corrected_count,
        human_metrics=human_metrics,
        low_confidence=low_confidence,
        recent_analyses=recent_analyses,
        low_confidence_items=low_confidence_items,
        activity_labels=activity_labels,
        activity_total=activity_total,
        activity_real=activity_real,
        activity_fake=activity_fake,
        registration_labels=registration_labels,
        registration_counts=registration_counts,
        recent_admin_actions=recent_admin_actions,
        model_loaded=truthlens_model is not None,
        model_error=model_load_error,
        threshold=get_fake_threshold(),
        model_version=SystemSetting.get('model_version', 'v1'),
    )


# ---------------------------------------------------------
# User Management
# ---------------------------------------------------------

@admin_bp.get('/users')
def users():
    page = request.args.get('page', 1, type=int)
    pagination = filtered_user_query().paginate(
        page=page, per_page=20, error_out=False
    )

    user_ids = [user.id for user in pagination.items]
    analysis_counts = {}

    if user_ids:
        analysis_counts = dict(
            db.session.query(
                AnalysisHistory.user_id,
                db.func.count(AnalysisHistory.id)
            ).filter(
                AnalysisHistory.user_id.in_(user_ids)
            ).group_by(AnalysisHistory.user_id).all()
        )

    return render_template(
        'admin/users.html',
        pagination=pagination,
        analysis_counts=analysis_counts,
    )


@admin_bp.get('/users/export.csv')
def export_users():
    users_list = filtered_user_query().all()
    counts = dict(
        db.session.query(
            AnalysisHistory.user_id,
            db.func.count(AnalysisHistory.id)
        ).group_by(AnalysisHistory.user_id).all()
    )

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        'id', 'name', 'email', 'role', 'status', 'analyses',
        'registered_at', 'last_login_at', 'last_login_ip',
    ])

    for user in users_list:
        writer.writerow([
            user.id,
            csv_safe(user.name),
            csv_safe(user.email),
            user.role,
            'active' if user.is_active else 'suspended',
            counts.get(user.id, 0),
            user.date_created.isoformat() if user.date_created else '',
            user.last_login_at.isoformat() if user.last_login_at else '',
            csv_safe(user.last_login_ip or ''),
        ])

    log_admin_action(
        'export.users',
        detail=f'Exported {len(users_list)} user rows',
    )
    db.session.commit()

    filename = f"truthlens_users_{datetime.utcnow():%Y%m%d_%H%M}.csv"
    return Response(
        '\ufeff' + buffer.getvalue(),
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': f'attachment; filename={filename}'},
    )


@admin_bp.get('/users/<int:user_id>')
def user_detail(user_id):
    user = db.get_or_404(User, user_id)
    page = request.args.get('page', 1, type=int)

    pagination = AnalysisHistory.query.filter_by(
        user_id=user.id
    ).order_by(
        AnalysisHistory.date_created.desc()
    ).paginate(page=page, per_page=15, error_out=False)

    base = AnalysisHistory.query.filter_by(user_id=user.id)
    avg_value = db.session.query(
        db.func.avg(AnalysisHistory.confidence)
    ).filter(AnalysisHistory.user_id == user.id).scalar()

    stats = {
        'total': base.count(),
        'real': base.filter(AnalysisHistory.prediction == 'Real').count(),
        'fake': base.filter(AnalysisHistory.prediction == 'Fake').count(),
        'corrected': base.filter(
            AnalysisHistory.corrected_label.isnot(None)
        ).count(),
        'avg_confidence': round(float(avg_value or 0), 2),
        'last_activity': db.session.query(
            db.func.max(AnalysisHistory.date_created)
        ).filter(AnalysisHistory.user_id == user.id).scalar(),
    }

    return render_template(
        'admin/user_detail.html',
        user=user,
        pagination=pagination,
        stats=stats,
    )


@admin_bp.post('/users/<int:user_id>/role')
def set_user_role(user_id):
    user = db.get_or_404(User, user_id)
    new_role = request.form.get('role', '')
    old_role = user.role

    if new_role not in ('user', 'admin'):
        abort(400)

    if user.id == g.admin.id:
        flash("You cannot change your own role.", "danger")
    elif new_role == old_role:
        flash("That account already has this role.", "info")
    elif (
        old_role == 'admin'
        and new_role == 'user'
        and user.is_active
        and User.query.filter_by(role='admin', is_active=True).count() <= 1
    ):
        flash("At least one active administrator must remain.", "danger")
    else:
        user.role = new_role
        log_admin_action(
            'user.set_role', 'user', user.id,
            f'{old_role} -> {new_role}',
        )
        db.session.commit()
        flash(f"Role updated to {new_role}.", "success")

    return redirect(url_for('admin.user_detail', user_id=user.id))


@admin_bp.post('/users/<int:user_id>/status')
def set_user_status(user_id):
    user = db.get_or_404(User, user_id)
    action = request.form.get('action', '')

    if action not in ('suspend', 'activate'):
        abort(400)

    activate = action == 'activate'

    if user.id == g.admin.id:
        flash("You cannot suspend your own account.", "danger")
    elif (
        not activate
        and user.role == 'admin'
        and User.query.filter_by(role='admin', is_active=True).count() <= 1
    ):
        flash("At least one active administrator must remain.", "danger")
    else:
        user.is_active = activate
        if activate:
            user.failed_logins = 0
            user.locked_until = None

        log_admin_action(
            'user.activate' if activate else 'user.suspend',
            'user', user.id,
            'Account reactivated' if activate else 'Account suspended',
        )
        db.session.commit()
        flash(
            "Account reactivated." if activate else "Account suspended.",
            "success",
        )

    return redirect(url_for('admin.user_detail', user_id=user.id))


@admin_bp.post('/users/<int:user_id>/reset-password')
def reset_user_password(user_id):
    user = db.get_or_404(User, user_id)
    temporary_password = secrets.token_urlsafe(9)

    user.password = generate_password_hash(temporary_password)
    user.failed_logins = 0
    user.locked_until = None

    log_admin_action(
        'user.reset_password', 'user', user.id,
        'Temporary password generated',
    )
    db.session.commit()

    flash(
        f"Temporary password for {user.email}: {temporary_password} "
        "(shown once — share it securely).",
        "warning",
    )
    return redirect(url_for('admin.user_detail', user_id=user.id))


@admin_bp.post('/users/<int:user_id>/delete')
def delete_user(user_id):
    user = db.get_or_404(User, user_id)

    if user.id == g.admin.id:
        flash("You cannot delete your own account.", "danger")
        return redirect(url_for('admin.users'))

    if user.role != 'user':
        flash("Administrator accounts are protected from deletion.", "danger")
        return redirect(url_for('admin.users'))

    email = user.email

    try:
        db.session.delete(user)
        log_admin_action(
            'user.delete', 'user', user.id,
            f'Deleted {email} and their analysis history',
        )
        db.session.commit()
    except Exception:
        db.session.rollback()
        app.logger.exception("Unable to delete user ID %s", user_id)
        flash("The user could not be deleted. Please try again.", "danger")
    else:
        flash(
            f"Deleted {email} and their analysis history.",
            "success",
        )

    return redirect(url_for('admin.users'))


# ---------------------------------------------------------
# Analysis Management & Corrected Labels
# ---------------------------------------------------------

@admin_bp.get('/analyses')
def analyses():
    page = request.args.get('page', 1, type=int)
    pagination = filtered_analysis_query().paginate(
        page=page, per_page=25, error_out=False
    )

    return render_template(
        'admin/analyses.html',
        pagination=pagination,
        users=User.query.order_by(User.name).all(),
    )


@admin_bp.get('/analyses/export.csv')
def export_analyses():
    items = filtered_analysis_query().all()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        'id', 'user_name', 'user_email', 'content', 'verdict', 'confidence',
        'corrected_label', 'corrected_by', 'corrected_at', 'source_type',
        'source_url', 'model_version', 'created_at',
    ])

    for item in items:
        writer.writerow([
            item.id,
            csv_safe(item.user.name if item.user else ''),
            csv_safe(item.user.email if item.user else ''),
            csv_safe(item.content),
            item.prediction,
            item.confidence,
            item.corrected_label or '',
            csv_safe(item.corrector.email if item.corrector else ''),
            item.corrected_at.isoformat() if item.corrected_at else '',
            item.source_type,
            csv_safe(item.source_url or ''),
            csv_safe(item.model_version),
            item.date_created.isoformat() if item.date_created else '',
        ])

    log_admin_action(
        'export.analyses',
        detail=f'Exported {len(items)} analysis rows',
    )
    db.session.commit()

    filename = f"truthlens_analyses_{datetime.utcnow():%Y%m%d_%H%M}.csv"
    return Response(
        '\ufeff' + buffer.getvalue(),
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': f'attachment; filename={filename}'},
    )


@admin_bp.get('/analyses/<int:analysis_id>')
def analysis_detail(analysis_id):
    analysis = db.get_or_404(AnalysisHistory, analysis_id)
    return render_template(
        'admin/analysis_detail.html',
        analysis=analysis,
        explanation=explain_prediction(analysis.content),
    )


@admin_bp.post('/analyses/<int:analysis_id>/corrected-label')
def set_corrected_label(analysis_id):
    analysis = db.get_or_404(AnalysisHistory, analysis_id)
    label = request.form.get('corrected_label', '').strip()

    if label not in ('', 'Real', 'Fake'):
        abort(400)

    previous = analysis.corrected_label
    analysis.corrected_label = label or None

    if label:
        analysis.corrected_by = g.admin.id
        analysis.corrected_at = datetime.utcnow()
    else:
        analysis.corrected_by = None
        analysis.corrected_at = None

    log_admin_action(
        'analysis.corrected_label', 'analysis', analysis.id,
        f'{previous or "none"} -> {label or "none"}',
    )
    db.session.commit()

    flash(
        "Corrected label saved." if label else "Corrected label cleared.",
        "success",
    )
    return redirect(url_for('admin.analysis_detail', analysis_id=analysis.id))


@admin_bp.post('/analyses/<int:analysis_id>/delete')
def delete_analysis(analysis_id):
    analysis = db.get_or_404(AnalysisHistory, analysis_id)
    preview = analysis.content[:80]

    db.session.delete(analysis)
    log_admin_action(
        'analysis.delete', 'analysis', analysis.id,
        f'Deleted analysis: {preview}',
    )
    db.session.commit()

    flash("Analysis entry deleted.", "success")
    return redirect(url_for('admin.analyses'))


# ---------------------------------------------------------
# Audit Log
# ---------------------------------------------------------

@admin_bp.get('/audit')
def audit():
    page = request.args.get('page', 1, type=int)
    action = request.args.get('action', 'all')
    admin_id = request.args.get('admin_id', type=int)

    q = AdminAuditLog.query

    if action and action != 'all':
        q = q.filter(AdminAuditLog.action == action)
    if admin_id:
        q = q.filter(AdminAuditLog.admin_id == admin_id)

    pagination = q.order_by(
        AdminAuditLog.created_at.desc()
    ).paginate(page=page, per_page=30, error_out=False)

    actions = [
        row[0] for row in db.session.query(
            AdminAuditLog.action
        ).distinct().order_by(AdminAuditLog.action).all()
    ]
    admins = User.query.filter_by(role='admin').order_by(User.name).all()

    return render_template(
        'admin/audit.html',
        pagination=pagination,
        actions=actions,
        admins=admins,
    )


# ---------------------------------------------------------
# Model & Health
# ---------------------------------------------------------

@admin_bp.get('/model')
def model():
    metrics = None
    metrics_path = BASE_DIR / 'training' / 'truthlens_metrics.json'

    if metrics_path.exists():
        try:
            loaded_metrics = json.loads(
                metrics_path.read_text(encoding='utf-8')
            )
            metrics = loaded_metrics if isinstance(loaded_metrics, dict) else None
        except (OSError, json.JSONDecodeError):
            metrics = None

    db_ok = True
    try:
        db.session.execute(text('SELECT 1')).scalar()
    except Exception:
        db_ok = False
        db.session.rollback()
        app.logger.exception("Admin health check DB query failed")

    confusion_matrix_path = (
        BASE_DIR / 'training' / 'truthlens_confusion_matrix.png'
    )

    return render_template(
        'admin/model.html',
        model_loaded=truthlens_model is not None,
        vectorizer_loaded=truthlens_vectorizer is not None,
        model_error=model_load_error,
        model_file=file_info(MODEL_PATH),
        vectorizer_file=file_info(VECTORIZER_PATH),
        model_classes=list(truthlens_model.classes_)
        if truthlens_model is not None else [],
        threshold=get_fake_threshold(),
        model_version=SystemSetting.get('model_version', 'v1'),
        metrics=metrics,
        db_ok=db_ok,
        confusion_matrix_exists=confusion_matrix_path.exists(),
        total_analyses=AnalysisHistory.query.count(),
        human_metrics=corrected_label_metrics(),
        corrected_count=AnalysisHistory.query.filter(
            AnalysisHistory.corrected_label.isnot(None)
        ).count(),
    )


@admin_bp.post('/model/settings')
def update_model_settings():
    raw_threshold = request.form.get('fake_threshold', '').strip()

    try:
        threshold = float(raw_threshold)
    except ValueError:
        flash("Threshold must be a number between 0.05 and 0.95.", "danger")
        return redirect(url_for('admin.model'))

    if not 0.05 <= threshold <= 0.95:
        flash("Threshold must be between 0.05 and 0.95.", "danger")
        return redirect(url_for('admin.model'))

    version = request.form.get('model_version', '').strip()[:20]

    SystemSetting.set('fake_threshold', f'{threshold:.2f}')
    if version:
        SystemSetting.set('model_version', version)

    log_admin_action(
        'model.update_settings',
        detail=(
            f'threshold={threshold:.2f}, '
            f'version={version or "unchanged"}'
        ),
    )
    db.session.commit()

    flash("Model settings saved.", "success")
    return redirect(url_for('admin.model'))


@admin_bp.get('/model/confusion-matrix')
def confusion_matrix():
    path = BASE_DIR / 'training' / 'truthlens_confusion_matrix.png'

    if not path.exists():
        abort(404)

    return send_file(path, mimetype='image/png', max_age=0)


app.register_blueprint(admin_bp)


# ---------------------------------------------------------
# Account: change password
# ---------------------------------------------------------

@app.route('/account/password', methods=['GET', 'POST'])
@login_required
def change_password():
    if 'user_csrf_token' not in session:
        session['user_csrf_token'] = secrets.token_urlsafe(32)

    if request.method == 'POST':
        expected = session['user_csrf_token']
        received = request.form.get('csrf_token', '')

        if not received or not received.isascii() or not secrets.compare_digest(
            expected, received
        ):
            abort(400)

        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        user = db.session.get(User, session['user_id'])

        if not user or not check_password_hash(
            user.password, current_password
        ):
            flash("Current password is incorrect.", "danger")
        elif len(new_password) < 8:
            flash("New password must be at least 8 characters.", "danger")
        elif new_password != confirm_password:
            flash("New passwords do not match.", "danger")
        else:
            user.password = generate_password_hash(new_password)
            db.session.commit()

            # Rotate the token after a successful credential change.
            session['user_csrf_token'] = secrets.token_urlsafe(32)
            flash("Password updated successfully.", "success")

            if session.get('user_role') == 'admin':
                return redirect(url_for('admin.overview'))
            return redirect(url_for('home'))

    return render_template(
        'change_password.html',
        csrf_token=session['user_csrf_token'],
    )


# ---------------------------------------------------------
# CLI: bootstrap an administrator
# ---------------------------------------------------------

@app.cli.command('create-admin')
@click.option('--name', prompt='Admin name')
@click.option('--email', prompt='Admin email')
@click.option(
    '--password',
    prompt=True,
    hide_input=True,
    confirmation_prompt=True,
)
def create_admin(name, email, password):
    """Create an administrator account."""

    email = email.strip().lower()
    name = name.strip()

    if len(name) < 2:
        raise click.ClickException("Name is too short.")
    if len(password) < 8:
        raise click.ClickException(
            "Password must be at least 8 characters."
        )
    if User.query.filter_by(email=email).first():
        raise click.ClickException(
            "An account with that email already exists."
        )

    db.session.add(User(
        name=name,
        email=email,
        password=generate_password_hash(password),
        role='admin',
    ))
    db.session.commit()

    click.echo(f"Admin account created for {email}.")


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

    # db.create_all() does not add columns to an existing table. Warn
    # clearly instead of failing later with an opaque SQL error.
    required_columns = {
        'user': {
            'is_active', 'last_login_at', 'last_login_ip',
            'failed_logins', 'locked_until',
        },
        'analysis_history': {
            'source_type', 'source_url', 'model_version', 'corrected_label',
            'corrected_by', 'corrected_at',
        },
    }

    try:
        inspector = inspect(db.engine)
        missing = {}

        for table, columns in required_columns.items():
            existing_columns = {
                column['name'] for column in inspector.get_columns(table)
            }
            absent = columns - existing_columns
            if absent:
                missing[table] = sorted(absent)

        if missing:
            detail = '; '.join(
                f"{table}: {', '.join(columns)}"
                for table, columns in missing.items()
            )
            print(
                "WARNING: The database is missing admin-phase columns "
                f"({detail}).\n"
                "         Run: python migrate_admin_schema.py"
            )
    except Exception as exc:
        print(f"Warning: could not verify database schema: {exc}")


# ---------------------------------------------------------
# Run Application
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")