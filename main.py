from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime


app = Flask(__name__)
import os

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
# Admin Dashboard
# ---------------------------------------------------------

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():

    total_users = User.query.count()
    total_admins = User.query.filter_by(role='admin').count()
    total_normal_users = User.query.filter_by(role='user').count()

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