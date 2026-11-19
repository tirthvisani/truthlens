# TruthLens AI

A Flask fake-news detector with a public analyzer and a full administration
workspace. Paste a claim or article URL, get a model verdict with a confidence
score and the words that influenced it, revisit results in history, and let
administrators add corrected labels, manage accounts, and audit sensitive
actions.

## Features

### Analyzer
- Paste article text or submit a public article URL.
- URL mode extracts the article body with BeautifulSoup and blocks private /
  internal addresses before making the request.
- Predictions are stored with verdict, confidence, source type, source URL, and
  model version.
- **Verdict explainability**: the analyzer returns and displays the top TF-IDF
  terms that pushed the prediction toward Fake or Real, based on feature value ×
  Logistic Regression coefficient.
- Per-user history with delete and clear actions.
- Configurable decision threshold: a claim is labelled **Fake** when the
  model's fake probability is at or above the admin-set threshold (default
  `0.50`).

### Admin workspace (`/admin`)
- **Overview**: user and analysis KPIs, human-label metrics (accuracy,
  precision, recall, F1 from corrected labels), 7-day activity, verdict split,
  registration trend, low-confidence queue, and recent admin activity.
- **Users**: search and filter by role/status, promote or demote, suspend or
  reactivate, reset a password to a one-time temporary password, delete a user
  with their history, and export the filtered list as CSV.
- **Analyses**: filter by From date, To date, user, verdict, minimum confidence,
  and maximum confidence; open an analysis to see its verdict explanation and
  set a corrected human label; delete an entry; export filtered rows for
  retraining.
- **Audit log**: every sensitive admin action with actor, target, detail, IP
  address, and timestamp, with action/admin filters.
- **Model & health**: model, vectorizer, and database status; file details;
  active model version; decision threshold; offline metrics from
  `training/truthlens_metrics.json`; human-label metrics; confusion matrix;
  corrected-label retraining exports.
- **Account security**: self-service password change.

### Security
- Passwords are hashed with Werkzeug.
- Five failed logins trigger a 15-minute lockout; suspended accounts cannot log
  in and existing sessions are revoked on the next request.
- Successful logins record the time and IP address.
- Admin routes require an active `admin` account on every request.
- Admin POSTs and the password-change form require a session-bound CSRF token.
- Admin responses are sent with `Cache-Control: no-store`,
  `X-Frame-Options: DENY`, and `X-Content-Type-Options: nosniff`.
- Production mode requires `SECRET_KEY` and marks session cookies `Secure`.
- Admin analytics exclude analyses created by admin accounts.

## Project structure

```text
main.py                     Flask app, models, routes, CLI
migrate_admin_schema.py     Adds the admin columns to an existing database
requirements.txt            Runtime dependencies
requirements-dev.txt        Runtime + test dependencies
.env.example                Environment variable template
templates/                  Public pages and admin templates
training/                   Model, vectorizer, evaluation image, training steps
tests/                      Pytest suite (in-memory SQLite)
```

## Requirements

- Python 3.10+
- MySQL 8 (or MariaDB) for the normal setup; tests use SQLite
- The model files `training/truthlens_model.pkl` and
  `training/truthlens_tfidf_vectorizer.pkl`

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate    # macOS / Linux
   pip install -r requirements.txt
   ```

2. Create the MySQL database:

   ```sql
   CREATE DATABASE truthlens_ai
     CHARACTER SET utf8mb4
     COLLATE utf8mb4_unicode_ci;
   ```

3. Create your environment file:

   ```bash
   cp .env.example .env
   ```

   Edit `.env`:

   ```dotenv
   SECRET_KEY=replace-with-a-long-random-string
   DATABASE_URL=mysql+pymysql://root:yourpassword@localhost/truthlens_ai
   FLASK_ENV=development
   FLASK_DEBUG=0
   ```

   Generate a secret key with:

   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

4. Initialise or upgrade the schema:

   ```bash
   python migrate_admin_schema.py
   ```

   This is safe to re-run. It creates missing tables and adds the admin-phase
   columns to an existing `user` / `analysis_history` table. If you skip this
   step against an older database, startup prints a warning and admin pages
   will fail until the columns exist.

5. Create the first administrator:

   ```bash
   flask --app main create-admin
   ```

   You will be prompted for a name, email, and password (minimum 8
   characters). Alternatively:

   ```bash
   python -m flask --app main create-admin
   ```

6. Run the app:

   ```bash
   python main.py
   ```

   Open <http://127.0.0.1:5000>. Administrators are redirected to `/admin`
   after login; normal users land on `/home`.

## Configuration

Values are read from the environment (and from `.env` when present).

| Variable | Default | Purpose |
| --- | --- | --- |
| `SECRET_KEY` | dev-only fallback | Session signing. **Required** when `FLASK_ENV=production`. |
| `DATABASE_URL` | `mysql+pymysql://root:@localhost/truthlens_ai` | SQLAlchemy connection string. |
| `FLASK_ENV` | `development` | Set to `production` for secure cookies and required secret. |
| `FLASK_DEBUG` | `0` | `1` enables the Flask debugger. Never use in production. |

The fake threshold and active model version are stored in the
`system_setting` table and edited from **Admin → Model & health**. They are
not environment variables.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite configures an in-memory SQLite database before importing the app, so
it does not touch MySQL and does not require a running server. It covers admin
access control, CSRF rejection, suspension and lockout, corrected-label workflow
and audit entries, explainability, CSV export, model settings, and user deletion.

## Training

The `training/` folder contains the pipeline scripts used to build the
classifier:

1. `truthlens_step2_preprocessing.py`
2. `truthlens_step3_train_model.py`
3. `truthlens_step4_evaluate_model.py`
4. `truthlens_step5_improve_model.py`
5. `truthlens_step6_audit.py`

The running app loads `truthlens_model.pkl` and
`truthlens_tfidf_vectorizer.pkl` at import time. If evaluation output is saved
to `training/truthlens_metrics.json`, the admin model page displays it
automatically.

## Deployment notes

- Set `FLASK_ENV=production`, a strong `SECRET_KEY`, and a production
  `DATABASE_URL`.
- Serve over HTTPS. Session cookies are only marked `Secure` in production.
- Do not run `python main.py` in production. Use a WSGI server, for example:

  ```bash
  # Linux
  gunicorn --workers 3 --bind 0.0.0.0:8000 main:app

  # Windows
  waitress-serve --listen=0.0.0.0:8000 main:app
  ```

- Run `python migrate_admin_schema.py` as part of each deploy that changes the
  schema.
- Back up the database before deleting accounts or running schema changes.
- Keep `.env` out of version control (it is already ignored).
