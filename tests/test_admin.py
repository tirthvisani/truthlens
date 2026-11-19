"""Admin-area regression tests.

These run against an isolated in-memory SQLite database configured in
``conftest.py``, so they never touch the real MySQL data.
"""

from datetime import datetime

from conftest import create_user, login_session

import main as truthlens

USER_ROLE_URL = "/admin/users/{user_id}/role"
USER_STATUS_URL = "/admin/users/{user_id}/status"


def add_analysis(user_id, content="A suspicious claim about the moon.",
                 prediction="Fake", label=1, confidence=88.5,
                 source_type="text", corrected_label=None,
                 corrected_by=None):
    with truthlens.app.app_context():
        analysis = truthlens.AnalysisHistory(
            user_id=user_id,
            content=content,
            prediction=prediction,
            label=label,
            confidence=confidence,
            source_type=source_type,
            corrected_label=corrected_label,
            corrected_by=corrected_by,
            corrected_at=(
                datetime.utcnow() if corrected_label else None
            ),
        )
        truthlens.db.session.add(analysis)
        truthlens.db.session.commit()
        return analysis.id


def test_anonymous_is_redirected_to_login(client):
    response = client.get("/admin/")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_normal_user_gets_forbidden(client, user_id):
    login_session(client, user_id, role="user")

    response = client.get("/admin/")

    assert response.status_code == 403


def test_admin_dashboard_renders(client, admin_id):
    login_session(client, admin_id, role="admin")

    response = client.get("/admin/")

    assert response.status_code == 200
    assert b"Admin Dashboard" in response.data
    assert response.headers["Cache-Control"] == "no-store"


def test_admin_post_without_csrf_token_is_rejected(client, admin_id, user_id):
    login_session(client, admin_id, role="admin")

    response = client.post(
        USER_STATUS_URL.format(user_id=user_id),
        data={"action": "suspend"},
    )

    assert response.status_code == 400
    assert "CSRF" in response.get_data(as_text=True)


def test_non_ascii_csrf_token_is_rejected_without_server_error(
    client, admin_id, user_id
):
    login_session(client, admin_id, role="admin")

    response = client.post(
        USER_STATUS_URL.format(user_id=user_id),
        data={"action": "suspend", "csrf_token": "tökén"},
    )

    assert response.status_code == 400


def test_all_admin_pages_render(client, admin_id, user_id):
    analysis_id = add_analysis(user_id)
    login_session(client, admin_id, role="admin")

    paths = [
        "/admin/",
        "/admin/dashboard",
        "/admin/users",
        f"/admin/users/{admin_id}",
        f"/admin/users/{user_id}",
        "/admin/analyses",
        "/admin/analyses?verdict=Fake&min_confidence=0&max_confidence=100",
        f"/admin/analyses/{analysis_id}",
        "/admin/audit",
        "/admin/model",
        "/account/password",
        "/home",
        "/history",
    ]

    for path in paths:
        response = client.get(path)
        assert response.status_code == 200, path


def test_admin_can_suspend_and_reactivate_user(client, admin_id, user_id):
    login_session(client, admin_id, role="admin")

    response = client.post(
        USER_STATUS_URL.format(user_id=user_id),
        data={"action": "suspend", "csrf_token": "test-csrf"},
    )
    assert response.status_code == 302

    with truthlens.app.app_context():
        assert truthlens.db.session.get(truthlens.User, user_id).is_active is False

    response = client.post(
        USER_STATUS_URL.format(user_id=user_id),
        data={"action": "activate", "csrf_token": "test-csrf"},
    )
    assert response.status_code == 302

    with truthlens.app.app_context():
        assert truthlens.db.session.get(truthlens.User, user_id).is_active is True
        actions = [
            row.action for row in truthlens.AdminAuditLog.query.all()
        ]
        assert "user.suspend" in actions
        assert "user.activate" in actions


def test_admin_cannot_change_own_role(client, admin_id):
    login_session(client, admin_id, role="admin")

    response = client.post(
        USER_ROLE_URL.format(user_id=admin_id),
        data={"role": "user", "csrf_token": "test-csrf"},
    )

    assert response.status_code == 302

    with truthlens.app.app_context():
        admin = truthlens.db.session.get(truthlens.User, admin_id)
        assert admin.role == "admin"
        assert truthlens.AdminAuditLog.query.count() == 0


def test_suspended_user_cannot_log_in(client, user_id):
    with truthlens.app.app_context():
        user = truthlens.db.session.get(truthlens.User, user_id)
        user.is_active = False
        truthlens.db.session.commit()

    response = client.post(
        "/login",
        data={"email": "user@example.com", "password": "password123"},
    )

    assert response.status_code == 200
    assert b"suspended" in response.data.lower()

    with client.session_transaction() as session:
        assert "user_id" not in session


def test_failed_logins_lock_the_account(client, user_id):
    for _ in range(truthlens.MAX_LOGIN_ATTEMPTS):
        client.post(
            "/login",
            data={"email": "user@example.com", "password": "wrong-password"},
        )

    with truthlens.app.app_context():
        user = truthlens.db.session.get(truthlens.User, user_id)
        assert user.locked_until is not None
        assert user.locked_until > datetime.utcnow()

    response = client.post(
        "/login",
        data={"email": "user@example.com", "password": "password123"},
    )

    assert response.status_code == 200
    assert b"Too many failed login attempts" in response.data


def test_corrected_label_saves_label_and_audit_entry(client, admin_id, user_id):
    analysis_id = add_analysis(user_id)
    login_session(client, admin_id, role="admin")

    response = client.post(
        f"/admin/analyses/{analysis_id}/corrected-label",
        data={"corrected_label": "Real", "csrf_token": "test-csrf"},
    )

    assert response.status_code == 302

    with truthlens.app.app_context():
        analysis = truthlens.db.session.get(
            truthlens.AnalysisHistory, analysis_id
        )
        assert analysis.corrected_label == "Real"
        assert analysis.corrected_by == admin_id
        assert analysis.corrected_at is not None

        entry = truthlens.AdminAuditLog.query.filter_by(
            action="analysis.corrected_label"
        ).one()
        assert entry.target_id == analysis_id
        assert entry.admin_id == admin_id


def test_corrected_label_can_be_cleared(client, admin_id, user_id):
    analysis_id = add_analysis(
        user_id, corrected_label="Fake", corrected_by=admin_id
    )
    login_session(client, admin_id, role="admin")

    response = client.post(
        f"/admin/analyses/{analysis_id}/corrected-label",
        data={"corrected_label": "", "csrf_token": "test-csrf"},
    )

    assert response.status_code == 302

    with truthlens.app.app_context():
        analysis = truthlens.db.session.get(
            truthlens.AnalysisHistory, analysis_id
        )
        assert analysis.corrected_label is None
        assert analysis.corrected_by is None
        assert analysis.corrected_at is None


def test_corrected_label_rejects_unknown_value(client, admin_id, user_id):
    analysis_id = add_analysis(user_id)
    login_session(client, admin_id, role="admin")

    response = client.post(
        f"/admin/analyses/{analysis_id}/corrected-label",
        data={"corrected_label": "Maybe", "csrf_token": "test-csrf"},
    )

    assert response.status_code == 400


def test_human_label_metrics_use_corrected_labels(client, admin_id, user_id):
    add_analysis(user_id, content="true fake", prediction="Fake",
                 label=1, corrected_label="Fake", corrected_by=admin_id)
    add_analysis(user_id, content="false fake", prediction="Fake",
                 label=1, corrected_label="Real", corrected_by=admin_id)
    add_analysis(user_id, content="missed fake", prediction="Real",
                 label=0, corrected_label="Fake", corrected_by=admin_id)
    add_analysis(user_id, content="true real", prediction="Real",
                 label=0, corrected_label="Real", corrected_by=admin_id)

    with truthlens.app.app_context():
        metrics = truthlens.corrected_label_metrics()

    assert metrics["total"] == 4
    assert metrics["correct"] == 2
    assert metrics["incorrect"] == 2
    assert metrics["accuracy"] == 50.0
    assert metrics["precision"] == 50.0
    assert metrics["recall"] == 50.0
    assert metrics["f1"] == 50.0
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (
        1, 1, 1, 1
    )


def test_explain_prediction_returns_ranked_terms():
    explanation = truthlens.explain_prediction(
        "Doctors are hiding a miracle cure from the public, sources claim."
    )

    assert isinstance(explanation, list)
    assert explanation
    assert set(explanation[0]) == {
        "term", "weight", "direction", "strength"
    }
    assert explanation[0]["direction"] in ("Fake", "Real")
    strengths = [item["strength"] for item in explanation]
    assert strengths == sorted(strengths, reverse=True)


def test_analyze_api_includes_explanation(client, user_id):
    login_session(client, user_id, role="user")

    response = client.post(
        "/api/analyze",
        json={
            "content": (
                "Scientists announce a new battery technology that "
                "could charge electric cars in minutes."
            )
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert isinstance(payload["explanation"], list)


def test_analysis_filters_ignore_removed_options(client, admin_id, user_id):
    add_analysis(user_id, content="Pasted source claim", source_type="text")
    add_analysis(user_id, content="URL source claim", source_type="url")
    login_session(client, admin_id, role="admin")

    response = client.get("/admin/analyses?source=url&q=URL&review=pending")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Pasted source claim" in body
    assert "URL source claim" in body


def test_analysis_filters_apply_verdict_and_confidence(client, admin_id, user_id):
    add_analysis(user_id, content="High confidence claim",
                 prediction="Fake", label=1, confidence=90)
    add_analysis(user_id, content="Low confidence claim",
                 prediction="Real", label=0, confidence=40)
    login_session(client, admin_id, role="admin")

    response = client.get(
        "/admin/analyses?verdict=Fake&min_confidence=80&max_confidence=95"
    )

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "High confidence claim" in body
    assert "Low confidence claim" not in body


def test_analysis_filters_apply_user_and_date(client, admin_id, user_id):
    other_user_id = create_user("user", "other@example.com")
    first_id = add_analysis(user_id, content="First user old claim")
    second_id = add_analysis(other_user_id, content="Other user new claim")

    with truthlens.app.app_context():
        truthlens.db.session.get(
            truthlens.AnalysisHistory, first_id
        ).date_created = datetime(2024, 1, 15, 12, 0, 0)
        truthlens.db.session.get(
            truthlens.AnalysisHistory, second_id
        ).date_created = datetime(2026, 1, 15, 12, 0, 0)
        truthlens.db.session.commit()

    login_session(client, admin_id, role="admin")

    response = client.get(
        f"/admin/analyses?user_id={user_id}"
        "&start=2024-01-01&end=2024-12-31"
    )

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "First user old claim" in body
    assert "Other user new claim" not in body


def test_overview_and_model_render_human_metrics(client, admin_id, user_id):
    add_analysis(
        user_id, content="Human-labelled claim", prediction="Fake",
        label=1, corrected_label="Real", corrected_by=admin_id,
    )
    login_session(client, admin_id, role="admin")

    for path in ("/admin/", "/admin/model"):
        response = client.get(path)
        assert response.status_code == 200, path
        body = response.get_data(as_text=True)
        assert "Human-label" in body
        assert "Precision (Fake)" in body


def test_analysis_csv_export_contains_rows_and_is_audited(
    client, admin_id, user_id
):
    analysis_id = add_analysis(
        user_id, content="Exported claim", corrected_label="Real",
        corrected_by=admin_id,
    )
    login_session(client, admin_id, role="admin")

    response = client.get("/admin/analyses/export.csv")

    assert response.status_code == 200
    assert "text/csv" in response.headers["Content-Type"]
    assert "attachment" in response.headers["Content-Disposition"]
    body = response.get_data(as_text=True)
    assert "corrected_label" in body
    assert "Exported claim" in body
    assert "user@example.com" in body

    with truthlens.app.app_context():
        assert truthlens.AdminAuditLog.query.filter_by(
            action="export.analyses"
        ).count() == 1
        assert truthlens.db.session.get(
            truthlens.AnalysisHistory, analysis_id
        ) is not None


def test_analysis_export_neutralises_formula_cells(
    client, admin_id, user_id
):
    add_analysis(user_id, content='=HYPERLINK("http://example.invalid")')
    login_session(client, admin_id, role="admin")

    response = client.get("/admin/analyses/export.csv")

    assert response.status_code == 200
    assert "\"'=HYPERLINK" in response.get_data(as_text=True)


def test_model_settings_are_saved(client, admin_id):
    login_session(client, admin_id, role="admin")

    response = client.post(
        "/admin/model/settings",
        data={
            "fake_threshold": "0.72",
            "model_version": "v2-test",
            "csrf_token": "test-csrf",
        },
    )

    assert response.status_code == 302

    with truthlens.app.app_context():
        assert truthlens.SystemSetting.get("fake_threshold") == "0.72"
        assert truthlens.SystemSetting.get("model_version") == "v2-test"
        assert truthlens.AdminAuditLog.query.filter_by(
            action="model.update_settings"
        ).count() == 1


def test_model_settings_reject_out_of_range_threshold(client, admin_id):
    login_session(client, admin_id, role="admin")

    response = client.post(
        "/admin/model/settings",
        data={"fake_threshold": "1.5", "csrf_token": "test-csrf"},
    )

    assert response.status_code == 302

    with truthlens.app.app_context():
        assert truthlens.SystemSetting.get("fake_threshold") is None
        assert truthlens.AdminAuditLog.query.count() == 0


def test_admin_can_delete_a_normal_user_and_history(client, admin_id, user_id):
    add_analysis(user_id)
    login_session(client, admin_id, role="admin")

    response = client.post(
        f"/admin/users/{user_id}/delete",
        data={"csrf_token": "test-csrf"},
    )

    assert response.status_code == 302

    with truthlens.app.app_context():
        assert truthlens.db.session.get(truthlens.User, user_id) is None
        assert truthlens.AnalysisHistory.query.filter_by(
            user_id=user_id
        ).count() == 0


def test_admin_cannot_delete_another_admin(client, admin_id):
    second_admin_id = create_user("admin", "second-admin@example.com")
    login_session(client, admin_id, role="admin")

    response = client.post(
        f"/admin/users/{second_admin_id}/delete",
        data={"csrf_token": "test-csrf"},
    )

    assert response.status_code == 302

    with truthlens.app.app_context():
        assert truthlens.db.session.get(
            truthlens.User, second_admin_id
        ) is not None
