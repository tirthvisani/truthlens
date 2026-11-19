"""Add the admin-phase schema to an existing TruthLens database.

`db.create_all()` only creates missing tables; it never adds columns to
tables that already exist. Run this once after pulling the admin update:

    python migrate_admin_schema.py

The script is safe to run more than once — it checks the live schema first.
New tables (admin_audit_log, system_setting) are created automatically.
"""

from sqlalchemy import inspect, text

from main import app, db


COLUMNS = {
    'user': {
        'is_active': 'BOOLEAN NOT NULL DEFAULT 1',
        'last_login_at': 'DATETIME NULL',
        'last_login_ip': 'VARCHAR(45) NULL',
        'failed_logins': 'INTEGER NOT NULL DEFAULT 0',
        'locked_until': 'DATETIME NULL',
    },
    'analysis_history': {
        'source_type': "VARCHAR(10) NOT NULL DEFAULT 'text'",
        'source_url': 'TEXT NULL',
        'model_version': "VARCHAR(20) NOT NULL DEFAULT 'v1'",
        'corrected_label': 'VARCHAR(10) NULL',
        'corrected_by': 'INTEGER NULL',
        'corrected_at': 'DATETIME NULL',
    },
}

INDEXES = {
    'ix_analysis_history_corrected_label': (
        'analysis_history', 'corrected_label'
    ),
}


def quote(identifier):
    if db.engine.dialect.name == 'mysql':
        return f'`{identifier}`'
    return f'"{identifier}"'


def main():
    with app.app_context():
        db.create_all()
        inspector = inspect(db.engine)
        existing_tables = set(inspector.get_table_names())
        changes = 0

        for table, columns in COLUMNS.items():
            if table not in existing_tables:
                print(f"Table {table} does not exist yet — created by create_all().")
                continue

            existing_columns = {
                column['name'] for column in inspector.get_columns(table)
            }

            for name, ddl in columns.items():
                if name in existing_columns:
                    continue

                statement = (
                    f'ALTER TABLE {quote(table)} '
                    f'ADD COLUMN {quote(name)} {ddl}'
                )
                print(f"Adding {table}.{name} ...")
                with db.engine.begin() as connection:
                    connection.execute(text(statement))
                changes += 1

        for index_name, (table, column) in INDEXES.items():
            if table not in existing_tables:
                continue

            existing_indexes = {
                index['name'] for index in inspector.get_indexes(table)
            }

            if index_name in existing_indexes:
                continue

            print(f"Creating index {index_name} ...")
            with db.engine.begin() as connection:
                connection.execute(text(
                    f'CREATE INDEX {quote(index_name)} '
                    f'ON {quote(table)} ({quote(column)})'
                ))
            changes += 1

        if changes:
            print(f"Migration complete - {changes} schema change(s) applied.")
        else:
            print("Schema already up to date - nothing to do.")


if __name__ == '__main__':
    main()
