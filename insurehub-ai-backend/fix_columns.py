# Run this ONCE from your project root (same folder as your app/ directory):
#     python fix_columns.py
#
# It adds the required_fields/obtained_fields/missing_fields columns to the
# EXISTING sessions table in your SQLite DB, without touching any data
# already in there. Safe to run more than once — it skips columns that are
# already present.

from sqlalchemy import inspect, text
from app.core.database import engine

NEW_COLUMNS = {
    "required_fields": "TEXT",
    "obtained_fields": "TEXT",
    "missing_fields": "TEXT",
}

inspector = inspect(engine)
existing_columns = {col["name"] for col in inspector.get_columns("sessions")}

with engine.connect() as conn:
    for col_name, col_type in NEW_COLUMNS.items():
        if col_name in existing_columns:
            print(f"'{col_name}' already exists — skipping.")
            continue
        print(f"Adding column '{col_name}'...")
        conn.execute(text(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_type}"))
    conn.commit()

print("Done — restart your backend now.")