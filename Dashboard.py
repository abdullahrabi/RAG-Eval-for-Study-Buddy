# dashboard.py

import os

from dotenv import load_dotenv
from trulens.core import TruSession
from trulens.dashboard import run_dashboard

load_dotenv()

DB_URL = os.getenv(
    "TRULENS_DB_URL",
    "sqlite:///data/default.sqlite"
)

PORT = int(os.getenv("PORT", "8080"))

os.makedirs("data", exist_ok=True)

print("=" * 60)
print("🚀 Starting TruLens Dashboard")
print("=" * 60)
print(f"📁 Database: {DB_URL}")
print(f"📡 Port: {PORT}")
print("=" * 60)

session = TruSession(database_url=DB_URL)

# IMPORTANT:
# Explicitly pass the port.
# Do NOT use force=True on Windows.
dashboard_process = run_dashboard(
    session=session,
    port=PORT,
    address="0.0.0.0"
)

print(f"📊 TruLens Dashboard running on port {PORT}")