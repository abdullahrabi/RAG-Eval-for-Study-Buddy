# dashboard.py

import os

# Set Streamlit config BEFORE importing trulens/streamlit
PORT = int(os.getenv("PORT", "8080"))
os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
os.environ.setdefault("STREAMLIT_SERVER_PORT", str(PORT))
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

from dotenv import load_dotenv
from trulens.core import TruSession
from trulens.dashboard import run_dashboard

load_dotenv()

DB_URL = os.getenv(
    "TRULENS_DB_URL",
    "sqlite:///data/default.sqlite",
)

os.makedirs("data", exist_ok=True)

print("=" * 60)
print("🚀 Starting TruLens Dashboard")
print("=" * 60)
print(f"📁 Database: {DB_URL}")
print(f"📡 Port: {PORT}")
print("=" * 60)

session = TruSession(database_url=DB_URL)

# No `address` kwarg — Streamlit picks up the env vars above
dashboard_process = run_dashboard(
    session=session,
    port=PORT,
)

print(f"📊 TruLens Dashboard running on port {PORT}")