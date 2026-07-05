# entrypoint.py
import subprocess
import sys
import os
import threading
import time

# Import your main script
from RAG_Eval import main as evaluator_main, create_dashboard_file

def run_dashboard():
    """Run Streamlit dashboard"""
    port = os.getenv("PORT", "8501")
    cmd = [
        "streamlit", "run", "dashboard.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--server.headless", "true"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    print("🚀 Starting RAG Evaluation System")
    
    # Create dashboard file
    create_dashboard_file()
    
    # Start evaluator in background
    eval_thread = threading.Thread(target=evaluator_main, daemon=True)
    eval_thread.start()
    
    # Give evaluator time to initialize
    time.sleep(2)
    
    # Run dashboard (this blocks)
    run_dashboard()