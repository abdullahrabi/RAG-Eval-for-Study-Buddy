# entrypoint.py

import os
import signal
import socket
import subprocess
import sys
import time


HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8080"))

dashboard_process = None
evaluation_process = None


def wait_for_port(host, port, timeout=60):
    print(f"⏳ Waiting for dashboard on {host}:{port}...")

    start_time = time.time()

    while time.time() - start_time < timeout:

        # Check whether dashboard process has already died.
        if dashboard_process is not None:
            return_code = dashboard_process.poll()

            if return_code is not None:
                print(
                    f"❌ Dashboard process stopped "
                    f"with exit code {return_code}"
                )
                return False

        try:
            with socket.create_connection(
                (host, port),
                timeout=1
            ):
                print(
                    f"✅ Dashboard is ready on "
                    f"http://localhost:{port}"
                )
                return True

        except (ConnectionRefusedError, TimeoutError, OSError):
            time.sleep(1)

    return False


def shutdown(signum=None, frame=None):
    global dashboard_process
    global evaluation_process

    print("\n🛑 Shutting down RAG Evaluation System...")

    if evaluation_process is not None:
        if evaluation_process.poll() is None:
            evaluation_process.terminate()

    if dashboard_process is not None:
        if dashboard_process.poll() is None:
            dashboard_process.terminate()

    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)

if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, shutdown)


if __name__ == "__main__":

    print("=" * 60)
    print("🚀 RAG Evaluation System")
    print("=" * 60)

    # =========================================================
    # 1. START DASHBOARD FIRST
    # =========================================================

    print("\n📊 Starting TruLens Dashboard...")

    dashboard_process = subprocess.Popen(
        [sys.executable, "Dashboard.py"]
    )

    # =========================================================
    # 2. WAIT FOR DASHBOARD
    # =========================================================

    dashboard_ready = wait_for_port(
        HOST,
        PORT,
        timeout=60
    )

    if not dashboard_ready:
        print("❌ Dashboard did not start.")
        shutdown()

    # =========================================================
    # 3. START RAG EVALUATION
    # =========================================================

    print("\n🧪 Starting RAG evaluation...")

    evaluation_process = subprocess.Popen(
        [sys.executable, "RAG_Eval.py"]
    )

    # Wait until evaluation finishes.
    evaluation_return_code = evaluation_process.wait()

    print(
        "\n🧪 RAG evaluation finished "
        f"with exit code {evaluation_return_code}"
    )

    # =========================================================
    # 4. KEEP DASHBOARD RUNNING
    # =========================================================

    print("\n" + "=" * 60)
    print("✅ Evaluation finished.")
    print("📊 TruLens Dashboard is still running.")
    print(
        f"🌐 Dashboard: http://localhost:{PORT}"
    )
    print("=" * 60)

    try:
        dashboard_process.wait()

    except KeyboardInterrupt:
        shutdown()