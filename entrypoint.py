# entrypoint.py - Direct Run
import os
import sys
import subprocess

if __name__ == "__main__":
    port = os.getenv("PORT", "8080")
    
    print("="*60)
    print("🚀 Starting RAG Evaluation System")
    print("="*60)
    print(f"📡 PORT: {port}")
    print("="*60)
    
    # Run the main file
    subprocess.run(["python", "RAG_Eval_All_Users.py"])