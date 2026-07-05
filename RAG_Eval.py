"""
RAG_Eval_All_Users_TruLens.py - Combined Version
Multi-user RAG evaluation with TruLens dashboard and SQLite database
Features:
- Background evaluation for all users
- TruLens dashboard with 5 metrics
- TruLens SQLite database (trulens.db)
- Real-time progress updates
- Railway compatible
"""

import os
import time
import re
import warnings
import pandas as pd
import json
import hashlib
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any
import sys
import concurrent.futures
import threading
import subprocess
import queue
import sqlite3
from collections import deque

warnings.filterwarnings('ignore')

load_dotenv()

print("="*60)
print("🔍 RAG EVALUATION WITH TRULENS DASHBOARD")
print("="*60)

# ============================================
# ENVIRONMENT VARIABLES
# ============================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")
PORT = int(os.getenv("PORT", 8080))

print(f"GEMINI_API_KEY: {'✅' if GEMINI_API_KEY else '❌'}")
print(f"PINECONE_API_KEY: {'✅' if PINECONE_API_KEY else '❌'}")
print(f"GROQ_API_KEY: {'✅' if GROQ_API_KEY else '❌'}")
print(f"INDEX_NAME: {INDEX_NAME}")
print(f"PORT: {PORT}")

if not GEMINI_API_KEY or not PINECONE_API_KEY or not GROQ_API_KEY:
    print("❌ Missing API keys!")
    sys.exit(1)

# TruLens environment setup
os.environ["TRULENS_OTEL_TRACING"] = "1"
os.environ["TRULENS_OTEL_ENABLED"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "false"

# ============================================
# IMPORTS
# ============================================

from pinecone import Pinecone
from google import genai
from google.genai import types
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.groq import Groq as LlamaGroq

# TruLens imports
from trulens.core import TruSession, Feedback
from trulens.apps.app import TruApp

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
APP_NAME = f"RAG_Eval_All_Users_{RUN_ID}"

# ============================================
# GEMINI EMBEDDING
# ============================================

class GeminiDirectEmbedding(BaseEmbedding):
    api_key: str
    model_name: str = "gemini-embedding-2"
    dimension: int = 768
    
    def __init__(self, api_key: str, model_name: str = "gemini-embedding-2", dimension: int = 768, **kwargs):
        super().__init__(api_key=api_key, model_name=model_name, dimension=dimension, **kwargs)
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client
    
    def _get_query_embedding(self, query: str) -> list:
        return self._embed_text(query)
    
    def _get_text_embedding(self, text: str) -> list:
        return self._embed_text(text)
    
    async def _aget_query_embedding(self, query: str) -> list:
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> list:
        return self._get_text_embedding(text)
    
    def _embed_text(self, text: str) -> list:
        try:
            if not text or not text.strip():
                return None
            if len(text) > 8000:
                text = text[:8000]
            result = self.client.models.embed_content(
                model=self.model_name, contents=[text],
                config=types.EmbedContentConfig(output_dimensionality=self.dimension)
            )
            if result and result.embeddings and len(result.embeddings) > 0:
                emb = result.embeddings[0].values
                norm = sum(v**2 for v in emb) ** 0.5
                if norm > 0:
                    return [v / norm for v in emb]
            return None
        except Exception:
            return None
    
    @classmethod
    def class_name(cls) -> str:
        return "GeminiDirectEmbedding"

# ============================================
# FETCH ALL USERS FROM PINECONE
# ============================================

def fetch_all_users(pinecone_index) -> List[Dict]:
    print("\n🔍 Fetching all users from Pinecone...")
    
    try:
        import random
        dummy_vector = [random.uniform(0.01, 0.02) for _ in range(768)]
        
        results = pinecone_index.query(
            vector=dummy_vector,
            top_k=1000,
            include_metadata=True,
            namespace="users",
            filter={"type": {"$eq": "user_auth"}}
        )
        
        users = []
        seen_users = set()
        
        for match in results.matches:
            if match.metadata:
                user_id = match.metadata.get('user_id')
                email = match.metadata.get('email')
                if user_id and user_id not in seen_users:
                    seen_users.add(user_id)
                    users.append({
                        'user_id': user_id,
                        'email': email,
                        'created_at': match.metadata.get('created_at', '')
                    })
        
        print(f"✅ Found {len(users)} users")
        return users
    except Exception as e:
        print(f"❌ Error fetching users: {e}")
        return []

# ============================================
# FETCH NOTES FOR A USER
# ============================================

def fetch_user_notes(pinecone_index, user_id: str) -> List[Dict]:
    try:
        import random
        dummy_vector = [random.uniform(0.01, 0.02) for _ in range(768)]
        
        results = pinecone_index.query(
            vector=dummy_vector,
            top_k=100,
            include_metadata=True,
            namespace="notes",
            filter={
                "user_id": {"$eq": user_id},
                "type": {"$eq": "notes"}
            }
        )
        
        notes = []
        for match in results.matches:
            if match.metadata:
                notes.append({
                    'id': match.id,
                    'text': match.metadata.get('text', ''),
                    'chunk_index': match.metadata.get('chunk_index', 0),
                    'timestamp': match.metadata.get('timestamp', 0),
                    'source': match.metadata.get('source', 'uploaded_notes')
                })
        
        notes.sort(key=lambda x: x.get('chunk_index', 0))
        return notes
    except Exception as e:
        print(f"⚠️ Error fetching notes for {user_id}: {e}")
        return []

# ============================================
# GENERATE QUESTIONS FROM NOTES
# ============================================

def generate_questions_from_notes(notes: List[Dict], num_questions: int = 10) -> List[str]:
    if not notes:
        return []
    
    full_text = " ".join([note['text'] for note in notes])
    
    try:
        from groq import Groq as GroqClient
        
        client = GroqClient(api_key=GROQ_API_KEY)
        
        if len(full_text) > 8000:
            full_text = full_text[:8000]
        
        prompt = f"""Based on the following educational content, generate {num_questions} thoughtful questions.

CONTENT:
{full_text}

Generate exactly {num_questions} questions as a numbered list:"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Generate exactly the number of questions requested. Output only the numbered questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        questions_text = response.choices[0].message.content.strip()
        
        questions = []
        for line in questions_text.split('\n'):
            line = line.strip()
            match = re.match(r'^(\d+)[\.\)]?\s*(.*)', line)
            if match:
                question_text = match.group(2).strip()
                if question_text:
                    questions.append(question_text)
            elif line and len(line) > 10 and '?' in line:
                questions.append(line)
        
        if len(questions) < num_questions:
            questions = generate_fallback_questions(full_text, num_questions)
        
        return questions[:num_questions]
        
    except Exception as e:
        print(f"⚠️ Error generating questions: {e}")
        return generate_fallback_questions(full_text, num_questions)

def generate_fallback_questions(text: str, num_questions: int = 10) -> List[str]:
    default_questions = [
        "What is automata theory?",
        "What is a finite automaton?",
        "What are the applications of finite automata?",
        "What is a formal language?",
        "What is the difference between a string and a language?",
        "What is the Kleene star operator?",
        "What is the role of Turing machines?",
        "What is the significance of the pumping lemma?",
        "What are regular expressions?",
        "What is the relationship between finite automata and regular languages?"
    ]
    
    questions = default_questions[:num_questions]
    while len(questions) < num_questions:
        questions.append(default_questions[len(questions) % len(default_questions)])
    
    return questions

# ============================================
# MODEL ROUTER FOR FEEDBACK
# ============================================

class ModelRouter:
    def __init__(self, api_key: str):
        from groq import Groq as GroqClient
        self.client = GroqClient(api_key=api_key)
    
    def call_model(self, prompt: str, model: str) -> float:
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Output ONLY a number between 0 and 1."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=10
                )
                text = response.choices[0].message.content.strip()
                nums = re.findall(r'(\d+\.?\d*)', text)
                if nums:
                    score = float(nums[0])
                    if score > 1 and score <= 100:
                        score = score / 100
                    return max(0.0, min(1.0, score))
                time.sleep(2)
            except Exception:
                time.sleep(5)
        return 0.5

# ============================================
# TRULENS FEEDBACK FUNCTIONS
# ============================================

router = None

def get_router():
    global router
    if router is None:
        router = ModelRouter(GROQ_API_KEY)
    return router

def relevance(input: str, output: str) -> float:
    return get_router().call_model(f"Score relevance 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.1-8b-instant")

def quality(input: str, output: str) -> float:
    return get_router().call_model(f"Score quality 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.1-8b-instant")

def groundedness(input: str, output: str) -> float:
    return get_router().call_model(f"Score groundedness 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.3-70b-versatile")

def context_relevance(input: str, output: str) -> float:
    return get_router().call_model(f"Score context relevance 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.1-8b-instant")

def correctness(input: str, output: str) -> float:
    return get_router().call_model(f"Score correctness 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.3-70b-versatile")

# ============================================
# OPTIMIZED RAG SYSTEM
# ============================================

class OptimizedRAG:
    def __init__(self, pinecone_index, embed_model, llm, user_id: str = None):
        self.pinecone_index = pinecone_index
        self.embed_model = embed_model
        self.llm = llm
        self.user_id = user_id
    
    def query(self, question: str) -> str:
        try:
            query_embedding = self.embed_model._embed_text(question)
            if not query_embedding:
                return "No relevant documents found."
            
            filter_dict = {}
            if self.user_id:
                filter_dict["user_id"] = {"$eq": self.user_id}
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                namespace="notes",
                filter=filter_dict
            )
            
            contexts = []
            for match in results.matches:
                if match.metadata and 'text' in match.metadata:
                    contexts.append(match.metadata['text'][:800])
            
            if not contexts:
                return "No relevant documents found for this user."
            
            prompt = f"""Answer based on context.

CONTEXT:
{chr(10).join(contexts)}

QUESTION:
{question}

ANSWER:"""
            
            response = self.llm.complete(prompt)
            return str(response).strip()
        except Exception as e:
            return f"Error: {e}"

# ============================================
# BACKGROUND EVALUATOR WITH TRULENS
# ============================================

class BackgroundEvaluatorWithTruLens:
    def __init__(self):
        self.running = False
        self.thread = None
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.total_evaluations = 0
        self.completed_evaluations = 0
        
        # Connect to TruLens database
        self.session = TruSession(database_url="sqlite:///trulens.db")
        print("✅ TruLens database connection established.")
        
    def start(self, users: List[Dict], pinecone_index, embed_model, llm):
        """Start background evaluation with TruLens"""
        if self.running:
            print("⚠️ Evaluation already running")
            return
        
        self.running = True
        self.users = users
        self.total_evaluations = len(users) * 10  # 10 questions per user
        
        print(f"\n📊 Total users: {len(users)}")
        print(f"📊 Total evaluations: {self.total_evaluations}")
        
        # Start background thread
        self.thread = threading.Thread(
            target=self._run_evaluation,
            args=(users, pinecone_index, embed_model, llm)
        )
        self.thread.daemon = True
        self.thread.start()
        
        print(f"✅ Background evaluation started (Run ID: {self.run_id})")
    
    def _run_evaluation(self, users: List[Dict], pinecone_index, embed_model, llm):
        """Run evaluation with TruLens in background"""
        try:
            self.completed_evaluations = 0
            
            for user_idx, user in enumerate(users):
                if not self.running:
                    break
                
                user_id = user['user_id']
                email = user['email']
                
                print(f"\n👤 Evaluating user {user_idx+1}/{len(users)}: {email}")
                
                # Get user notes
                notes = fetch_user_notes(pinecone_index, user_id)
                
                if not notes:
                    print(f"⚠️ No notes found for {email}, skipping...")
                    continue
                
                # Generate questions
                questions = generate_questions_from_notes(notes, num_questions=10)
                print(f"📝 Generated {len(questions)} questions")
                
                # Create RAG instance
                rag = OptimizedRAG(pinecone_index, embed_model, llm, user_id=user_id)
                
                # Create RAG wrapper for TruLens
                class RAGWrapper:
                    def __init__(self, rag_instance):
                        self.rag = rag_instance
                    
                    def respond(self, question: str) -> str:
                        return self.rag.query(question)
                
                rag_wrapper = RAGWrapper(rag)
                
                # Setup TruLens feedback functions
                f_relevance = Feedback(relevance, name="Relevance").on_input_output()
                f_quality = Feedback(quality, name="Quality").on_input_output()
                f_groundedness = Feedback(groundedness, name="Groundedness").on_input_output()
                f_context_relevance = Feedback(context_relevance, name="Context Relevance").on_input_output()
                f_correctness = Feedback(correctness, name="Correctness").on_input_output()
                
                # Create TruLens app for this user
                tru_app = TruApp(
                    rag_wrapper,
                    app_name=f"{APP_NAME}_{email}",
                    app_version="v1.0",
                    feedbacks=[f_relevance, f_quality, f_groundedness, f_context_relevance, f_correctness],
                    main_method=rag_wrapper.respond
                )
                
                # Run evaluation with TruLens
                print(f"🔄 Running {len(questions)} questions for {email}...")
                
                with tru_app as recording:
                    for q_idx, question in enumerate(questions, 1):
                        if not self.running:
                            break
                        
                        print(f"  {q_idx}/{len(questions)}: {question[:50]}...")
                        rag_wrapper.respond(question)
                        self.completed_evaluations += 1
                
                print(f"✅ Completed evaluation for {email}")
            
            print("\n" + "="*60)
            print("✅ All evaluations complete!")
            print(f"📊 Total evaluations: {self.completed_evaluations}")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
    
    def stop(self):
        """Stop background evaluation"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("⏹️ Evaluation stopped")

# ============================================
# LAUNCH TRULENS DASHBOARD
# ============================================

def launch_trulens_dashboard():
    """Launch the TruLens dashboard"""
    
    print("\n" + "="*60)
    print("📊 Launching TruLens Dashboard...")
    print("="*60)
    print(f"✅ Dashboard available at: http://localhost:{PORT}")
    print(f"📁 Using TruLens database: trulens.db")
    print("="*60)
    
    try:
        # Try to use TruLens built-in dashboard
        from trulens.dashboard import run_dashboard
        
        # Create session
        session = TruSession(database_url="sqlite:///trulens.db")
        print("Starting TruLens dashboard on port", PORT)
        
        # Run the dashboard
        run_dashboard(session=session, port=PORT)
        
    except Exception as e:
        print(f"⚠️ TruLens dashboard error: {e}")
        print("Trying alternative method...")
        
        # Alternative: Use custom Streamlit dashboard
        try:
            create_trulens_standalone_dashboard()
            
            cmd = [
                "streamlit", "run", "trulens_standalone_dashboard.py",
                "--server.port", str(PORT),
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false"
            ]
            subprocess.run(cmd)
            
        except Exception as e2:
            print(f"❌ Alternative dashboard also failed: {e2}")
            print("\n💡 Dashboard could not be started automatically.")
            print("You can still access your results via:")
            print("  - TruLens SQLite database: trulens.db")
            print(f"  - Run: streamlit run trulens_standalone_dashboard.py --server.port {PORT}")

# ============================================
# STANDALONE TRULENS DASHBOARD
# ============================================

def create_trulens_standalone_dashboard():
    """Create a standalone TruLens dashboard"""
    
    dashboard_code = '''
"""
TruLens Standalone Dashboard - Shows all evaluations from trulens.db
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="TruLens RAG Evaluation Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .score-high { color: #4CAF50; }
    .score-medium { color: #FF9800; }
    .score-low { color: #f44336; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🔍 TruLens RAG Evaluation Dashboard</h1></div>', unsafe_allow_html=True)

DB_PATH = "trulens.db"

@st.cache_data(ttl=5)
def get_data(query):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# Get summary from TruLens database
summary = get_data("""
    SELECT 
        COUNT(DISTINCT r.record_id) as total,
        AVG(f.result) as avg_score,
        f.name as metric_name
    FROM records r
    JOIN feedback f ON r.record_id = f.record_id
    GROUP BY f.name
""")

if summary.empty:
    st.warning("⚠️ No evaluation data found in trulens.db")
    st.info("💡 Run the RAG evaluation first to generate data")
    st.stop()

# Display metrics
st.subheader("📊 Evaluation Metrics")

cols = st.columns(5)
metrics = {
    "Relevance": 0,
    "Quality": 0,
    "Groundedness": 0,
    "Context Relevance": 0,
    "Correctness": 0
}

for _, row in summary.iterrows():
    if row['metric_name'] in metrics:
        metrics[row['metric_name']] = row['avg_score']

for idx, (name, value) in enumerate(metrics.items()):
    with cols[idx]:
        color = "score-high" if value > 0.7 else "score-medium" if value > 0.4 else "score-low"
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.8rem;color:#666;">{name}</div>
                <div class="metric-value {color}">{value*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

# Get recent evaluations
recent = get_data("""
    SELECT 
        r.timestamp, 
        r.input as question,
        f.name as metric,
        f.result as score
    FROM records r
    JOIN feedback f ON r.record_id = f.record_id
    ORDER BY r.timestamp DESC
    LIMIT 50
""")

if not recent.empty:
    st.markdown("---")
    st.subheader("📋 Recent Evaluations")
    
    # Pivot for display
    pivot_df = recent.pivot_table(
        index=['timestamp', 'question'],
        columns='metric',
        values='score'
    ).reset_index()
    
    st.dataframe(pivot_df, use_container_width=True)

# Auto-refresh
if st.sidebar.checkbox("Auto-refresh", value=True):
    time.sleep(5)
    st.rerun()
'''

    with open("trulens_standalone_dashboard.py", "w") as f:
        f.write(dashboard_code)
    print("✅ TruLens standalone dashboard created")

# ============================================
# MAIN
# ============================================

def main():
    """Main function - starts TruLens dashboard and background evaluation"""
    
    print("\n" + "="*60)
    print("🚀 Starting TruLens RAG Evaluation System")
    print("="*60)
    
    # Initialize Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)
        print(f"✅ Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        print(f"❌ Failed to connect to Pinecone: {e}")
        return
    
    # Fetch users
    users = fetch_all_users(pinecone_index)
    
    if not users:
        print("❌ No users found in Pinecone!")
        return
    
    print(f"\n👥 Found {len(users)} users:")
    for user in users[:5]:  # Show first 5
        print(f"  - {user['email']} ({user['user_id']})")
    if len(users) > 5:
        print(f"  ... and {len(users) - 5} more")
    
    # Initialize embedding and LLM
    try:
        embed_model = GeminiDirectEmbedding(api_key=GEMINI_API_KEY)
        llm = LlamaGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.3)
        Settings.embed_model = embed_model
        Settings.llm = llm
        print("✅ Initialized embedding and LLM")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Start background evaluation
    evaluator = BackgroundEvaluatorWithTruLens()
    evaluator.start(users, pinecone_index, embed_model, llm)
    
    print("\n" + "="*60)
    print("📊 Launching TruLens Dashboard...")
    print("="*60)
    print(f"✅ Dashboard will be available at: http://localhost:{PORT}")
    print("🔄 Evaluation is running in the background!")
    print("📈 Results will update in real-time as evaluation progresses")
    print("="*60)
    
    # Launch TruLens dashboard
    launch_trulens_dashboard()

if __name__ == "__main__":
    main()