# RAG_Eval_All_Users.py - Live Dashboard with Background Evaluation (Railway Compatible)
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
print("🔍 RAG EVALUATION WITH LIVE DASHBOARD")
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

# ============================================
# IMPORTS
# ============================================

from pinecone import Pinecone
from google import genai
from google.genai import types
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.groq import Groq as LlamaGroq

# ============================================
# DATABASE FOR REAL-TIME UPDATES
# ============================================

class EvalDatabase:
    def __init__(self, db_path="data/eval_results.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                user_id TEXT,
                email TEXT,
                question TEXT,
                response TEXT,
                relevance REAL,
                quality REAL,
                groundedness REAL,
                context_relevance REAL,
                correctness REAL,
                timestamp TEXT,
                status TEXT DEFAULT 'pending'
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eval_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                total_users INTEGER DEFAULT 0,
                completed_users INTEGER DEFAULT 0,
                total_questions INTEGER DEFAULT 0,
                completed_questions INTEGER DEFAULT 0,
                current_user TEXT,
                status TEXT DEFAULT 'idle',
                message TEXT,
                updated_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_evaluation(self, data: Dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO evaluations (
                run_id, user_id, email, question, response,
                relevance, quality, groundedness, context_relevance, correctness,
                timestamp, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get('run_id'),
            data.get('user_id'),
            data.get('email'),
            data.get('question'),
            data.get('response'),
            data.get('relevance', 0),
            data.get('quality', 0),
            data.get('groundedness', 0),
            data.get('context_relevance', 0),
            data.get('correctness', 0),
            data.get('timestamp', datetime.now().isoformat()),
            data.get('status', 'completed')
        ))
        
        conn.commit()
        conn.close()
    
    def update_status(self, status: Dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO eval_status (
                id, run_id, total_users, completed_users,
                total_questions, completed_questions, current_user,
                status, message, updated_at
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            status.get('run_id'),
            status.get('total_users', 0),
            status.get('completed_users', 0),
            status.get('total_questions', 0),
            status.get('completed_questions', 0),
            status.get('current_user', ''),
            status.get('status', 'idle'),
            status.get('message', ''),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_status(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM eval_status ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'run_id': row[1],
                'total_users': row[2],
                'completed_users': row[3],
                'total_questions': row[4],
                'completed_questions': row[5],
                'current_user': row[6],
                'status': row[7],
                'message': row[8],
                'updated_at': row[9]
            }
        return None
    
    def get_results(self, limit: int = 100):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM evaluations ORDER BY timestamp DESC LIMIT ?",
            conn, params=(limit,)
        )
        conn.close()
        return df
    
    def get_summary(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(relevance) as avg_relevance,
                AVG(quality) as avg_quality,
                AVG(groundedness) as avg_groundedness,
                AVG(context_relevance) as avg_context_relevance,
                AVG(correctness) as avg_correctness
            FROM evaluations
        """)
        row = cursor.fetchone()
        conn.close()
        
        return {
            'total': row[0] or 0,
            'avg_relevance': row[1] or 0,
            'avg_quality': row[2] or 0,
            'avg_groundedness': row[3] or 0,
            'avg_context_relevance': row[4] or 0,
            'avg_correctness': row[5] or 0
        }

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

def generate_questions_from_notes(notes: List[Dict], num_questions: int = 20) -> List[str]:
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

def generate_fallback_questions(text: str, num_questions: int = 20) -> List[str]:
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
# FEEDBACK FUNCTIONS
# ============================================

router = None

def get_router():
    global router
    if router is None:
        router = ModelRouter(GROQ_API_KEY)
    return router

def evaluate_relevance(input: str, output: str) -> float:
    return get_router().call_model(f"Score relevance 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.1-8b-instant")

def evaluate_quality(input: str, output: str) -> float:
    return get_router().call_model(f"Score quality 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.1-8b-instant")

def evaluate_groundedness(input: str, output: str) -> float:
    return get_router().call_model(f"Score groundedness 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.3-70b-versatile")

def evaluate_context_relevance(input: str, output: str) -> float:
    return get_router().call_model(f"Score context relevance 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.1-8b-instant")

def evaluate_correctness(input: str, output: str) -> float:
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
# BACKGROUND EVALUATOR
# ============================================

class BackgroundEvaluator:
    def __init__(self):
        self.db = EvalDatabase()
        self.running = False
        self.thread = None
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def start(self, users: List[Dict], pinecone_index, embed_model, llm):
        """Start background evaluation"""
        if self.running:
            print("⚠️ Evaluation already running")
            return
        
        self.running = True
        
        # Initialize status
        self.db.update_status({
            'run_id': self.run_id,
            'total_users': len(users),
            'completed_users': 0,
            'total_questions': 0,
            'completed_questions': 0,
            'current_user': '',
            'status': 'running',
            'message': 'Starting evaluation...'
        })
        
        # Start background thread
        self.thread = threading.Thread(
            target=self._run_evaluation,
            args=(users, pinecone_index, embed_model, llm)
        )
        self.thread.daemon = True
        self.thread.start()
        
        print(f"✅ Background evaluation started (Run ID: {self.run_id})")
    
    def _run_evaluation(self, users: List[Dict], pinecone_index, embed_model, llm):
        """Run evaluation in background"""
        try:
            total_questions = 0
            
            for user_idx, user in enumerate(users):
                if not self.running:
                    break
                
                user_id = user['user_id']
                email = user['email']
                
                self.db.update_status({
                    'run_id': self.run_id,
                    'total_users': len(users),
                    'completed_users': user_idx,
                    'total_questions': total_questions,
                    'completed_questions': len([q for q in range(total_questions)]),
                    'current_user': email,
                    'status': 'running',
                    'message': f'Evaluating user: {email}'
                })
                
                # Get user notes
                notes = fetch_user_notes(pinecone_index, user_id)
                
                if not notes:
                    continue
                
                # Generate questions
                questions = generate_questions_from_notes(notes, num_questions=10)
                
                # Create RAG instance
                rag = OptimizedRAG(pinecone_index, embed_model, llm, user_id=user_id)
                
                for q_idx, question in enumerate(questions):
                    if not self.running:
                        break
                    
                    # Get response
                    response = rag.query(question)
                    
                    # Evaluate
                    scores = {
                        'relevance': evaluate_relevance(question, response),
                        'quality': evaluate_quality(question, response),
                        'groundedness': evaluate_groundedness(question, response),
                        'context_relevance': evaluate_context_relevance(question, response),
                        'correctness': evaluate_correctness(question, response)
                    }
                    
                    # Save to database
                    eval_data = {
                        'run_id': self.run_id,
                        'user_id': user_id,
                        'email': email,
                        'question': question,
                        'response': response[:500],
                        'timestamp': datetime.now().isoformat(),
                        'status': 'completed',
                        **scores
                    }
                    
                    self.db.save_evaluation(eval_data)
                    total_questions += 1
                    
                    # Update status
                    self.db.update_status({
                        'run_id': self.run_id,
                        'total_users': len(users),
                        'completed_users': user_idx + 1,
                        'total_questions': total_questions,
                        'completed_questions': total_questions,
                        'current_user': email,
                        'status': 'running',
                        'message': f'Evaluated {total_questions} questions for {email}'
                    })
                
                self.db.update_status({
                    'run_id': self.run_id,
                    'total_users': len(users),
                    'completed_users': user_idx + 1,
                    'total_questions': total_questions,
                    'completed_questions': total_questions,
                    'current_user': email,
                    'status': 'running',
                    'message': f'✅ Completed user: {email}'
                })
            
            self.db.update_status({
                'run_id': self.run_id,
                'total_users': len(users),
                'completed_users': len(users),
                'total_questions': total_questions,
                'completed_questions': total_questions,
                'current_user': '',
                'status': 'completed',
                'message': f'✅ Evaluation complete! {total_questions} questions evaluated'
            })
            
        except Exception as e:
            self.db.update_status({
                'run_id': self.run_id,
                'total_users': len(users) if users else 0,
                'completed_users': 0,
                'total_questions': 0,
                'completed_questions': 0,
                'current_user': '',
                'status': 'error',
                'message': f'❌ Error: {str(e)}'
            })
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
# DASHBOARD
# ============================================

def create_dashboard_file():
    """Create the dashboard Streamlit file"""
    
    dashboard_code = '''
import streamlit as st
import pandas as pd
import sqlite3
import time
import os
from datetime import datetime

st.set_page_config(
    page_title="Live RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Live RAG Evaluation Dashboard")

# Database connection
DB_PATH = "data/eval_results.db"

def get_status():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM eval_status ORDER BY id DESC LIMIT 1", conn)
        conn.close()
        return df.iloc[0] if not df.empty else None
    except:
        return None

def get_summary():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("""
            SELECT 
                COUNT(*) as total,
                AVG(relevance) as avg_relevance,
                AVG(quality) as avg_quality,
                AVG(groundedness) as avg_groundedness,
                AVG(context_relevance) as avg_context_relevance,
                AVG(correctness) as avg_correctness
            FROM evaluations
        """, conn)
        conn.close()
        return df.iloc[0] if not df.empty else None
    except:
        return None

def get_recent_results(limit=20):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM evaluations ORDER BY timestamp DESC LIMIT ?",
            conn, params=(limit,)
        )
        conn.close()
        return df
    except:
        return pd.DataFrame()

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 3)

# Status
status = get_status()

if status is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Users", f"{status.get('completed_users', 0)}/{status.get('total_users', 0)}")
    with col2:
        st.metric("❓ Questions", status.get('completed_questions', 0))
    with col3:
        st.metric("⏳ Status", status.get('status', 'idle').upper())
    with col4:
        st.metric("📋 Current", status.get('current_user', 'Waiting...'))
    
    # Progress bar
    if status.get('total_users', 0) > 0:
        progress = status.get('completed_users', 0) / status.get('total_users', 0)
        st.progress(progress, text=f"{progress*100:.1f}% - {status.get('message', '')}")
else:
    st.info("⏳ Waiting for evaluation to start...")

# Summary
summary = get_summary()
if summary is not None:
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎯 Relevance", f"{summary.get('avg_relevance', 0)*100:.1f}%")
    with col2:
        st.metric("⭐ Quality", f"{summary.get('avg_quality', 0)*100:.1f}%")
    with col3:
        st.metric("📚 Groundedness", f"{summary.get('avg_groundedness', 0)*100:.1f}%")
    with col4:
        st.metric("🔗 Context Relevance", f"{summary.get('avg_context_relevance', 0)*100:.1f}%")
    with col5:
        st.metric("✅ Correctness", f"{summary.get('avg_correctness', 0)*100:.1f}%")

# Recent results
st.markdown("---")
st.subheader("📋 Recent Evaluations")

df = get_recent_results()
if not df.empty:
    display_df = df[['timestamp', 'email', 'question', 'relevance', 'quality', 'correctness']].head(20)
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
    display_df['relevance'] = display_df['relevance'] * 100
    display_df['quality'] = display_df['quality'] * 100
    display_df['correctness'] = display_df['correctness'] * 100
    
    st.dataframe(
        display_df,
        column_config={
            'timestamp': 'Time',
            'email': 'User',
            'question': 'Question',
            'relevance': st.column_config.NumberColumn('Relevance', format='%.1f%%'),
            'quality': st.column_config.NumberColumn('Quality', format='%.1f%%'),
            'correctness': st.column_config.NumberColumn('Correctness', format='%.1f%%')
        },
        use_container_width=True
    )
else:
    st.info("📭 No evaluation data yet. Waiting for first results...")

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
'''

    with open("dashboard.py", "w") as f:
        f.write(dashboard_code)
    print("✅ Dashboard file created")

# ============================================
# MAIN
# ============================================

def main():
    """Main function - starts dashboard immediately and evaluation in background"""
    
    print("\n" + "="*60)
    print("🚀 Starting Live RAG Evaluation System")
    print("="*60)
    
    # Create dashboard file
    create_dashboard_file()
    
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
    for user in users:
        print(f"  - {user['email']} ({user['user_id']})")
    
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
    evaluator = BackgroundEvaluator()
    evaluator.start(users, pinecone_index, embed_model, llm)
    
    print("\n" + "="*60)
    print("📊 Starting Dashboard...")
    print("="*60)
    print(f"Dashboard will be available at: http://localhost:{PORT}")
    print("Results will update in real-time as evaluation runs!")
    
    # Start Streamlit dashboard
    try:
        cmd = ["streamlit", "run", "dashboard.py", "--server.port", str(PORT), "--server.address", "0.0.0.0", "--server.headless", "true"]
        subprocess.Popen(cmd)
    except Exception as e:
        print(f"⚠️ Dashboard error: {e}")
    
    # Keep process alive
    print("\n⏳ System running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping system...")
        evaluator.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()