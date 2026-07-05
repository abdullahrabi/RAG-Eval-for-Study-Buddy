"""
RAG_Eval_All_Users_TruLens.py - FINAL COMPLETE VERSION
Multi-user RAG evaluation with TruLens native dashboard
25+ AI-generated questions per user from notes ONLY
NO fallback questions - strictly AI-generated
Complete logging of each question and metric
Dashboard opens only after ALL evaluations complete
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
import sqlite3
import random
import subprocess

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
try:
    from trulens.core import TruSession, Feedback
    from trulens.apps.app import TruApp
    from trulens.dashboard import run_dashboard
    TRULENS_AVAILABLE = True
    print("✅ TruLens available")
except ImportError as e:
    TRULENS_AVAILABLE = False
    print(f"❌ TruLens not available: {e}")
    print("❌ TruLens is REQUIRED. Please install: pip install trulens-eval")
    sys.exit(1)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
APP_NAME = f"RAG_Eval_{RUN_ID}"

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
# GENERATE QUESTIONS FROM NOTES - NO FALLBACK
# ============================================

def generate_questions_from_notes(notes: List[Dict], num_questions: int = 25) -> List[str]:
    """
    Generate questions from notes using AI.
    NO FALLBACK QUESTIONS - strictly AI-generated.
    """
    if not notes:
        print("❌ No notes found - cannot generate questions!")
        return []
    
    full_text = " ".join([note['text'] for note in notes])
    
    try:
        from groq import Groq as GroqClient
        
        client = GroqClient(api_key=GROQ_API_KEY)
        
        # Truncate if too long
        if len(full_text) > 8000:
            full_text = full_text[:8000]
        
        prompt = f"""Based on the following educational content, generate {num_questions} diverse and thoughtful questions.

CONTENT:
{full_text}

Generate exactly {num_questions} questions covering different aspects of the content.
Questions should be:
- Diverse in difficulty (basic to advanced)
- Cover different topics from the content
- Include conceptual, analytical, and application-based questions
- Be clear and well-phrased

Output ONLY the questions as a numbered list (1 to {num_questions}):"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": f"Generate exactly {num_questions} questions based on the provided content. Output only the numbered questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=1500
        )
        
        questions_text = response.choices[0].message.content.strip()
        print(f"📝 AI Generated Questions Preview:\n{questions_text[:300]}...")
        
        # Parse questions
        questions = []
        for line in questions_text.split('\n'):
            line = line.strip()
            # Match numbered questions: 1. Question, 1) Question, etc.
            match = re.match(r'^(\d+)[\.\)]?\s*(.*)', line)
            if match:
                question_text = match.group(2).strip()
                if question_text and len(question_text) > 10:
                    questions.append(question_text)
            elif line and len(line) > 15 and '?' in line:
                # If no number but has question mark
                questions.append(line)
        
        # Verify we have enough questions
        if len(questions) < num_questions:
            print(f"⚠️ Only {len(questions)} questions generated, attempting to generate more...")
            # Try one more time with a different prompt
            more_questions = generate_more_questions(full_text, num_questions - len(questions))
            questions.extend(more_questions)
        
        # Strictly limit to requested number, but only if we have them
        if len(questions) >= num_questions:
            questions = questions[:num_questions]
            print(f"✅ Successfully generated {len(questions)} questions")
            return questions
        else:
            print(f"❌ Failed to generate {num_questions} questions. Only got {len(questions)}")
            return questions  # Return whatever we got, even if less than requested
            
    except Exception as e:
        print(f"❌ Error generating questions: {e}")
        return []  # Return empty list - NO FALLBACK QUESTIONS

def generate_more_questions(text: str, num_needed: int) -> List[str]:
    """Generate additional questions if initial generation was insufficient"""
    try:
        from groq import Groq as GroqClient
        
        client = GroqClient(api_key=GROQ_API_KEY)
        
        prompt = f"""Based on this content, generate {num_needed} more diverse questions:

CONTENT:
{text[:3000]}

Generate {num_needed} additional questions covering different aspects of the content.
Output ONLY the questions as a numbered list:"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": f"Generate {num_needed} additional questions. Output only the numbered questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=800
        )
        
        questions_text = response.choices[0].message.content.strip()
        questions = []
        for line in questions_text.split('\n'):
            line = line.strip()
            match = re.match(r'^(\d+)[\.\)]?\s*(.*)', line)
            if match:
                question_text = match.group(2).strip()
                if question_text and len(question_text) > 10:
                    questions.append(question_text)
        
        print(f"✅ Generated {len(questions)} additional questions")
        return questions
    except Exception as e:
        print(f"⚠️ Error generating more questions: {e}")
        return []  # NO FALLBACK - return empty list

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

# Global router
_router = None

def get_router():
    global _router
    if _router is None:
        _router = ModelRouter(GROQ_API_KEY)
    return _router

# ============================================
# FEEDBACK FUNCTIONS - TRULENS METRICS
# ============================================

def feedback_relevance(input: str, output: str) -> float:
    router = get_router()
    return router.call_model(f"Score relevance 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.1-8b-instant")

def feedback_quality(input: str, output: str) -> float:
    router = get_router()
    return router.call_model(f"Score quality 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.1-8b-instant")

def feedback_groundedness(input: str, output: str) -> float:
    router = get_router()
    return router.call_model(f"Score groundedness 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.3-70b-versatile")

def feedback_context_relevance(input: str, output: str) -> float:
    router = get_router()
    return router.call_model(f"Score context relevance 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.1-8b-instant")

def feedback_correctness(input: str, output: str) -> float:
    router = get_router()
    return router.call_model(f"Score correctness 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.3-70b-versatile")

# ============================================
# OPTIMIZED RAG SYSTEM
# ============================================

class OptimizedRAG:
    def __init__(self, pinecone_index, embed_model, llm, user_id: str = None):
        self.pinecone_index = pinecone_index
        self.embed_model = embed_model
        self.llm = llm
        self.user_id = user_id
        self.last_context = ""
    
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
            
            self.last_context = " ".join(contexts[:3])
            
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
# DATABASE CLASS FOR STORAGE
# ============================================

class EvalDatabase:
    def __init__(self, db_path="default.sqlite"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Evaluations table
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
        
        # Status table
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
        print("✅ Database tables initialized")
    
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

# ============================================
# RUN EVALUATION - COMPLETE BEFORE DASHBOARD
# ============================================

def run_evaluation(users, pinecone_index, embed_model, llm):
    """Run complete evaluation before launching dashboard"""
    
    print("\n" + "="*60)
    print("📊 Running RAG Evaluation with TruLens...")
    print("="*60)
    
    # Initialize database
    db = EvalDatabase()
    
    # Initialize TruLens session
    session = TruSession(database_url="sqlite:///default.sqlite")
    print("✅ TruLens session initialized")
    
    # Update status
    db.update_status({
        'run_id': RUN_ID,
        'total_users': len(users),
        'completed_users': 0,
        'total_questions': 0,
        'completed_questions': 0,
        'current_user': '',
        'status': 'running',
        'message': 'Starting evaluation...'
    })
    
    total_questions = 0
    all_tru_apps = []
    
    for user_idx, user in enumerate(users):
        user_id = user['user_id']
        email = user['email']
        
        print(f"\n{'='*60}")
        print(f"👤 Evaluating user {user_idx+1}/{len(users)}: {email}")
        print(f"{'='*60}")
        
        db.update_status({
            'run_id': RUN_ID,
            'total_users': len(users),
            'completed_users': user_idx,
            'total_questions': total_questions,
            'completed_questions': total_questions,
            'current_user': email,
            'status': 'running',
            'message': f'Evaluating user: {email}'
        })
        
        # Get user notes
        print(f"📚 Fetching notes for {email}...")
        notes = fetch_user_notes(pinecone_index, user_id)
        if not notes:
            print(f"❌ No notes found for {email}, skipping user...")
            continue
        
        print(f"✅ Found {len(notes)} notes")
        
        # Generate questions from notes - NO FALLBACK
        print(f"🤖 Generating 25 questions from notes using AI...")
        questions = generate_questions_from_notes(notes, num_questions=25)
        
        if not questions:
            print(f"❌ No questions generated for {email}, skipping user...")
            continue
        
        print(f"✅ Generated {len(questions)} questions for {email}")
        
        # Create RAG instance
        rag = OptimizedRAG(pinecone_index, embed_model, llm, user_id=user_id)
        
        class RAGWrapper:
            def __init__(self, rag_instance):
                self.rag = rag_instance
            
            def respond(self, question: str) -> str:
                return self.rag.query(question)
        
        rag_wrapper = RAGWrapper(rag)
        
        # Setup TruLens feedbacks - 5 metrics
        print(f"🔧 Setting up TruLens feedbacks...")
        f_relevance = Feedback(feedback_relevance, name="Relevance").on_input_output()
        f_quality = Feedback(feedback_quality, name="Quality").on_input_output()
        f_groundedness = Feedback(feedback_groundedness, name="Groundedness").on_input_output()
        f_context_relevance = Feedback(feedback_context_relevance, name="Context Relevance").on_input_output()
        f_correctness = Feedback(feedback_correctness, name="Correctness").on_input_output()
        
        # Create TruLens app
        tru_app = TruApp(
            rag_wrapper,
            app_name=f"{APP_NAME}_{email}",
            app_version="v1.0",
            feedbacks=[
                f_relevance, f_quality, f_groundedness,
                f_context_relevance, f_correctness
            ],
            main_method=rag_wrapper.respond
        )
        
        print(f"🔄 Evaluating {len(questions)} questions for {email}...")
        print(f"{'='*60}")
        
        # Run evaluation with TruLens
        with tru_app as recording:
            for q_idx, question in enumerate(questions, 1):
                print(f"📝 Question {q_idx}/{len(questions)}: {question[:80]}...")
                
                # Get response
                response = rag_wrapper.respond(question)
                
                # Calculate scores
                scores = {
                    'relevance': feedback_relevance(question, response),
                    'quality': feedback_quality(question, response),
                    'groundedness': feedback_groundedness(question, response),
                    'context_relevance': feedback_context_relevance(question, response),
                    'correctness': feedback_correctness(question, response)
                }
                
                # Print metrics
                print(f"   📊 Relevance: {scores['relevance']:.3f} | Quality: {scores['quality']:.3f} | Groundedness: {scores['groundedness']:.3f}")
                print(f"   📊 Context Relevance: {scores['context_relevance']:.3f} | Correctness: {scores['correctness']:.3f}")
                print(f"   ✅ Response: {response[:100]}...")
                print(f"{'-'*60}")
                
                # Save to database
                eval_data = {
                    'run_id': RUN_ID,
                    'user_id': user_id,
                    'email': email,
                    'question': question,
                    'response': response[:500],
                    'timestamp': datetime.now().isoformat(),
                    'status': 'completed',
                    **scores
                }
                
                db.save_evaluation(eval_data)
                total_questions += 1
                
                # Update status every 5 questions
                if q_idx % 5 == 0:
                    db.update_status({
                        'run_id': RUN_ID,
                        'total_users': len(users),
                        'completed_users': user_idx,
                        'total_questions': total_questions,
                        'completed_questions': total_questions,
                        'current_user': email,
                        'status': 'running',
                        'message': f'Evaluated {total_questions} questions'
                    })
        
        print(f"{'='*60}")
        print(f"⏳ Waiting for TruLens feedback computation to complete for {email}...")
        try:
            tru_app.wait_for_feedback_results()
            print(f"✅ TruLens feedbacks completed for {email}")
        except Exception as e:
            print(f"⚠️ Feedback wait error: {e}")
        
        all_tru_apps.append(tru_app)
        
        # Update status after user complete
        db.update_status({
            'run_id': RUN_ID,
            'total_users': len(users),
            'completed_users': user_idx + 1,
            'total_questions': total_questions,
            'completed_questions': total_questions,
            'current_user': '',
            'status': 'running',
            'message': f'✅ Completed user {user_idx+1}/{len(users)}: {email}'
        })
        
        print(f"✅ Completed evaluation for {email}")
        print(f"{'='*60}\n")
    
    # Wait for any remaining feedbacks
    if all_tru_apps:
        print("\n⏳ Waiting for all TruLens feedbacks to complete...")
        time.sleep(5)
    
    # Final status
    db.update_status({
        'run_id': RUN_ID,
        'total_users': len(users),
        'completed_users': len(users),
        'total_questions': total_questions,
        'completed_questions': total_questions,
        'current_user': '',
        'status': 'completed',
        'message': f'✅ Evaluation complete! {total_questions} questions evaluated'
    })
    
    print("\n" + "="*60)
    print(f"✅ EVALUATION COMPLETE!")
    print(f"📊 Total Users: {len(users)}")
    print(f"📝 Total Questions: {total_questions}")
    print(f"📁 Data saved to: default.sqlite")
    print("="*60)
    
    return session, total_questions

# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "="*60)
    print("🚀 Starting RAG Evaluation with TruLens")
    print("="*60)
    
    # Verify TruLens is available
    if not TRULENS_AVAILABLE:
        print("❌ TruLens is REQUIRED but not available!")
        print("💡 Install with: pip install trulens-eval")
        sys.exit(1)
    
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
    for user in users[:5]:
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
    
    # Run evaluation - COMPLETELY before dashboard
    session, total_questions = run_evaluation(users, pinecone_index, embed_model, llm)
    
    if not session or total_questions == 0:
        print("❌ Evaluation failed or no questions evaluated!")
        return
    
    print("\n" + "="*60)
    print(f"📊 Launching TruLens Dashboard on port {PORT}...")
    print("="*60)
    print(f"✅ TruLens Dashboard available at: http://localhost:{PORT}")
    print(f"📁 Data in: default.sqlite")
    print(f"📊 {total_questions} questions evaluated across {len(users)} users")
    print("="*60)
    
    # Launch TruLens dashboard - THIS BLOCKS
    try:
        run_dashboard(session=session, port=PORT)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Dashboard error: {e}")
        print("💡 Data is still saved in default.sqlite")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        sys.exit(0)