# RAG_Eval_All_Users.py - Evaluate RAG for All Users
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
import threading
import concurrent.futures
from collections import defaultdict

warnings.filterwarnings('ignore')

load_dotenv()

print("="*60)
print("🔍 RAG EVALUATION FOR ALL USERS")
print("="*60)

# ============================================
# ENVIRONMENT VARIABLES
# ============================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")

print(f"GEMINI_API_KEY: {'✅' if GEMINI_API_KEY else '❌'}")
print(f"PINECONE_API_KEY: {'✅' if PINECONE_API_KEY else '❌'}")
print(f"GROQ_API_KEY: {'✅' if GROQ_API_KEY else '❌'}")

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

# TruLens imports
from trulens.core import TruSession, Feedback
from trulens.apps.app import TruApp

# ============================================
# RUN ID
# ============================================

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
    """Fetch all unique user IDs from Pinecone"""
    print("\n🔍 Fetching all users from Pinecone...")
    
    try:
        # Query with a dummy vector to get all user entries
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
    """Fetch all notes for a specific user"""
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
        
        # Sort by chunk index
        notes.sort(key=lambda x: x.get('chunk_index', 0))
        return notes
    except Exception as e:
        print(f"⚠️ Error fetching notes for {user_id}: {e}")
        return []

# ============================================
# GENERATE QUESTIONS FROM NOTES
# ============================================

def generate_questions_from_notes(notes: List[Dict], num_questions: int = 20) -> List[str]:
    """Generate questions based on user's notes content"""
    if not notes:
        return []
    
    # Combine all note text
    full_text = " ".join([note['text'] for note in notes])
    
    # Use LLM to generate questions
    try:
        from groq import Groq as GroqClient
        
        client = GroqClient(api_key=GROQ_API_KEY)
        
        # Truncate text if too long
        if len(full_text) > 8000:
            full_text = full_text[:8000]
        
        prompt = f"""Based on the following educational content about Automata Theory and Languages, generate {num_questions} thoughtful questions that test understanding of the material.

CONTENT:
{full_text}

Requirements:
1. Questions should cover different topics from the content
2. Include a mix of conceptual questions, definitions, and applications
3. Questions should be answerable from the provided content
4. Format as a numbered list (1. Question text?)

Generate exactly {num_questions} questions:"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates educational questions from content. Only output the numbered questions, nothing else."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        questions_text = response.choices[0].message.content.strip()
        
        # Parse numbered questions
        questions = []
        for line in questions_text.split('\n'):
            line = line.strip()
            # Match patterns like "1." or "1)" or "1. "
            match = re.match(r'^(\d+)[\.\)]?\s*(.*)', line)
            if match:
                question_text = match.group(2).strip()
                if question_text:
                    questions.append(question_text)
            elif line and len(line) > 10 and '?' in line:
                # If line contains a question mark, it's likely a question
                questions.append(line)
        
        # If we didn't get enough questions, generate more
        if len(questions) < num_questions:
            questions = generate_fallback_questions(full_text, num_questions)
        
        return questions[:num_questions]
        
    except Exception as e:
        print(f"⚠️ Error generating questions: {e}")
        return generate_fallback_questions(full_text, num_questions)

def generate_fallback_questions(text: str, num_questions: int = 20) -> List[str]:
    """Generate fallback questions using pattern matching"""
    questions = []
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s for s in sentences if len(s) > 20]
    
    if not sentences:
        return [
            "What is automata theory?",
            "What is a finite automaton?",
            "What is the purpose of studying automata theory?",
            "What are the applications of finite automata?"
        ][:num_questions]
    
    # Generate questions from important sentences
    for i, sentence in enumerate(sentences[:num_questions]):
        if i % 3 == 0:
            questions.append(f"What is the definition of '{sentence[:30]}...'?")
        elif i % 3 == 1:
            questions.append(f"What does the text say about '{sentence[:30]}...'?")
        else:
            questions.append(f"How is '{sentence[:30]}...' related to automata theory?")
    
    # Add some default questions
    default_questions = [
        "What are the primary goals of studying automata theory?",
        "What is a finite automaton?",
        "What are the applications of finite automata?",
        "What is a formal language?",
        "What is the difference between a string and a language?",
        "What is the Kleene star operator?",
        "What is the role of Turing machines in automata theory?",
        "What is the significance of the pumping lemma?",
        "What are regular expressions and how are they used?",
        "What is the relationship between finite automata and regular languages?"
    ]
    
    # Fill remaining with default questions
    while len(questions) < num_questions:
        idx = len(questions) % len(default_questions)
        questions.append(default_questions[idx])
    
    return questions[:num_questions]

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
            
            # Filter by user_id if provided
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
                    messages=[{"role": "system", "content": "Output ONLY a number 0-1."}, {"role": "user", "content": prompt}],
                    temperature=0, max_tokens=10
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
# EVALUATE SINGLE USER
# ============================================

def evaluate_user(user_data: Dict, pinecone_index, embed_model, llm, session) -> Dict:
    """Evaluate a single user's RAG performance"""
    user_id = user_data['user_id']
    email = user_data['email']
    
    print(f"\n📊 Evaluating user: {email} ({user_id})")
    
    # Fetch user's notes
    notes = fetch_user_notes(pinecone_index, user_id)
    
    if not notes:
        print(f"  ⚠️ No notes found for {email}")
        return {
            'user_id': user_id,
            'email': email,
            'notes_count': 0,
            'questions': [],
            'success': False,
            'error': 'No notes found'
        }
    
    print(f"  📝 Found {len(notes)} note chunks")
    
    # Generate questions from notes
    questions = generate_questions_from_notes(notes, num_questions=20)
    
    if not questions:
        print(f"  ⚠️ Could not generate questions for {email}")
        return {
            'user_id': user_id,
            'email': email,
            'notes_count': len(notes),
            'questions': [],
            'success': False,
            'error': 'No questions generated'
        }
    
    print(f"  ❓ Generated {len(questions)} questions")
    
    # Create RAG instance for this user
    rag = OptimizedRAG(pinecone_index, embed_model, llm, user_id=user_id)
    
    class RAGWrapper:
        def respond(self, question: str) -> str:
            return rag.query(question)
    
    rag_wrapper = RAGWrapper()
    
    # Setup feedback functions
    f_relevance = Feedback(relevance, name="Relevance").on_input_output()
    f_quality = Feedback(quality, name="Quality").on_input_output()
    f_groundedness = Feedback(groundedness, name="Groundedness").on_input_output()
    f_context_relevance = Feedback(context_relevance, name="Context Relevance").on_input_output()
    f_correctness = Feedback(correctness, name="Correctness").on_input_output()
    
    # Create TruApp for this user
    user_app_name = f"{APP_NAME}_{user_id}"
    
    tru_app = TruApp(
        rag_wrapper,
        app_name=user_app_name,
        app_version="v1.0",
        feedbacks=[f_relevance, f_quality, f_groundedness, f_context_relevance, f_correctness],
        main_method=rag_wrapper.respond
    )
    
    # Run evaluation
    print(f"  🔄 Running evaluation with {len(questions)} questions...")
    
    with tru_app as recording:
        for i, q in enumerate(questions, 1):
            print(f"    {i}/{len(questions)}: {q[:40]}...")
            rag_wrapper.respond(q)
    
    return {
        'user_id': user_id,
        'email': email,
        'notes_count': len(notes),
        'questions': questions,
        'success': True,
        'app_name': user_app_name
    }

# ============================================
# PARALLEL EVALUATION
# ============================================

def run_parallel_evaluation(users: List[Dict], pinecone_index, embed_model, llm, session, max_workers: int = 3):
    """Run evaluation for multiple users in parallel"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_user = {
            executor.submit(evaluate_user, user, pinecone_index, embed_model, llm, session): user
            for user in users
        }
        
        for future in concurrent.futures.as_completed(future_to_user):
            user = future_to_user[future]
            try:
                result = future.result(timeout=600)  # 10 minute timeout per user
                results.append(result)
                print(f"✅ Completed evaluation for {user['email']}")
            except Exception as e:
                print(f"❌ Failed evaluation for {user['email']}: {e}")
                results.append({
                    'user_id': user['user_id'],
                    'email': user['email'],
                    'success': False,
                    'error': str(e)
                })
    
    return results

# ============================================
# MAIN EVALUATION
# ============================================

def run_evaluation():
    """Run the complete evaluation for all users"""
    
    print("\n" + "="*60)
    print("🚀 RAG EVALUATION FOR ALL USERS")
    print("="*60)
    
    # Initialize Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)
        print(f"✅ Connected to Pinecone index: {INDEX_NAME}")
    except Exception as e:
        print(f"❌ Failed to connect to Pinecone: {e}")
        return None
    
    # Fetch all users
    users = fetch_all_users(pinecone_index)
    
    if not users:
        print("❌ No users found in Pinecone!")
        return None
    
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
        return None
    
    # Initialize TruLens session
    try:
        session = TruSession(database_url="sqlite:///trulens.db")
        print("✅ Connected to TruLens database")
    except Exception as e:
        print(f"❌ Failed to connect to TruLens: {e}")
        return None
    
    # Run parallel evaluation
    print(f"\n📊 Running parallel evaluation with up to 3 users at a time...")
    print("="*60)
    
    results = run_parallel_evaluation(users, pinecone_index, embed_model, llm, session, max_workers=3)
    
    # Summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    total_questions = sum(r.get('questions', []) for r in successful)
    print(f"📝 Total questions generated: {len(total_questions)}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"evaluation_results_{RUN_ID}.csv", index=False)
    print(f"💾 Results saved to: evaluation_results_{RUN_ID}.csv")
    
    return session, results

# ============================================
# DASHBOARD LAUNCHER
# ============================================

def launch_dashboard(session):
    """Launch the TruLens dashboard"""
    print("\n" + "="*60)
    print("📊 Launching TruLens Dashboard...")
    print("="*60)
    
    try:
        from trulens.dashboard import run_dashboard
        
        print("Starting dashboard on port 8502...")
        run_dashboard(session=session, port=8502)
        
    except ImportError:
        print("❌ TruLens dashboard module not found")
        print("💡 Try running: streamlit run trulens/dashboard/app.py")
    except Exception as e:
        print(f"⚠️ Dashboard error: {e}")
        print("\n💡 You can view results using:")
        print(f"  - CSV file: evaluation_results_{RUN_ID}.csv")
        print("  - TruLens database: sqlite3 trulens.db")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    session, results = run_evaluation()
    
    if session:
        launch_dashboard(session)
    else:
        print("\n❌ Evaluation failed. Check the logs above.")