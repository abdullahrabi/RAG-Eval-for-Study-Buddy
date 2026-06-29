# RAG_Eval_API.py - Fixed with proper data storage
import os
import time
import re
import warnings
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import json
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import threading
import subprocess
import sys

warnings.filterwarnings('ignore')

# ============================================
# ENVIRONMENT VARIABLE LOADING
# ============================================

load_dotenv()

print("="*60)
print("🔍 ENVIRONMENT VARIABLE CHECK")
print("="*60)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME") or os.environ.get("INDEX_NAME", "studybuddy")

print(f"GEMINI_API_KEY loaded: {'✅' if GEMINI_API_KEY else '❌'}")
print(f"PINECONE_API_KEY loaded: {'✅' if PINECONE_API_KEY else '❌'}")
print(f"GROQ_API_KEY loaded: {'✅' if GROQ_API_KEY else '❌'}")
print(f"INDEX_NAME: {INDEX_NAME}")

if not GEMINI_API_KEY or not PINECONE_API_KEY or not GROQ_API_KEY:
    print("\n⚠️ SOME ENVIRONMENT VARIABLES ARE MISSING!")
    print("="*60)
else:
    print("✅ All required environment variables loaded successfully!")
    print("="*60)

os.environ["TRULENS_OTEL_TRACING"] = "1"
os.environ["TRULENS_OTEL_ENABLED"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "false"

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
# FLASK APP
# ============================================

app = Flask(__name__)
CORS(app)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
APP_NAME = f"RAG_Eval_API_{RUN_ID}"

# Global variables
rag_wrapper = None
tru_app = None
session = None

# ============================================
# EMBEDDING
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
# RAG
# ============================================

class OptimizedRAG:
    def __init__(self, pinecone_index, embed_model, llm):
        self.pinecone_index = pinecone_index
        self.embed_model = embed_model
        self.llm = llm
    
    def query(self, question: str) -> tuple:
        try:
            query_embedding = self.embed_model._embed_text(question)
            if not query_embedding:
                return "No relevant documents found.", []
            
            results = self.pinecone_index.query(vector=query_embedding, top_k=5, include_metadata=True)
            
            contexts = []
            for match in results.matches:
                if match.metadata and 'text' in match.metadata:
                    contexts.append(match.metadata['text'][:800])
            
            if not contexts:
                return "No relevant documents found.", []
            
            prompt = f"""Answer based on context.

CONTEXT:
{chr(10).join(contexts)}

QUESTION:
{question}

ANSWER:"""
            
            response = self.llm.complete(prompt)
            return str(response).strip(), contexts
        except Exception as e:
            return f"Error: {e}", []
    
    def query_with_context(self, question: str, contexts: list) -> str:
        try:
            if not contexts:
                return "No context provided."
            
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
# MODEL ROUTER
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
# TRULENS FEEDBACK
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
# INITIALIZE RAG SYSTEM
# ============================================

def initialize_rag():
    global rag_wrapper, tru_app, session
    
    print("\n" + "="*60)
    print("🔍 Initializing RAG System...")
    print("="*60)
    
    if not GEMINI_API_KEY or not PINECONE_API_KEY or not GROQ_API_KEY:
        print("❌ ERROR: Missing API keys!")
        return False
    
    try:
        embed_model = GeminiDirectEmbedding(api_key=GEMINI_API_KEY)
        llm = LlamaGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.3)
        Settings.embed_model = embed_model
        Settings.llm = llm
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)
        rag = OptimizedRAG(pinecone_index, embed_model, llm)
        
        class RAGWrapper:
            def respond(self, question: str) -> str:
                answer, _ = rag.query(question)
                return answer
            
            def respond_with_context(self, question: str, contexts: list) -> str:
                return rag.query_with_context(question, contexts)
        
        rag_wrapper = RAGWrapper()
        
        session = TruSession(database_url="sqlite:///trulens.db")
        print("✅ Database connection established.")
        
        f_relevance = Feedback(relevance, name="Relevance").on_input_output()
        f_quality = Feedback(quality, name="Quality").on_input_output()
        f_groundedness = Feedback(groundedness, name="Groundedness").on_input_output()
        f_context_relevance = Feedback(context_relevance, name="Context Relevance").on_input_output()
        f_correctness = Feedback(correctness, name="Correctness").on_input_output()
        
        tru_app = TruApp(
            rag_wrapper,
            app_name=APP_NAME,
            app_version="v1.0",
            feedbacks=[f_relevance, f_quality, f_groundedness, f_context_relevance, f_correctness],
            main_method=rag_wrapper.respond
        )
        
        print("✅ RAG system initialized successfully!")
        print("="*60)
        return True
    except Exception as e:
        print(f"❌ Error initializing RAG: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head><title>RAG Evaluation API</title></head>
    <body>
        <h1>RAG Evaluation API</h1>
        <p>Status: Running</p>
        <p><a href="/results">View Results</a></p>
        <p><a href="/health">Health Check</a></p>
    </body>
    </html>
    """)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

# ============================================
# FIXED EVALUATE ENDPOINT - PROPERLY SAVES DATA
# ============================================

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing 'question' in request body",
                "status": "error"
            }), 400
        
        question = data['question']
        answer = data.get('answer', 'No answer provided')
        contexts = data.get('contexts', [])
        user_id = data.get('user_id', 'default')
        
        print(f"\n📝 Evaluating: {question[:50]}...")
        print(f"   Answer length: {len(answer)} chars")
        print(f"   Contexts: {len(contexts)} passages")
        
        # ============================================
        # FIX: Direct database insert using Record
        # ============================================
        
        from trulens.core.record import Record
        
        # Create a proper TruLens record
        record = Record(
            app_id=APP_NAME,
            main_input=question,
            main_output=answer,
            meta={
                "contexts": contexts,
                "user_id": user_id,
                "context_count": len(contexts),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Insert into database
        try:
            session.records.insert(record)
            session.connector.db.commit()
            print("   ✅ Record saved to database!")
        except Exception as e:
            print(f"   ⚠️ Insert error: {e}")
            # Try alternative method
            try:
                db = session.connector.db
                from trulens.core.database.orm import Record as ORMRecord
                
                new_record = ORMRecord(
                    app_id=APP_NAME,
                    main_input=question,
                    main_output=answer,
                    meta=json.dumps({
                        "contexts": contexts,
                        "user_id": user_id
                    })
                )
                db.add(new_record)
                db.commit()
                print("   ✅ Record saved via ORM fallback!")
            except Exception as e2:
                print(f"   ❌ All insert methods failed: {e2}")
                return jsonify({
                    "error": f"Database insert failed: {str(e2)}",
                    "status": "error"
                }), 500
        
        # Wait for feedback computation
        time.sleep(2)
        
        # Verify
        try:
            records_df, _ = session.get_records_and_feedback(app_name=APP_NAME)
            count = len(records_df) if records_df is not None else 0
            print(f"   📊 Database now has {count} records")
        except:
            pass
        
        return jsonify({
            "status": "success",
            "question": question,
            "answer": answer[:200] + "..." if len(answer) > 200 else answer,
            "contexts_count": len(contexts),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in evaluate: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/evaluate_batch', methods=['POST'])
def evaluate_batch():
    try:
        data = request.get_json()
        
        if not data or 'questions' not in data:
            return jsonify({
                "error": "Missing 'questions' in request body",
                "status": "error"
            }), 400
        
        questions = data['questions']
        results = []
        
        from trulens.core.record import Record
        
        for q_data in questions:
            question = q_data['question']
            contexts = q_data.get('contexts', [])
            provided_answer = q_data.get('answer', None)
            
            if provided_answer:
                answer = provided_answer
            else:
                if contexts:
                    answer = rag_wrapper.respond_with_context(question, contexts)
                else:
                    embed_model = GeminiDirectEmbedding(api_key=GEMINI_API_KEY)
                    llm = LlamaGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.3)
                    pc = Pinecone(api_key=PINECONE_API_KEY)
                    pinecone_index = pc.Index(INDEX_NAME)
                    temp_rag = OptimizedRAG(pinecone_index, embed_model, llm)
                    answer, contexts = temp_rag.query(question)
            
            # Save each record directly
            record = Record(
                app_id=APP_NAME,
                main_input=question,
                main_output=answer,
                meta={
                    "contexts": contexts,
                    "timestamp": datetime.now().isoformat()
                }
            )
            session.records.insert(record)
            session.connector.db.commit()
            
            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts
            })
        
        time.sleep(5)
        
        return jsonify({
            "status": "success",
            "total": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error in batch evaluate: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/results', methods=['GET'])
def get_results():
    try:
        records_df, feedback_names = session.get_records_and_feedback(app_name=APP_NAME)
        
        if records_df is not None and len(records_df) > 0:
            records = records_df.to_dict(orient='records')
            return jsonify({
                "status": "success",
                "count": len(records),
                "records": records,
                "feedback_names": feedback_names
            })
        else:
            return jsonify({
                "status": "success",
                "count": 0,
                "records": [],
                "message": "No records found"
            })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/results/csv', methods=['GET'])
def get_results_csv():
    try:
        records_df, feedback_names = session.get_records_and_feedback(app_name=APP_NAME)
        
        if records_df is not None and len(records_df) > 0:
            csv = records_df.to_csv(index=False)
            return jsonify({
                "status": "success",
                "csv": csv,
                "filename": f"trulens_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            })
        else:
            return jsonify({
                "status": "success",
                "csv": None,
                "message": "No records found"
            })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    if initialize_rag():
        print("\n" + "="*60)
        print("🚀 Starting Flask API Server...")
        print("="*60)
        print("\n📌 Available Endpoints:")
        print("  GET  /                 - Home")
        print("  GET  /health           - Health check")
        print("  POST /evaluate         - Evaluate single question")
        print("  POST /evaluate_batch   - Evaluate multiple questions")
        print("  GET  /results          - Get all results")
        print("  GET  /results/csv      - Download results as CSV")
        print("\n🌐 API: https://rag-eval-for-study-buddy-production.up.railway.app")
        print("="*60)
        
        app.run(host='0.0.0.0', port=8501, debug=False)
    else:
        print("\n❌ Failed to initialize RAG system.")
        print("Please check your environment variables.")