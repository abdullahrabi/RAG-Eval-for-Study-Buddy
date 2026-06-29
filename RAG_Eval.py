# RAG_Eval.py
import os
import time
import re
import warnings
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

warnings.filterwarnings('ignore')

os.environ["TRULENS_OTEL_TRACING"] = "1"
os.environ["TRULENS_OTEL_ENABLED"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "false"

# ===== FIX: Delete old TruLens DB on cloud =====
if os.path.exists("default.sqlite"):
    try:
        os.remove("default.sqlite")
        print("🗑️ Cleared old TruLens database")
    except:
        pass
# =============================================

from pinecone import Pinecone
from google import genai
from google.genai import types
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.groq import Groq as LlamaGroq
from trulens.core import TruSession, Feedback
from trulens.apps.app import TruApp

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
APP_NAME = f"RAG_Eval_{RUN_ID}"

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
    
    def query(self, question: str) -> str:
        try:
            query_embedding = self.embed_model._embed_text(question)
            if not query_embedding:
                return "No relevant documents found."
            
            results = self.pinecone_index.query(vector=query_embedding, top_k=5, include_metadata=True)
            
            contexts = []
            for match in results.matches:
                if match.metadata and 'text' in match.metadata:
                    contexts.append(match.metadata['text'][:800])
            
            if not contexts:
                return "No relevant documents found."
            
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
            return 0.5
        except Exception:
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
    r = get_router()
    return r.call_model(f"Score relevance 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.1-8b-instant")

def quality(input: str, output: str) -> float:
    r = get_router()
    return r.call_model(f"Score quality 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.1-8b-instant")

def groundedness(input: str, output: str) -> float:
    r = get_router()
    return r.call_model(f"Score groundedness 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.3-70b-versatile")

def context_relevance(input: str, output: str) -> float:
    r = get_router()
    return r.call_model(f"Score context relevance 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.1-8b-instant")

def correctness(input: str, output: str) -> float:
    r = get_router()
    return r.call_model(f"Score correctness 0-1.\nQ: {input[:300]}\nA: {output[:300]}\nScore:", "llama-3.3-70b-versatile")

# ============================================
# MAIN
# ============================================

def main():
    print(f"\n🔍 TruLens Evaluation: {APP_NAME}")
    
    embed_model = GeminiDirectEmbedding(api_key=GEMINI_API_KEY)
    llm = LlamaGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.3)
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)
    rag = OptimizedRAG(pinecone_index, embed_model, llm)
    
    class RAGWrapper:
        def respond(self, question: str) -> str:
            return rag.query(question)
    
    rag_wrapper = RAGWrapper()
    
    # TruLens with fresh DB
    session = TruSession()
    
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
    
    questions = [
        
        "Given the regular expression [A-Z][a-z]* [ ][A-Z][A-Z], what pattern does it represent and what is its limitation?",
        "List three software applications using automata.",
    ]
    
    print("\n📊 Recording with TruLens...")
    with tru_app as recording:
        for i, q in enumerate(questions, 1):
            print(f"  {i}/{len(questions)}: {q[:50]}...")
            rag_wrapper.respond(q)
    
    print("✅ Recording complete")
    print("\n⏳ Waiting for TruLens feedback...")
    time.sleep(30)
    
    try:
        tru_app.wait_for_feedback_results()
        records_df, _ = session.get_records_and_feedback(app_name=APP_NAME)
        
        if records_df is not None and len(records_df) > 0:
            records_df.to_csv(f"trulens_records_{RUN_ID}.csv", index=False)
            print(f"✅ TruLens records: {len(records_df)}")
            
            feedback_cols = ['relevance', 'quality', 'groundedness', 'context_relevance', 'correctness']
            for col in feedback_cols:
                if col in records_df.columns:
                    print(f"  {col}: {records_df[col].mean():.3f}")
    except Exception as e:
        print(f"⚠️ Error: {e}")
    
    print(f"\n📱 App: {APP_NAME}")

if __name__ == "__main__":
    main()