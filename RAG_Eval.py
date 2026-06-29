# RAG_Eval.py - Simple version using Streamlit's built-in port forwarding
import os
import time
import re
import warnings
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import subprocess
import threading
import socket

warnings.filterwarnings('ignore')

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
from trulens.core.database.connector.default import DefaultDBConnector
from trulens.core.database.sqlalchemy import SQLAlchemyDB
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
# DASHBOARD LAUNCHER - SIMPLE VERSION
# ============================================

def find_free_port():
    """Find a free port to run the dashboard"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def launch_dashboard():
    """Launch TruLens dashboard on a free port"""
    try:
        # Find a free port
        port = find_free_port()
        st.info(f"Starting TruLens dashboard on port {port}...")
        
        # Start dashboard in background
        process = subprocess.Popen(
            ["trulens", "dashboard", "--port", str(port), "--no-browser"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for it to start
        time.sleep(5)
        
        # Get the local URL
        local_url = f"http://localhost:{port}"
        
        # Store in session state
        st.session_state['dashboard_process'] = process
        st.session_state['dashboard_port'] = port
        st.session_state['dashboard_url'] = local_url
        st.session_state['dashboard_launched'] = True
        
        return local_url
    except Exception as e:
        st.error(f"Failed to launch dashboard: {e}")
        return None

def run_evaluation():
    """Run the RAG evaluation"""
    
    with st.spinner("Initializing RAG system..."):
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
    
    # Use persistent database
    db_path = "trulens.db"
    db = SQLAlchemyDB.from_db_url(f"sqlite:///{db_path}")
    
    # Handle different API versions
    try:
        connector = DefaultDBConnector(db)
    except TypeError:
        connector = DefaultDBConnector(db=db)
    
    session = TruSession(connector=connector)
    
    # Setup feedback functions
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
    
    # Questions to evaluate
    questions = [
        "Given the regular expression [A-Z][a-z]* [ ][A-Z][A-Z], what pattern does it represent and what is its limitation?",
        "List three software applications using automata.",
    ]
    
    # Run evaluation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with tru_app as recording:
        for i, q in enumerate(questions, 1):
            status_text.text(f"Processing question {i}/{len(questions)}: {q[:50]}...")
            rag_wrapper.respond(q)
            progress_bar.progress(i / len(questions))
    
    status_text.text("Computing feedback scores...")
    time.sleep(30)
    
    try:
        tru_app.wait_for_feedback_results()
        records_df, feedback_names = session.get_records_and_feedback(app_name=APP_NAME)
        
        if records_df is not None and len(records_df) > 0:
            st.session_state['records_df'] = records_df
            st.session_state['feedback_names'] = feedback_names
            st.session_state['app_name'] = APP_NAME
            st.session_state['evaluation_done'] = True
            
            records_df.to_csv(f"trulens_records_{RUN_ID}.csv", index=False)
            
            status_text.text("✅ Evaluation complete!")
            progress_bar.progress(1.0)
            return True
        else:
            st.error("No records found from evaluation")
            return False
    except Exception as e:
        st.error(f"Error during evaluation: {str(e)[:200]}")
        return False

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(page_title="RAG Evaluation with TruLens", layout="wide")

def main():
    st.title("🔍 RAG Evaluation with TruLens")
    
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.info(f"Index: {INDEX_NAME}")
    st.sidebar.info(f"Run ID: {RUN_ID}")
    
    # Initialize session state
    if 'evaluation_done' not in st.session_state:
        st.session_state['evaluation_done'] = False
    
    if 'dashboard_launched' not in st.session_state:
        st.session_state['dashboard_launched'] = False
    
    # Show evaluation button
    if not st.session_state['evaluation_done']:
        st.info("Click the button below to start the RAG evaluation.")
        
        with st.expander("📝 Evaluation Questions"):
            questions = [
                "Given the regular expression [A-Z][a-z]* [ ][A-Z][A-Z], what pattern does it represent and what is its limitation?",
                "List three software applications using automata.",
            ]
            for i, q in enumerate(questions, 1):
                st.write(f"{i}. {q}")
        
        if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
            success = run_evaluation()
            if success:
                st.rerun()
    else:
        st.success("✅ Evaluation completed!")
        
        # Show TruLens Dashboard button
        if not st.session_state['dashboard_launched']:
            if st.button("📊 Open TruLens Dashboard", type="primary", use_container_width=True):
                url = launch_dashboard()
                if url:
                    st.session_state['dashboard_launched'] = True
                    st.rerun()
        else:
            # Show dashboard URL - THIS WILL WORK WITH STREAMLIT CLOUD
            if 'dashboard_url' in st.session_state:
                st.markdown(f"""
                ### ✅ TruLens Dashboard is running!
                
                The dashboard is available at:
                
                🔗 **`{st.session_state['dashboard_url']}`**
                
                💡 **Note:** Since this is running on Streamlit Cloud, the URL above will be automatically forwarded. 
                Click the link or open it in a new tab.
                """)
                
                # Create a clickable link
                st.markdown(f"""
                [**Click here to open TruLens Dashboard**]({st.session_state['dashboard_url']})
                """)
                
                st.info("📌 The dashboard will remain active until you close this session.")
            
            # Show option to restart dashboard
            if st.button("🔄 Restart Dashboard"):
                st.session_state['dashboard_launched'] = False
                if 'dashboard_process' in st.session_state:
                    try:
                        st.session_state['dashboard_process'].terminate()
                    except:
                        pass
                st.rerun()
        
        # Show summary of results
        if 'records_df' in st.session_state:
            records_df = st.session_state['records_df']
            st.subheader("📊 Evaluation Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            metric_cols = ['relevance', 'quality', 'groundedness', 'context_relevance', 'correctness']
            available_metrics = [col for col in metric_cols if col in records_df.columns]
            
            with col1:
                st.metric("Total Questions", len(records_df))
            
            if available_metrics:
                avg_score = records_df[available_metrics].mean().mean()
                with col2:
                    st.metric("Average Score", f"{avg_score:.3f}")
                
                # Show individual metrics
                st.subheader("📈 Individual Metrics")
                metric_df = records_df[available_metrics].mean().reset_index()
                metric_df.columns = ['Metric', 'Score']
                st.dataframe(metric_df, use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Built with TruLens • {datetime.now().strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()