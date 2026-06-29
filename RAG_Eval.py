# RAG_Eval.py - Completely fixed version
import os
import time
import re
import warnings
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st

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

# TruLens imports - FIXED
from trulens.core import TruSession, Feedback
from trulens.core.database.sqlalchemy import SQLAlchemyDB
from trulens.core.database.connector import DBConnector
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
# EVALUATION FUNCTIONS
# ============================================

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
    
    # Create SQLAlchemy DB
    try:
        db = SQLAlchemyDB.from_db_url(f"sqlite:///{db_path}")
        connector = DBConnector(db=db)
    except Exception as e:
        st.error(f"Database error: {e}")
        return False
    
    # Create session
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
    time.sleep(30)  # Allow time for feedback computation
    
    try:
        tru_app.wait_for_feedback_results()
        records_df, feedback_names = session.get_records_and_feedback(app_name=APP_NAME)
        
        if records_df is not None and len(records_df) > 0:
            # Store in session state
            st.session_state['records_df'] = records_df
            st.session_state['feedback_names'] = feedback_names
            st.session_state['app_name'] = APP_NAME
            st.session_state['evaluation_done'] = True
            
            # Save to CSV
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

def show_dashboard():
    """Display the evaluation dashboard"""
    
    if 'records_df' not in st.session_state:
        st.warning("No evaluation data found. Please run the evaluation first.")
        return
    
    records_df = st.session_state['records_df']
    
    st.header("📊 TruLens Evaluation Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    metric_cols = ['relevance', 'quality', 'groundedness', 'context_relevance', 'correctness']
    available_metrics = [col for col in metric_cols if col in records_df.columns]
    
    with col1:
        st.metric("Total Records", len(records_df))
    
    if available_metrics:
        with col2:
            avg_score = records_df[available_metrics].mean().mean()
            st.metric("Average Score", f"{avg_score:.3f}")
        
        with col3:
            min_score = records_df[available_metrics].min().min()
            st.metric("Min Score", f"{min_score:.3f}")
        
        with col4:
            max_score = records_df[available_metrics].max().max()
            st.metric("Max Score", f"{max_score:.3f}")
    
    # Metrics visualization
    st.subheader("📈 Performance Metrics")
    
    if available_metrics:
        # Create columns for metrics
        cols = st.columns(len(available_metrics))
        
        for idx, metric in enumerate(available_metrics):
            with cols[idx]:
                mean_val = records_df[metric].mean()
                median_val = records_df[metric].median()
                std_val = records_df[metric].std()
                
                # Color coding
                color = "🟢" if mean_val >= 0.7 else "🟡" if mean_val >= 0.5 else "🔴"
                
                st.metric(
                    f"{color} {metric}",
                    f"{mean_val:.3f}",
                    delta=f"±{std_val:.3f}",
                    delta_color="off"
                )
                st.caption(f"Median: {median_val:.3f}")
    
    # Score distribution
    if available_metrics:
        st.subheader("📊 Score Distribution")
        
        # Create a bar chart
        chart_data = pd.DataFrame()
        for metric in available_metrics:
            chart_data[metric] = records_df[metric]
        
        # Show statistics
        st.dataframe(chart_data.describe(), use_container_width=True)
        
        # Individual records
        st.subheader("📝 Detailed Records")
        
        with st.expander("View all records"):
            display_cols = ['input', 'output'] + available_metrics
            available_display_cols = [col for col in display_cols if col in records_df.columns]
            
            if available_display_cols:
                display_df = records_df[available_display_cols].copy()
                for col in ['input', 'output']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].str[:200] + "..."
                
                st.dataframe(display_df, use_container_width=True, height=400)
        
        # Export option
        col1, col2 = st.columns(2)
        with col1:
            csv = records_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"trulens_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("🔄 Run New Evaluation", use_container_width=True):
                st.session_state['evaluation_done'] = False
                st.rerun()
    
    # App info
    if 'app_name' in st.session_state:
        st.sidebar.success(f"📱 App: {st.session_state['app_name']}")
    
    if 'feedback_names' in st.session_state and st.session_state['feedback_names']:
        st.sidebar.subheader("📋 Feedback Functions")
        for fb in st.session_state['feedback_names']:
            st.sidebar.write(f"✅ {fb}")

# ============================================
# MAIN STREAMLIT APP
# ============================================

st.set_page_config(page_title="RAG Evaluation with TruLens", layout="wide")

def main():
    st.title("🔍 RAG Evaluation with TruLens")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.info(f"Index: {INDEX_NAME}")
    st.sidebar.info(f"Run ID: {RUN_ID}")
    
    # Initialize session state
    if 'evaluation_done' not in st.session_state:
        st.session_state['evaluation_done'] = False
    
    # Show content based on state
    if not st.session_state['evaluation_done']:
        st.info("Click the button below to start the RAG evaluation.")
        
        # Questions preview
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
        # Show the dashboard
        show_dashboard()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Built with TruLens • {datetime.now().strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()