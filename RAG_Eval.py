# RAG_Eval.py
import os
import time
import re
import warnings
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

warnings.filterwarnings('ignore')

from pinecone import Pinecone
from google import genai
from google.genai import types
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.groq import Groq as LlamaGroq

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
# RAG WITH SENTENCE WINDOW
# ============================================

class OptimizedRAG:
    def __init__(self, pinecone_index, embed_model, llm):
        self.pinecone_index = pinecone_index
        self.embed_model = embed_model
        self.llm = llm
    
    def retrieve_with_sentence_window(self, query: str, top_k: int = 5, window_size: int = 3) -> list:
        try:
            query_embedding = self.embed_model._embed_text(query)
            if not query_embedding:
                return []
            
            results = self.pinecone_index.query(vector=query_embedding, top_k=15, include_metadata=True)
            
            documents = []
            for match in results.matches:
                if match.metadata and 'text' in match.metadata:
                    documents.append({'text': match.metadata['text'], 'score': match.score})
            
            expanded_docs = []
            seen_texts = set()
            keywords = {'alphabet', 'string', 'language', 'automata', 'state', 'symbol', 'empty', 'kleene', 'regular', 'expression', 'formal', 'closure', 'transition', 'dfa', 'nfa', 'epsilon'}
            
            for doc in documents:
                text = doc['text']
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                
                sentences = re.split(r'(?<=[.!?])\s+', text)
                if len(sentences) <= window_size * 2:
                    expanded_docs.append(doc)
                    continue
                
                query_terms = set(query.lower().split())
                sentence_scores = []
                for i, sentence in enumerate(sentences):
                    sentence_terms = set(sentence.lower().split())
                    overlap = len(query_terms & sentence_terms)
                    bonus = len(sentence_terms & keywords) * 0.5
                    sentence_scores.append((i, overlap + bonus))
                
                sentence_scores.sort(key=lambda x: x[1], reverse=True)
                
                expanded_texts = []
                for idx, score in sentence_scores[:2]:
                    start = max(0, idx - window_size)
                    end = min(len(sentences), idx + window_size + 1)
                    expanded_texts.append(' '.join(sentences[start:end]))
                
                expanded_docs.append({'text': ' ... '.join(expanded_texts), 'score': doc['score']})
            
            expanded_docs.sort(key=lambda x: x['score'], reverse=True)
            return expanded_docs[:top_k]
        except Exception:
            return []
    
    def query(self, question: str) -> str:
        try:
            documents = self.retrieve_with_sentence_window(question)
            if not documents:
                return "No relevant documents found."
            
            context_text = "\n\n".join([doc['text'] for doc in documents[:5]])
            
            prompt = f"""You are an expert in automata theory. Answer based on context.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""
            
            response = self.llm.complete(prompt)
            return str(response).strip()
        except Exception as e:
            return f"Error: {e}"

# ============================================
# TRULENS-STYLE METRICS (No TruLens DB)
# ============================================

class TruLensStyleEvaluator:
    """TruLens-compatible metrics without TruLens database"""
    
    def __init__(self, api_key: str):
        from groq import Groq as GroqClient
        self.client = GroqClient(api_key=api_key)
        self.metrics = {}
    
    def score(self, prompt: str, model: str) -> float:
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
    
    def relevance(self, question: str, answer: str) -> float:
        return self.score(f"Score relevance 0-1.\nQ: {question[:300]}\nA: {answer[:300]}\nScore:", "llama-3.1-8b-instant")
    
    def quality(self, question: str, answer: str) -> float:
        return self.score(f"Score quality 0-1.\nQ: {question[:300]}\nA: {answer[:300]}\nScore:", "llama-3.1-8b-instant")
    
    def groundedness(self, question: str, answer: str) -> float:
        return self.score(f"Score groundedness 0-1.\nQ: {question[:300]}\nA: {answer[:300]}\nScore:", "llama-3.3-70b-versatile")
    
    def context_relevance(self, question: str, answer: str) -> float:
        return self.score(f"Score context relevance 0-1.\nQ: {question[:300]}\nA: {answer[:300]}\nScore:", "llama-3.1-8b-instant")
    
    def correctness(self, question: str, answer: str) -> float:
        return self.score(f"Score correctness 0-1.\nQ: {question[:300]}\nA: {answer[:300]}\nScore:", "llama-3.3-70b-versatile")

# ============================================
# MAIN
# ============================================

def main():
    print(f"\n{'='*60}")
    print(f"🔍 TRULENS-STYLE RAG EVALUATION")
    print(f"{'='*60}")
    print(f"📅 Run ID: {RUN_ID}")
    print(f"📱 App: {APP_NAME}")
    print(f"🎯 Strategy: Sentence Window Retrieval")
    print(f"{'='*60}")
    
    embed_model = GeminiDirectEmbedding(api_key=GEMINI_API_KEY)
    llm = LlamaGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.3)
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)
    rag = OptimizedRAG(pinecone_index, embed_model, llm)
    
    evaluator = TruLensStyleEvaluator(GROQ_API_KEY)
    
    questions = [
       
        "Given the regular expression [A-Z][a-z]* [ ][A-Z][A-Z], what pattern does it represent and what is its limitation?",
        "List three software applications using automata.",
    ]
    
    # Phase 1: Generate Answers
    print(f"\n{'='*60}")
    print("📝 PHASE 1: GENERATING ANSWERS")
    print(f"{'='*60}")
    
    answers, latencies = [], []
    for i, q in enumerate(questions, 1):
        start = time.time()
        a = rag.query(q)
        lat = time.time() - start
        answers.append(a)
        latencies.append(lat)
        print(f"  {i}/7 | {lat:.2f}s | {a[:80]}...")
    
    # Phase 2: TruLens-Style Metrics
    print(f"\n{'='*60}")
    print("📊 PHASE 2: TRULENS METRICS")
    print(f"{'='*60}")
    
    results = []
    for i, (q, a, lat) in enumerate(zip(questions, answers, latencies), 1):
        rel = evaluator.relevance(q, a)
        qua = evaluator.quality(q, a)
        gro = evaluator.groundedness(q, a)
        ctx = evaluator.context_relevance(q, a)
        cor = evaluator.correctness(q, a)
        
        results.append({
            'question': q,
            'answer': a[:200] + "...",
            'relevance': rel,
            'quality': qua,
            'groundedness': gro,
            'context_relevance': ctx,
            'correctness': cor,
            'latency': lat,
            'run_id': RUN_ID,
            'strategy': 'sentence_window'
        })
        
        print(f"\n  Q{i}: {q[:50]}...")
        print(f"    Relevance: {rel:.3f} | Quality: {qua:.3f}")
        print(f"    Groundedness: {gro:.3f} | Context: {ctx:.3f} | Correctness: {cor:.3f}")
    
    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(f"trulens_results_{RUN_ID}.csv", index=False)
    df.to_csv("trulens_results_latest.csv", index=False)
    
    # History
    history_file = "trulens_results_history.csv"
    if os.path.exists(history_file):
        df_history = pd.read_csv(history_file)
        df_history = pd.concat([df_history, df], ignore_index=True)
    else:
        df_history = df
    df_history.to_csv(history_file, index=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("📈 TRULENS METRICS SUMMARY")
    print(f"{'='*60}")
    
    for metric in ['relevance', 'quality', 'groundedness', 'context_relevance', 'correctness']:
        avg = df[metric].mean()
        status = "🟢 HIGH" if avg >= 0.7 else "🟡 MEDIUM" if avg >= 0.5 else "🔴 LOW"
        print(f"  {metric.replace('_', ' ').title():20s}: {avg:.3f}  {status}")
    
    print(f"  {'Average Latency':20s}: {df['latency'].mean():.2f}s")
    print(f"\n💾 Results: trulens_results_{RUN_ID}.csv")
    print(f"📁 History: trulens_results_history.csv")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()