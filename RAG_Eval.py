# rag_eval_final_complete.py
import os
import time
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

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

# ============================================
# 1. CUSTOM EMBEDDING CLASS
# ============================================

class GeminiDirectEmbedding(BaseEmbedding):
    """Direct Gemini embedding without LlamaIndex wrapper"""
    
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
                model=self.model_name,
                contents=[text],
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
# 2. OPTIMIZED RAG WITH SENTENCE WINDOW
# ============================================

class OptimizedRAG:
    """RAG with Sentence Window - Best for Groundedness & Correctness"""
    
    def __init__(self, pinecone_index, embed_model, llm):
        self.pinecone_index = pinecone_index
        self.embed_model = embed_model
        self.llm = llm
    
    def retrieve_with_sentence_window(self, query: str, top_k: int = 5, window_size: int = 3) -> list:
        """Sentence Window Retrieval"""
        try:
            query_embedding = self.embed_model._embed_text(query)
            if not query_embedding:
                return []
            
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=15,
                include_metadata=True
            )
            
            documents = []
            for match in results.matches:
                if match.metadata and 'text' in match.metadata:
                    documents.append({
                        'text': match.metadata['text'],
                        'score': match.score,
                        'metadata': match.metadata
                    })
            
            expanded_docs = []
            seen_texts = set()
            automata_keywords = {'alphabet', 'string', 'language', 'automata', 'state', 
                                'symbol', 'empty', 'kleene', 'regular', 'expression',
                                'formal', 'closure', 'transition', 'dfa', 'nfa', 'epsilon'}
            
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
                    bonus = len(sentence_terms & automata_keywords) * 0.5
                    sentence_scores.append((i, overlap + bonus))
                
                sentence_scores.sort(key=lambda x: x[1], reverse=True)
                
                expanded_texts = []
                for idx, score in sentence_scores[:2]:
                    start = max(0, idx - window_size)
                    end = min(len(sentences), idx + window_size + 1)
                    window_text = ' '.join(sentences[start:end])
                    expanded_texts.append(window_text)
                
                expanded_text = ' ... '.join(expanded_texts)
                expanded_docs.append({
                    'text': expanded_text,
                    'score': doc['score'],
                    'metadata': {**doc['metadata'], 'strategy': 'sentence_window'}
                })
            
            expanded_docs.sort(key=lambda x: x['score'], reverse=True)
            return expanded_docs[:top_k]
        except Exception:
            return []
    
    def query(self, question: str) -> str:
        """Generate answer using Sentence Window retrieval"""
        try:
            documents = self.retrieve_with_sentence_window(question)
            
            if not documents:
                return "No relevant documents found."
            
            top_docs = documents[:5]
            context_text = "\n\n".join([doc['text'] for doc in top_docs])
            
            prompt = f"""You are an expert in automata theory. Answer based on the context provided.

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
# 3. MODEL ROUTING STRATEGY
# ============================================

MODEL_CONFIGS = {
    'fast': {'model': 'llama-3.1-8b-instant', 'use_for': ['relevance', 'quality']},
    'primary': {'model': 'llama-3.3-70b-versatile', 'use_for': ['groundedness', 'correctness']},
    'context': {'model': 'llama-3.1-8b-instant', 'use_for': ['context_relevance']}
}

class ModelRouter:
    """Routes evaluation tasks to dedicated models"""
    
    def __init__(self, api_key: str):
        from groq import Groq as GroqClient
        self.client = GroqClient(api_key=api_key)
        
    def clean_score(self, text: str) -> float:
        if not text:
            return None
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if not cleaned:
            cleaned = text
        patterns = [
            r'(?:score|rating)?\s*[:=]?\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*\/\s*(?:1|10|100)',
            r'(\d+\.?\d*)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, cleaned.lower())
            if matches:
                try:
                    score = float(matches[0])
                    if score > 1 and score <= 10:
                        score = score / 10
                    elif score > 10 and score <= 100:
                        score = score / 100
                    return max(0.0, min(1.0, score))
                except:
                    continue
        return None
    
    def call_model(self, prompt: str, model: str, max_retries: int = 3) -> float:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Output ONLY a number between 0 and 1."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0, max_tokens=10
                )
                
                score = self.clean_score(response.choices[0].message.content.strip())
                if score is not None:
                    return score
                elif attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return 0.5
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    return 0.5
        return 0.5
    
    def evaluate(self, task_type: str, prompt: str) -> float:
        task_model_map = {
            'relevance': MODEL_CONFIGS['fast'],
            'quality': MODEL_CONFIGS['fast'],
            'groundedness': MODEL_CONFIGS['primary'],
            'context_relevance': MODEL_CONFIGS['context'],
            'correctness': MODEL_CONFIGS['primary']
        }
        config = task_model_map.get(task_type, MODEL_CONFIGS['fast'])
        return self.call_model(prompt, config['model'])

# ============================================
# 4. METRIC EVALUATORS
# ============================================

def evaluate_relevance(input: str, output: str, router: ModelRouter) -> float:
    prompt = f"""Score relevance from 0 to 1. Output only the number.
Question: {input[:300]}
Answer: {output[:300] if output else "None"}
Score:"""
    return router.evaluate('relevance', prompt)

def evaluate_quality(input: str, output: str, router: ModelRouter) -> float:
    prompt = f"""Score quality from 0 to 1. Output only the number.
Question: {input[:300]}
Answer: {output[:300] if output else "None"}
Score:"""
    return router.evaluate('quality', prompt)

def evaluate_groundedness(input: str, output: str, router: ModelRouter) -> float:
    prompt = f"""Score factual reliability from 0 to 1. Output only the number.
Question: {input[:300]}
Answer: {output[:300] if output else "None"}
Score:"""
    return router.evaluate('groundedness', prompt)

def evaluate_context_relevance(input: str, output: str, router: ModelRouter) -> float:
    prompt = f"""Score context relevance from 0 to 1. Output only the number.
Question: {input[:300]}
Answer: {output[:300] if output else "None"}
Score:"""
    return router.evaluate('context_relevance', prompt)

def evaluate_correctness(input: str, output: str, router: ModelRouter) -> float:
    prompt = f"""Score correctness from 0 to 1. Output only the number.
Question: {input[:300]}
Answer: {output[:300] if output else "None"}
Score:"""
    return router.evaluate('correctness', prompt)

# ============================================
# 5. MAIN EXECUTION
# ============================================

def run_evaluation(eval_questions=None):
    """Run evaluation and return results DataFrame"""
    
    if eval_questions is None:
        eval_questions = [
            
            "Given the regular expression [A-Z][a-z]* [ ][A-Z][A-Z], what pattern does it represent and what is its limitation?",
            "List three software applications using automata.",
        ]
    
    embed_model = GeminiDirectEmbedding(api_key=GEMINI_API_KEY, model_name="gemini-embedding-2", dimension=768)
    llm = LlamaGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.3)
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        raise ValueError(f"Index '{INDEX_NAME}' not found!")
    
    pinecone_index = pc.Index(INDEX_NAME)
    rag = OptimizedRAG(pinecone_index, embed_model, llm)
    
    # Phase 1: Generate answers
    answers, latencies = [], []
    for q in eval_questions:
        start_time = time.time()
        answer = rag.query(q)
        latency = time.time() - start_time
        answers.append(answer)
        latencies.append(latency)
    
    # Phase 2: Calculate metrics
    router = ModelRouter(GROQ_API_KEY)
    metrics_data = []
    
    for q, a, lat in zip(eval_questions, answers, latencies):
        metrics_data.append({
            'question': q,
            'answer': a[:200] + "...",
            'relevance': evaluate_relevance(q, a, router),
            'quality': evaluate_quality(q, a, router),
            'groundedness': evaluate_groundedness(q, a, router),
            'context_relevance': evaluate_context_relevance(q, a, router),
            'correctness': evaluate_correctness(q, a, router),
            'latency': lat,
            'run_id': RUN_ID,
            'strategy': 'sentence_window'
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Save files
    df_metrics.to_csv(f"trulens_results_{RUN_ID}.csv", index=False)
    df_metrics.to_csv("trulens_results_latest.csv", index=False)
    
    history_file = "trulens_results_history.csv"
    if os.path.exists(history_file):
        df_history = pd.read_csv(history_file)
        df_history = pd.concat([df_history, df_metrics], ignore_index=True)
    else:
        df_history = df_metrics
    df_history.to_csv(history_file, index=False)
    
    return df_metrics

def main():
    """Run evaluation and save results"""
    df = run_evaluation()
    
    print("\n📈 Overall Averages:")
    for metric in ['relevance', 'quality', 'groundedness', 'context_relevance', 'correctness']:
        avg = df[metric].mean()
        status = "✅ high" if avg >= 0.7 else "⚠️ medium" if avg >= 0.5 else "🛑 low"
        print(f"  {metric.replace('_', ' ').title()}: {avg:.3f} {status}")
    print(f"  Average Latency: {df['latency'].mean():.2f}s")
    print(f"\n💾 Results saved: trulens_results_{RUN_ID}.csv")

if __name__ == "__main__":
    main()