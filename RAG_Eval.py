"""
RAG_Eval.py
============================================================

StudyBuddy RAG Evaluation System

Architecture:
    - TruLens is the ONLY evaluation database.
    - No custom SQLite evaluation wrapper.
    - No RAGWrapper.
    - Direct OptimizedRAG app instrumentation.
    - Questions are loaded from per-user cache files.
    - Existing questions can be skipped on reruns.
    - New questions are automatically added to TruLens.
    - Dashboard can already be running while evaluation happens.
    - Per-user question caching: loads existing questions, generates only the missing ones.
    - Automatic top_k scaling based on corpus size.

Evaluation metrics:
    1. Answer Relevance
    2. Answer Quality (context-aware)
    3. Context Relevance
    4. Groundedness
    5. Correctness (context-aware)

Models:
    RAG generation:
        openai/gpt-oss-20b (Groq) — light operation

    Question generation:
        openai/gpt-oss-120b (Groq) — light operation

    All 5 evaluation metrics:
        @cf/openai/gpt-oss-120b (Cloudflare Workers AI) — heavy workload
        Fallback: Groq if Cloudflare credentials missing

Embeddings:
    Gemini Embedding 2
"""

import os
import re
import sys
import time
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# SILENCE LITELLM COST-CALC TRACEBACKS
# ============================================================

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CLOUDFLARE_API_KEY = os.getenv("CLOUDFLARE_API_KEY-1")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID-1")

INDEX_NAME = os.getenv("INDEX_NAME", "studybuddy")

PORT = int(os.getenv("PORT", "8080"))

QUESTIONS_FILE = os.getenv("QUESTIONS_FILE", "generated_questions.txt")
QUESTIONS_DIR = os.getenv("QUESTIONS_DIR", "questions")

TRULENS_DB_URL = os.getenv("TRULENS_DB_URL", "sqlite:///data/default.sqlite")

APP_NAME = os.getenv("TRULENS_APP_NAME", "StudyBuddy RAG Evaluation")
APP_VERSION = os.getenv("TRULENS_APP_VERSION", "v3")

SKIP_EXISTING_QUESTIONS = (
    os.getenv("SKIP_EXISTING_QUESTIONS", "true").lower() == "true"
)

# ------------------------------------------------------------
# MODEL ASSIGNMENT
# ------------------------------------------------------------

RAG_MODEL = os.getenv("RAG_MODEL", "openai/gpt-oss-20b")
QUESTION_MODEL = os.getenv("QUESTION_MODEL", "openai/gpt-oss-120b")

FAST_EVAL_MODEL = os.getenv("FAST_EVAL_MODEL", "openai/gpt-oss-20b")
DEEP_EVAL_MODEL = os.getenv("DEEP_EVAL_MODEL", "openai/gpt-oss-120b")

CLOUDFLARE_MODEL = os.getenv(
    "CLOUDFLARE_MODEL",
    "@cf/openai/gpt-oss-120b"
)

FALLBACK_QUESTION_MODEL = os.getenv(
    "FALLBACK_QUESTION_MODEL",
    "moonshotai/kimi-k2-instruct"
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

QUESTIONS_PER_USER = int(os.getenv("QUESTIONS_PER_USER", "25"))
INTER_QUESTION_DELAY = int(os.getenv("INTER_QUESTION_DELAY", "15"))

# ------------------------------------------------------------
# MAX_TOKENS LIMITS
# ------------------------------------------------------------

RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "2000"))
QUESTION_MAX_TOKENS = int(os.getenv("QUESTION_MAX_TOKENS", "2500"))
CLOUDFLARE_MAX_TOKENS = int(os.getenv("CLOUDFLARE_MAX_TOKENS", "500"))

print("=" * 70)
print("🔍 STUDYBUDDY RAG EVALUATION WITH TRULENS")
print("=" * 70)

print(f"GEMINI_API_KEY: {'✅' if GEMINI_API_KEY else '❌'}")
print(f"PINECONE_API_KEY: {'✅' if PINECONE_API_KEY else '❌'}")
print(f"GROQ_API_KEY: {'✅' if GROQ_API_KEY else '❌'}")
print(f"CLOUDFLARE_API_KEY: {'✅' if CLOUDFLARE_API_KEY else '❌'}")
print(f"CLOUDFLARE_ACCOUNT_ID: {'✅' if CLOUDFLARE_ACCOUNT_ID else '❌'}")
print(f"INDEX_NAME: {INDEX_NAME}")
print(f"QUESTIONS_FILE (legacy): {QUESTIONS_FILE}")
print(f"QUESTIONS_DIR: {QUESTIONS_DIR}")
print(f"TRULENS_DB_URL: {TRULENS_DB_URL}")
print(f"APP_NAME: {APP_NAME}")
print(f"APP_VERSION: {APP_VERSION}")
print(f"RAG_MODEL: {RAG_MODEL}")
print(f"RAG_MAX_TOKENS: {RAG_MAX_TOKENS}")
print(f"QUESTION_MODEL: {QUESTION_MODEL}")
print(f"QUESTION_MAX_TOKENS: {QUESTION_MAX_TOKENS}")
print(f"CLOUDFLARE_MODEL: {CLOUDFLARE_MODEL}")
print(f"CLOUDFLARE_MAX_TOKENS: {CLOUDFLARE_MAX_TOKENS}")
print(f"FAST_EVAL_MODEL (Groq fallback): {FAST_EVAL_MODEL}")
print(f"DEEP_EVAL_MODEL (Groq fallback): {DEEP_EVAL_MODEL}")
print(f"FALLBACK_QUESTION_MODEL: {FALLBACK_QUESTION_MODEL}")
print(f"QUESTIONS_PER_USER: {QUESTIONS_PER_USER}")
print(f"INTER_QUESTION_DELAY: {INTER_QUESTION_DELAY}s")
print("=" * 70)

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY is missing.")
    sys.exit(1)

if not PINECONE_API_KEY:
    print("❌ PINECONE_API_KEY is missing.")
    sys.exit(1)

if not GROQ_API_KEY:
    print("❌ GROQ_API_KEY is missing.")
    sys.exit(1)

if not CLOUDFLARE_API_KEY or not CLOUDFLARE_ACCOUNT_ID:
    print("⚠️ Cloudflare credentials missing.")
    print("   All 5 metrics will fall back to Groq (may hit TPM limits).")

os.makedirs(QUESTIONS_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


# ============================================================
# TRULENS / OTEL CONFIGURATION
# ============================================================

os.environ["TRULENS_OTEL_TRACING"] = "1"
os.environ["TRULENS_OTEL_ENABLED"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "false"
os.environ["TRULENS_FEEDBACK_MAX_WORKERS"] = "1"


# ============================================================
# IMPORTS
# ============================================================

from pinecone import Pinecone

from google import genai
from google.genai import types

from groq import Groq
from openai import OpenAI

from trulens.core import (
    TruSession,
    Metric,
    Selector,
)

from trulens.apps.app import TruApp

from trulens.core.otel.instrument import instrument

from trulens.otel.semconv.trace import SpanAttributes

import logging
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)


# ============================================================
# TOP_K AUTO-SCALING
# ============================================================

def calculate_top_k(total_chunks: int) -> int:
    """Auto-scale top_k based on corpus size."""
    if total_chunks <= 20:
        return 3
    elif total_chunks <= 50:
        return 5
    elif total_chunks <= 100:
        return 8
    elif total_chunks <= 300:
        return 15
    elif total_chunks <= 600:
        return 25
    else:
        return 40


# ============================================================
# CLOUDFLARE WORKERS AI PROVIDER
# ============================================================

class CloudflareProvider:
    """
    Custom provider for Cloudflare Workers AI with all 5 TruLens metrics.

    Correctness and Answer Quality now receive the retrieved context
    so the judge can assess correctness/quality WITH the source material,
    not just the answer in isolation.
    """

    def __init__(
        self,
        api_key: str,
        account_id: str,
        model: str,
        max_tokens: int = CLOUDFLARE_MAX_TOKENS,
    ):
        self.model = model
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=api_key,
            base_url=(
                f"https://api.cloudflare.com/client/v4/"
                f"accounts/{account_id}/ai/v1"
            ),
        )

    def _call(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    def _parse_score(self, text: str) -> float:
        match = re.search(
            r'(?:score[:\s]+)?(\d+\.?\d*)',
            text,
            re.IGNORECASE,
        )
        if not match:
            return 0.5
        try:
            score = float(match.group(1))
            if score > 1.0 and score <= 100.0:
                score = score / 100.0
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.5

    # --------------------------------------------------------
    # TruLens-compatible metric methods
    # --------------------------------------------------------

    def relevance_with_cot_reasons(
        self,
        prompt: str,
        response: str,
    ) -> Tuple[float, Dict]:
        """Answer Relevance score (question vs answer)."""
        full_prompt = f"""You are an evaluation assistant.
Score how directly the ANSWER addresses the QUESTION.

QUESTION:
{prompt[:1200]}

ANSWER:
{response[:2000]}

Output a single number between 0 and 1.
Format:
Score: <number>
Reason: <brief explanation>"""

        text = self._call(full_prompt)
        score = self._parse_score(text)
        return score, {"reason": text[:500]}

    def coherence(
        self,
        source: str,
        text: str,
    ) -> Tuple[float, Dict]:
        """
        Answer Quality / coherence score — CONTEXT-AWARE.

        The judge now sees the retrieved context, so it can
        properly assess whether the answer is complete, useful,
        and appropriate given the source material.
        """
        full_prompt = f"""You are an evaluation assistant.
Score the overall QUALITY of the ANSWER from 0 to 1.
Consider clarity, completeness, usefulness, and whether the answer
fully addresses what the context supports.

Do NOT penalize short answers if the answer is complete.
Do NOT penalize answers that correctly refuse when the context
does not contain the information.

CONTEXT (what the answer was based on):
{source[:3000]}

ANSWER:
{text[:2000]}

Output a single number between 0 and 1.
Format:
Score: <number>
Reason: <brief explanation>"""

        response_text = self._call(full_prompt)
        score = self._parse_score(response_text)
        return score, {"reason": response_text[:500]}

    def context_relevance_with_cot_reasons(
        self,
        question: str,
        context: str,
    ) -> Tuple[float, Dict]:
        """Context Relevance score (question vs retrieved chunks)."""
        full_prompt = f"""You are an evaluation assistant.
Score how RELEVANT the retrieved CONTEXT is to answering the QUESTION.
1 = highly relevant. 0 = unrelated.

QUESTION:
{question[:1200]}

CONTEXT:
{context[:5000]}

Output a single number between 0 and 1.
Format:
Score: <number>
Reason: <brief explanation>"""

        text = self._call(full_prompt)
        score = self._parse_score(text)
        return score, {"reason": text[:500]}

    def groundedness_measure_with_cot_reasons(
        self,
        source: str,
        statement: str,
    ) -> Tuple[float, Dict]:
        """Groundedness score (context vs answer claims)."""
        full_prompt = f"""You are an evaluation assistant.
Score GROUNDEDNESS from 0 to 1.
Every substantive claim in the ANSWER should be supported by the CONTEXT.
Penalize hallucinations or unsupported claims.

CONTEXT:
{source[:5000]}

ANSWER:
{statement[:3000]}

Output a single number between 0 and 1.
Format:
Score: <number>
Reason: <brief explanation>"""

        text = self._call(full_prompt)
        score = self._parse_score(text)
        return score, {"reason": text[:500]}

    def correctness(
        self,
        source: str,
        text: str,
    ) -> Tuple[float, Dict]:
        """
        Correctness score — CONTEXT-AWARE.

        The judge now sees the retrieved context, so it can verify
        whether the answer's facts match the source material rather
        than trying to assess correctness in isolation.

        This eliminates the "short answer penalization" problem
        where terse but accurate answers were scored low.
        """
        full_prompt = f"""You are an evaluation assistant.
Score CORRECTNESS from 0 to 1.
Judge whether the ANSWER is factually correct AND consistent
with the CONTEXT.

A correct answer will match the facts in the context.
If the context does not contain the information, saying so is correct.

CONTEXT (source material):
{source[:3000]}

ANSWER:
{text[:2000]}

Output a single number between 0 and 1.
Format:
Score: <number>
Reason: <brief explanation>"""

        response_text = self._call(full_prompt)
        score = self._parse_score(response_text)
        return score, {"reason": response_text[:500]}


# ============================================================
# GEMINI EMBEDDING
# ============================================================

class GeminiDirectEmbedding:
    """Direct Gemini Embedding 2 client (768-dim)."""

    QUERY_PREFIX = "task: search result | query: "

    def __init__(
        self,
        api_key: str,
        model_name: str = EMBEDDING_MODEL,
        dimension: int = EMBEDDING_DIMENSION,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.dimension = dimension
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def embed_text(self, text: str) -> Optional[List[float]]:
        if not text or not text.strip():
            return None

        prefixed = (self.QUERY_PREFIX + text.strip())[:8000]

        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=[prefixed],
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimension
                ),
            )

            if not result or not result.embeddings:
                return None

            embedding = result.embeddings[0].values
            if not embedding:
                return None

            norm = sum(v * v for v in embedding) ** 0.5
            if norm == 0:
                return None

            return [v / norm for v in embedding]

        except Exception as exc:
            print(f"⚠️ Gemini embedding error: {exc}")
            return None

    def embed_query(self, text: str) -> Optional[List[float]]:
        return self.embed_text(text)


# ============================================================
# PINECONE USER DISCOVERY
# ============================================================

def fetch_all_users(pinecone_index) -> List[Dict]:
    print()
    print("🔍 Fetching users from Pinecone...")

    try:
        dummy_vector = [0.01 for _ in range(EMBEDDING_DIMENSION)]

        results = pinecone_index.query(
            vector=dummy_vector,
            top_k=10000,
            include_metadata=True,
            namespace="users",
            filter={"type": {"$eq": "user_auth"}},
        )

        users = []
        seen_users = set()

        for match in results.matches:
            metadata = match.metadata
            if not metadata:
                continue

            user_id = metadata.get("user_id")
            email = metadata.get("email")

            if not user_id or user_id in seen_users:
                continue

            seen_users.add(user_id)
            users.append({
                "user_id": user_id,
                "email": email or user_id,
                "created_at": metadata.get("created_at", ""),
            })

        print(f"✅ Found {len(users)} users")
        return users

    except Exception as exc:
        print(f"❌ Failed to fetch users: {exc}")
        return []


# ============================================================
# USER NOTES
# ============================================================

def fetch_user_notes(pinecone_index, user_id: str) -> List[Dict]:
    try:
        dummy_vector = [0.01 for _ in range(EMBEDDING_DIMENSION)]

        results = pinecone_index.query(
            vector=dummy_vector,
            top_k=10000,
            include_metadata=True,
            namespace="notes",
            filter={
                "user_id": {"$eq": user_id},
                "type": {"$eq": "notes"},
            },
        )

        notes = []

        for match in results.matches:
            metadata = match.metadata
            if not metadata:
                continue

            text = metadata.get("text", "")
            if not text:
                continue

            notes.append({
                "id": match.id,
                "text": text,
                "chunk_index": metadata.get("chunk_index", 0),
                "timestamp": metadata.get("timestamp", 0),
                "source": metadata.get("source", "uploaded_notes"),
            })

        notes.sort(key=lambda item: item.get("chunk_index", 0))
        return notes

    except Exception as exc:
        print(f"⚠️ Failed to fetch notes for {user_id}: {exc}")
        return []


# ============================================================
# PER-USER QUESTION FILE MANAGEMENT
# ============================================================

def get_questions_file_path(user_id: str) -> str:
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)
    return os.path.join(QUESTIONS_DIR, f"questions_{safe_id}.txt")


def parse_question_file(path: str) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    questions = []

    try:
        content = file_path.read_text(encoding="utf-8")

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue

            line = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", line)
            line = line.strip(" -*\t")

            if len(line) < 10:
                continue

            questions.append(line)

        unique_questions = []
        seen = set()

        for question in questions:
            key = question.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            unique_questions.append(question)

        return unique_questions

    except Exception as exc:
        print(f"❌ Failed reading {path}: {exc}")
        return []


def save_questions_to_file(user_id: str, questions: List[str]):
    path = get_questions_file_path(user_id)

    try:
        with open(path, "w", encoding="utf-8") as f:
            for i, q in enumerate(questions, 1):
                f.write(f"{i}. {q}\n")
        print(f"💾 Saved {len(questions)} questions to {path}")
    except Exception as exc:
        print(f"⚠️ Failed to write {path}: {exc}")


# ============================================================
# AI QUESTION GENERATION
# ============================================================

def generate_questions_from_notes(
    notes: List[Dict],
    num_questions: int = 25,
) -> List[str]:

    if not notes or num_questions <= 0:
        return []

    full_text = " ".join(note["text"] for note in notes)
    if len(full_text) > 12000:
        full_text = full_text[:12000]

    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
Based ONLY on the educational content below,
generate exactly {num_questions} evaluation questions.

CONTENT:

{full_text}

Requirements:

- Questions must be answerable from the content.
- Cover different parts of the content.
- Mix basic, conceptual, analytical and application questions.
- Do not invent information outside the content.
- Do not provide answers.
- Output ONLY a numbered list.
"""

    for model_name in [QUESTION_MODEL, FALLBACK_QUESTION_MODEL]:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Generate evaluation questions only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=QUESTION_MAX_TOKENS,
            )

            text = (response.choices[0].message.content or "").strip()

            questions = []
            for line in text.splitlines():
                line = line.strip()
                match = re.match(r"^\s*\d+\s*[\.\)]\s*(.+)", line)
                if match:
                    question = match.group(1).strip()
                    if len(question) >= 10:
                        questions.append(question)

            result = []
            seen = set()
            for question in questions:
                key = question.lower()
                if key in seen:
                    continue
                seen.add(key)
                result.append(question)

            print(f"🤖 AI generated {len(result)} questions using {model_name}")
            return result[:num_questions]

        except Exception as exc:
            print(f"⚠️ Question generation failed with {model_name}: {exc}")
            if model_name == FALLBACK_QUESTION_MODEL:
                return []
            print("   Trying fallback model...")
            continue

    return []


# ============================================================
# QUESTION SOURCE
# ============================================================

def get_or_generate_questions(
    notes: List[Dict],
    user_id: str,
    target: int = 25,
) -> List[str]:

    print(f"\n🧠 Question planning for user {user_id} (target: {target})")

    path = get_questions_file_path(user_id)
    existing = parse_question_file(path)

    if existing:
        print(f"📄 Loaded {len(existing)} existing questions from {path}")
    else:
        print(f"📄 No existing question file for {user_id} (will create one)")

    if len(existing) >= target:
        print(f"✅ Already have {len(existing)} questions (>= {target}), using file as-is")
        return existing[:target]

    missing = target - len(existing)
    print(f"📊 Have {len(existing)} questions, need {missing} more")

    if not notes:
        print("❌ No notes available to generate questions from!")
        return existing if existing else []

    print(f"🤖 Generating {missing} new questions from notes...")
    new_questions = generate_questions_from_notes(notes, num_questions=missing)

    if not new_questions:
        print("⚠️ AI generation returned no questions")
        return existing

    combined = list(existing)
    seen = set(q.lower().strip() for q in existing)

    added = 0
    for q in new_questions:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            combined.append(q)
            added += 1

    print(f"➕ Added {added} new unique questions (total: {len(combined)})")

    save_questions_to_file(user_id, combined)
    return combined[:target]


# ============================================================
# RAG APPLICATION
# ============================================================

class OptimizedRAG:

    def __init__(
        self,
        pinecone_index,
        embed_model,
        groq_client,
        user_id: str,
        total_notes: int = 0,
    ):
        self.pinecone_index = pinecone_index
        self.embed_model = embed_model
        self.groq_client = groq_client
        self.user_id = user_id
        self.total_notes = total_notes

        # LOCKED top_k=3 — proven optimal across 3 corpora
        self.top_k = 3

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "question",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve_context(self, question: str) -> List[str]:

        embedding = self.embed_model.embed_query(question)
        if not embedding:
            return []

        results = self.pinecone_index.query(
            vector=embedding,
            top_k=self.top_k,
            include_metadata=True,
            namespace="notes",
            filter={"user_id": {"$eq": self.user_id}},
        )

        contexts = []
        for match in results.matches:
            metadata = match.metadata
            if not metadata:
                continue
            text = metadata.get("text")
            if not text:
                continue
            contexts.append(text[:800])

        return contexts

    @instrument(
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "question",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, question: str) -> str:

        contexts = self.retrieve_context(question)

        if not contexts:
            return "I could not find relevant information in the user's notes."

        context_text = "\n\n".join(contexts)

        # SOFTER PROMPT — allows partial answers, only refuses when
        # context is completely unrelated. Preserves groundedness.
        prompt = f"""
You are StudyBuddy, an educational AI assistant.

Answer the user's question using ONLY the retrieved context.

If the context fully answers the question, answer completely.
If the context partially answers the question, give the partial answer
and explicitly note what's missing.
Only say "the notes don't contain this information" if the context is
completely unrelated to the question.

Never invent facts outside the context.

RETRIEVED CONTEXT:

{context_text}

QUESTION:

{question}

ANSWER:
"""

        response = self.groq_client.chat.completions.create(
            model=RAG_MODEL,
            messages=[
                {"role": "system", "content": "Answer using only the supplied context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=RAG_MAX_TOKENS,
        )

        return (response.choices[0].message.content or "").strip()


# ============================================================
# TRULENS METRICS
# ============================================================

def create_metrics():
    """
    All 5 metrics → Cloudflare Workers AI.
    Correctness + Answer Quality are now CONTEXT-AWARE.
    """
    if CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID:
        provider = CloudflareProvider(
            api_key=CLOUDFLARE_API_KEY,
            account_id=CLOUDFLARE_ACCOUNT_ID,
            model=CLOUDFLARE_MODEL,
            max_tokens=CLOUDFLARE_MAX_TOKENS,
        )
        source = f"Cloudflare {CLOUDFLARE_MODEL}"

        answer_relevance = Metric(
            implementation=provider.relevance_with_cot_reasons,
            name="Answer Relevance",
            selectors={
                "prompt": Selector.select_record_input(),
                "response": Selector.select_record_output(),
            },
        )

        # CONTEXT-AWARE: Answer Quality now receives retrieved context
        answer_quality = Metric(
            implementation=provider.coherence,
            name="Answer Quality",
            selectors={
                "source": Selector.select_context(collect_list=True),
                "text": Selector.select_record_output(),
            },
        )

        context_relevance = Metric(
            implementation=provider.context_relevance_with_cot_reasons,
            name="Context Relevance",
            selectors={
                "question": Selector.select_record_input(),
                "context": Selector.select_context(collect_list=False),
            },
        )

        groundedness = Metric(
            implementation=provider.groundedness_measure_with_cot_reasons,
            name="Groundedness",
            selectors={
                "source": Selector.select_context(collect_list=True),
                "statement": Selector.select_record_output(),
            },
        )

        # CONTEXT-AWARE: Correctness now receives retrieved context
        correctness = Metric(
            implementation=provider.correctness,
            name="Correctness",
            selectors={
                "source": Selector.select_context(collect_list=True),
                "text": Selector.select_record_output(),
            },
        )

        print(f"   All 5 metrics → {source}")
        print(f"   Correctness + Answer Quality: CONTEXT-AWARE")

        return [
            answer_relevance,
            answer_quality,
            context_relevance,
            groundedness,
            correctness,
        ]

    # --------------------------------------------------------
    # Fallback: Groq via LiteLLM
    # --------------------------------------------------------
    from trulens.providers.litellm import LiteLLM

    fast_provider = LiteLLM(
        model_engine=f"groq/{FAST_EVAL_MODEL}",
        api_key=GROQ_API_KEY,
    )

    deep_provider = LiteLLM(
        model_engine=f"groq/{DEEP_EVAL_MODEL}",
        api_key=GROQ_API_KEY,
    )

    # LiteLLM's correctness/coherence accept source + text arguments
    return [
        Metric(
            implementation=fast_provider.relevance_with_cot_reasons,
            name="Answer Relevance",
            selectors={
                "prompt": Selector.select_record_input(),
                "response": Selector.select_record_output(),
            },
        ),
        Metric(
            implementation=fast_provider.coherence,
            name="Answer Quality",
            selectors={
                "source": Selector.select_context(collect_list=True),
                "text": Selector.select_record_output(),
            },
        ),
        Metric(
            implementation=fast_provider.context_relevance_with_cot_reasons,
            name="Context Relevance",
            selectors={
                "question": Selector.select_record_input(),
                "context": Selector.select_context(collect_list=False),
            },
        ),
        Metric(
            implementation=deep_provider.groundedness_measure_with_cot_reasons,
            name="Groundedness",
            selectors={
                "source": Selector.select_context(collect_list=True),
                "statement": Selector.select_record_output(),
            },
        ),
        Metric(
            implementation=deep_provider.correctness,
            name="Correctness",
            selectors={
                "source": Selector.select_context(collect_list=True),
                "text": Selector.select_record_output(),
            },
        ), 
    ]


# ============================================================
# EXISTING QUESTIONS
# ============================================================

def get_existing_questions(session: TruSession) -> set:

    if not SKIP_EXISTING_QUESTIONS:
        return set()

    try:
        records, _ = session.get_records_and_feedback(
            app_name=APP_NAME,
            app_versions=[APP_VERSION],
        )

        if records is None or records.empty:
            return set()

        existing = set()

        for col in ["input", "main_input"]:
            if col in records.columns:
                for value in records[col]:
                    if isinstance(value, str):
                        existing.add(value.strip().lower())

        return existing

    except Exception as exc:
        print(f"⚠️ Could not read existing questions: {exc}")
        return set()


# ============================================================
# EVALUATE ONE USER
# ============================================================

def evaluate_user(
    session: TruSession,
    pinecone_index,
    embedding_model,
    groq_client,
    user: Dict,
    metrics,
) -> int:

    user_id = user["user_id"]
    email = user["email"]

    print()
    print("=" * 70)
    print(f"👤 User: {email}")
    print(f"🆔 User ID: {user_id}")
    print("=" * 70)

    notes = fetch_user_notes(pinecone_index, user_id)

    if not notes:
        print("❌ No notes found. Skipping user.")
        return 0

    total_notes = len(notes)
    print(f"📚 Found {total_notes} note chunks")

    questions = get_or_generate_questions(
        notes=notes,
        user_id=user_id,
        target=QUESTIONS_PER_USER,
    )

    if not questions:
        print("❌ No evaluation questions.")
        return 0

    existing_questions = get_existing_questions(session)

    if existing_questions:
        original_count = len(questions)
        questions = [
            q for q in questions
            if q.strip().lower() not in existing_questions
        ]
        skipped = original_count - len(questions)
        if skipped:
            print(f"⏭️ Skipping {skipped} already evaluated questions.")

    if not questions:
        print("✅ No new questions to evaluate.")
        return 0

    print(f"📝 Questions to evaluate: {len(questions)}")

    rag = OptimizedRAG(
        pinecone_index=pinecone_index,
        embed_model=embedding_model,
        groq_client=groq_client,
        user_id=user_id,
        total_notes=total_notes,
    )

    tru_app = TruApp(
        rag,
        app_name=f"{APP_NAME} - {email}",
        app_versions=APP_VERSION,
        main_method=rag.query,
        feedbacks=metrics,
    )

    completed = 0

    for index, question in enumerate(questions, start=1):
        print()
        print(f"📝 Question {index}/{len(questions)}")
        print(f"   {question}")

        try:
            with tru_app(conversation_id=f"{user_id}") as recording:
                response = rag.query(question)

            print(f"   ✅ Response: {response[:250]}...")

            try:
                recording.retrieve_feedback_results()
                print("   📊 TruLens feedback completed.")
            except Exception as feedback_error:
                print(f"   ⚠️ Feedback is still processing: {feedback_error}")

            completed += 1

        except Exception as exc:
            print(f"   ❌ Question failed: {exc}")

        if index < len(questions):
            print(f"   ⏳ Waiting {INTER_QUESTION_DELAY}s...")
            time.sleep(INTER_QUESTION_DELAY)

        print("-" * 70)

    try:
        session.force_flush()
    except Exception:
        pass

    print()
    print(f"✅ Completed {completed}/{len(questions)} questions for {email}")

    return completed


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print()
    print("=" * 70)
    print("🚀 STARTING STUDYBUDDY RAG EVALUATION")
    print("=" * 70)

    try:
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pinecone_client.Index(INDEX_NAME)
        print(f"✅ Connected to Pinecone: {INDEX_NAME}")
    except Exception as exc:
        print(f"❌ Pinecone connection failed: {exc}")
        sys.exit(1)

    users = fetch_all_users(pinecone_index)

    if not users:
        print("❌ No users discovered.")
        sys.exit(1)

    print()
    print(f"👥 Total users: {len(users)}")

    for user in users:
        print(f"   - {user['email']} ({user['user_id']})")

    embedding_model = GeminiDirectEmbedding(api_key=GEMINI_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)

    print("✅ Gemini Embedding 2 initialized (with task prefix)")
    print(f"✅ RAG model: {RAG_MODEL}")

    print()
    print("🦑 Connecting to TruLens database...")

    session = TruSession(database_url=TRULENS_DB_URL)

    print("✅ TruLens session connected")
    print("🔧 Creating TruLens metrics...")

    metrics = create_metrics()

    print(f"✅ Created {len(metrics)} TruLens metrics")

    total_completed = 0

    for user in users:
        completed = evaluate_user(
            session=session,
            pinecone_index=pinecone_index,
            embedding_model=embedding_model,
            groq_client=groq_client,
            user=user,
            metrics=metrics,
        )
        total_completed += completed

    try:
        session.force_flush()
    except Exception:
        pass

    print()
    print("=" * 70)
    print("🎉 EVALUATION FINISHED")
    print("=" * 70)
    print(f"👥 Users discovered: {len(users)}")
    print(f"📝 New questions evaluated: {total_completed}")
    print(f"🦑 TruLens database: {TRULENS_DB_URL}")
    print("=" * 70)

    print()
    print("📊 The TruLens dashboard remains available.")
    print(
        f"💡 Add new questions to "
        f"{QUESTIONS_DIR}/questions_<user_id>.txt "
        "and rerun the evaluator."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Evaluation stopped.")
        sys.exit(0)
    except Exception as exc:
        print()
        print(f"❌ Fatal error: {exc}")
        sys.exit(1)