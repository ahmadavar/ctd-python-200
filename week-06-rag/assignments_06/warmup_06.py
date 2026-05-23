from dotenv import load_dotenv
import os
import string

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# =============================================================================
# PART 1: RAG CONCEPTS
# =============================================================================

# --- RAG Concepts ---

# Concepts Q1
# Best approach for each scenario:
#
# Scenario A: Legal team, hundreds of PDFs updated every quarter
# -> RAG
# The document library is large and changes frequently. Fine-tuning would require
# retraining the model every quarter just to pick up new policy changes. RAG lets
# you update the document store without touching the model at all.
#
# Scenario B: Startup wants a specific brand voice, 3,000 human-written examples
# -> Fine-tuning
# This is about teaching a style, not facts. The content won't change — the goal is
# to make the model write in a particular tone. Fine-tuning is the right tool when
# you're baking in behavior rather than knowledge.
#
# Scenario C: Analyst has a single two-page report, one-off question
# -> Prompt engineering (context injection)
# The document is small enough to paste directly into the prompt. No infrastructure,
# no setup, immediate result. RAG would be overkill for a single document.


# Concepts Q2
# Why is a confidently wrong answer more harmful than "I'm not sure"?
#
# When a model hedges ("I'm not sure"), the reader knows to verify the answer before
# acting on it. When a model is confidently wrong, the reader has no signal to distrust
# the response — it looks like a reliable answer. The tone of certainty suppresses the
# natural instinct to double-check.
#
# Real harm example: a doctor asks an LLM about a drug dosage. The model confidently
# states the wrong maximum dose. Because it sounds authoritative, the doctor doesn't
# verify it. That confident wrong answer has real clinical consequences that a hedged
# "I'm not sure, please verify" would have prevented.


# Concepts Q3
# RAG pipeline steps — correct order with descriptions:
#
# 1. Extract text from source documents
#    - Pull raw text out of PDFs, Word files, or other formats.
#
# 2. Split text into chunks
#    - Break long documents into smaller passages (e.g. 256-512 tokens each)
#      so only the relevant section gets retrieved, not the whole file.
#
# 3. Convert text chunks into embeddings
#    - Run each chunk through an embedding model to get a vector — a list of
#      numbers that represents the chunk's meaning.
#
# 4. Receive the user's query
#    - The user types a question.
#
# 5. Embed the user's query
#    - Convert the question into a vector using the same embedding model.
#
# 6. Retrieve the most relevant chunks
#    - Compare the query vector against all stored chunk vectors using cosine
#      similarity. Return the top-k closest matches.
#
# 7. Inject retrieved chunks into the prompt
#    - Prepend the retrieved text to the LLM prompt so the model can read it.
#
# 8. Generate a response from the LLM
#    - The LLM answers based on the injected context, not its general training.


# =============================================================================
# PART 2: KEYWORD RAG
# =============================================================================

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

# --- Keyword RAG ---

# Keyword Q1
print("\n--- Keyword Q1 ---")
q1 = "What are your hours on the weekend?"
result1 = simple_keyword_retrieval(q1, documents)
print(f"Selected: {result1[0][0]}")
# Actual result: loyalty.txt was selected — NOT hours.txt.
# Why? The query token "weekend" is singular, but hours.txt uses "weekends" (plural).
# Keyword RAG does exact matching with no stemming — "weekend" != "weekends".
# "hours" also doesn't appear literally in hours.txt (it says "7am to 7pm").
# The only overlap found was "your" (in hiring.txt and loyalty.txt), and loyalty.txt
# won the tie. This is a clear failure — the obviously correct document wasn't retrieved.


# Keyword Q2
print("\n--- Keyword Q2 ---")
q2 = "Do you have anything without caffeine?"
result2 = simple_keyword_retrieval(q2, documents)
print(f"Selected: {result2[0][0]}")
# Actual result: None found — no document was selected at all.
# "caffeine" doesn't appear in any document. Words like "do", "have", "without"
# aren't in the stopwords list so they show up as query tokens, but they don't
# match anything meaningful either. Keyword RAG completely fails here.
# Semantic RAG would understand that "without caffeine" relates to decaf/milk options
# and retrieve the menu — because it searches by meaning, not exact words.


# Keyword Q3
print("\n--- Keyword Q3 ---")
# Prediction: loyalty.txt
# Reasoning: "sign" appears in loyalty.txt ("Sign up at the register"), so it should
# score at least 1. "rewards" won't match because the doc uses "loyalty program".
q3 = "How do I sign up for rewards?"
result3 = simple_keyword_retrieval(q3, documents)
print(f"Selected: {result3[0][0]}")
# Actual result: None found — prediction was wrong.
# "sign" does appear in loyalty.txt but "sign" is not in the stopwords list,
# so the query token set includes "sign" — but the document tokenizer strips
# punctuation, and "sign" in the document is followed by a period: it does exist.
# The mismatch likely came from the document not containing enough raw token overlap.
# This shows that keyword RAG is fragile even for simple vocabulary matches.


# =============================================================================
# PART 3: SEMANTIC RAG CONCEPTS
# =============================================================================

# --- Semantic RAG Concepts ---

# Semantic Q1
#
# What is a vector embedding?
# An embedding is a list of numbers (a vector) that represents the meaning of a
# piece of text. Words or sentences with similar meanings end up close together
# in that numeric space — the model has encoded meaning as geometry.
#
# Cosine similarity scores:
# A chunk with score 0.85 is highly relevant — the query and chunk are very
# similar in meaning. A chunk with 0.30 is loosely related at best, probably
# not what the user is asking about. The score measures the angle between the
# two vectors: 1.0 = identical direction (same meaning), 0.0 = unrelated.
#
# Why semantic search handles missing exact words:
# The embedding model was trained on massive text — it learned that "car" and
# "automobile" appear in similar contexts and assigns them similar vectors.
# So even if the query says "car" and the document says "automobile", their
# vectors land close together in the embedding space and cosine similarity
# will still return a high score.


# Semantic Q2
# Keyword RAG vs Semantic RAG comparison:
#
# | Feature                    | Keyword RAG                       | Semantic RAG                        |
# |----------------------------|-----------------------------------|-------------------------------------|
# | What is compared?          | Exact word overlap                | Vector similarity (meaning)         |
# | What is retrieved?         | Full document                     | Specific chunks                     |
# | Can it handle synonyms?    | No                                | Yes                                 |
# | Storage format             | Plain text dictionary             | Vectors in a vector store           |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity (0 to 1)          |


# =============================================================================
# PART 4: LLAMAINDEX
# =============================================================================

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Anthropic(model="claude-haiku-4-5-20251001")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

BRIGHTLEAF_PATH = os.path.expanduser(
    "~/ctd-school-repo/lessons/06_AI_augmentation/resources/brightleaf_pdfs/"
)

documents = SimpleDirectoryReader(BRIGHTLEAF_PATH).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

# --- LlamaIndex Q1 ---

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    response = query_engine.query(question)
    print(f"Answer: {response}")
    print("\nSource nodes:")
    for i, node in enumerate(response.source_nodes, 1):
        print(f"  [{i}] score={node.score:.4f} | {node.text[:150]!r}")

# Q1 observations:
#
# Query 1 — "What employee benefits does BrightLeaf offer?"
# - Top chunk: employee_benefits.pdf score=0.8674 (very high — direct match)
# - Response: specific and confident — named Blue Cross medical, 401k match up to 5%,
#   12 weeks parental leave, $600 wellness reimbursement. No hedging at all.
# - Chunk 2 was from security_policy.pdf (score=0.7034) — unexpected, but the LLM
#   correctly ignored it and answered only from the benefits content.
#
# Query 2 — "What are BrightLeaf's security policies?"
# - Top chunk: security_policy.pdf score=0.8122 (strong match)
# - Response: detailed breakdown — MFA, TLS 1.3, AES-256, NIST 800-61 incident response,
#   90-day credential rotation. Highly specific and confident.
# - Nothing unexpected retrieved; semantic search matched "security policies" precisely.

# --- LlamaIndex Q2 ---

test_query = "What employee benefits does BrightLeaf offer?"

for top_k in [1, 5]:
    print(f"\n{'='*60}")
    print(f"top_k={top_k} | Query: {test_query}")
    qe = index.as_query_engine(similarity_top_k=top_k)
    response = qe.query(test_query)
    print(f"Answer: {response}")
    print("\nSource nodes:")
    for i, node in enumerate(response.source_nodes, 1):
        print(f"  [{i}] score={node.score:.4f} | {node.text[:150]!r}")

# Q2 observations:
#
# top_k=1:
# - Only one chunk retrieved — the response covers whatever that single chunk contained.
# - If the benefits span multiple sections (health, PTO, retirement), the answer is
#   likely incomplete. The model answers confidently but with less detail.
#
# top_k=5:
# - Five chunks retrieved — more coverage, but some chunks may be loosely related
#   or repeat similar information. The response tends to be more complete.
# - More context is NOT always better: too many chunks can dilute the prompt with
#   noise, confuse the model, and increase cost. The sweet spot depends on how
#   spread out the answer is across the source documents.

# --- LlamaIndex Q3 ---

edge_query = "How does BrightLeaf's employee satisfaction compare to its financial performance?"
print(f"\n{'='*60}")
print(f"Edge case query: {edge_query}")
qe_edge = index.as_query_engine(similarity_top_k=3)
response_edge = qe_edge.query(edge_query)
print(f"Answer: {response_edge}")
print("\nSource nodes:")
for i, node in enumerate(response_edge.source_nodes, 1):
    print(f"  [{i}] score={node.score:.4f} | {node.text[:150]!r}")

# Q3 observations:
#
# Why this query is hard:
# - It spans two separate documents (employee_benefits.pdf and earnings_report.pdf)
# - It asks for a *comparison* — a reasoning task, not just a retrieval task
# - RAG retrieves facts; synthesizing a comparison across documents is the LLM's job,
#   but only if both relevant chunks are retrieved in the same top_k window
#
# Expected behavior: the model hedges ("based on the provided context...") or gives
# a one-sided answer because both documents rarely land in the same top_k=3 result.
#
# What would improve this:
# - Increase top_k so both documents have a chance to be retrieved
# - Use query decomposition: split into two sub-queries, retrieve separately,
#   then combine — LlamaIndex supports this via SubQuestionQueryEngine

# --- LlamaIndex Q4 ---

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

judge_llm = Anthropic(model="claude-haiku-4-5-20251001")
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

# Good query — answer clearly exists in the documents
q_good = "What employee benefits does BrightLeaf offer?"
response_good = query_engine.query(q_good)
faith_good = faithfulness_evaluator.evaluate_response(query=q_good, response=response_good)
relev_good = relevancy_evaluator.evaluate_response(query=q_good, response=response_good)
print(f"\nGood query: {q_good}")
print(f"  Faithfulness: {faith_good.score} | Relevancy: {relev_good.score}")

# Bad query — answer does NOT exist in the documents
q_bad = "What is BrightLeaf's policy on cryptocurrency payments?"
response_bad = query_engine.query(q_bad)
faith_bad = faithfulness_evaluator.evaluate_response(query=q_bad, response=response_bad)
relev_bad = relevancy_evaluator.evaluate_response(query=q_bad, response=response_bad)
print(f"\nBad query: {q_bad}")
print(f"  Faithfulness: {faith_bad.score} | Relevancy: {relev_bad.score}")

# Q4 observations:
#
# Faithfulness score of 1.0: every claim in the answer is supported by the retrieved chunks.
# Faithfulness score of 0.0: the model hallucinated — it said things not in the context.
#
# Relevancy measures whether the retrieved chunks actually match the question.
# It judges the retriever. Faithfulness judges the LLM's answer given those chunks.
# You can have high faithfulness but low relevancy (model stuck to irrelevant chunks)
# or high relevancy but low faithfulness (right chunks retrieved, model still hallucinated).
#
# Scores between queries (actual results):
# - Good query: faithfulness=1.0, relevancy=1.0 — expected. Document exists,
#   retrieval was correct, model answered entirely from the context.
# - Bad query (cryptocurrency): faithfulness=1.0, relevancy=1.0 — surprising.
#   The judge LLM scored it as faithful because the model correctly said
#   "I cannot find this in the documents" — that statement IS faithful to the context.
#   Relevancy=1.0 may reflect that the retrieved chunks were topically adjacent
#   (security/finance docs) and the judge considered them relevant enough.
#   This shows a limitation: evaluator scores don't always catch retrieval failure
#   when the LLM gracefully declines to answer.
#
# LLM-as-a-judge: instead of comparing to a fixed correct answer (which RAG doesn't have),
# a second LLM reads the query + context + response and answers YES/NO on faithfulness/relevancy.
# This works because evaluation is a reasoning task — it needs language understanding,
# not just string matching.
