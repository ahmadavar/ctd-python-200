from dotenv import load_dotenv
import os
from pathlib import Path

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# =============================================================================
# STEP 1: SETUP
# =============================================================================

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Anthropic(model="claude-haiku-4-5-20251001")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

DOCS_PATH = Path(os.path.expanduser(
    "~/ctd-school-repo/lessons/06_AI_augmentation/resources/groundwork_docs"
))

assert DOCS_PATH.exists(), f"Document directory not found: {DOCS_PATH}"
print(f"Documents directory confirmed: {DOCS_PATH}")


# =============================================================================
# STEP 2: LOAD THE DOCUMENTS
# =============================================================================

documents = SimpleDirectoryReader(str(DOCS_PATH)).load_data()
print(f"\nLoaded {len(documents)} documents:")
for doc in documents:
    print(f"  - {doc.metadata.get('file_name', 'unknown')}")


# =============================================================================
# STEP 3: BUILD THE INDEX AND QUERY ENGINE
# =============================================================================

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
print("\nIndex built successfully. Ready to answer questions.")


# =============================================================================
# STEP 4: QUERY THE ASSISTANT
# =============================================================================

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

print("\n" + "=" * 60)
print("STEP 4: QUERY RESULTS")
print("=" * 60)

for question in questions:
    print(f"\nQuestion: {question}")
    response = query_engine.query(question)
    print(f"Answer: {response}")
    top_node = response.source_nodes[0]
    print(f"Top source: [{top_node.metadata.get('file_name', 'unknown')}] "
          f"score={top_node.score:.4f}")
    print(f"  Chunk: {top_node.text[:200]!r}")

# Step 4 reflection:
# All five queries returned confident, specific, accurate answers — no hedging.
# The loyalty program answer was exact: "1 point per dollar, 100 points = free drink."
# The catering answer was the most impressive — it pulled from wholesale_catering.txt
# and gave a complete breakdown of packages, pricing process, and email contact.
# The "how did Groundwork start" query scored 0.8157 — the highest of any query —
# because our_story.txt is a tight, focused document. Nothing unexpected was retrieved.


# =============================================================================
# STEP 5: FIND A FAILURE
# =============================================================================

failure_query = "What is Groundwork's refund policy for online orders?"

print("\n" + "=" * 60)
print("STEP 5: FAILURE CASE")
print("=" * 60)
print(f"\nQuery: {failure_query}")
failure_response = query_engine.query(failure_query)
print(f"Answer: {failure_response}")
print("\nAll retrieved source nodes:")
for i, node in enumerate(failure_response.source_nodes, 1):
    print(f"  [{i}] {node.metadata.get('file_name', 'unknown')} "
          f"score={node.score:.4f}")
    print(f"       {node.text[:200]!r}")

# Step 5 analysis:
#
# What I asked and why it's hard:
# None of the five Groundwork documents mention refunds or online orders.
# The topic simply doesn't exist in the knowledge base.
#
# What went wrong:
# LlamaIndex retrieved wholesale_catering.txt, faq.txt, and menu.txt — none of
# which contain refund information. The retriever returned the top_k most similar
# chunks regardless of whether the answer exists.
#
# Did the model's tone change?
# Yes — and this is the best-case scenario. Claude said "I don't have information
# about Groundwork's refund policy" and redirected to hello@groundworkcoffee.com.
# The tone was honest rather than confident-but-wrong. However, this behavior
# is not guaranteed — other LLMs (or Claude with different prompting) might
# hallucinate a plausible-sounding refund policy instead of admitting ignorance.
#
# What would I change:
# Add a similarity score threshold — if the top chunk scores below ~0.75,
# return "I don't have information about that" instead of guessing.
# LlamaIndex supports this via a custom postprocessor or similarity_cutoff.


# =============================================================================
# STEP 6: REFLECTION
# =============================================================================

# Q1: How many lines did LlamaIndex take vs manual RAG?
#
# The full LlamaIndex pipeline — load, chunk, embed, index, query — took 3 lines:
#   SimpleDirectoryReader(...).load_data()
#   VectorStoreIndex.from_documents(documents)
#   index.as_query_engine(similarity_top_k=3)
# A manual semantic RAG (chunking + embedding calls + cosine similarity + retrieval)
# would take 100+ lines and require managing API calls, vector math, and storage by hand.
# Frameworks exist to encode best practices so engineers focus on the problem, not the plumbing.

# Q2: A different use case where this adds genuine value:
#
# A hospital system with hundreds of internal clinical guidelines, drug formularies,
# and discharge protocols. Doctors and nurses could ask natural-language questions
# ("what is the dosing protocol for vancomycin in renal failure?") and get answers
# grounded in the actual approved documents — not the model's general training.
# The RAG constraint (only answer from retrieved docs) is critical here: hallucinated
# medical advice is dangerous, but grounded retrieval from verified sources is safe.

# Q3: One failure mode RAG cannot fully prevent:
#
# Confident answers on out-of-scope questions. If the user asks something not covered
# by any document, the retriever still returns the top_k most similar chunks —
# and the LLM still generates an answer from that weak context.
# RAG prevents hallucination about *document content*, but it does not prevent
# the model from extrapolating or fabricating when the right information simply
# isn't there. A similarity cutoff helps but doesn't eliminate this risk entirely.
