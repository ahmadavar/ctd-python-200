# --- LLMs as Transform ---

# Q1
# For each task: LLM or deterministic code, and why.

# Task 1: Parse "Jan 5th, 2024" → "2024-01-05"
# Deterministic code — date formats follow fixed rules; dateutil parses any format
# exactly, with zero latency and zero cost.

# Task 2: Classify "my card was charged twice" → billing / technical / general
# LLM — requires understanding natural language intent, not a calculable rule.

# Task 3: Calculate the average of a list of numbers
# Deterministic code — pure arithmetic; sum(lst) / len(lst) is exact, instant, and free.

# Task 4: Extract company name from "Sr. Data Eng @ Acme Corp (contract)"
# LLM — freeform text with no fixed pattern; an LLM handles variations a regex would miss.

# Task 5: Determine whether a product review is more than 100 words long
# Deterministic code — len(text.split()) > 100 is exact, instant, and free.


# Q2
# Problem with: system = "Summarize this product review in a few sentences."
#
# In a pipeline, output is saved automatically to a file or database — no human reads it.
# An open-ended prompt returns unpredictable output: sometimes 2 sentences, sometimes 5,
# sometimes starting with "Sure!", sometimes "This review...". Code that parses or stores
# that output will break silently or produce inconsistent records downstream.
#
# Fixed prompt — forces a single predictable, parseable output every time:

FIXED_PROMPT = (
    "Summarize the following product review in exactly one sentence. "
    "Reply with that sentence only — no preamble, no punctuation beyond the sentence itself."
)


# Q3
# 1. Sequential processing time:
#    50,000 records × 1 second = 50,000 seconds ÷ 3,600 = ~13.9 hours
#
# 2. Strategy to handle this more efficiently without changing models:
#    Use parallelism / batching — send many records simultaneously instead of waiting
#    for each response before sending the next. OpenAI's Batch API supports this natively
#    and is 50% cheaper than synchronous calls for large-scale workloads.


# --- Azure OpenAI ---

# Q1 (Azure OpenAI Question 1)
# Two reasons an organization uses Azure OpenAI instead of the OpenAI API directly:
#
# 1. Data residency and compliance — requests stay within Azure infrastructure and never
#    reach OpenAI's servers. Regulated industries (healthcare, finance, legal) require
#    that sensitive data not leave their controlled environment.
#
# 2. Contractual data assurances — Microsoft enterprise agreements explicitly guarantee
#    that customer data will not be used to train future models. The standard OpenAI API
#    does not make this promise to individual users.


# Q2 (Azure OpenAI Question 2)
# Three Azure-specific parameters for AzureOpenAI client initialization
# (not including api_key):
#
# 1. azure_endpoint — the URL of your Azure OpenAI resource
#    e.g. "https://your-resource.openai.azure.com/"
#    Tells the client which Azure resource to route requests to.
#
# 2. api_version — the API version date string, e.g. "2024-02-01"
#    Azure requires an explicit version so behavior stays stable across updates.
#
# 3. azure_deployment (passed via the model parameter at call time) —
#    the deployment name your org created in Azure AI Foundry.
#    This replaces the model name used in standard OpenAI calls.

# NOTE ON CLIENT SETUP — INTENTIONAL DEVIATION FROM SPEC:
# The assignment asks for the OpenAI client (openai.OpenAI / openai.AzureOpenAI).
# I do not have an OpenAI API key; I am using the Anthropic key set up in Weeks 5–7.
# The Anthropic client (anthropic.Anthropic) provides identical functionality for this
# warmup — structured prompts, single-word responses, fallback handling.
# If an OpenAI key is available, replace with:
#   from openai import AzureOpenAI
#   client = AzureOpenAI(azure_endpoint=..., api_key=..., api_version=...)


# Q3 (Azure OpenAI Question 3)
# When using AzureOpenAI, the model parameter in chat.completions.create() does NOT
# take a value like "gpt-4o-mini". It takes a deployment name — a custom name chosen
# by your organization's infrastructure team when they deployed the model in Azure AI Foundry.
# You find the correct value in: Azure AI Foundry → Deployments section.
