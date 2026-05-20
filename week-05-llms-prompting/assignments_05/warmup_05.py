from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

# --- Completions API ---

# API Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("API Q1 — Response:", response.choices[0].message.content)
print("API Q1 — Model used:", response.model)
print("API Q1 — Total tokens used:", response.usage.total_tokens)

# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"API Q2 — Temperature {temp}: {resp.choices[0].message.content}")

# At temperature 0 the output is deterministic — same name every run.
# At 1.5 the output is highly varied and sometimes unexpected.
# Use temperature 0 when you need consistent, reproducible output (e.g. structured extraction in a pipeline).

# API Q3
response_n3 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response_n3.choices):
    print(f"API Q3 — Completion {i + 1}: {choice.message.content}")

# API Q4
response_capped = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)

print("API Q4 — Capped response:", response_capped.choices[0].message.content)
# The response cuts off mid-sentence because the token budget ran out.
# Use max_tokens to control cost, but also to enforce output shape —
# e.g. a one-word classifier where a long response would break your parser.

# --- System Messages and Personas ---

# System Q1
messages_tutor = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response_tutor = client.chat.completions.create(model="gpt-4o-mini", messages=messages_tutor)
print("System Q1 — Tutor persona:", response_tutor.choices[0].message.content)

messages_skeptic = [
    {"role": "system", "content": "You are a brutally honest senior engineer who thinks most programmers rely on abstractions they don't understand. You answer technically and without sugarcoating."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response_skeptic = client.chat.completions.create(model="gpt-4o-mini", messages=messages_skeptic)
print("System Q1 — Skeptic persona:", response_skeptic.choices[0].message.content)
# The tone, vocabulary, and structure changed completely — same question, different system message.
# System messages are a privileged channel separate from user input,
# which prevents users from accidentally overriding the app's instructions.

# System Q2
messages_history = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response_history = client.chat.completions.create(model="gpt-4o-mini", messages=messages_history)
print("System Q2 — Stateless memory:", response_history.choices[0].message.content)
# The model knows Jordan's name because it's in the messages list we passed — not because it remembered it.
# The API is stateless: no memory between calls. To build a chatbot you must store and
# pass the full conversation history yourself on every request.

# --- Prompt Engineering ---

# Prompt Q1 — Zero-Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Classify the sentiment of this review as positive, negative, or mixed.\n\nReview: {review}"}
        ]
    )
    print(f"Prompt Q1 — Review {i + 1} sentiment: {resp.choices[0].message.content}")

# Prompt Q2 — One-Shot
one_shot_example = 'Example:\nReview: "Fast shipping but the item arrived damaged."\nSentiment: mixed\n\n'

for i, review in enumerate(reviews):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Classify the sentiment of this review as positive, negative, or mixed.\n\n{one_shot_example}Review: {review}\nSentiment:"}
        ]
    )
    print(f"Prompt Q2 — Review {i + 1} sentiment: {resp.choices[0].message.content}")

# Adding one example locked the output format (just the label, no extra explanation).
# Format consistency improved over Q1; accuracy still depends on the model's training.

# Prompt Q3 — Few-Shot
few_shot_examples = (
    'Review: "The team was incredibly supportive throughout the entire process."\nSentiment: positive\n\n'
    'Review: "Product broke after one day and customer service was useless."\nSentiment: negative\n\n'
    'Review: "Fast shipping but the item arrived damaged."\nSentiment: mixed\n\n'
)

for i, review in enumerate(reviews):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Classify the sentiment of this review as positive, negative, or mixed.\n\n{few_shot_examples}Review: {review}\nSentiment:"}
        ]
    )
    print(f"Prompt Q3 — Review {i + 1} sentiment: {resp.choices[0].message.content}")

# Zero-shot: no format, model guesses structure — inconsistent output.
# One-shot: format is locked but model only sees one pattern.
# Few-shot: format is locked AND model sees the full range of labels — best for ambiguous tasks.
# Use zero-shot for simple well-known tasks, one-shot when format matters,
# few-shot when the task has nuance or multiple categories.

# Prompt Q4 — Chain of Thought
cot_prompt = """
A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?

Think through this step by step, then clearly label your final answer.
"""

response_cot = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": cot_prompt}]
)
print("Prompt Q4 — Chain of Thought:\n", response_cot.choices[0].message.content)
# Asking the model to reason step by step forces it to write each intermediate result as a token.
# The next step is predicted from that written context, not from scratch —
# effectively giving the model working memory it wouldn't otherwise have.

# Prompt Q5 — Structured Output
review = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)

structured_prompt = f"""Analyze the review below and return ONLY valid JSON with exactly these keys:
- sentiment: "positive", "negative", or "mixed"
- confidence: a float from 0 to 1
- reason: one sentence explaining the sentiment

Review: {review}"""

response_json = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": structured_prompt}]
)

raw = response_json.choices[0].message.content
print("Prompt Q5 — Raw response:", raw)

try:
    parsed = json.loads(raw)
    print("Prompt Q5 — Sentiment:", parsed["sentiment"])
    print("Prompt Q5 — Confidence:", parsed["confidence"])
    print("Prompt Q5 — Reason:", parsed["reason"])
except json.JSONDecodeError:
    print("Prompt Q5 — Failed to parse JSON. Raw response:", raw)

# Prompt Q6 — Delimiters
user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
)

prompt_steps = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response_steps = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt_steps}]
)
print("Prompt Q6 — Steps detected:\n", response_steps.choices[0].message.content)

prose_text = "The Sahara Desert spans several countries across northern Africa and is one of the hottest places on Earth."

prompt_no_steps = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{prose_text}```
"""

response_no_steps = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt_no_steps}]
)
print("Prompt Q6 — No steps:", response_no_steps.choices[0].message.content)
# Delimiters create a clear boundary between your instructions and the user's content.
# Without them, user text that looks like an instruction (prompt injection) could
# override your prompt and make the model do something unintended.

# --- Local Models with Ollama ---

# Ollama Q1
# Ollama was not available in this environment so the local run could not be executed.
# Expected terminal command:
#   ollama run qwen3:0.6b "Explain what a large language model is in two sentences."
#
# Placeholder for Ollama output (would paste actual terminal output here):
# """
# A large language model is a neural network trained on massive amounts of text data
# to predict and generate human-like language. It learns statistical patterns across
# billions of tokens to answer questions, summarize, translate, and more.
# """

response_ollama_compare = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}]
)
print("Ollama Q1 — OpenAI response:", response_ollama_compare.choices[0].message.content)

# Differences: cloud model (GPT-4o-mini) tends to be more fluent and detailed;
# local model (qwen3:0.6b) is smaller so responses may be shorter or less precise.
# Advantage of local: data never leaves your machine — critical for sensitive/private data.
# Disadvantage of local: smaller models, slower on consumer hardware, lower quality output.
