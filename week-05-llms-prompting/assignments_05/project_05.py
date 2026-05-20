from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

# --- Task 1: Setup and System Prompt ---

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


# Deliberately named the role "career transition coach" rather than "assistant" —
# specificity here narrows the model's behavior to job application topics only
# and makes the constraints (review before submitting, industry humility) feel natural.
SYSTEM_PROMPT = """
You are a career transition coach specializing in job application materials.
You help people who are switching fields rewrite resume bullet points, draft cover letter openings,
and answer questions about how to present their experience to a new industry.

Rules you must always follow:
1. Stay focused on job application materials. If the user asks about anything unrelated, politely redirect.
2. Always remind the user to review and edit your output before submitting it to any employer.
3. Acknowledge that you may not know the norms of every specific industry, and encourage the user to use their own judgment.
4. Never invent credentials, achievements, or facts the user has not provided.
"""

# --- Task 2: Bullet Point Rewriter ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list with no other text. Each item must have exactly two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    raw = get_completion(messages, temperature=0.3)

    try:
        results = json.loads(raw)
    except json.JSONDecodeError:
        print("Bullet rewriter: failed to parse JSON. Raw response:\n", raw)
        return []

    for item in results:
        print(f"  Original : {item['original']}")
        print(f"  Improved : {item['improved']}")
        print()

    return results


# Test
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

print("Task 2 — Bullet Rewriter:")
rewrite_bullets(bullets)
# These bullets are weak because they have no action verb, no specific result, and no measurable impact.
# The model should add strong verbs, clarify what was actually done, and imply an outcome.

# --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
    Never use phrases like "I am excited to leverage my unique skills."

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages, temperature=0.7)


# Test
job_title = "Junior Data Engineer"
background = (
    "Five years of experience as a middle school math teacher; recently completed "
    "a Python course and built data pipelines using Prefect and Pandas."
)

print("Task 3 — Cover Letter Generator:")
cover_letter = generate_cover_letter(job_title, background)
print(cover_letter)
# The two examples were chosen to show career changers framing their old experience as an asset,
# not a liability. Both use a specific detail from the background to open with confidence.
# Few-shot prompting controls tone here — without examples the model defaults to generic openers.

# --- Task 4: Moderation Check ---

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    if flagged:
        print("Job Application Helper: I can't respond to that. Please rephrase your message and keep it focused on your job application.")
    return not flagged


# Test
print("Task 4 — Moderation Check:")
print("Safe input:", is_safe("Can you help me rewrite my resume bullet points?"))
print("Flagged input:", is_safe("I want to hurt my coworker who rejected my application."))

# --- Task 5: Chatbot Loop ---

def run_chatbot():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            print("\nJob Application Helper: Here are your rewritten bullets:\n")
            rewrite_bullets(raw_bullets)
            print("Job Application Helper: Remember to review and edit these before submitting.\n")

        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            print("\nJob Application Helper: Here's a draft opening paragraph:\n")
            result = generate_cover_letter(job_title, background)
            print(result)
            print("\nJob Application Helper: Please review and personalize this before submitting.\n")

        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}\n")
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()

# --- Task 6: Ethics Reflection ---
# Format: Option A — Comment block

# Question 1 — Bias in training data:
# This model was trained on text that likely overrepresents English-speaking, Western professional
# norms — corporate job titles, formal writing styles, and career paths common in North America or Europe.
# A career changer from a different cultural background might get advice that feels foreign or
# misrepresents how they naturally communicate. For example, bullet point styles that emphasize
# individual achievement ("I single-handedly...") may not reflect collaborative cultures where
# team credit is the norm. The bot has no way to know this, and neither will the user unless
# they already have that industry context to push back on it.

# Question 2 — Risk of submitting output directly:
# If a job-seeker submits the bot's cover letter without editing, the employer receives text that
# doesn't sound like that person — and that mismatch becomes obvious in an interview.
# Beyond voice, the model might subtly misrepresent the candidate's experience by using confident
# language around skills the user only partly has. A human reviewer would catch this; the model
# won't flag it. Disclosing AI assistance is also becoming an expectation in many hiring processes,
# and submitting unreviewed AI output without disclosure raises an integrity concern.
# The safest practice: treat the bot's output as a first draft, not a finished product.
