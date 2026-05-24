# Week 7 Warmup — AI Agents
# Run: python warmup_07.py

import os
from datetime import datetime
from dotenv import load_dotenv
import anthropic
import pandas as pd
from scipy import stats

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ---------------------------------------------------------------------------
# Q1: celsius_to_fahrenheit — function + JSON schema + direct calls
# ---------------------------------------------------------------------------

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

celsius_to_fahrenheit_schema = {
    "name": "celsius_to_fahrenheit",
    "description": "Converts a temperature from Celsius to Fahrenheit.",
    "input_schema": {
        "type": "object",
        "properties": {
            "celsius": {
                "type": "number",
                "description": "Temperature in Celsius to convert."
            }
        },
        "required": ["celsius"]
    }
}

# Direct calls — no agent yet
for temp in [0, 100, -40]:
    result = celsius_to_fahrenheit(temp)
    print(f"{temp}°C = {result}°F")

# ---------------------------------------------------------------------------
# Q2: run_agent with get_current_time only — test with a Celsius query
# ---------------------------------------------------------------------------

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

get_current_time_schema = {
    "name": "get_current_time",
    "description": "Returns the current date and time.",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

def run_agent(user_message: str, tools: list, tool_functions: dict, max_rounds: int = 10) -> str:
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_rounds):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        if response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            break

        # Collect all tool calls in this round and run them
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = tool_functions[block.name](**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return response.content[0].text, messages

print("\n--- Q2: agent with get_current_time only ---")

# Prediction:
# Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call?
# NO — the only available tool is get_current_time, which has nothing to do with temperature.
# The model has no celsius_to_fahrenheit tool to call, so it will answer from its own training
# knowledge (100°C = 212°F is common knowledge) without making any tool call.
# How many API calls will be made? Just ONE — the model responds directly with text.

reply, _ = run_agent(
    "Convert 100 degrees Celsius to Fahrenheit",
    tools=[get_current_time_schema],
    tool_functions={"get_current_time": get_current_time}
)
print(reply)
# Prediction confirmed: no tool call was triggered. The model answered from training data.

# ---------------------------------------------------------------------------
# Q3: Extend agent with both tools — routing kicks in
# ---------------------------------------------------------------------------

all_tools = [get_current_time_schema, celsius_to_fahrenheit_schema]
all_functions = {
    "get_current_time": get_current_time,
    "celsius_to_fahrenheit": celsius_to_fahrenheit
}

print("\n--- Q3: response_a — temperature query (tool WILL be called) ---")
response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?", tools=all_tools, tool_functions=all_functions)
print("Response A:", response_a[0])
# celsius_to_fahrenheit WAS called here — the query is a direct conversion request
# and the model now has a matching tool, so it invokes it.

print("\n--- Q3: response_b — factual query (no tool needed) ---")
response_b = run_agent("What is the boiling point of water in plain English?", tools=all_tools, tool_functions=all_functions)
print("Response B:", response_b[0])
# No tool was called here — the model answered from training knowledge.
# "Boiling point of water" is general scientific knowledge; no conversion tool is required
# because the user asked for a plain English explanation, not a numeric conversion.

# ---------------------------------------------------------------------------
# Q4: CsvManager agent — load_csv, summarize_column, compute_correlation
# ---------------------------------------------------------------------------

CSV_PATH = "outputs/bike_commute.csv"
csv_df = None  # shared state across tool calls

def load_csv(filepath: str) -> str:
    global csv_df
    csv_df = pd.read_csv(filepath)
    return f"Loaded {len(csv_df)} rows, columns: {list(csv_df.columns)}"

def summarize_column(column: str) -> str:
    if csv_df is None:
        return "No CSV loaded. Call load_csv first."
    if column not in csv_df.columns:
        return f"Column '{column}' not found. Available: {list(csv_df.columns)}"
    s = csv_df[column].describe()
    return f"{column} — mean: {s['mean']:.2f}, std: {s['std']:.2f}, min: {s['min']:.2f}, max: {s['max']:.2f}"

def compute_correlation(col1: str, col2: str) -> str:
    if csv_df is None:
        return "No CSV loaded. Call load_csv first."
    r, p = stats.pearsonr(csv_df[col1], csv_df[col2])
    return f"Pearson r({col1}, {col2}) = {r:.4f}, p-value = {p:.4f}"

load_csv_schema = {
    "name": "load_csv",
    "description": "Loads a CSV file into memory for analysis.",
    "input_schema": {
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to the CSV file."}
        },
        "required": ["filepath"]
    }
}

summarize_column_schema = {
    "name": "summarize_column",
    "description": "Returns descriptive statistics for a single column in the loaded CSV.",
    "input_schema": {
        "type": "object",
        "properties": {
            "column": {"type": "string", "description": "Column name to summarize."}
        },
        "required": ["column"]
    }
}

compute_correlation_schema = {
    "name": "compute_correlation",
    "description": "Computes the Pearson correlation coefficient between two columns in the loaded CSV.",
    "input_schema": {
        "type": "object",
        "properties": {
            "col1": {"type": "string", "description": "First column name."},
            "col2": {"type": "string", "description": "Second column name."}
        },
        "required": ["col1", "col2"]
    }
}

csv_tools = [load_csv_schema, summarize_column_schema, compute_correlation_schema]
csv_functions = {
    "load_csv": load_csv,
    "summarize_column": summarize_column,
    "compute_correlation": compute_correlation
}

print("\n--- Q4: CsvManager — correlation query ---")
reply, _ = run_agent(
    f"Load the CSV at '{CSV_PATH}', then compute the correlation between avg_speed_kmh and avg_heart_rate.",
    tools=csv_tools,
    tool_functions=csv_functions
)
print(reply)

# ---------------------------------------------------------------------------
# Q5: The scenario that previously hit the tool-round limit — now succeeds
# ---------------------------------------------------------------------------

csv_df = None  # reset shared state

print("\n--- Q5: multi-step scenario (previously failed, now succeeds) ---")
reply, _ = run_agent(
    f"Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.",
    tools=csv_tools,
    tool_functions=csv_functions
)
print(reply)

# ---------------------------------------------------------------------------
# Q6: Print full messages list — inspect the ReAct loop
# ---------------------------------------------------------------------------

csv_df = None  # reset shared state

print("\n--- Q6: full messages list ---")
reply, messages = run_agent(
    f"Load the CSV at '{CSV_PATH}', then compute the correlation between duration_min and avg_heart_rate.",
    tools=csv_tools,
    tool_functions=csv_functions
)

for i, msg in enumerate(messages):
    role = msg["role"]
    content = msg["content"]
    if isinstance(content, str):
        # plain text message
        print(f"\n[{i}] role={role}: {content[:120]}")
    elif isinstance(content, list):
        for block in content:
            if hasattr(block, "type"):
                # assistant block (SDK object)
                if block.type == "tool_use":
                    print(f"\n[{i}] role={role} | tool_use: {block.name}({block.input})")
                elif block.type == "text":
                    print(f"\n[{i}] role={role} | text: {block.text[:120]}")
            elif isinstance(block, dict):
                # user tool_result dict
                print(f"\n[{i}] role={role} | tool_result: {str(block.get('content',''))[:120]}")

# Comments explaining each role in the ReAct loop:
# role=user (first message)  — the human's original question
# role=assistant (tool_use)  — LLM reasoning: decides which tool to call and with what args
# role=user (tool_result)    — the observation: your code ran the tool, result sent back as user turn
# role=assistant (text)      — final answer: LLM has all info it needs, writes the response

# ---------------------------------------------------------------------------
# Q7: Re-wrap compute_correlation as smolagents @tool — auto schema generation
# ---------------------------------------------------------------------------

from smolagents import tool as smolagents_tool

@smolagents_tool
def compute_correlation_smol(col1: str, col2: str) -> str:
    """Computes the Pearson correlation coefficient between two columns in the loaded CSV.

    Args:
        col1: First column name.
        col2: Second column name.
    """
    r, p = stats.pearsonr(csv_df[col1], csv_df[col2])
    return f"Pearson r({col1}, {col2}) = {r:.4f}, p-value = {p:.4f}"

print("\n--- Q7: smolagents @tool auto-generated schema ---")
import json
smol_schema = {
    "name": compute_correlation_smol.name,
    "description": compute_correlation_smol.description,
    "inputs": compute_correlation_smol.inputs,
    "output_type": compute_correlation_smol.output_type,
}
print(json.dumps(smol_schema, indent=2, default=str))

print("\n--- Q7: manual schema for comparison ---")
print(json.dumps(compute_correlation_schema, indent=2))

# Comparison:
# @tool reads type hints (str) and the Args: docstring to build the schema automatically
# Manual schema requires you to write the dict by hand — more control, more boilerplate
# Both produce equivalent JSON that the LLM reads to understand what the tool does

# ---------------------------------------------------------------------------
# Q8: ToolCallingAgent vs CodeAgent — scatter plot test
# ---------------------------------------------------------------------------

from smolagents import ToolCallingAgent, CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="anthropic/claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Load data so tools can access it
csv_df = pd.read_csv(CSV_PATH)

scatter_prompt = (
    "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot "
    "with green dots. Save the result to outputs/scatter_heartrate_duration.png"
)

print("\n--- Q8: ToolCallingAgent (no plotting tool) ---")
tool_agent = ToolCallingAgent(tools=[compute_correlation_smol], model=model)
try:
    result = tool_agent.run(scatter_prompt)
    print(result)
except Exception as e:
    print(f"ToolCallingAgent failed: {e}")

print("\n--- Q8: CodeAgent (writes matplotlib code) ---")
code_agent = CodeAgent(
    tools=[compute_correlation_smol],
    model=model,
    additional_authorized_imports=["matplotlib", "matplotlib.pyplot", "pandas"]
)
try:
    result = code_agent.run(scatter_prompt)
    print(result)
except Exception as e:
    print(f"CodeAgent failed: {e}")

# Q8 comparison:
# ToolCallingAgent: has no plotting tool, so it CANNOT produce a chart at all.
# It also cannot change the dot color to green — it can only call the tools it was given.
# CodeAgent: writes matplotlib code on the fly, sets color="green" explicitly, saves the PNG.
# Key insight: ToolCallingAgent is constrained to a pre-approved menu of actions.
# CodeAgent can solve open-ended tasks by generating arbitrary code — including visual styling
# that no tool covers. Use ToolCallingAgent when you need control; CodeAgent when you need flexibility.

# ---------------------------------------------------------------------------
# Q9: When to prefer ToolCallingAgent — and the one big risk of CodeAgent
# ---------------------------------------------------------------------------

# USE ToolCallingAgent WHEN:
# 1. Controlled environment — you know every action in advance and pre-approve it
#    (e.g. customer support bot that may only call lookup_order() or send_email())
# 2. Auditability matters — every action is a named tool call you can log and trace
# 3. Sandboxing is hard — restricting a list of tools is far simpler than restricting
#    arbitrary Python execution

# USE CodeAgent WHEN:
# - The task is open-ended and can't be solved by a fixed menu of tools
# - You need the agent to solve novel problems (e.g. generate a plot, run a calculation)
# - You can sandbox the execution environment (Docker, isolated subprocess)

# THE ONE BIG RISK OF CodeAgent:
# Arbitrary code execution — the agent writes and runs real Python on your machine.
# If the LLM hallucinates or receives an adversarial/injected prompt, it can:
#   - Read secrets from environment variables
#   - Delete files
#   - Make unauthorized network calls
# Production mitigation: run CodeAgent inside a Docker container or restricted
# subprocess with no access to the host filesystem, secrets, or network.
