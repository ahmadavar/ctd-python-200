import os
import glob
import pandas as pd
from scipy import stats
from dotenv import load_dotenv
from smolagents import tool, CodeAgent, LiteLLMModel

load_dotenv()

# Absolute paths derived from this file's location — script runs correctly from any directory
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
DATA_PATH   = os.path.join(REPO_ROOT, "week-01-analysis-pipelines", "outputs", "merged_happiness.csv")
YEARLY_GLOB = os.path.join(REPO_ROOT, "assignments", "resources", "happiness_project", "*.csv")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Task 1: Four @tool decorated functions
# ---------------------------------------------------------------------------

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory and return its structure.

    Loads from a pre-merged CSV if it exists; otherwise merges all yearly CSVs
    found in the assignments_01/resources/happiness_project/ directory.

    Returns:
        A dict with keys 'shape' (tuple), 'columns' (list of str), and
        'years' (list of int) describing the loaded dataset.
    """
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        yearly_files = sorted(glob.glob(YEARLY_GLOB))
        if not yearly_files:
            return {"error": f"No data found at {DATA_PATH} or {YEARLY_GLOB}"}
        df = pd.concat([pd.read_csv(f) for f in yearly_files], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
    years = sorted(df["year"].dropna().unique().astype(int).tolist())
    return {"shape": list(df.shape), "columns": df.columns.tolist(), "years": years}


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a numeric column in the happiness dataset.

    Args:
        column: The name of the numeric column to summarize (e.g. 'happiness_score').

    Returns:
        A dict containing the describe() statistics for the column, or an error key
        if the column is not found or no data is loaded.
    """
    if not os.path.exists(DATA_PATH):
        return {"error": "No data loaded. Call load_happiness_data first."}
    df = pd.read_csv(DATA_PATH)
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {df.columns.tolist()}"}
    return df[column].describe().round(4).to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: First numeric column name.
        col2: Second numeric column name.

    Returns:
        A dict with keys 'col1', 'col2', 'pearson_r', and 'p_value' (floats rounded to
        4 decimal places), or an error key if a column is missing or data is not loaded.
    """
    if not os.path.exists(DATA_PATH):
        return {"error": "No data loaded. Call load_happiness_data first."}
    df = pd.read_csv(DATA_PATH)
    if col1 not in df.columns:
        return {"error": f"Column '{col1}' not found."}
    if col2 not in df.columns:
        return {"error": f"Column '{col2}' not found."}
    df = df.dropna(subset=[col1, col2])
    r, p = stats.pearsonr(df[col1], df[col2])
    return {"col1": col1, "col2": col2, "pearson_r": round(r, 4), "p_value": round(p, 4)}


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The numeric column to rank by (e.g. 'happiness_score').
        year: The year to filter on (e.g. 2019).
        n: Number of top countries to return (default 5).

    Returns:
        A dict with key 'results' containing a list of dicts, each with 'country'
        and the requested column value. Returns an error key on bad input.
    """
    if not os.path.exists(DATA_PATH):
        return {"error": "No data loaded. Call load_happiness_data first."}
    df = pd.read_csv(DATA_PATH)
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    filtered = df[df["year"] == year].dropna(subset=[column])
    if filtered.empty:
        return {"error": f"No data found for year {year}."}
    top = filtered.nlargest(n, column)[["country", column]].reset_index(drop=True)
    return {"results": top.to_dict(orient="records")}


# ---------------------------------------------------------------------------
# Task 2: Build the CodeAgent
#
# NOTE ON MODEL CHOICE — INTENTIONAL DEVIATION FROM SPEC:
# The assignment asks for: OpenAIServerModel(model_id="gpt-4o-mini")
# I do not have an OpenAI API key; I am using the Anthropic key set up in Week 5.
# Substitution used: LiteLLMModel(model_id="anthropic/claude-haiku-4-5-20251001")
# LiteLLMModel implements the same smolagents ModelBase interface as OpenAIServerModel,
# so tool calls, CodeAgent execution, and all agent behavior are identical in practice.
# If an OpenAI key is available, replace the block below with:
#   from smolagents import OpenAIServerModel
#   model = OpenAIServerModel(api_key=os.getenv("OPENAI_API_KEY"), model_id="gpt-4o-mini")
# ---------------------------------------------------------------------------

model = LiteLLMModel(
    model_id="anthropic/claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)

# ---------------------------------------------------------------------------
# Task 3: Five guided queries (reset=False retains context across turns)
# ---------------------------------------------------------------------------

queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    (
        "Plot happiness_score over the years as a line chart, with one line per region. "
        f"Save the plot to {os.path.join(OUTPUTS_DIR, 'happiness_by_region.png')}."
    ),
]

# ---------------------------------------------------------------------------
# Task 4: Custom queries
# ---------------------------------------------------------------------------

# Custom Query 1: year-over-year delta — requires the agent to write merge + subtract
# code on the fly. No tool covers cross-year comparisons, so this is pure code generation.
my_query_1 = (
    "Using the happiness dataset, find which countries improved their happiness score "
    "the most between 2015 and 2023. Load the data, filter for both years, merge on country, "
    "compute the delta (2023 score minus 2015 score), and return the top 5 most improved "
    "countries with their delta values."
)

# Custom Query 2: scatter plot with regional color grouping — requires matplotlib code
# with per-group coloring and a legend. Triggers code generation (no plotting tool exists).
my_query_2 = (
    "Create a scatter plot of happiness_score vs gdp_per_capita for the year 2023, "
    "with each point colored by regional_indicator. Add a title, axis labels, and a legend. "
    f"Save the plot to {os.path.join(OUTPUTS_DIR, 'happiness_gdp_scatter.png')}."
)


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("TASK 3: Guided Queries")
    print("#"*60)
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        result = agent.run(query, reset=False)
        print(f"\nAnswer: {result}")

    print("\n" + "#"*60)
    print("TASK 4: Custom Queries")
    print("#"*60)

    print(f"\n{'='*60}")
    print(f"Custom Query 1: {my_query_1}")
    print('='*60)
    result_1 = agent.run(my_query_1, reset=False)
    print(f"\nAnswer: {result_1}")
    # Comment: triggered code generation — agent wrote pandas merge + delta logic on the fly.
    # No tool covers cross-year subtraction, so this was 100% generated code, not tool calls.

    print(f"\n{'='*60}")
    print(f"Custom Query 2: {my_query_2}")
    print('='*60)
    result_2 = agent.run(my_query_2, reset=False)
    print(f"\nAnswer: {result_2}")
    # Comment: triggered code generation — agent wrote matplotlib scatter with per-region
    # color grouping and a legend. No plotting tool exists, so this was pure code generation.


# ---------------------------------------------------------------------------
# Task 5: Reflection
# ---------------------------------------------------------------------------

# Q1: In Query 3, how did the agent communicate whether the correlation was statistically
#     significant? Did it use the p-value correctly? What threshold did it apply?
#
# The agent returned r = 0.6218 with p-value = 1.6234e-146 — essentially zero.
# It correctly flagged this as statistically significant using the standard p < 0.05
# threshold. A p-value that small means there is virtually zero chance the correlation
# is due to random chance across 1,362 observations. The agent communicated this clearly
# and applied the threshold correctly.

# Q2: Did any of the agent's responses surprise you — either by being more capable
#     than you expected, or less? Describe one specific example.
#
# Custom Query 1 was the standout. I expected the agent to struggle with a multi-step
# pandas operation it had never been shown: filter two years, merge on country, subtract
# scores, sort by delta. It nailed it in one pass — returning Slovakia (+0.48) as the
# top improver with correct delta values for all 5 countries. The capability ceiling
# was higher than expected for open-ended data wrangling.

# Q3: What one additional tool would make this agent meaningfully more useful?
#     Describe what it would do and what kind of question it would help the agent answer.
#
# A get_regional_average(column, year) tool that groups the dataset by regional_indicator
# and returns the mean of a given column per region for a specific year. This would let
# the agent answer "which region is happiest on average in 2022?" without writing groupby
# code each time — making regional comparisons faster, cleaner, and easier to audit.
