import os
import json
import anthropic
from dotenv import load_dotenv
from src.storage import query

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"
MAX_TOOL_ROUNDS = 5  # guard against infinite loops

SYSTEM_PROMPT = """You are a fair lending compliance analyst with deep expertise in HMDA \
(Home Mortgage Disclosure Act) data analysis and regulatory examination practices.

You have access to 2024 HMDA data for home purchase loans across five major U.S. metropolitan \
areas: Chicago, Houston, Atlanta, Los Angeles, and New York. The data covers applications from \
all HMDA filers active in these markets.

DATABASE TABLES AVAILABLE VIA YOUR TOOLS:
- denial_rates: denial rate by lender (LEI) × race × MSA
- disparity_flags: denial rates + peer benchmarks + disparity ratios + Red/Yellow flags
- denial_reasons: denial reason breakdown by lender × race × MSA
- peer_benchmarks: average denial rate across all lenders by race × MSA

REGULATORY CONTEXT — critical for interpreting results:
- Disparity ratio = lender denial rate / peer average denial rate for that demographic × geography
- Ratio ≥ 1.5x (Yellow): warrants internal review under OCC and CFPB fair lending exam guidance
- Ratio ≥ 2.0x (Red): threshold that typically triggers examiner scrutiny and enforcement referral
- Home purchase lending carries the highest regulatory weight — it directly affects access to \
wealth-building through homeownership
- Denial rate denominator: denied / (originated + approved-not-accepted + denied)

BEHAVIOR RULES:
1. Always call a tool to query real data before answering — never rely on assumptions
2. Every response must end with a RECOMMENDED ACTION — what should the compliance officer do \
next, and by when
3. Be specific: cite actual disparity ratios, LEI codes, MSA names, and application counts
4. Frame findings in terms of regulatory risk, not just statistics
5. Lenders are identified by their LEI — if a user provides a bank name, query for available \
lenders and identify the matching LEI
6. If a query returns no results, say so clearly and suggest alternate filters

Tone: direct, professional, action-oriented. You are an operator compliance tool, not a chatbot."""


# ── Tool definitions (sent to Claude as the tools list) ────────────────────────

TOOLS = [
    {
        "name": "get_denial_rates",
        "description": (
            "Returns denial rates by lender, race, and/or MSA. Use this to look up raw denial "
            "rate figures for a specific lender or demographic segment. Returns up to 50 rows "
            "ordered by denial rate descending."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lei": {
                    "type": "string",
                    "description": "Legal Entity Identifier of the lender (optional). "
                                   "Leave blank to see all lenders.",
                },
                "msa_name": {
                    "type": "string",
                    "description": "Partial MSA name to filter by, e.g. 'Chicago' or 'Houston' (optional).",
                },
                "race": {
                    "type": "string",
                    "description": "Partial race/ethnicity string to filter by, e.g. "
                                   "'Black' or 'Hispanic' (optional).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "compare_to_peers",
        "description": (
            "Benchmarks a specific lender's denial rates against the peer average for each "
            "race × MSA segment. Returns disparity ratios and Red/Yellow flags. Use this when "
            "an operator wants to assess one lender's relative exposure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lei": {
                    "type": "string",
                    "description": "Legal Entity Identifier of the lender to benchmark.",
                },
                "msa_name": {
                    "type": "string",
                    "description": "Partial MSA name to narrow results (optional).",
                },
            },
            "required": ["lei"],
        },
    },
    {
        "name": "flag_disparities",
        "description": (
            "Scans all lenders and surfaces segments where the disparity ratio exceeds a given "
            "threshold. Use this to identify the highest-risk lender × demographic × MSA "
            "combinations across the full dataset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Minimum disparity ratio to flag. Default 2.0 (Red). "
                                   "Use 1.5 to include Yellow flags as well.",
                },
                "msa_name": {
                    "type": "string",
                    "description": "Partial MSA name to narrow results (optional).",
                },
                "race": {
                    "type": "string",
                    "description": "Partial race string to filter by (optional).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_denial_reasons",
        "description": (
            "Returns the breakdown of denial reasons for a given lender × race × MSA. "
            "Use this after identifying a disparity to understand *why* applications are being "
            "denied — critical for root cause analysis and remediation recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lei": {
                    "type": "string",
                    "description": "Legal Entity Identifier of the lender (optional).",
                },
                "msa_name": {
                    "type": "string",
                    "description": "Partial MSA name to filter by (optional).",
                },
                "race": {
                    "type": "string",
                    "description": "Partial race string to filter by (optional).",
                },
            },
            "required": [],
        },
    },
]


# ── Tool execution — each function queries DuckDB and returns a string ──────────

def _safe_query(sql: str) -> str:
    """Run SQL and return a clean string. Returns a message if no rows found."""
    df = query(sql)
    if df.empty:
        return "No results found for these filters."
    return df.to_string(index=False)


def _get_denial_rates(lei=None, msa_name=None, race=None) -> str:
    conditions = []
    if lei:
        conditions.append(f"UPPER(lei) = '{lei.upper().strip()}'")
    if msa_name:
        conditions.append(f"msa_name ILIKE '%{msa_name}%'")
    if race:
        conditions.append(f"derived_race ILIKE '%{race}%'")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT lei, msa_name, derived_race,
               total_applications,
               total_denied,
               ROUND(denial_rate * 100, 1) AS denial_rate_pct
        FROM denial_rates
        {where}
        ORDER BY denial_rate DESC
        LIMIT 50
    """
    return _safe_query(sql)


def _compare_to_peers(lei: str, msa_name=None) -> str:
    conditions = [f"UPPER(lei) = '{lei.upper().strip()}'"]
    if msa_name:
        conditions.append(f"msa_name ILIKE '%{msa_name}%'")

    where = "WHERE " + " AND ".join(conditions)

    sql = f"""
        SELECT lei, msa_name, derived_race,
               ROUND(denial_rate * 100, 1)          AS lender_denial_rate_pct,
               ROUND(peer_avg_denial_rate * 100, 1)  AS peer_avg_pct,
               ROUND(disparity_ratio, 2)             AS disparity_ratio,
               disparity_flag,
               total_applications
        FROM disparity_flags
        {where}
        ORDER BY disparity_ratio DESC
        LIMIT 50
    """
    return _safe_query(sql)


def _flag_disparities(threshold=2.0, msa_name=None, race=None) -> str:
    conditions = [f"disparity_ratio >= {float(threshold)}"]
    if msa_name:
        conditions.append(f"msa_name ILIKE '%{msa_name}%'")
    if race:
        conditions.append(f"derived_race ILIKE '%{race}%'")

    where = "WHERE " + " AND ".join(conditions)

    sql = f"""
        SELECT lei, msa_name, derived_race,
               ROUND(denial_rate * 100, 1)          AS lender_denial_rate_pct,
               ROUND(peer_avg_denial_rate * 100, 1)  AS peer_avg_pct,
               ROUND(disparity_ratio, 2)             AS disparity_ratio,
               disparity_flag,
               total_applications
        FROM disparity_flags
        {where}
        ORDER BY disparity_ratio DESC
        LIMIT 50
    """
    return _safe_query(sql)


def _get_denial_reasons(lei=None, msa_name=None, race=None) -> str:
    conditions = []
    if lei:
        conditions.append(f"UPPER(lei) = '{lei.upper().strip()}'")
    if msa_name:
        conditions.append(f"msa_name ILIKE '%{msa_name}%'")
    if race:
        conditions.append(f"derived_race ILIKE '%{race}%'")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT lei, msa_name, derived_race, denial_reason, count
        FROM denial_reasons
        {where}
        ORDER BY count DESC
        LIMIT 50
    """
    return _safe_query(sql)


def _execute_tool(name: str, inputs: dict) -> str:
    """Dispatch a tool call to the appropriate function."""
    if name == "get_denial_rates":
        return _get_denial_rates(**inputs)
    elif name == "compare_to_peers":
        return _compare_to_peers(**inputs)
    elif name == "flag_disparities":
        return _flag_disparities(**inputs)
    elif name == "get_denial_reasons":
        return _get_denial_reasons(**inputs)
    else:
        return f"Unknown tool: {name}"


# ── Public interface ────────────────────────────────────────────────────────────

def ask(question: str) -> str:
    """
    Send a compliance question to Claude. Claude will call tools against DuckDB
    as needed and return a recommendation grounded in real data.
    """
    messages = [{"role": "user", "content": question}]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # No more tool calls — return the final text
        if response.stop_reason == "end_turn":
            return _extract_text(response)

        # Execute all tool calls in this round
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = _execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Append assistant turn + tool results and loop
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    # Fallback: return whatever text we have after max rounds
    return _extract_text(response)


def _extract_text(response) -> str:
    """Pull text blocks out of an API response object."""
    return "\n".join(
        block.text for block in response.content if hasattr(block, "text")
    )


if __name__ == "__main__":
    # Quick smoke test
    print(ask("Which lender × demographic segment has the highest disparity ratio in the dataset?"))
