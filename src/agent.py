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

You have access to 2024 HMDA data for home purchase loans across 15 major U.S. metropolitan \
areas: New York, Los Angeles, Chicago, Dallas-Fort Worth, Houston, Washington DC, Miami, \
Philadelphia, Atlanta, Phoenix, Seattle, Boston, Charlotte, Minneapolis, and Denver. \
The data covers applications from all HMDA filers active in these markets.

DATABASE TABLES AVAILABLE VIA YOUR TOOLS:
- denial_rates: denial rate by lender (LEI) × race × MSA
- disparity_flags: denial rates + peer benchmarks + disparity ratios + Red/Yellow flags
- denial_reasons: denial reason breakdown by lender × race × MSA
- peer_benchmarks: average denial rate across all lenders by race × MSA
- institutions: lender profiles — institution name, type (Depository vs. Non-Depository), \
charter type, assets (in $thousands), and ultimate parent company

INSTITUTION TYPE CONTEXT:
- Depository institutions (banks, credit unions, savings associations) are subject to CRA \
and primary federal regulator supervision — fair lending violations carry heightened consequences
- Non-Depository institutions (independent mortgage companies, affiliated mortgage companies) \
are supervised primarily by the CFPB and state regulators — they often serve non-QM or \
specialty markets and can show higher denial rate variance
- When interpreting disparity patterns, always note whether flagged lenders are depository \
or non-depository, as this affects regulatory risk and remediation approach

REGULATORY CONTEXT — critical for interpreting results:
- Disparity ratio = lender denial rate / peer average denial rate for that demographic × geography
- Ratio ≥ 1.5x (Yellow): warrants internal review under OCC and CFPB fair lending exam guidance
- Ratio ≥ 2.0x (Red): threshold that typically triggers examiner scrutiny and enforcement referral
- Home purchase lending carries the highest regulatory weight — it directly affects access to \
wealth-building through homeownership
- Denial rate denominator: denied / (originated + approved-not-accepted + denied)

TOOL ROUTING — always use the correct tool for the question type:
- "Which MSA has the most flags / highest concentration?" → summarize_flags_by_msa
- "Are flagged lenders banks or mortgage companies?" → summarize_flags_by_lender_type
- "What is the denial rate for [demographic] in [MSA]?" → summarize_denial_rates_by_race
- "What are the most common denial reasons for [lender/MSA/demographic]?" → summarize_denial_reasons
- "Show me the flagged segments in [specific MSA or demographic]" → flag_disparities WITH a filter
- "What is lender X's exposure?" → compare_to_peers
- "Who is lender X / what type of institution?" → get_lender_profile
- "What are denial rates for [specific lender or segment]?" → get_denial_rates WITH a filter

BEHAVIOR RULES:
1. Always call a tool to query real data before answering — never rely on assumptions
2. Every response must end with a RECOMMENDED ACTION — what should the compliance officer do \
next, and by when
3. Be specific: cite actual disparity ratios, institution names, MSA names, and application counts
4. Frame findings in terms of regulatory risk, not just statistics
5. Lenders are identified by their LEI — if a user provides a bank name, use get_lender_profile \
to find the matching LEI first
6. If a query returns no results, say so clearly and suggest alternate filters
7. Never perform any arithmetic — counting, summing, averaging, or computing ratios — on \
row-level tool results. Every number you cite must come directly from a tool response. \
If a question requires a total, average, or count, call the appropriate summary tool \
(summarize_flags_by_msa, summarize_flags_by_lender_type, summarize_denial_rates_by_race, \
summarize_denial_reasons) which return pre-computed SQL aggregates

Tone: direct, professional, action-oriented. You are an operator compliance tool, not a chatbot."""


# ── Tool definitions (sent to Claude as the tools list) ────────────────────────

TOOLS = [
    {
        "name": "summarize_flags_by_msa",
        "description": (
            "Returns the count of Red and Yellow disparity flag segments per MSA, along with "
            "the number of distinct lenders flagged. Use this for any question about which "
            "markets have the highest concentration of risk — it runs a deterministic GROUP BY "
            "query and returns authoritative counts. Never use flag_disparities to answer "
            "concentration questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "race": {
                    "type": "string",
                    "description": "Partial race string to scope the summary to one demographic (optional).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "summarize_flags_by_lender_type",
        "description": (
            "Returns Red and Yellow flag counts broken down by lender type (Depository vs. "
            "Non-Depository), with distinct lender counts. Use this for any question about "
            "whether flagged institutions are banks or mortgage companies. Optionally scope "
            "to a single MSA or demographic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "msa_name": {
                    "type": "string",
                    "description": "Partial MSA name to scope the summary (optional).",
                },
                "race": {
                    "type": "string",
                    "description": "Partial race string to scope the summary (optional).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "summarize_denial_rates_by_race",
        "description": (
            "Returns the market-level denial rate per racial/ethnic group for a given MSA — "
            "weighted by application volume (SUM denied / SUM applications), which is the "
            "correct methodology. Use this for any question about overall denial rates by "
            "demographic. Never average denial_rate values from get_denial_rates manually."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "msa_name": {
                    "type": "string",
                    "description": "Partial MSA name to scope the summary (optional — omit for national view).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "summarize_denial_reasons",
        "description": (
            "Returns total denial reason citation counts aggregated across all races, ranked "
            "by frequency. Use this for any question about which denial reasons are most common "
            "for a lender or market. Never sum the count column from get_denial_reasons manually."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lei": {
                    "type": "string",
                    "description": "LEI to scope to a specific lender (optional).",
                },
                "msa_name": {
                    "type": "string",
                    "description": "Partial MSA name to filter (optional).",
                },
                "race": {
                    "type": "string",
                    "description": "Partial race string to filter (optional).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_denial_rates",
        "description": (
            "Drill-down tool: returns raw denial rates for a specific lender, MSA, or "
            "demographic segment. Always provide at least one filter (lei, msa_name, or race). "
            "Do not use this for broad market overviews or counting — use summary tools instead."
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
            "Drill-down tool: returns individual flagged lender × demographic × MSA segments "
            "above a disparity threshold. Always use with at least one filter (msa_name or race) "
            "to scope the results. Do NOT use this to count flags or compare markets — use "
            "summarize_flags_by_msa or summarize_flags_by_lender_type for those questions."
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
        "name": "get_lender_profile",
        "description": (
            "Look up institution details for one or more lenders — name, type (Depository vs. "
            "Non-Depository), charter type, asset size, and ultimate parent company. Use this "
            "after identifying a flagged LEI to understand who the lender is and what kind of "
            "institution they are. Also use to filter lenders by type across an MSA."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lei": {
                    "type": "string",
                    "description": "LEI to look up (optional).",
                },
                "institution_name": {
                    "type": "string",
                    "description": "Partial institution name to search, e.g. 'Chase' or 'Wells' (optional).",
                },
                "broad_type": {
                    "type": "string",
                    "description": "Filter by 'Depository' or 'Non-Depository' (optional).",
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
        conditions.append(f"UPPER(dr.lei) = '{lei.upper().strip()}'")
    if msa_name:
        conditions.append(f"dr.msa_name ILIKE '%{msa_name}%'")
    if race:
        conditions.append(f"dr.derived_race ILIKE '%{race}%'")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT dr.lei,
               COALESCE(i.institution_name, 'Unknown') AS institution_name,
               COALESCE(i.broad_type, 'Unknown')       AS lender_type,
               dr.msa_name, dr.derived_race,
               dr.total_applications,
               dr.total_denied,
               ROUND(dr.denial_rate * 100, 1) AS denial_rate_pct
        FROM denial_rates dr
        LEFT JOIN institutions i ON dr.lei = i.lei
        {where}
        ORDER BY dr.denial_rate DESC
        LIMIT 50
    """
    return _safe_query(sql)


def _compare_to_peers(lei: str, msa_name=None) -> str:
    conditions = [f"UPPER(df.lei) = '{lei.upper().strip()}'"]
    if msa_name:
        conditions.append(f"df.msa_name ILIKE '%{msa_name}%'")

    where = "WHERE " + " AND ".join(conditions)

    sql = f"""
        SELECT df.lei,
               COALESCE(i.institution_name, 'Unknown') AS institution_name,
               COALESCE(i.broad_type, 'Unknown')       AS lender_type,
               df.msa_name, df.derived_race,
               ROUND(df.denial_rate * 100, 1)         AS lender_denial_rate_pct,
               ROUND(df.peer_avg_denial_rate * 100, 1) AS peer_avg_pct,
               ROUND(df.disparity_ratio, 2)            AS disparity_ratio,
               df.disparity_flag,
               df.total_applications,
               COUNT(CASE WHEN df.disparity_flag = 'Red'    THEN 1 END)
                   OVER (PARTITION BY df.lei) AS lender_total_red_flags,
               COUNT(CASE WHEN df.disparity_flag = 'Yellow' THEN 1 END)
                   OVER (PARTITION BY df.lei) AS lender_total_yellow_flags
        FROM disparity_flags df
        LEFT JOIN institutions i ON df.lei = i.lei
        {where}
        ORDER BY df.disparity_ratio DESC
        LIMIT 50
    """
    return _safe_query(sql)


def _flag_disparities(threshold=2.0, msa_name=None, race=None) -> str:
    if not msa_name and not race:
        return (
            "filter_required: flag_disparities must be called with at least one filter "
            "(msa_name or race). For market-wide counts use summarize_flags_by_msa instead."
        )
    conditions = [f"df.disparity_ratio >= {float(threshold)}"]
    if msa_name:
        conditions.append(f"df.msa_name ILIKE '%{msa_name}%'")
    if race:
        conditions.append(f"df.derived_race ILIKE '%{race}%'")

    where = "WHERE " + " AND ".join(conditions)

    sql = f"""
        SELECT df.lei,
               COALESCE(i.institution_name, 'Unknown') AS institution_name,
               COALESCE(i.broad_type, 'Unknown')       AS lender_type,
               df.msa_name, df.derived_race,
               ROUND(df.denial_rate * 100, 1)         AS lender_denial_rate_pct,
               ROUND(df.peer_avg_denial_rate * 100, 1) AS peer_avg_pct,
               ROUND(df.disparity_ratio, 2)            AS disparity_ratio,
               df.disparity_flag,
               df.total_applications
        FROM disparity_flags df
        LEFT JOIN institutions i ON df.lei = i.lei
        {where}
        ORDER BY df.disparity_ratio DESC
        LIMIT 50
    """
    return _safe_query(sql)


def _get_denial_reasons(lei=None, msa_name=None, race=None) -> str:
    conditions = []
    if lei:
        conditions.append(f"UPPER(dr.lei) = '{lei.upper().strip()}'")
    if msa_name:
        conditions.append(f"dr.msa_name ILIKE '%{msa_name}%'")
    if race:
        conditions.append(f"dr.derived_race ILIKE '%{race}%'")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT dr.lei,
               COALESCE(i.institution_name, 'Unknown') AS institution_name,
               COALESCE(i.broad_type, 'Unknown')       AS lender_type,
               dr.msa_name, dr.derived_race,
               dr.denial_reason, dr.count
        FROM denial_reasons dr
        LEFT JOIN institutions i ON dr.lei = i.lei
        {where}
        ORDER BY dr.count DESC
        LIMIT 50
    """
    return _safe_query(sql)


def _get_lender_profile(lei=None, institution_name=None, broad_type=None) -> str:
    conditions = []
    if lei:
        conditions.append(f"UPPER(lei) = '{lei.upper().strip()}'")
    if institution_name:
        conditions.append(f"institution_name ILIKE '%{institution_name}%'")
    if broad_type:
        conditions.append(f"broad_type ILIKE '%{broad_type}%'")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT lei, institution_name, institution_type, broad_type,
               state,
               CASE
                   WHEN assets IS NULL THEN 'N/A'
                   WHEN assets >= 1000000 THEN '$' || ROUND(assets / 1000000.0, 1)::VARCHAR || 'B'
                   WHEN assets >= 1000    THEN '$' || ROUND(assets / 1000.0, 1)::VARCHAR || 'M'
                   ELSE '$' || assets::VARCHAR || 'K'
               END AS total_assets,
               top_holder
        FROM institutions
        {where}
        ORDER BY assets DESC NULLS LAST
        LIMIT 50
    """
    return _safe_query(sql)


def _summarize_denial_rates_by_race(msa_name=None) -> str:
    msa_condition = f"WHERE dr.msa_name ILIKE '%{msa_name}%'" if msa_name else ""
    sql = f"""
        SELECT dr.derived_race,
               SUM(dr.total_applications)                                  AS total_applications,
               SUM(dr.total_denied)                                        AS total_denied,
               ROUND(SUM(dr.total_denied) * 100.0 / SUM(dr.total_applications), 1)
                                                                           AS denial_rate_pct,
               COUNT(DISTINCT dr.lei)                                      AS lender_count
        FROM denial_rates dr
        {msa_condition}
        GROUP BY dr.derived_race
        ORDER BY denial_rate_pct DESC
    """
    return _safe_query(sql)


def _summarize_denial_reasons(lei=None, msa_name=None, race=None) -> str:
    conditions = []
    if lei:
        conditions.append(f"UPPER(dr.lei) = '{lei.upper().strip()}'")
    if msa_name:
        conditions.append(f"dr.msa_name ILIKE '%{msa_name}%'")
    if race:
        conditions.append(f"dr.derived_race ILIKE '%{race}%'")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT dr.denial_reason,
               SUM(dr.count)  AS total_citations,
               COUNT(DISTINCT dr.derived_race) AS races_affected
        FROM denial_reasons dr
        {where}
        GROUP BY dr.denial_reason
        ORDER BY total_citations DESC
    """
    return _safe_query(sql)


def _summarize_flags_by_msa(race=None) -> str:
    race_condition = f"AND df.derived_race ILIKE '%{race}%'" if race else ""
    sql = f"""
        SELECT df.msa_name,
               COUNT(CASE WHEN df.disparity_flag = 'Red'    THEN 1 END) AS red_segments,
               COUNT(CASE WHEN df.disparity_flag = 'Yellow' THEN 1 END) AS yellow_segments,
               COUNT(DISTINCT df.lei)                                    AS distinct_lenders
        FROM disparity_flags df
        WHERE df.disparity_flag IN ('Red', 'Yellow')
        {race_condition}
        GROUP BY df.msa_name
        ORDER BY red_segments DESC, yellow_segments DESC
    """
    return _safe_query(sql)


def _summarize_flags_by_lender_type(msa_name=None, race=None) -> str:
    conditions = ["df.disparity_flag IN ('Red', 'Yellow')"]
    if msa_name:
        conditions.append(f"df.msa_name ILIKE '%{msa_name}%'")
    if race:
        conditions.append(f"df.derived_race ILIKE '%{race}%'")

    where = "WHERE " + " AND ".join(conditions)

    sql = f"""
        SELECT COALESCE(i.broad_type, 'Unknown')                         AS lender_type,
               COUNT(CASE WHEN df.disparity_flag = 'Red'    THEN 1 END) AS red_segments,
               COUNT(CASE WHEN df.disparity_flag = 'Yellow' THEN 1 END) AS yellow_segments,
               COUNT(DISTINCT df.lei)                                    AS distinct_lenders
        FROM disparity_flags df
        LEFT JOIN institutions i ON df.lei = i.lei
        {where}
        GROUP BY COALESCE(i.broad_type, 'Unknown')
        ORDER BY red_segments DESC
    """
    return _safe_query(sql)


def _execute_tool(name: str, inputs: dict) -> str:
    """Dispatch a tool call to the appropriate function."""
    if name == "summarize_flags_by_msa":
        return _summarize_flags_by_msa(**inputs)
    elif name == "summarize_flags_by_lender_type":
        return _summarize_flags_by_lender_type(**inputs)
    elif name == "summarize_denial_rates_by_race":
        return _summarize_denial_rates_by_race(**inputs)
    elif name == "summarize_denial_reasons":
        return _summarize_denial_reasons(**inputs)
    elif name == "get_denial_rates":
        return _get_denial_rates(**inputs)
    elif name == "compare_to_peers":
        return _compare_to_peers(**inputs)
    elif name == "flag_disparities":
        return _flag_disparities(**inputs)
    elif name == "get_denial_reasons":
        return _get_denial_reasons(**inputs)
    elif name == "get_lender_profile":
        return _get_lender_profile(**inputs)
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
