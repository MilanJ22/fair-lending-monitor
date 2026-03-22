# Fair Lending Compliance Monitor

An AI-powered compliance tool that surfaces racial disparity patterns in mortgage lending data — built for fair lending officers who need exam-ready answers in seconds, not weeks.

Built to demonstrate the **Palantir AIP pattern**: raw regulatory data in, LLM-powered operator decision support out.

---

## What It Does

Fair lending examiners use HMDA data to identify lenders whose denial rates for minority applicants are disproportionately high relative to peers. Historically, surfacing these patterns required weeks of manual analysis by analysts.

This tool ingests 253,000+ home purchase loan records directly from the CFPB HMDA API, computes peer-benchmarked disparity ratios across every lender × race × MSA combination, and puts a compliance analyst (Claude) on top — so a fair lending officer can ask questions in plain English and get regulator-ready answers backed by real data.

**Example queries the system handles:**

- *"Flag all lender × demographic segments with a disparity ratio above 2.0x"*
- *"Which MSA has the highest concentration of Red-flag disparity segments?"*
- *"Show me the denial reasons for the highest-disparity lender in Houston"*
- *"Are there any lenders with a near-100% denial rate for Black applicants?"*

---

## Architecture

```
CFPB HMDA API
     │
     ▼
src/ingest.py       ← fetch 2024 LAR data for 5 MSAs via HMDA Data Browser API v2
     │
     ▼
src/transform.py    ← clean, compute denial rates, peer benchmarks, disparity flags
     │
     ▼
src/storage.py      ← persist to DuckDB (4 tables)
     │
     ▼
src/agent.py        ← Claude (claude-sonnet-4-6) with 4 structured tools over DuckDB
     │
     ▼
app.py              ← Streamlit operator UI
```

**DuckDB tables:**

| Table | Description |
|---|---|
| `applications` | Cleaned LAR records (~253k rows) |
| `denial_rates` | Denial rate by lender × race × MSA |
| `peer_benchmarks` | Average denial rate across all lenders by race × MSA |
| `disparity_flags` | Disparity ratios + Red/Yellow flags per lender segment |
| `denial_reasons` | Denial reason breakdown by lender × race × MSA |

**Claude tools:**

| Tool | Purpose |
|---|---|
| `get_denial_rates` | Look up raw denial rates for any lender/demographic/MSA filter |
| `compare_to_peers` | Benchmark one lender against the peer average — returns disparity ratios |
| `flag_disparities` | Scan all lenders for segments exceeding a disparity threshold |
| `get_denial_reasons` | Root-cause breakdown: why is a lender denying a specific group? |

---

## Disparity Methodology

Disparity ratio = `lender denial rate ÷ peer average denial rate` for the same race × MSA segment.

Thresholds follow OCC and CFPB fair lending examination guidance:

| Flag | Threshold | Meaning |
|---|---|---|
| 🟡 Yellow | ≥ 1.5x | Warrants internal review |
| 🔴 Red | ≥ 2.0x | Threshold for examiner scrutiny and potential enforcement referral |

**Statistical floor:** Lender × race × MSA segments with fewer than 20 applications are excluded to prevent ratio inflation from small sample sizes.

**Peer benchmark:** Average of individual lender denial rates per race × MSA (not the pooled market rate), consistent with how examiners construct peer groups.

---

## Dataset

- **Source:** [CFPB HMDA Data Browser API v2](https://ffiec.cfpb.gov/v2/data-browser-api/view)
- **Year:** 2024
- **Loan type:** Home purchase (loan_purpose=1)
- **Outcomes included:** Originated, approved-not-accepted, denied (action_taken=1,2,3)
- **MSAs:** Chicago · Houston · Atlanta · Los Angeles · New York

---

## Setup

**Prerequisites:** Python 3.10+, Anthropic API key

```bash
git clone https://github.com/MilanJ22/fair-lending-monitor.git
cd fair-lending-monitor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_key_here
```

**Run the pipeline** (downloads ~253k records from CFPB, takes 2-3 minutes):

```bash
python pipeline.py
```

**Launch the UI:**

```bash
streamlit run app.py
```

---

## Key Finding

All 143 Red-flag and 64 Yellow-flag disparity segments are concentrated in the **Houston MSA**. The tool surfaces this immediately — and Claude correctly frames the follow-up question: is this a data quality issue, a non-QM/specialty lender market structure artifact, or a genuine fair lending pattern? That's the question that would take a compliance team weeks to arrive at. The tool gets you there in one query.

---

## Stack

- **Data:** Python · Pandas · CFPB HMDA Data Browser API v2
- **Storage:** DuckDB
- **AI:** Claude (`claude-sonnet-4-6`) via Anthropic Python SDK — tool use pattern
- **UI:** Streamlit
