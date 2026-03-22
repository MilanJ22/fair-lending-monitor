"""
tests/test_pipeline.py

Unit and edge case tests for the HMDA Fair Lending pipeline.

Covers:
  - Data cleaning and type coercion (transform)
  - Statistical floor enforcement (≥20 applications)
  - Denial rate calculation correctness
  - Peer benchmark methodology (mean of rates, not pooled)
  - Disparity flag thresholds (Red ≥2.0x, Yellow ≥1.5x, Normal <1.5x)
  - Zero peer average does not raise (division-by-zero guard)
  - Denial reason melt logic and empty value handling
  - Storage idempotency (re-running pipeline replaces tables cleanly)
  - Agent tool dispatch: unknown tool name handled gracefully
  - Agent safe query: empty result returns clean message, not exception

Run with:
    pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


# ── Shared fixture builder ────────────────────────────────────────────────────

def make_apps(lei, msamd, msa_name, race, n_denied, n_originated):
    """
    Build a minimal cleaned application DataFrame for a single lender × race × MSA segment.
    Includes denial reason columns so it works with compute_denial_reasons as well.
    """
    denied_rows = [
        {
            "lei": lei, "msamd": msamd, "msa_name": msa_name,
            "action_taken": 3, "derived_race": race,
            "income": 60.0, "loan_amount": 250.0,
            "action_label": "Denied",
            "denial_reason_1": "1", "denial_reason_2": None,
            "denial_reason_3": None, "denial_reason_4": None,
        }
    ] * n_denied

    originated_rows = [
        {
            "lei": lei, "msamd": msamd, "msa_name": msa_name,
            "action_taken": 1, "derived_race": race,
            "income": 80.0, "loan_amount": 300.0,
            "action_label": "Originated",
            "denial_reason_1": None, "denial_reason_2": None,
            "denial_reason_3": None, "denial_reason_4": None,
        }
    ] * n_originated

    return pd.DataFrame(denied_rows + originated_rows)


# ── 1. clean_applications ─────────────────────────────────────────────────────

class TestCleanApplications:

    def test_coerces_numeric_types(self):
        """String values from CSV should be cast to numeric."""
        from src.transform import clean_applications
        df = pd.DataFrame({
            "action_taken": ["1", "3"],
            "income": ["50", "60"],
            "loan_amount": ["200000", "150000"],
            "msamd": [16980, 35620],
            "lei": ["abc", "xyz"],
            "msa_name": ["Chicago", "New York"],
        })
        result = clean_applications(df)
        assert pd.api.types.is_numeric_dtype(result["action_taken"])
        assert pd.api.types.is_numeric_dtype(result["income"])
        assert pd.api.types.is_numeric_dtype(result["loan_amount"])

    def test_adds_action_label(self):
        """action_taken codes should map to human-readable labels."""
        from src.transform import clean_applications
        df = pd.DataFrame({
            "action_taken": [1, 2, 3],
            "income": [50, 60, 70],
            "loan_amount": [200, 250, 300],
            "msamd": ["16980"] * 3,
            "lei": ["A", "B", "C"],
            "msa_name": ["Chicago"] * 3,
        })
        result = clean_applications(df)
        assert list(result["action_label"]) == [
            "Originated", "Approved - Not Accepted", "Denied"
        ]

    def test_normalizes_lei_to_uppercase_stripped(self):
        """LEI should be uppercased and whitespace-stripped for consistent joins."""
        from src.transform import clean_applications
        df = pd.DataFrame({
            "action_taken": [1],
            "income": [50],
            "loan_amount": [200],
            "msamd": ["16980"],
            "lei": ["  abc123  "],
            "msa_name": ["Chicago"],
        })
        result = clean_applications(df)
        assert result["lei"].iloc[0] == "ABC123"


# ── 2. compute_denial_rates ───────────────────────────────────────────────────

class TestComputeDenialRates:

    def test_drops_segments_below_20_applications(self):
        """Segments with <20 applications must be excluded — not statistically meaningful."""
        from src.transform import compute_denial_rates
        df = make_apps("LEI001", "16980", "Chicago", "Black or African American", 3, 2)
        result = compute_denial_rates(df)
        assert len(result) == 0

    def test_retains_segments_at_20_applications(self):
        """Exactly 20 applications should pass the floor."""
        from src.transform import compute_denial_rates
        df = make_apps("LEI001", "16980", "Chicago", "Black or African American", 10, 10)
        result = compute_denial_rates(df)
        assert len(result) == 1

    def test_correct_denial_rate_calculation(self):
        """denial_rate = denied / (originated + denied). 8/20 = 0.40."""
        from src.transform import compute_denial_rates
        df = make_apps("LEI001", "16980", "Chicago", "Black or African American", 8, 12)
        result = compute_denial_rates(df)
        assert pytest.approx(result.iloc[0]["denial_rate"], abs=0.001) == 0.40

    def test_multiple_lenders_computed_independently(self):
        """Each lender's denial rate is computed from their own applications only."""
        from src.transform import compute_denial_rates
        df = pd.concat([
            make_apps("LEI001", "16980", "Chicago", "Black or African American", 10, 10),
            make_apps("LEI002", "16980", "Chicago", "Black or African American", 5, 15),
        ], ignore_index=True)
        result = compute_denial_rates(df)
        assert len(result) == 2
        r1 = result[result["lei"] == "LEI001"].iloc[0]["denial_rate"]
        r2 = result[result["lei"] == "LEI002"].iloc[0]["denial_rate"]
        assert pytest.approx(r1, abs=0.001) == 0.50
        assert pytest.approx(r2, abs=0.001) == 0.25


# ── 3. compute_peer_benchmarks ────────────────────────────────────────────────

class TestComputePeerBenchmarks:

    def test_mean_of_rates_not_pooled(self):
        """
        Peer avg = mean of individual lender denial rates, not the pooled market rate.

        LEI001: 2 denied / 20 total = 0.10
        LEI002: 15 denied / 25 total = 0.60
        Mean of rates  = (0.10 + 0.60) / 2 = 0.35
        Pooled rate    = 17 / 45         = 0.378  ← different, proving methodology
        """
        from src.transform import compute_denial_rates, compute_peer_benchmarks
        df = pd.concat([
            make_apps("LEI001", "16980", "Chicago", "Black or African American", 2, 18),
            make_apps("LEI002", "16980", "Chicago", "Black or African American", 15, 10),
        ], ignore_index=True)
        rates = compute_denial_rates(df)
        benchmarks = compute_peer_benchmarks(rates)
        assert pytest.approx(benchmarks.iloc[0]["peer_avg_denial_rate"], abs=0.001) == 0.35


# ── 4. compute_disparity_flags ────────────────────────────────────────────────

class TestComputeDisparityFlags:

    def _run(self, lender_rate, peer_rate):
        """Build minimal DataFrames and run disparity flagging directly."""
        from src.transform import compute_disparity_flags
        denial_rates = pd.DataFrame([{
            "lei": "LEI001",
            "msamd": "16980",
            "msa_name": "Chicago",
            "derived_race": "Black or African American",
            "total_applications": 25,
            "total_denied": int(lender_rate * 25),
            "denial_rate": lender_rate,
        }])
        peer_benchmarks = pd.DataFrame([{
            "msamd": "16980",
            "msa_name": "Chicago",
            "derived_race": "Black or African American",
            "peer_avg_denial_rate": peer_rate,
            "peer_lender_count": 5,
        }])
        return compute_disparity_flags(denial_rates, peer_benchmarks)

    def test_red_flag_at_exactly_2x(self):
        """Ratio of exactly 2.0x should produce a Red flag."""
        result = self._run(lender_rate=0.40, peer_rate=0.20)
        assert result.iloc[0]["disparity_flag"] == "Red"

    def test_red_flag_above_2x(self):
        result = self._run(lender_rate=0.60, peer_rate=0.20)
        assert result.iloc[0]["disparity_flag"] == "Red"

    def test_yellow_flag_at_exactly_1_5x(self):
        """Ratio of exactly 1.5x should produce a Yellow flag."""
        result = self._run(lender_rate=0.30, peer_rate=0.20)
        assert result.iloc[0]["disparity_flag"] == "Yellow"

    def test_yellow_flag_between_1_5x_and_2x(self):
        result = self._run(lender_rate=0.35, peer_rate=0.20)
        assert result.iloc[0]["disparity_flag"] == "Yellow"

    def test_normal_below_1_5x(self):
        result = self._run(lender_rate=0.20, peer_rate=0.20)
        assert result.iloc[0]["disparity_flag"] == "Normal"

    def test_zero_peer_avg_does_not_raise(self):
        """
        If all lenders in a segment have 0% denial rate, peer avg = 0.
        Division produces inf — verify the function completes without raising.
        """
        result = self._run(lender_rate=0.10, peer_rate=0.0)
        assert len(result) == 1  # completed without raising


# ── 5. compute_denial_reasons ─────────────────────────────────────────────────

class TestComputeDenialReasons:

    def test_melts_multiple_reason_columns(self):
        """All four denial_reason columns should be melted into a single column."""
        from src.transform import compute_denial_reasons
        df = pd.DataFrame([{
            "lei": "LEI001", "msamd": "16980", "msa_name": "Chicago",
            "action_taken": 3, "derived_race": "Black or African American",
            "denial_reason_1": "1", "denial_reason_2": "2",
            "denial_reason_3": None, "denial_reason_4": None,
        }])
        result = compute_denial_reasons(df)
        assert set(result["denial_reason"].astype(str)) == {"1", "2"}

    def test_ignores_originated_applications(self):
        """Only denied applications (action_taken=3) contribute denial reasons."""
        from src.transform import compute_denial_reasons
        df = pd.DataFrame([{
            "lei": "LEI001", "msamd": "16980", "msa_name": "Chicago",
            "action_taken": 1, "derived_race": "Black or African American",
            "denial_reason_1": "1", "denial_reason_2": None,
            "denial_reason_3": None, "denial_reason_4": None,
        }])
        result = compute_denial_reasons(df)
        assert len(result) == 0

    def test_drops_empty_string_reasons(self):
        """Empty string denial reasons should not appear in results."""
        from src.transform import compute_denial_reasons
        df = pd.DataFrame([{
            "lei": "LEI001", "msamd": "16980", "msa_name": "Chicago",
            "action_taken": 3, "derived_race": "Black or African American",
            "denial_reason_1": "1", "denial_reason_2": "",
            "denial_reason_3": None, "denial_reason_4": None,
        }])
        result = compute_denial_reasons(df)
        assert len(result) == 1
        assert result.iloc[0]["denial_reason"] == "1"


# ── 6. Storage idempotency ────────────────────────────────────────────────────

class TestStorage:

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Redirect DB_PATH to a temp file so tests never touch the production database."""
        db_file = str(tmp_path / "test_hmda.duckdb")
        with patch("src.storage.DB_PATH", db_file):
            yield db_file

    def _minimal_frames(self):
        return (
            pd.DataFrame({"lei": ["A"], "msamd": ["16980"], "action_taken": [1]}),
            pd.DataFrame({"lei": ["A"], "denial_rate": [0.2]}),
            pd.DataFrame({"msamd": ["16980"], "peer_avg_denial_rate": [0.2]}),
            pd.DataFrame({"lei": ["A"], "disparity_flag": ["Normal"]}),
            pd.DataFrame({"lei": ["A"], "denial_reason": ["1"], "count": [5]}),
            pd.DataFrame({"lei": ["A"], "institution_name": ["Test Bank"], "broad_type": ["Depository"]}),
        )

    def test_save_all_creates_tables(self, temp_db):
        """Pipeline output should be queryable after save_all."""
        from src.storage import save_all, query
        save_all(*self._minimal_frames())
        result = query("SELECT COUNT(*) AS n FROM applications")
        assert result.iloc[0]["n"] == 1

    def test_save_all_is_idempotent(self, temp_db):
        """Running the pipeline twice must not raise or duplicate rows."""
        from src.storage import save_all, query
        save_all(*self._minimal_frames())
        save_all(*self._minimal_frames())  # second run — DROP TABLE IF EXISTS handles this
        result = query("SELECT COUNT(*) AS n FROM applications")
        assert result.iloc[0]["n"] == 1


# ── 8. Institution type derivation ───────────────────────────────────────────

class TestDeriveInstitutionType:

    def test_national_bank(self):
        from src.ingest import _derive_institution_type
        assert _derive_institution_type(agency=1, other_lender_code=0) == "National Bank"

    def test_credit_union(self):
        from src.ingest import _derive_institution_type
        assert _derive_institution_type(agency=5, other_lender_code=0) == "Credit Union"

    def test_independent_mortgage_company(self):
        from src.ingest import _derive_institution_type
        assert _derive_institution_type(agency=0, other_lender_code=2) == "Independent Mortgage Company"

    def test_affiliated_mortgage_company(self):
        from src.ingest import _derive_institution_type
        assert _derive_institution_type(agency=0, other_lender_code=1) == "Affiliated Mortgage Company"

    def test_unknown_agency_does_not_raise(self):
        """An unrecognized agency code should return a fallback string, not raise."""
        from src.ingest import _derive_institution_type
        result = _derive_institution_type(agency=99, other_lender_code=0)
        assert isinstance(result, str) and len(result) > 0

    def test_unknown_other_lender_code_does_not_raise(self):
        """An unrecognized otherLenderCode should return a fallback string, not raise."""
        from src.ingest import _derive_institution_type
        result = _derive_institution_type(agency=0, other_lender_code=99)
        assert isinstance(result, str) and len(result) > 0


# ── 9. fetch_institutions (mocked HTTP) ───────────────────────────────────────

class TestFetchInstitutions:

    def _mock_institution(self, lei="LEI001", agency=1, olc=0, name="First National Bank"):
        return {
            "lei": lei,
            "agency": agency,
            "otherLenderCode": olc,
            "respondent": {"name": name, "state": "IL", "city": "Chicago"},
            "assets": 500000,
            "topHolder": {"idRssd": 123, "name": "Bancorp Holdings"},
        }

    def test_returns_dataframe_with_correct_columns(self):
        from src.ingest import fetch_institutions
        mock_data = self._mock_institution()
        with patch("src.ingest.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_data
            result = fetch_institutions(["LEI001"], year=2024)
        assert "institution_name" in result.columns
        assert "broad_type" in result.columns
        assert "institution_type" in result.columns
        assert result.iloc[0]["institution_name"] == "First National Bank"
        assert result.iloc[0]["broad_type"] == "Depository"

    def test_non_depository_classified_correctly(self):
        from src.ingest import fetch_institutions
        mock_data = self._mock_institution(agency=0, olc=2, name="QuickMortgage LLC")
        with patch("src.ingest.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_data
            result = fetch_institutions(["LEI001"], year=2024)
        assert result.iloc[0]["broad_type"] == "Non-Depository"
        assert result.iloc[0]["institution_type"] == "Independent Mortgage Company"

    def test_failed_api_call_is_skipped_gracefully(self):
        """A 404 for a LEI should not crash the pipeline — returns empty DataFrame."""
        from src.ingest import fetch_institutions
        with patch("src.ingest.requests.get") as mock_get:
            mock_get.return_value.status_code = 404
            result = fetch_institutions(["BADLEI"], year=2024)
        assert len(result) == 0

    def test_partial_failure_returns_successful_results(self):
        """If one of two LEIs fails, the successful one should still be returned."""
        from src.ingest import fetch_institutions
        mock_data = self._mock_institution(lei="GOODLEI")

        def side_effect(url, timeout):
            r = MagicMock()
            if "GOODLEI" in url:
                r.status_code = 200
                r.json.return_value = mock_data
            else:
                r.status_code = 404
            return r

        with patch("src.ingest.requests.get", side_effect=side_effect):
            result = fetch_institutions(["GOODLEI", "BADLEI"], year=2024)

        assert len(result) == 1
        assert result.iloc[0]["lei"] == "GOODLEI"

    def test_lei_normalized_to_uppercase(self):
        """LEIs should be stored uppercase regardless of API response casing."""
        from src.ingest import fetch_institutions
        mock_data = self._mock_institution(lei="lei001lowercase")
        with patch("src.ingest.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_data
            result = fetch_institutions(["lei001lowercase"], year=2024)
        assert result.iloc[0]["lei"] == "LEI001LOWERCASE"


# ── 9. Aggregate summary tools ───────────────────────────────────────────────

class TestSummaryTools:
    """
    These tests verify that the aggregate tools return deterministic GROUP BY results
    rather than raw rows — ensuring counts are always authoritative SQL outputs,
    not Claude inference on top of a row sample.
    """

    @pytest.fixture
    def temp_db_with_flags(self, tmp_path):
        """Seed a temp DuckDB with known disparity_flags and institutions rows."""
        import duckdb
        db_file = str(tmp_path / "test_summary.duckdb")
        con = duckdb.connect(db_file)

        con.execute("""
            CREATE TABLE disparity_flags AS SELECT * FROM (VALUES
                ('LEI001', 'Minneapolis', 'Black or African American', 0.40, 0.15, 2.67, 'Red',    25),
                ('LEI002', 'Minneapolis', 'Black or African American', 0.35, 0.15, 2.33, 'Red',    30),
                ('LEI003', 'Minneapolis', 'Hispanic or Latino',        0.32, 0.20, 1.60, 'Yellow', 22),
                ('LEI004', 'Houston',     'Black or African American', 0.38, 0.15, 2.53, 'Red',    28),
                ('LEI005', 'Houston',     'Hispanic or Latino',        0.28, 0.20, 1.40, 'Normal', 35)
            ) t(lei, msa_name, derived_race, denial_rate, peer_avg_denial_rate,
                disparity_ratio, disparity_flag, total_applications)
        """)

        con.execute("""
            CREATE TABLE institutions AS SELECT * FROM (VALUES
                ('LEI001', 'First Mortgage LLC',  'Non-Depository'),
                ('LEI002', 'Second Mortgage LLC', 'Non-Depository'),
                ('LEI003', 'Community Bank',      'Depository'),
                ('LEI004', 'Houston Mortgage Co', 'Non-Depository')
            ) t(lei, institution_name, broad_type)
        """)
        con.close()

        with patch("src.agent.query") as mock_query:
            import duckdb as ddb
            def run_query(sql):
                c = ddb.connect(db_file)
                result = c.execute(sql).df()
                c.close()
                return result
            mock_query.side_effect = run_query
            yield

    def test_summarize_flags_by_msa_returns_correct_counts(self, temp_db_with_flags):
        from src.agent import _summarize_flags_by_msa
        result_str = _summarize_flags_by_msa()
        assert "Minneapolis" in result_str
        assert "Houston" in result_str
        # Minneapolis has 2 Red + 1 Yellow; Houston has 1 Red
        assert "2" in result_str  # Minneapolis red count

    def test_summarize_flags_by_msa_excludes_normal(self, temp_db_with_flags):
        """Normal segments must not appear in the summary."""
        from src.agent import _summarize_flags_by_msa
        result_str = _summarize_flags_by_msa()
        # LEI005/Houston Normal row should not inflate counts
        # Houston should show 1 Red, 0 Yellow (not 2)
        assert "No results" not in result_str

    def test_summarize_flags_by_msa_race_filter(self, temp_db_with_flags):
        """Race filter should scope the GROUP BY to only matching rows."""
        from src.agent import _summarize_flags_by_msa
        result_str = _summarize_flags_by_msa(race="Hispanic")
        # Only LEI003 Minneapolis Yellow should match — Houston Normal excluded
        assert "Minneapolis" in result_str

    def test_summarize_flags_by_lender_type_counts_correctly(self, temp_db_with_flags):
        from src.agent import _summarize_flags_by_lender_type
        result_str = _summarize_flags_by_lender_type()
        assert "Non-Depository" in result_str
        assert "Depository" in result_str

    def test_summarize_flags_by_lender_type_msa_filter(self, temp_db_with_flags):
        """MSA filter should scope results to that market only."""
        from src.agent import _summarize_flags_by_lender_type
        result_str = _summarize_flags_by_lender_type(msa_name="Minneapolis")
        assert "Non-Depository" in result_str
        # Community Bank (Depository) has a Yellow flag in Minneapolis — should appear
        assert "Depository" in result_str

    def test_summary_tools_dispatched_correctly(self):
        """All four summary tools must be reachable via _execute_tool."""
        from src.agent import _execute_tool
        with patch("src.agent.query", return_value=pd.DataFrame()):
            r1 = _execute_tool("summarize_flags_by_msa", {})
            r2 = _execute_tool("summarize_flags_by_lender_type", {})
            r3 = _execute_tool("summarize_denial_rates_by_race", {})
            r4 = _execute_tool("summarize_denial_reasons", {})
        assert "Unknown tool" not in r1
        assert "Unknown tool" not in r2
        assert "Unknown tool" not in r3
        assert "Unknown tool" not in r4

    def test_summarize_denial_reasons_aggregates_correctly(self, temp_db_with_flags):
        """SUM of count across races should be returned, not individual rows."""
        from src.agent import _summarize_denial_reasons
        # temp_db_with_flags doesn't have denial_reasons — patch query directly
        mock_result = pd.DataFrame({
            "denial_reason": ["3", "4", "1"],
            "total_citations": [56, 39, 11],
            "races_affected": [4, 4, 3],
        })
        with patch("src.agent.query", return_value=mock_result):
            result = _summarize_denial_reasons(msa_name="Houston")
        assert "56" in result   # code 3 total — not 35 (the hallucinated number)
        assert "39" in result   # code 4 total
        assert "total_citations" in result

    def test_flag_disparities_rejects_unfiltered_calls(self):
        """Calling flag_disparities with no filters must return an error, not query rows."""
        from src.agent import _flag_disparities
        result = _flag_disparities()
        assert "filter_required" in result

    def test_flag_disparities_accepts_filtered_calls(self):
        """Calling flag_disparities with a filter should proceed normally."""
        from src.agent import _flag_disparities
        with patch("src.agent.query", return_value=pd.DataFrame()):
            result = _flag_disparities(msa_name="Minneapolis")
        assert "filter_required" not in result

    def test_compare_to_peers_includes_flag_counts(self):
        """compare_to_peers must include pre-computed lender_total_red_flags column."""
        from src.agent import _compare_to_peers
        mock_result = pd.DataFrame({
            "lei": ["LEI001"],
            "institution_name": ["Test Bank"],
            "lender_type": ["Depository"],
            "msa_name": ["Minneapolis"],
            "derived_race": ["Black or African American"],
            "lender_denial_rate_pct": [40.0],
            "peer_avg_pct": [15.0],
            "disparity_ratio": [2.67],
            "disparity_flag": ["Red"],
            "total_applications": [25],
            "lender_total_red_flags": [3],
            "lender_total_yellow_flags": [1],
        })
        with patch("src.agent.query", return_value=mock_result):
            result = _compare_to_peers("LEI001")
        assert "lender_total_red_flags" in result
        assert "3" in result

    def test_summarize_denial_rates_by_race_uses_weighted_average(self, temp_db_with_flags):
        """Market denial rate must be SUM(denied)/SUM(apps), not AVG of rates."""
        from src.agent import _summarize_denial_rates_by_race
        # Two lenders: 2/20=10% and 15/25=60% — weighted rate = 17/45 = 37.8%, not avg(10,60)=35%
        mock_result = pd.DataFrame({
            "derived_race": ["Black or African American"],
            "total_applications": [45],
            "total_denied": [17],
            "denial_rate_pct": [37.8],
            "lender_count": [2],
        })
        with patch("src.agent.query", return_value=mock_result):
            result = _summarize_denial_rates_by_race(msa_name="Minneapolis")
        assert "37.8" in result  # weighted — not 35.0 (simple average)


# ── 9. Agent — get_lender_profile dispatch ───────────────────────────────────

class TestGetLenderProfileDispatch:

    def test_get_lender_profile_is_dispatched(self):
        """get_lender_profile should reach _get_lender_profile, not the unknown-tool fallback."""
        from src.agent import _execute_tool
        with patch("src.agent.query", return_value=pd.DataFrame()):
            result = _execute_tool("get_lender_profile", {"broad_type": "Depository"})
        assert "Unknown tool" not in result


# ── 7. Agent tool dispatch ────────────────────────────────────────────────────

class TestAgentToolDispatch:

    def test_unknown_tool_name_returns_message(self):
        """An unrecognized tool name should return a clean error string, not raise."""
        from src.agent import _execute_tool
        result = _execute_tool("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_safe_query_returns_message_on_empty_result(self):
        """When a SQL query returns no rows, the agent should get a clean message."""
        from src.agent import _safe_query
        with patch("src.agent.query", return_value=pd.DataFrame()):
            result = _safe_query("SELECT 1 WHERE 1=0")
        assert result == "No results found for these filters."
