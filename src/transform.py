import pandas as pd

# Action taken codes → readable labels
ACTION_LABELS = {
    1: "Originated",
    2: "Approved - Not Accepted",
    3: "Denied",
}

# Disparity ratio thresholds — based on OCC/CFPB fair lending exam guidance
DISPARITY_YELLOW = 1.5  # >1.5x peer average = yellow flag (warrants review)
DISPARITY_RED = 2.0     # >2.0x peer average = examiner attention warranted


def clean_applications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize types and add readable labels.
    Institution names are resolved at query time via LEI — LEI is the stable identifier.
    """
    df = df.copy()

    df["action_taken"] = pd.to_numeric(df["action_taken"], errors="coerce")
    df["income"] = pd.to_numeric(df["income"], errors="coerce")
    df["loan_amount"] = pd.to_numeric(df["loan_amount"], errors="coerce")
    df["msamd"] = df["msamd"].astype(str)
    df["lei"] = df["lei"].astype(str).str.strip().str.upper()
    df["action_label"] = df["action_taken"].map(ACTION_LABELS)

    return df


def compute_denial_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute denial rates by lender × race × MSA.

    Denial rate = denied / (originated + approved_not_accepted + denied)
    This is the standard HMDA denominator used in regulatory fair lending exams.
    Rows with fewer than 20 applications are dropped — not statistically meaningful.
    """
    decisive = df[df["action_taken"].isin([1, 2, 3])].copy()
    decisive["is_denied"] = (decisive["action_taken"] == 3).astype(int)

    grouped = (
        decisive.groupby(["lei", "msamd", "msa_name", "derived_race"])
        .agg(
            total_applications=("is_denied", "count"),
            total_denied=("is_denied", "sum"),
        )
        .reset_index()
    )

    grouped["denial_rate"] = grouped["total_denied"] / grouped["total_applications"]

    # Drop statistically thin cells
    grouped = grouped[grouped["total_applications"] >= 20]

    return grouped


def compute_peer_benchmarks(denial_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the peer average denial rate by race × MSA across all lenders.
    This becomes the benchmark each individual lender is compared against.
    """
    benchmarks = (
        denial_rates.groupby(["msamd", "msa_name", "derived_race"])
        .agg(
            peer_avg_denial_rate=("denial_rate", "mean"),
            peer_lender_count=("lei", "count"),
            peer_total_applications=("total_applications", "sum"),
        )
        .reset_index()
    )

    return benchmarks


def compute_disparity_flags(
    denial_rates: pd.DataFrame, benchmarks: pd.DataFrame
) -> pd.DataFrame:
    """
    Join denial rates with peer benchmarks and flag disparity.

    Disparity ratio = lender denial rate / peer average denial rate
    - Red:    >= 2.0x → examiner attention warranted
    - Yellow: >= 1.5x → warrants internal review
    - Normal: < 1.5x
    """
    merged = denial_rates.merge(
        benchmarks[["msamd", "derived_race", "peer_avg_denial_rate", "peer_lender_count"]],
        on=["msamd", "derived_race"],
        how="left",
    )

    merged["disparity_ratio"] = (merged["denial_rate"] / merged["peer_avg_denial_rate"]).round(4)

    merged["disparity_flag"] = "Normal"
    merged.loc[merged["disparity_ratio"] >= DISPARITY_YELLOW, "disparity_flag"] = "Yellow"
    merged.loc[merged["disparity_ratio"] >= DISPARITY_RED, "disparity_flag"] = "Red"

    return merged.sort_values("disparity_ratio", ascending=False).reset_index(drop=True)


def compute_denial_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize denial reasons by lender × race × MSA.
    HMDA allows up to 4 denial reasons per application — we melt and count.
    """
    denied = df[df["action_taken"] == 3].copy()

    reason_cols = ["denial_reason_1", "denial_reason_2", "denial_reason_3", "denial_reason_4"]
    available = [c for c in reason_cols if c in denied.columns]

    melted = denied.melt(
        id_vars=["lei", "msamd", "msa_name", "derived_race"],
        value_vars=available,
        value_name="denial_reason",
    ).dropna(subset=["denial_reason"])

    melted = melted[melted["denial_reason"].astype(str).str.strip() != ""]

    reason_counts = (
        melted.groupby(
            ["lei", "msamd", "msa_name", "derived_race", "denial_reason"]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    return reason_counts
