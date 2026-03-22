import io
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view"
INSTITUTION_API = "https://ffiec.cfpb.gov/v2/public/institutions"

# agency code → human-readable institution type (for depository institutions)
_AGENCY_LABELS = {
    1: "National Bank",
    2: "State Member Bank",
    3: "State Nonmember Bank",
    4: "Savings Association",
    5: "Credit Union",
    7: "Non-Depository (CFPB)",
    9: "HUD-supervised",
}

# otherLenderCode → human-readable type (for non-depository institutions)
_OTHER_LENDER_LABELS = {
    1: "Affiliated Mortgage Company",
    2: "Independent Mortgage Company",
    3: "Other Non-Depository",
}

# 15 major U.S. MSAs — broad geographic and demographic coverage for fair lending analysis
TARGET_MSAS = {
    "16980": "Chicago-Naperville-Elgin, IL-IN-WI",
    "26420": "Houston-The Woodlands-Sugar Land, TX",
    "12060": "Atlanta-Sandy Springs-Roswell, GA",
    "31080": "Los Angeles-Long Beach-Anaheim, CA",
    "35620": "New York-Newark-Jersey City, NY-NJ-PA",
    "19100": "Dallas-Fort Worth-Arlington, TX",
    "38060": "Phoenix-Mesa-Chandler, AZ",
    "33100": "Miami-Fort Lauderdale-Pompano Beach, FL",
    "37980": "Philadelphia-Camden-Wilmington, PA-NJ-DE",
    "47900": "Washington-Arlington-Alexandria, DC-VA-MD-WV",
    "42660": "Seattle-Tacoma-Bellevue, WA",
    "14460": "Boston-Cambridge-Newton, MA-NH",
    "16740": "Charlotte-Concord-Gastonia, NC-SC",
    "33460": "Minneapolis-St. Paul-Bloomington, MN-WI",
    "19740": "Denver-Aurora-Lakewood, CO",
}

# Columns we care about — drop the 80+ others in the raw LAR
# Note: HMDA uses hyphens in some column names (e.g. derived_msa-md, denial_reason-1)
KEEP_COLS = [
    "lei",
    "activity_year",
    "derived_msa-md",
    "census_tract",
    "action_taken",
    "loan_purpose",
    "loan_type",
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
    "income",
    "loan_amount",
    "denial_reason-1",
    "denial_reason-2",
    "denial_reason-3",
    "denial_reason-4",
]


def fetch_applications(year: int = 2024) -> pd.DataFrame:
    """
    Fetch home purchase loan applications from the CFPB HMDA API.
    Scoped to target MSAs, home purchase loans (purpose=1), and decisive
    action types (originated, approved not accepted, denied).
    """
    params = {
        "years": year,
        "msamds": ",".join(TARGET_MSAS.keys()),
        "action_taken": "1,2,3",  # originated, approved-not-accepted, denied
        "loan_purpose": "1",      # home purchase only
    }

    print(f"Fetching HMDA {year} applications for {len(TARGET_MSAS)} MSAs...")
    print("  This may take 60-120 seconds for a large CSV download...")

    response = requests.get(
        f"{BASE_URL}/csv",
        params=params,
        timeout=300,
        stream=True,
    )
    response.raise_for_status()

    # Stream into memory in 1MB chunks
    content = b""
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        content += chunk

    df = pd.read_csv(io.BytesIO(content), low_memory=False)
    print(f"  Raw records fetched: {len(df):,}")

    # Retain only the columns we need
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()

    # Normalize MSA column name and attach human-readable name
    df = df.rename(columns={"derived_msa-md": "msamd"})
    df["msa_name"] = df["msamd"].astype(str).map(TARGET_MSAS)

    # Normalize denial reason column names (hyphens → underscores)
    df = df.rename(columns={
        "denial_reason-1": "denial_reason_1",
        "denial_reason-2": "denial_reason_2",
        "denial_reason-3": "denial_reason_3",
        "denial_reason-4": "denial_reason_4",
    })

    print(f"  Columns retained: {len(df.columns)}")
    return df


def _derive_institution_type(agency: int, other_lender_code: int) -> str:
    """Map agency + otherLenderCode to a human-readable institution type label."""
    if other_lender_code == 0:
        return _AGENCY_LABELS.get(agency, f"Depository (agency {agency})")
    return _OTHER_LENDER_LABELS.get(other_lender_code, f"Non-Depository (code {other_lender_code})")


def _fetch_one_institution(lei: str, year: int) -> dict | None:
    """Fetch a single institution profile from the FFIEC public API. Returns None on failure."""
    try:
        r = requests.get(f"{INSTITUTION_API}/{lei}/year/{year}", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def fetch_institutions(leis: list, year: int = 2024) -> pd.DataFrame:
    """
    Fetch institution profiles for a list of LEIs from the FFIEC public API.
    Uses concurrent requests to minimize fetch time. Failed lookups are skipped
    gracefully — those LEIs will have NULL institution columns in the database.
    """
    print(f"Fetching institution profiles for {len(leis)} unique lenders...")

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_one_institution, lei, year): lei for lei in leis}
        for future in as_completed(futures):
            data = future.result()
            if data:
                results.append(data)

    if not results:
        print("  Warning: No institution profiles retrieved.")
        return pd.DataFrame(columns=[
            "lei", "institution_name", "state", "agency", "other_lender_code",
            "institution_type", "broad_type", "assets", "top_holder",
        ])

    rows = []
    for d in results:
        olc = d.get("otherLenderCode", 0)
        agency = d.get("agency", 0)
        rows.append({
            "lei":              d.get("lei", "").strip().upper(),
            "institution_name": d.get("respondent", {}).get("name", ""),
            "state":            d.get("respondent", {}).get("state", ""),
            "agency":           agency,
            "other_lender_code": olc,
            "institution_type": _derive_institution_type(agency, olc),
            "broad_type":       "Depository" if olc == 0 else "Non-Depository",
            "assets":           d.get("assets"),
            "top_holder":       d.get("topHolder", {}).get("name", ""),
        })

    df = pd.DataFrame(rows)
    type_counts = df["broad_type"].value_counts().to_dict()
    print(f"  Retrieved {len(df):,} profiles — {type_counts}")
    return df


if __name__ == "__main__":
    apps = fetch_applications(2024)
    print(apps["action_taken"].value_counts())
    print(apps["derived_race"].value_counts().head(10))
