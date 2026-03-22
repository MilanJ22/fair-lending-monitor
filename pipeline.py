from src.ingest import fetch_applications, fetch_institutions
from src.transform import (
    clean_applications,
    compute_denial_rates,
    compute_peer_benchmarks,
    compute_disparity_flags,
    compute_denial_reasons,
)
from src.storage import save_all


def run():
    print("=" * 55)
    print("  HMDA Fair Lending Pipeline — 2024 Data")
    print("=" * 55)

    # Step 1: Ingest
    print("\n[1/5] Ingesting from CFPB HMDA API...")
    raw_apps = fetch_applications(year=2024)

    # Step 2: Clean
    print("\n[2/5] Cleaning data...")
    apps = clean_applications(raw_apps)

    # Step 3: Denial rates
    print("\n[3/6] Computing denial rates by lender × race × MSA...")
    denial_rates = compute_denial_rates(apps)

    # Step 4: Benchmarks + disparity flags
    print("\n[4/6] Computing peer benchmarks and disparity flags...")
    peer_benchmarks = compute_peer_benchmarks(denial_rates)
    disparity_flags = compute_disparity_flags(denial_rates, peer_benchmarks)

    # Step 5: Denial reasons
    denial_reasons = compute_denial_reasons(apps)

    # Step 6: Institution profiles (only for lenders that cleared the 20-app floor)
    print("\n[5/6] Fetching institution profiles from FFIEC API...")
    unique_leis = denial_rates["lei"].unique().tolist()
    institutions = fetch_institutions(unique_leis, year=2024)

    # Step 7: Store
    print("\n[6/6] Writing to DuckDB...")
    save_all(apps, denial_rates, peer_benchmarks, disparity_flags, denial_reasons, institutions)

    # Summary
    red_flags = disparity_flags[disparity_flags["disparity_flag"] == "Red"]
    yellow_flags = disparity_flags[disparity_flags["disparity_flag"] == "Yellow"]

    print("\n" + "=" * 55)
    print("  Pipeline Complete")
    print("=" * 55)
    print(f"  Total applications:    {len(apps):,}")
    print(f"  Denial rate records:   {len(denial_rates):,}")
    print(f"  Institution profiles:  {len(institutions):,}")
    print(f"  Red flags (>=2.0x):    {len(red_flags):,}")
    print(f"  Yellow flags (>=1.5x): {len(yellow_flags):,}")
    print()


if __name__ == "__main__":
    run()
