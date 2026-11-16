"""Command-line explanation of the hypothetical credit-increase scenario for one customer.

Usage:
    python main.py explain --customer 102 --increase 30000             # an example absolute amount
    python main.py explain --customer 102 --times-typical-payment 3    # 3 x the customer's own typical payment
    python main.py explain --customer 102 --share-of-limit 0.25        # 25% of the customer's own limit

Requires notebooks 01-07 to have been run (they produce the feature tables and calibrated PDs).
Graph evidence is read from Neo4j when it is reachable; otherwise the explanation is printed without it.
DEMONSTRATIVE ONLY: the increase is hypothetical and the score is not an underwriting decision.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from credit_kg import scoring as S  # noqa: E402


def explain(customer: int, increase: float | None, times_payment: float | None, share_of_limit: float | None) -> None:
    cfg = S.load_config()
    inp = S.load_scoring_inputs()
    row_df = inp[inp["customer_id"] == customer]
    if row_df.empty:
        sys.exit(f"customer {customer} not found (valid IDs: 1-{int(inp.customer_id.max())})")
    row = row_df.iloc[0]
    if times_payment is not None:
        increase = float(S.relative_increase(row_df, "payment_multiple", times_payment).iloc[0])
        print(f"Relative scenario: {times_payment:g} x typical monthly payment ({row.pay_median:,.0f}) = {increase:,.0f}\n")
    elif share_of_limit is not None:
        increase = float(S.relative_increase(row_df, "limit_share", share_of_limit).iloc[0])
        print(f"Relative scenario: {share_of_limit:.0%} of current limit ({row.credit_limit:,.0f}) = {increase:,.0f}\n")
    elif increase is None:
        increase = cfg["default_increase"]
    res = S.score(row_df, increase, cfg).iloc[0]
    evidence = None
    try:
        from credit_kg.graph import get_driver
        driver = get_driver()
        evidence = S.graph_evidence(driver, customer, row)
        driver.close()
    except Exception as exc:  # Neo4j offline or not loaded: explanation still works from stored features
        print(f"(graph evidence unavailable: {type(exc).__name__})\n")
    print(S.explain(row, res, cfg, evidence))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("explain", help="explain the scenario score for one customer")
    ex.add_argument("--customer", type=int, required=True, help="customer ID (1-30000)")
    how = ex.add_mutually_exclusive_group()
    how.add_argument("--increase", type=float, help="hypothetical increase as an absolute example amount")
    how.add_argument("--times-typical-payment", type=float, help="hypothetical increase = k x the customer's own typical payment")
    how.add_argument("--share-of-limit", type=float, help="hypothetical increase = p x the customer's own current limit")
    args = parser.parse_args()
    if args.command == "explain":
        explain(args.customer, args.increase, args.times_typical_payment, args.share_of_limit)


if __name__ == "__main__":
    main()
