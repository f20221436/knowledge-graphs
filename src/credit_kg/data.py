"""Phase 1: load, check and reshape the raw UCI table.

Two facts found by inspecting the data drive this module:
* Months are renamed to chronological m1 (Apr) .. m6 (Sep); PAY_0 / BILL_AMT1 / PAY_AMT1 are September.
* PAY_AMT for month t pays the statement of month t-1 (PAY_AMTk matches BILL_AMT(k+1) ~6x more often
  than BILL_AMTk), so payment coverage is pay_t / bill_(t-1).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C

RAW_COLUMNS = (
    ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
     "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
    + ["default.payment.next.month"]
)
STATUS_RAW = {6: "PAY_0", 5: "PAY_2", 4: "PAY_3", 3: "PAY_4", 2: "PAY_5", 1: "PAY_6"}

STATES = {
    "NO_USE": (-2, "No consumption / no balance to pay (code -2)"),
    "PAID_FULL": (-1, "Paid duly in full (code -1)"),
    "REVOLVING": (0, "Minimum paid, revolving credit used (code 0)"),
    "LATE_1": (1, "Payment delayed ~1 month (code 1)"),
    "LATE_2PLUS": (2, "Payment delayed 2+ months (codes 2-9)"),
}


def status_to_state(code):
    """Map raw PAY_* codes to named repayment states (codes are ordinal labels, not month counters)."""
    code = np.asarray(code)
    return np.select(
        [code <= -2, code == -1, code == 0, code == 1],
        ["NO_USE", "PAID_FULL", "REVOLVING", "LATE_1"],
        default="LATE_2PLUS",
    )


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(C.RAW_CSV)
    assert list(df.columns) == RAW_COLUMNS, "unexpected column layout"
    assert df["ID"].is_unique, "duplicate customer IDs"
    assert df.notna().all().all(), "missing values present"
    assert set(df["default.payment.next.month"].unique()) <= {0, 1}
    return df


def data_quality(df: pd.DataFrame) -> pd.DataFrame:
    bills = df[[f"BILL_AMT{i}" for i in range(1, 7)]]
    pays = df[[f"PAY_AMT{i}" for i in range(1, 7)]]
    util = bills.div(df["LIMIT_BAL"], axis=0)
    rows = {
        "customers": len(df),
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_ids": int(df["ID"].duplicated().sum()),
        "duplicate_rows_excluding_id": int(df.drop(columns="ID").duplicated().sum()),
        "default_rate": round(df["default.payment.next.month"].mean(), 4),
        "negative_bill_cells": int((bills < 0).sum().sum()),
        "rows_with_negative_bill": int((bills < 0).any(axis=1).sum()),
        "all_zero_bill_rows": int((bills == 0).all(axis=1).sum()),
        "all_zero_payment_rows": int((pays == 0).all(axis=1).sum()),
        "utilization_above_1_cells": int((util > 1).sum().sum()),
        "max_utilization": round(float(util.max().max()), 2),
        "undocumented_education_codes_rows": int(df["EDUCATION"].isin([0, 5, 6]).sum()),
        "undocumented_marriage_code_rows": int((df["MARRIAGE"] == 0).sum()),
    }
    return pd.DataFrame({"check": list(rows), "value": list(rows.values())})


def alignment_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Count exact matches of each payment against same-month vs previous-month bill."""
    out = []
    for k in range(1, 6):
        pay = df[f"PAY_AMT{k}"]
        out.append({
            "payment_column": f"PAY_AMT{k}",
            "matches_same_month_bill": int(((pay == df[f"BILL_AMT{k}"]) & (pay > 0)).sum()),
            "matches_previous_month_bill": int(((pay == df[f"BILL_AMT{k + 1}"]) & (pay > 0)).sum()),
        })
    return pd.DataFrame(out)


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer, chronological month columns, readable names."""
    w = pd.DataFrame({
        "customer_id": df["ID"].astype(int),
        "credit_limit": df["LIMIT_BAL"].astype(float),
        "sex": df["SEX"],
        # Undocumented codes are grouped into the documented 'other' bucket.
        "education": df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}),
        "marriage": df["MARRIAGE"].replace({0: 3}),
        "age": df["AGE"],
    })
    for m in C.MONTHS:
        w[f"status_m{m}"] = df[STATUS_RAW[m]].astype(int)
    for m in C.MONTHS:
        w[f"bill_m{m}"] = df[f"BILL_AMT{7 - m}"].astype(float)
    for m in C.MONTHS:
        w[f"pay_m{m}"] = df[f"PAY_AMT{7 - m}"].astype(float)
    w[C.TARGET] = df["default.payment.next.month"].astype(int)
    return w


def coverage_matrix(wide: pd.DataFrame) -> np.ndarray:
    """(n, 5) payment coverage for months 2..6: pay_t / bill_(t-1), capped at 1, NaN if no balance due."""
    prev_bill = wide[[f"bill_m{m}" for m in range(1, 6)]].to_numpy()
    pay = wide[[f"pay_m{m}" for m in range(2, 7)]].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = np.where(prev_bill > 0, np.minimum(pay / prev_bill, 1.0), np.nan)
    return cov


def to_monthly(wide: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per customer-month (180k rows), with the aligned coverage of the
    payment made in that month against the previous statement."""
    cov = coverage_matrix(wide)
    frames = []
    for m in C.MONTHS:
        bill = wide[f"bill_m{m}"]
        status = wide[f"status_m{m}"]
        frames.append(pd.DataFrame({
            "customer_id": wide["customer_id"],
            "record_id": wide["customer_id"].astype(str) + "-" + str(m),
            "month_index": m,
            "month": C.MONTH_LABELS[m],
            "bill_amount": bill,
            "payment_amount": wide[f"pay_m{m}"],
            "repayment_status": status,
            "state": status_to_state(status),
            "utilization": bill.clip(lower=0) / wide["credit_limit"],
            "is_late": status >= 1,
            "credit_balance": bill < 0,
            "prev_bill_amount": wide[f"bill_m{m - 1}"] if m > 1 else np.nan,
            "coverage_of_prev_bill": cov[:, m - 2] if m > 1 else np.nan,
        }))
    return pd.concat(frames, ignore_index=True).sort_values(["customer_id", "month_index"], ignore_index=True)


def make_split(wide: pd.DataFrame) -> pd.DataFrame:
    train_ids, test_ids = train_test_split(
        wide["customer_id"], test_size=C.TEST_SIZE, stratify=wide[C.TARGET], random_state=C.SEED
    )
    split = pd.concat([
        pd.DataFrame({"customer_id": train_ids, "split": "train"}),
        pd.DataFrame({"customer_id": test_ids, "split": "test"}),
    ]).sort_values("customer_id", ignore_index=True)
    return split


def load_processed():
    """Return (wide, monthly, split) written by notebook 01."""
    wide = pd.read_parquet(C.PROCESSED / "customers.parquet")
    monthly = pd.read_parquet(C.PROCESSED / "monthly.parquet")
    split = pd.read_csv(C.PROCESSED / "split.csv")
    return wide, monthly, split
