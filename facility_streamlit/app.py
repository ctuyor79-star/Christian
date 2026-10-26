# -*- coding: utf-8 -*-
"""
Streamlit version of:
Laboratory Examination — Data Mining and Systems Analysis Integration
Course: Systems Analysis and Design (SAD) | Program: BS Information Systems
Student: Christian P. Tuyor | Section: BSIS-3A

Run with:   streamlit run app.py
"""
import io
import sqlite3
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.data_generator import generate_facility_reservations as _generate
from src.cleaning import (REQUIRED_COLS, LEAD_ORDER, VALID_RANGES, clean_data as _clean)
from src.diagram import build_use_case_diagram

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, f1_score, classification_report,
                             roc_auc_score)

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

st.set_page_config(page_title="Facility Reservation — Data Mining Lab Exam", page_icon="🏫", layout="wide")


# =====================================================================
# Helpers
# =====================================================================
def md_table(frame, index=False):
    """DataFrame -> Markdown table."""
    f = frame.reset_index() if index else frame
    head = "| " + " | ".join(map(str, f.columns)) + " |"
    sep = "|" + "---|" * len(f.columns)
    rows = ["| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in r) + " |"
            for r in f.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# =====================================================================
# Sidebar — settings
# =====================================================================
st.sidebar.title("⚙️ Settings")
data_source = st.sidebar.radio("Data source", ["Generate synthetic data", "Upload my own CSV"])
RANDOM_STATE = int(st.sidebar.number_input("Random seed", value=42, step=1))
uploaded = None
n_rows = 420
if data_source == "Generate synthetic data":
    n_rows = int(st.sidebar.slider("Number of transactions", 300, 500, 420, 10))
else:
    uploaded = st.sidebar.file_uploader("facility_reservations.csv", type="csv")

st.sidebar.markdown("---")
st.sidebar.subheader("Decision Tree")
TEST_SIZE = st.sidebar.slider("Test set share", 0.10, 0.40, 0.20, 0.05)
MAX_DEPTH = int(st.sidebar.slider("max_depth", 2, 10, 4))
MIN_LEAF = int(st.sidebar.slider("min_samples_leaf", 1, 40, 15))
st.sidebar.subheader("Risk levels (dashboard)")
RISK_LOW = st.sidebar.slider("Low below", 0.05, 0.60, 0.35, 0.05)
RISK_HIGH = st.sidebar.slider("High from", 0.30, 0.95, 0.60, 0.05)

st.title("🏫 Laboratory Examination — Data Mining and Systems Analysis Integration")
st.markdown(
    "**Course:** Systems Analysis and Design (SAD) | **Program:** BS Information Systems  \n"
    "**Student:** Christian P. Tuyor | **Section:** BSIS-3A | **Date:** 10-12-2026"
)


# =====================================================================
# Section 0 — Synthetic dataset
# =====================================================================
generate_facility_reservations = st.cache_data(show_spinner=False)(_generate)


# =====================================================================
# Load the "CSV" (round-trips through CSV text exactly like the notebook did through disk)
# =====================================================================
if data_source == "Generate synthetic data":
    csv_text = generate_facility_reservations(n_rows, RANDOM_STATE).to_csv(index=False)
else:
    if uploaded is None:
        st.info("⬅️ Upload a `facility_reservations.csv` in the sidebar, or switch to *Generate synthetic data*.")
        st.stop()
    csv_text = uploaded.getvalue().decode("utf-8")

df_raw = pd.read_csv(io.StringIO(csv_text))
missing_cols = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing_cols:
    st.error(f"The CSV is missing required columns: {missing_cols}")
    st.stop()


# =====================================================================
# Part II helpers — cleaning (cached)
# =====================================================================
clean_data = st.cache_data(show_spinner=False)(_clean)

df, cleaning_log = clean_data(df_raw)
df_original = df_raw.copy(deep=True)


def to_csv_bytes(frame):
    out = frame.copy()
    for c in ["Lead_Time_Days", "Participants", "Start_Hour", "Prior_Cancellations",
              "Actual_Attendance", "Cancellation_Days_Before"]:
        out[c] = out[c].round().astype("Int64")
    return out.to_csv(index=False).encode("utf-8")


# =====================================================================
# Tabs
# =====================================================================
tabs = st.tabs(["📘 Overview", "I · Understanding", "II · Cleaning", "III · EDA", "IV · Modelling",
                "V · Requirements", "VI · Decision support", "⬇️ Downloads"])

# ---------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------
with tabs[0]:
    st.markdown("""
**Scenario.** The university's prototype *Campus Facility Reservation and Monitoring System* is being turned into a
data-driven decision-support platform. This app plays both roles: **Data Analyst** (clean → explore → predict) and
**Systems Analyst** (evidence-based requirements → use cases → dashboard).

**Pipeline:** Historical data → cleaning → analysis → prediction → evidence-based requirements → use cases → decision support

| Part | Content | Pts |
|---|---|---|
| 0 | Generate the synthetic dataset (instructor-preparation step) | – |
| I | Dataset acquisition and understanding (Tasks 1–3) | 15 |
| II | Data cleaning and feature engineering (Tasks 4–5) | 20 |
| III | Exploratory data analysis (Task 6) | 15 |
| IV | Data mining — Decision Tree + evaluation (Tasks 7–8) | 25 |
| V | Requirements and use-case modelling (Tasks 9–10) | 15 |
| VI | Decision support and critical analysis (Tasks 11–12) | 10 |

> All data are **synthetic**. No real student, staff or personal information is used.

### Section 0 — Synthetic dataset
* **~420 synthetic transactions** + a few duplicates, per the 300–500 row recommendation.
* Outcomes follow realistic-but-noisy rules (longer lead time, repeat cancellers and social events → higher chance of
  *not* being used), so the data-mining steps have a genuine signal to find, **not** a perfect one.
* **Deliberate problems** are injected for Part II: missing values, exact and near-duplicate transactions, inconsistent
  category spellings (`"computer lab"`, `"Comp Lab "` …) and invalid numbers (negative participants, 48-hour durations …).
* A fixed random seed (sidebar) makes every run reproducible.
""")
    st.success(f"✅ facility_reservations.csv ready → {df_raw.shape[0]} rows × {df_raw.shape[1]} columns "
               f"({'synthetic' if data_source.startswith('Generate') else 'uploaded'})")

# ---------------------------------------------------------------------
# PART I
# ---------------------------------------------------------------------
with tabs[1]:
    st.header("Part I — Dataset Acquisition and Understanding (15 pts)")
    st.subheader("Task 1 (7 pts) — Load the CSV, preview, shape, types, summary statistics")
    st.markdown("**▶ First 10 records**")
    st.dataframe(df_raw.head(10))
    c1, c2 = st.columns(2)
    c1.metric("Rows", df_raw.shape[0])
    c2.metric("Columns", df_raw.shape[1])
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**▶ Variable types**")
        st.dataframe(df_raw.dtypes.astype(str).rename("dtype").to_frame())
    with col_b:
        st.markdown("**▶ Summary statistics — numerical variables**")
        st.dataframe(df_raw.describe().T.round(2))
    st.markdown("**▶ Summary statistics — categorical / text variables**")
    st.dataframe(df_raw.describe(include="object").T)

    st.subheader("Task 2 (5 pts) — Predictors, target, variable roles, and data dictionary")
    data_dictionary = pd.DataFrame([
        ("Reservation_ID",           "Identifier",  "Key (not a predictor)",        "Unique transaction code; used to detect duplicate transactions", "Yes"),
        ("Request_Date",             "Date",        "Support field",                "Date the request was submitted; used to verify / recompute lead time", "Yes"),
        ("Reservation_Date",         "Date",        "Support field → weekday",      "Date the facility is reserved for", "Yes"),
        ("Lead_Time_Days",           "Numerical",   "PREDICTOR (numerical)",        "Days between request date and reservation date", "Yes"),
        ("Facility_Type",            "Categorical", "PREDICTOR (categorical)",      "Type of facility requested", "Yes"),
        ("User_Category",            "Categorical", "PREDICTOR (categorical)",      "Requester group (role only — no personal identity)", "Yes"),
        ("Purpose",                  "Categorical", "PREDICTOR",                    "Activity type of the reservation", "Yes"),
        ("Participants",             "Numerical",   "PREDICTOR (numerical)",        "Expected number of participants declared on the request", "Yes"),
        ("Duration_Hours",           "Numerical",   "PREDICTOR",                    "Requested duration in hours", "Yes"),
        ("Start_Hour",               "Numerical",   "PREDICTOR",                    "Start time on a 24-hour clock", "Yes"),
        ("Requires_Equipment",       "Categorical", "PREDICTOR",                    "Whether AV / sports equipment was requested", "Yes"),
        ("Prior_Cancellations",      "Numerical",   "PREDICTOR",                    "Requester's cancellations recorded BEFORE this request", "Yes"),
        ("Usage_Status",             "Categorical", "Source of the TARGET",         "Used / Cancelled / No-Show — known only after the event date", "No"),
        ("Actual_Attendance",        "Numerical",   "Post-outcome (leakage risk)",  "Head-count recorded at the event; used for Attendance_Ratio", "No"),
        ("Cancellation_Days_Before", "Numerical",   "Post-outcome (leakage risk)",  "Days before the event that a cancellation was filed", "No"),
        ("Reservation_Outcome",      "Categorical", "TARGET (derived in Task 5)",   "Completed vs Not Completed", "No"),
    ], columns=["Variable", "Type", "Role in this study", "Description", "Known at submission?"])
    st.markdown("**Data dictionary**")
    st.dataframe(data_dictionary, hide_index=True)

    st.markdown("""
**Task 2 answers**

| Item | Selection | Why / role |
|---|---|---|
| **Five+ predictor candidates** | `Lead_Time_Days`, `Facility_Type`, `User_Category`, `Participants`, `Duration_Hours` (also `Purpose`, `Start_Hour`, `Requires_Equipment`, `Prior_Cancellations`) | Every one is known **when the request is submitted**, so a model can use them to estimate risk *before* the event happens. |
| **Target variable** | `Reservation_Outcome` (derived: *Completed* vs *Not Completed*) | The thing the administrator wants to anticipate. It is built from `Usage_Status`, which is only known after the event. |
| **Two categorical variables** | `Facility_Type`, `User_Category` | Nominal groupings; they need encoding (one-hot) for the Decision Tree and are the natural "slice by" fields in the dashboard. |
| **Two numerical variables** | `Lead_Time_Days`, `Participants` | Measured quantities; `Lead_Time_Days` is also the source of the derived `Lead_Time_Category`, and `Participants` is the denominator of `Attendance_Ratio`. |
""")

    st.subheader("Task 3 (3 pts) — How the same data could be obtained from the university system")
    st.markdown("**Option A — Database query (SQL).** The reservation database (e.g. MySQL/PostgreSQL) would be queried "
                "with a read-only account. Only role-level fields are pulled — never student names or ID numbers.")
    st.code("""SELECT r.reservation_id, r.request_date, r.reservation_date,
       DATEDIFF(r.reservation_date, r.request_date) AS lead_time_days,
       f.facility_type, u.user_category, r.purpose, r.participants,
       r.duration_hours, r.start_hour, r.requires_equipment, r.prior_cancellations,
       r.usage_status, r.actual_attendance, r.cancellation_days_before
FROM   reservations r
JOIN   facilities  f ON f.facility_id = r.facility_id
JOIN   users       u ON u.user_id     = r.user_id
WHERE  r.reservation_date BETWEEN '2026-02-01' AND '2026-09-30';""", language="sql")
    st.markdown("**Option B — REST API.** The system exposes a JSON endpoint; Python pages through it and flattens the "
                "JSON (illustrative — not executed here because there is no real server):")
    st.code("""import requests, pandas as pd
rows, page = [], 1
while True:
    r = requests.get("https://reservations.university.edu/api/v1/reservations",
                     params={"from": "2026-02-01", "to": "2026-09-30", "page": page},
                     headers={"Authorization": "Bearer <TOKEN>"}, timeout=30)
    r.raise_for_status()
    batch = r.json()["data"]
    if not batch: break
    rows += batch; page += 1
df = pd.json_normalize(rows)""", language="python")
    st.markdown("**Validation check during collection (one required):** *cross-field consistency* — `Lead_Time_Days` must "
                "equal `Reservation_Date − Request_Date` and can never be negative. Records that fail are rejected or flagged "
                "at the source instead of being stored silently. Below: (1) a real database query against an in-memory "
                "SQLite copy of the CSV and (2) the validation check.")

    con = sqlite3.connect(":memory:")
    df_raw.to_sql("reservations", con, index=False, if_exists="replace")
    query = """SELECT Facility_Type, COUNT(*) AS reservations
FROM reservations
WHERE Facility_Type IS NOT NULL
GROUP BY Facility_Type
ORDER BY reservations DESC"""
    st.code(query, language="sql")
    st.markdown("**▶ SQL result** (raw spellings are still inconsistent — cleaning comes in Part II)")
    st.dataframe(pd.read_sql_query(query, con), hide_index=True)
    con.close()

    rd = pd.to_datetime(df_raw["Reservation_Date"], format="%Y-%m-%d", errors="coerce")
    qd = pd.to_datetime(df_raw["Request_Date"], format="%Y-%m-%d", errors="coerce")
    checkable = df_raw["Lead_Time_Days"].notna() & rd.notna() & qd.notna()
    mismatch = checkable & ((df_raw["Lead_Time_Days"] != (rd - qd).dt.days) | (df_raw["Lead_Time_Days"] < 0))
    st.markdown(f"**▶ Validation check:** {int(mismatch.sum())} of {int(checkable.sum())} checkable records violate the lead-time rule")
    st.dataframe(df_raw.loc[mismatch, ["Reservation_ID", "Request_Date", "Reservation_Date", "Lead_Time_Days"]].head(),
                 hide_index=True)

# ---------------------------------------------------------------------
# PART II
# ---------------------------------------------------------------------
with tabs[2]:
    st.header("Part II — Data Cleaning and Feature Engineering (20 pts)")
    st.subheader("Task 4 (10 pts) — Detect and fix missing values, duplicates, inconsistent categories, invalid numbers")
    st.caption("The untouched original is preserved (`df_original`, downloadable in the last tab); all cleaning happens on a separate copy.")

    st.markdown("#### Detect")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**▶ Missing values per column (raw)**")
        miss = pd.DataFrame({"missing": df_raw.isna().sum(), "missing_%": (df_raw.isna().mean() * 100).round(1)})
        st.dataframe(miss[miss["missing"] > 0])
    with d2:
        st.markdown("**▶ Duplicates (raw)**")
        st.write(f"Exact duplicate rows: **{int(df_raw.duplicated().sum())}**")
        st.write(f"Repeated Reservation_IDs: **{int(df_raw['Reservation_ID'].duplicated().sum())}**")
        st.markdown("**▶ Invalid numerical values (raw) — outside the allowed range**")
        inv = {c: int((df_raw[c].notna() & ~df_raw[c].between(lo, hi)).sum()) for c, (lo, hi) in VALID_RANGES.items()}
        inv["Lead_Time_Days (<0)"] = int((df_raw["Lead_Time_Days"] < 0).sum())
        st.dataframe(pd.DataFrame({"allowed range": {**{c: f"{lo} – {hi}" for c, (lo, hi) in VALID_RANGES.items()},
                                                    "Lead_Time_Days (<0)": ">= 0"},
                                   "invalid count": inv}))

    st.markdown("**▶ Inconsistent category spellings (raw) — distinct values per column**")
    for c in ["Facility_Type", "User_Category", "Requires_Equipment", "Usage_Status"]:
        st.write(f"`{c}` — **{df_raw[c].nunique()}** distinct → {sorted(df_raw[c].dropna().unique())}")

    st.markdown("#### Fix")
    st.write("Distinct values after standardising:", cleaning_log["distinct_after"],
             "| unmapped values:", cleaning_log["unmapped_total"])
    st.write(f"Duplicates removed: **{cleaning_log['exact_duplicates_removed']}** exact/near + "
             f"**{cleaning_log['conflicting_id_duplicates_removed']}** same-ID conflicts → **{len(df)}** unique transactions remain")
    st.write(f"Lead_Time_Days recomputed from dates: **{cleaning_log['lead_time_recomputed_from_dates']}** | "
             f"left missing (dates not available): **{cleaning_log['lead_time_left_missing']}**")

    st.markdown("**▶ Missing values AFTER cleaning** (deliberately kept as blanks — see decisions table)")
    miss_after = pd.DataFrame({"missing": pd.Series(cleaning_log["missing_after"])})
    miss_after["missing_%"] = (miss_after["missing"] / len(df) * 100).round(1)
    st.dataframe(miss_after[miss_after["missing"] > 0])

    decisions = pd.DataFrame([
        ("Inconsistent categories", f"{sum(cleaning_log['category_values_standardised'].values())} extra spellings across 4 columns",
         "Trim, lower-case, remove punctuation, map to ONE standard label", "Same real-world value must be counted once, otherwise counts and rates are split across spellings."),
        ("Duplicate transactions", f"{cleaning_log['exact_duplicates_removed']} removed",
         "Drop exact duplicates after standardising; keep first per Reservation_ID", "A transaction is one event; duplicates inflate counts and bias the model."),
        ("Invalid numbers", f"{sum(cleaning_log['invalid_set_to_missing'].values())} values (negative, zero or impossible)",
         "Set to missing (NaN)", "The true value is unknown, so inventing one would be fabricating data."),
        ("Contradicting / missing lead time", f"{cleaning_log['lead_time_recomputed_from_dates']} recomputed, {cleaning_log['lead_time_left_missing']} left blank",
         "Recompute from Request/Reservation dates only when both are valid", "The record itself supports the value; otherwise it stays missing."),
        ("Missing categorical / numeric values", "see table above",
         "Kept blank in the cleaned CSV; imputed only inside the model pipeline (train data only)", "Imputing in the dataset would hide uncertainty and leak test information."),
        ("Missing Usage_Status", f"{int(df['Usage_Status'].isna().sum())} records",
         "Outcome left blank; record excluded from outcome analysis and modelling", "Cannot label a reservation Completed / Not Completed without evidence."),
        ("Missing attendance on 'Used' rows", f"{int((df['Usage_Status'].eq('Used') & df['Actual_Attendance'].isna()).sum())} records",
         "Keep blank → Attendance_Ratio stays blank", "Attendance was never recorded; no value is supported."),
    ], columns=["Issue", "Detected", "Action taken", "Reason"])
    st.markdown("**Cleaning decisions**")
    st.dataframe(decisions, hide_index=True)

    st.subheader("Task 5 (10 pts) — Derived variables")
    st.markdown("""
| Variable | Rule implemented |
|---|---|
| `Lead_Time_Category` | 0–2 days = Last Minute · 3–7 = Short Lead · 8–14 = Medium Lead · >14 = Early Booking |
| `Reservation_Outcome` | *Completed* if `Usage_Status = Used`; *Not Completed* if Cancelled or No-Show; blank if status unknown |
| `Attendance_Ratio` | `Actual_Attendance / Participants`; blank when the denominator is zero / missing or attendance is missing |
""")
    st.write("Lead_Time_Category counts:", df["Lead_Time_Category"].value_counts(dropna=False).to_dict())
    st.write("Reservation_Outcome counts:", df["Reservation_Outcome"].value_counts(dropna=False).to_dict())
    st.write(f"Attendance_Ratio: valid = **{int(df['Attendance_Ratio'].notna().sum())}** | blank = "
             f"**{int(df['Attendance_Ratio'].isna().sum())}** | ratio > 1 (more attended than declared) = "
             f"**{int((df['Attendance_Ratio'] > 1).sum())}**")
    st.markdown("**Cleaned data (first 8 rows)**")
    st.dataframe(df.head(8))

    dq = pd.DataFrame({
        "missing (raw)": df_raw.isna().sum(),
        "invalid → set to missing": pd.Series(cleaning_log["invalid_set_to_missing"]),
        "missing (clean)": df.drop(columns=["Lead_Time_Category", "Reservation_Outcome", "Attendance_Ratio"]).isna().sum(),
    }).fillna(0).astype(int)
    dq = dq[(dq != 0).any(axis=1)]

    st.markdown(f"""
### Data-quality summary
| Measure | Value |
|---|---|
| Rows in raw file | **{len(df_raw)}** |
| Duplicate transactions removed | **{cleaning_log['exact_duplicates_removed'] + cleaning_log['conflicting_id_duplicates_removed']}** |
| Rows in clean file | **{len(df)}** |
| Category spellings merged | **{sum(cleaning_log['category_values_standardised'].values())}** |
| Invalid numeric values set to missing | **{sum(cleaning_log['invalid_set_to_missing'].values())}** |
| Lead times recomputed from dates / left missing | **{cleaning_log['lead_time_recomputed_from_dates']} / {cleaning_log['lead_time_left_missing']}** |
| Records with known outcome (usable for analysis & modelling) | **{int(df['Reservation_Outcome'].notna().sum())}** |
| Records with unknown outcome (excluded, not guessed) | **{int(df['Reservation_Outcome'].isna().sum())}** |
""")
    st.dataframe(dq)

# ---------------------------------------------------------------------
# PART III
# ---------------------------------------------------------------------
known = df[df["Reservation_Outcome"].notna()].copy()
fac = df["Facility_Type"].dropna()
fac_counts = fac.value_counts()
N_fac = len(fac)

u = known.dropna(subset=["User_Category"])
cat_tbl = u.groupby("User_Category").agg(
    total=("Reservation_Outcome", "size"),
    not_completed=("Reservation_Outcome", lambda s: int((s == "Not Completed").sum())),
    cancelled=("Usage_Status", lambda s: int((s == "Cancelled").sum())),
    no_show=("Usage_Status", lambda s: int((s == "No-Show").sum())))
cat_tbl["pct"] = cat_tbl["not_completed"] / cat_tbl["total"] * 100
cat_tbl = cat_tbl.sort_values("pct", ascending=False)
overall_nc, overall_n = int((u["Reservation_Outcome"] == "Not Completed").sum()), len(u)
overall_pct = overall_nc / overall_n * 100

lt = known.dropna(subset=["Lead_Time_Category"])
ct = pd.crosstab(lt["Lead_Time_Category"], lt["Reservation_Outcome"]).reindex(
    index=LEAD_ORDER, columns=["Completed", "Not Completed"]).fillna(0)
lead_tbl = pd.DataFrame({"total": ct.sum(axis=1), "not_completed": ct["Not Completed"]})
lead_tbl["pct"] = lead_tbl["not_completed"] / lead_tbl["total"] * 100
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
x = np.arange(len(LEAD_ORDER))
hi, lo = cat_tbl.index[0], cat_tbl.index[-1]
hi_l, lo_l = lead_tbl["pct"].idxmax(), lead_tbl["pct"].idxmin()

with tabs[3]:
    st.header("Part III — Exploratory Data Analysis (15 pts)")
    st.subheader("Task 6 — Business questions answered with Python")
    st.caption("Three questions, three labelled charts. Every percentage states its denominator. Records with a missing "
               "value in the field being analysed are excluded from that analysis only, and the count is reported.")
    st.info(f"Records in clean data: **{len(df)}** | with a known outcome: **{len(known)}** | "
            f"outcome unknown (excluded from outcome questions): **{len(df) - len(known)}**")

    # ---- Q1
    st.markdown("### Q1 — Which facility type has the highest reservation count?")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(fac_counts.index, fac_counts.values, color="#4C78A8")
    ax.bar_label(bars, labels=[f"{v}\n({v / N_fac:.1%})" for v in fac_counts.values], padding=3, fontsize=9)
    ax.set_title(f"Chart 1 — Reservation Count by Facility Type (N = {N_fac} reservations with a known facility)")
    ax.set_xlabel("Facility type"); ax.set_ylabel("Number of reservations")
    ax.set_ylim(0, fac_counts.max() * 1.22)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right"); fig.tight_layout()
    show_fig(fig)

    top_fac = fac_counts[fac_counts == fac_counts.max()].index.tolist()
    runner = fac_counts.index[len(top_fac)] if len(top_fac) < len(fac_counts) else None
    st.markdown(f"""**Interpretation (Chart 1).** **{' and '.join(top_fac)}** had the highest reservation count with
**{fac_counts.max()} of {N_fac}** reservations (**{fac_counts.max() / N_fac:.1%}** of the {N_fac} records with a known facility type; {len(df) - N_fac} records had no facility and were excluded).
{f"The next most requested was {runner} with {fac_counts[runner]} ({fac_counts[runner] / N_fac:.1%})." if runner else ""}
The least requested was **{fac_counts.index[-1]}** ({fac_counts.iloc[-1]}, {fac_counts.iloc[-1] / N_fac:.1%}).""")

    # ---- Q2
    st.markdown("### Q2 — Which user category has the highest cancellation / no-show percentage?")
    st.dataframe(cat_tbl.rename(columns={"total": "reservations (denominator)", "not_completed": "cancelled + no-show",
                                         "pct": "% not completed"}).round(1))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = ["#E15759" if i == 0 else "#9AA5B1" for i in range(len(cat_tbl))]
    bars = ax.barh(cat_tbl.index[::-1], cat_tbl["pct"][::-1], color=colors[::-1])
    ax.bar_label(bars, labels=[f"{p:.1f}%  ({k}/{n})" for p, k, n in
                               zip(cat_tbl["pct"][::-1], cat_tbl["not_completed"][::-1], cat_tbl["total"][::-1])],
                 padding=3, fontsize=9)
    ax.axvline(overall_pct, color="black", ls="--", lw=1, label=f"Overall rate {overall_pct:.1f}% ({overall_nc}/{overall_n})")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_xlim(0, max(cat_tbl["pct"]) * 1.35)
    ax.set_title("Chart 2 — Cancellation + No-Show Rate by User Category")
    ax.set_xlabel("% of reservations Not Completed  (label = % and not-completed / reservations in that category)")
    fig.tight_layout()
    show_fig(fig)

    small = cat_tbl[cat_tbl["total"] < 30].index.tolist()
    st.markdown(f"""**Interpretation (Chart 2).** **{hi}** had the highest cancellation/no-show rate: **{cat_tbl.loc[hi, 'pct']:.1f}%**
({cat_tbl.loc[hi, 'not_completed']} of {cat_tbl.loc[hi, 'total']} reservations). The lowest was **{lo}** at **{cat_tbl.loc[lo, 'pct']:.1f}%**
({cat_tbl.loc[lo, 'not_completed']} of {cat_tbl.loc[lo, 'total']}). Across all {overall_n} reservations with a known outcome and category, the overall rate is **{overall_pct:.1f}%** ({overall_nc}/{overall_n}).
{"⚠️ Caution: " + ", ".join(small) + " have fewer than 30 reservations, so their percentages are less stable." if small else ""}""")

    # ---- Q3
    st.markdown("### Q3 — How does lead-time category relate to reservation outcome?")
    st.dataframe(ct.astype(int).assign(**{"total (denominator)": lead_tbl["total"].astype(int),
                                          "% not completed": lead_tbl["pct"].round(1)}))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bottom = np.zeros(len(LEAD_ORDER))
    for outcome, colr in [("Completed", "#59A14F"), ("Not Completed", "#E15759")]:
        vals = ct_pct[outcome].to_numpy()
        b = ax.bar(x, vals, bottom=bottom, label=outcome, color=colr)
        ax.bar_label(b, labels=[f"{v:.1f}%" for v in vals], label_type="center", color="white", fontsize=9)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n = {int(lead_tbl.loc[c, 'total'])})" for c in LEAD_ORDER])
    ax.set_ylabel("% of reservations in the lead-time category"); ax.set_ylim(0, 108)
    ax.set_title(f"Chart 3 — Reservation Outcome by Lead-Time Category (N = {int(lead_tbl['total'].sum())})")
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    show_fig(fig)

    seq = lead_tbl["pct"].to_numpy()
    trend = ("rises steadily as lead time gets longer" if np.all(np.diff(seq) > 0) else
             "falls steadily as lead time gets longer" if np.all(np.diff(seq) < 0) else
             "does not change in one consistent direction across the categories")
    st.markdown(f"""**Interpretation (Chart 3).** The Not-Completed rate {trend}: it ranges from **{lead_tbl.loc[lo_l, 'pct']:.1f}%** for
**{lo_l}** ({int(lead_tbl.loc[lo_l, 'not_completed'])}/{int(lead_tbl.loc[lo_l, 'total'])}) to **{lead_tbl.loc[hi_l, 'pct']:.1f}%** for **{hi_l}**
({int(lead_tbl.loc[hi_l, 'not_completed'])}/{int(lead_tbl.loc[hi_l, 'total'])}). Denominator: {int(lead_tbl['total'].sum())} reservations with both a known outcome and a lead time
({len(df) - int(lead_tbl['total'].sum())} excluded because one of the two was missing). This is an association in historical data, not proof that lead time *causes* cancellation.""")

# ---------------------------------------------------------------------
# PART IV — modelling (computed outside the tab so later tabs can use results)
# ---------------------------------------------------------------------
FEATURES_CAT = ["Facility_Type", "User_Category", "Purpose", "Requires_Equipment", "Reservation_Weekday"]
FEATURES_NUM = ["Lead_Time_Days", "Participants", "Duration_Hours", "Start_Hour", "Prior_Cancellations"]
LEAKY = ["Usage_Status", "Reservation_Outcome", "Actual_Attendance", "Attendance_Ratio",
         "Cancellation_Days_Before", "Reservation_ID"]
assert not set(LEAKY) & set(FEATURES_CAT + FEATURES_NUM), "Leakage: a post-outcome column is in the predictor list!"

model_df = df[df["Reservation_Outcome"].notna()].copy()
model_df["Reservation_Weekday"] = model_df["Reservation_Date"].dt.day_name().astype(object)
X = model_df[FEATURES_CAT + FEATURES_NUM].copy()
for c in FEATURES_CAT:
    X[c] = X[c].astype(object)
y = (model_df["Reservation_Outcome"] == "Not Completed").astype(int)

if y.nunique() < 2 or len(X) < 20:
    st.error("Not enough labelled records (or only one outcome class) to train a model with this data.")
    st.stop()

can_stratify = y.value_counts().min() >= 2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if can_stratify else None)

preprocess = ColumnTransformer([
    ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), FEATURES_CAT),
    ("num", SimpleImputer(strategy="median"), FEATURES_NUM),
])
clf_pipe = Pipeline([
    ("pre", preprocess),
    ("tree", DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, class_weight="balanced",
                                    random_state=RANDOM_STATE)),
])
clf_pipe.fit(X_train, y_train)

y_pred = clf_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

# Baseline, AUC, cross-validation
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
base_acc = accuracy_score(y_test, base.predict(X_test))
base_rec = recall_score(y_test, base.predict(X_test), pos_label=1, zero_division=0)
auc = roc_auc_score(y_test, clf_pipe.predict_proba(X_test)[:, 1]) if y_test.nunique() == 2 else float("nan")

n_splits = int(min(5, y.value_counts().min()))
cv_ok = n_splits >= 2
if cv_ok:
    cv = cross_validate(clf_pipe, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE),
                        scoring=["accuracy", "precision", "recall", "f1"])
    cv_f1_m, cv_f1_s = cv["test_f1"].mean(), cv["test_f1"].std()
    cv_rec_m = cv["test_recall"].mean()
    cv_acc_m = cv["test_accuracy"].mean()
else:
    cv_f1_m = cv_f1_s = cv_rec_m = cv_acc_m = float("nan")

criteria = {
    "Test F1 ≥ 0.60": f1 >= 0.60,
    "Test recall ≥ 0.60 (catches most at-risk bookings)": rec >= 0.60,
    "Test ROC-AUC ≥ 0.65 (ranks at-risk bookings better than chance)": bool(auc >= 0.65),
    "CV F1 ≥ 0.55 and stable (std ≤ 0.10)": bool(cv_ok and cv_f1_m >= 0.55 and cv_f1_s <= 0.10),
}
advisory_ok = all(criteria.values())

# Top-3 predictors (aggregate one-hot columns back to their original feature)
_imp = pd.Series(clf_pipe.named_steps["tree"].feature_importances_,
                 index=clf_pipe.named_steps["pre"].get_feature_names_out())
_agg = {}
for _n, _v in _imp.items():
    _base = (next(c for c in FEATURES_CAT if _n[5:].startswith(c + "_")) if _n.startswith("cat__") else _n[5:])
    _agg[_base] = _agg.get(_base, 0) + _v
feat_importance = pd.Series(_agg).sort_values(ascending=False)
top3 = feat_importance.head(3).index.tolist()

with tabs[4]:
    st.header("Part IV — Data Mining and Predictive Modelling (25 pts)")
    st.subheader("Task 7 (20 pts) — Decision Tree classifier for `Reservation_Outcome`")
    st.markdown("""
**Goal:** at the moment a reservation is *submitted*, estimate whether it will end as **Not Completed**
(cancelled or no-show) — the "positive" class an administrator cares about.

**Leakage control (critical rule).** A predictor is allowed only if it is known **at submission time**. The code uses an
explicit allow-list and an `assert` that blocks the forbidden columns:

| Excluded column | Why it would leak the answer |
|---|---|
| `Usage_Status` | The target is derived directly from it |
| `Reservation_Outcome` | It *is* the target |
| `Actual_Attendance` | Recorded only after the event happens |
| `Attendance_Ratio` | Computed from `Actual_Attendance` |
| `Cancellation_Days_Before` | Exists only for reservations that were already cancelled |
| `Reservation_ID` | Identifier, carries no behaviour |

`Prior_Cancellations` counts the requester's cancellations **before** the current request, so it is legitimately known at submission.
""")
    st.success("✅ Leakage check passed. Predictors used: " + ", ".join(FEATURES_CAT + FEATURES_NUM))
    st.write(f"Modelling records: **{len(X)}** | Not Completed = **{int(y.sum())}** ({y.mean():.1%}) | Completed = **{int((1 - y).sum())}**")
    st.write(f"Train: **{len(X_train)}** rows | Test: **{len(X_test)}** rows | classes in test set: {sorted(y_test.unique().tolist())}")
    st.write(f"✅ Decision Tree trained (max_depth={MAX_DEPTH}, min_samples_leaf={MIN_LEAF}, class_weight='balanced', random_state={RANDOM_STATE})")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1-score", f"{f1:.3f}")

    cc1, cc2 = st.columns([1, 1.4])
    with cc1:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ConfusionMatrixDisplay(cm, display_labels=["Completed", "Not Completed"]).plot(
            ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title("Confusion Matrix — test set"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.tight_layout()
        show_fig(fig)
    with cc2:
        st.markdown(f"""**Test-set metrics (positive class = *Not Completed*, n = {len(y_test)})**

| Metric | Value | Meaning |
|---|---|---|
| Accuracy | **{acc:.3f}** | {tn + tp} of {len(y_test)} test reservations classified correctly |
| Precision | **{prec:.3f}** | Of {tp + fp} reservations flagged Not Completed, {tp} really were |
| Recall | **{rec:.3f}** | Of {tp + fn} reservations that really were Not Completed, {tp} were caught |
| F1-score | **{f1:.3f}** | Harmonic mean of precision and recall |

Confusion matrix cells → TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}.""")

    notes = []
    if y_test.nunique() < 2:
        notes.append("The test split contains only ONE class, so precision/recall/F1 are not meaningful. They are reported as 0.0 (zero_division=0) and the split should be redrawn or more data collected.")
    if tp + fp == 0:
        notes.append("Precision is UNDEFINED (0/0) because the model predicted no 'Not Completed' cases; it is reported as 0.0.")
    if tp + fn == 0:
        notes.append("Recall is UNDEFINED (0/0) because the test set has no actual 'Not Completed' cases; it is reported as 0.0.")
    st.markdown("**Undefined-metric handling:** " + (" ".join(notes) if notes else
                "none needed — both classes appear in the test set and the model made both kinds of prediction, so every metric is defined."))
    st.code(classification_report(y_test, y_pred, labels=[0, 1], target_names=["Completed", "Not Completed"], zero_division=0))

    st.markdown("**Feature importance** (one-hot columns summed back to the original predictor)")
    fi1, fi2 = st.columns(2)
    with fi1:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        top_imp = feat_importance[feat_importance > 0].head(8)[::-1]
        ax.barh(top_imp.index, top_imp.values, color="#4C78A8")
        ax.set_xlabel("Importance"); fig.tight_layout()
        show_fig(fig)
    with fi2:
        with st.expander("Show the fitted tree"):
            fig, ax = plt.subplots(figsize=(14, 6))
            plot_tree(clf_pipe.named_steps["tree"],
                      feature_names=clf_pipe.named_steps["pre"].get_feature_names_out(),
                      class_names=["Completed", "Not Completed"], filled=True, fontsize=6, ax=ax, max_depth=3)
            show_fig(fig)

    st.subheader("Task 8 (5 pts) — Is the model reliable enough for administrator decision support?")
    st.caption("The test set is small, so one split can be lucky or unlucky. The checks below add a majority-class "
               "baseline and k-fold cross-validation before judging the model.")
    st.markdown(f"""| Check | Result |
|---|---|
| Majority-class baseline (always predicts "Completed") | accuracy **{base_acc:.3f}**, but recall **{base_rec:.3f}** — it never flags an at-risk booking, so accuracy alone is misleading on imbalanced classes |
| Decision Tree test accuracy | **{acc:.3f}** |
| Test ROC-AUC (0.5 = guessing) | **{auc:.3f}** |
| Test precision / recall / F1 (Not Completed) | **{prec:.3f} / {rec:.3f} / {f1:.3f}** |
| {n_splits}-fold CV accuracy (mean) | **{cv_acc_m:.3f}** |
| {n_splits}-fold CV recall (mean) | **{cv_rec_m:.3f}** |
| {n_splits}-fold CV F1 (mean ± std) | **{cv_f1_m:.3f} ± {cv_f1_s:.3f}** |
| Test set size / actual Not-Completed cases in it | **{len(y_test)} / {int(y_test.sum())}** |""")

    st.markdown("**Reliability criteria (advisory use):**\n\n" +
                "\n".join(f"- {'✅' if v else '❌'} {k}" for k, v in criteria.items()))

    verdict = ("**Conditionally reliable — as an ADVISORY risk flag only.** The model clears the minimum bar and is clearly "
               "better than guessing the majority class, so it can help an administrator decide where to send reminders "
               "or double-check bookings." if advisory_ok else
               "**Not yet reliable enough** — at least one criterion above failed, so the model should be shown to "
               "administrators only as an experimental hint, if at all.")
    st.markdown(f"""### Verdict
{verdict}
Either way, it is **not** reliable enough to approve or reject reservations automatically: a wrong 'high risk' label would deny a legitimate booking, and a wrong 'low risk' label gives false reassurance.

**One limitation:** the training data are small (**{len(X_train)} training rows**, only **{int(y_test.sum())} Not-Completed cases in the test set**), so every metric has a wide margin of error, and a single Decision Tree is unstable — a few different rows can change its splits. The data are also synthetic, so real behaviour (exam weeks, weather, events) is not captured.

**One improvement:** collect at least a full semester of real, de-identified reservations and replace the single tree with a cross-validated, tuned Random Forest or gradient-boosted model (with probability calibration), then re-evaluate with stratified cross-validation and monitor accuracy as new data arrive.""")

# ---------------------------------------------------------------------
# PART V — requirements + use-case diagram
# ---------------------------------------------------------------------
with tabs[5]:
    st.header("Part V — Requirements Analysis and Use Case Modelling (15 pts)")
    st.subheader("Task 9 (8 pts) — Three functional requirements grounded in the actual findings")
    st.caption("The Finding / Evidence column is filled from the numbers computed in Parts III and IV of this run.")

    ev1 = (f"Chart 3 (Part III): {hi_l} reservations were Not Completed {lead_tbl.loc[hi_l, 'pct']:.1f}% of the time "
           f"({int(lead_tbl.loc[hi_l, 'not_completed'])}/{int(lead_tbl.loc[hi_l, 'total'])}) vs {lead_tbl.loc[lo_l, 'pct']:.1f}% for {lo_l} "
           f"({int(lead_tbl.loc[lo_l, 'not_completed'])}/{int(lead_tbl.loc[lo_l, 'total'])}).")
    ev2 = (f"Chart 2 (Part III): {hi} had the highest cancellation/no-show rate, {cat_tbl.loc[hi, 'pct']:.1f}% "
           f"({cat_tbl.loc[hi, 'not_completed']}/{cat_tbl.loc[hi, 'total']}), against an overall {overall_pct:.1f}% ({overall_nc}/{overall_n}).")
    ev3 = (f"Task 7 (Part IV): the Decision Tree's top predictors were {', '.join(top3)}; test F1 = {f1:.2f}, recall = {rec:.2f} "
           f"(n = {len(y_test)}); verdict: {'advisory use only' if advisory_ok else 'experimental only'}.")

    requirements = pd.DataFrame([
        (ev1, "FR-DM-01",
         f"The system shall flag every reservation request whose lead-time category is '{hi_l}' on the administrator's review screen and shall send the requester a confirmation reminder before the reservation date."),
        (ev2, "FR-DM-02",
         "The system shall calculate the cancellation/no-show rate, with the number of reservations used, for each user category and shall highlight on the administrator dashboard any category whose rate is above the overall rate."),
        (ev3, "FR-DM-03",
         f"The system shall display a predicted risk level (Low / Medium / High) and its main contributing factors (such as {', '.join(top3)}) for each pending reservation as advisory information, and shall not approve or reject a reservation automatically on the basis of the prediction."),
    ], columns=["Finding / Evidence", "Requirement ID", "Functional Requirement"])
    st.dataframe(requirements, hide_index=True)

    st.subheader("Task 10 (7 pts) — Updated use case diagram")
    st.markdown("""
The diagram below is drawn in the app and is also available as a **`.drawio`** file (open at <https://app.diagrams.net>,
*File ▸ Open from ▸ Device*) and a PNG in the **Downloads** tab.

**System boundary:** *Campus Facility Reservation and Monitoring System*

| Element | Item | Status |
|---|---|---|
| **Actor** | Requester (Student / Faculty / Staff) | existing |
| **Actor** | Facility Administrator | existing |
| Use case | Submit Reservation Request · Check Facility Availability · Cancel Reservation · Review Reservation Request · Record Facility Usage | existing |
| Use case | **View Analytics Dashboard** | **new — analytical** |
| Use case | **View Cancellation Risk Prediction** | **new — analytical** |
| Use case | **Show Risk Explanation** | **new — analytical** |

**Associations:** Requester — Submit Reservation Request, Cancel Reservation · Facility Administrator — Review Reservation Request, Record Facility Usage, View Analytics Dashboard.

**Justification of the new analytical functions**
* **View Analytics Dashboard** — supports FR-DM-02: administrators need the observed cancellation/no-show rates per category (Part III evidence).
* **View Cancellation Risk Prediction** — supports FR-DM-01 and FR-DM-03: the Part IV model produces a risk level at submission time.

**Include / extend — used only where semantically justified**
* `Submit Reservation Request` **«include»** `Check Facility Availability` — availability is *always* checked as part of submitting, so the behaviour is mandatory.
* `View Cancellation Risk Prediction` **«extend»** `Review Reservation Request` — the review is complete without a prediction; the risk view is *optional, advisory* behaviour the administrator may trigger (the model is only conditionally reliable — Task 8).
* `View Cancellation Risk Prediction` **«include»** `Show Risk Explanation` — a risk level must never be shown without its reasons (Task 12c), so the explanation is mandatory whenever the prediction is shown.
""")
    uc_fig, drawio_xml, uc_png = build_use_case_diagram()
    show_fig(uc_fig)

# ---------------------------------------------------------------------
# PART VI — decision support
# ---------------------------------------------------------------------
def risk_level(p, low=RISK_LOW, high=RISK_HIGH):
    return "Low" if p < low else ("Medium" if p < high else "High")


proba = clf_pipe.predict_proba(X_test)
p_not = proba[:, list(clf_pipe.classes_).index(1)] if 1 in clf_pipe.classes_ else np.zeros(len(X_test))
pred_tbl = pd.DataFrame({"Reservation_ID": model_df.loc[X_test.index, "Reservation_ID"],
                         "Facility_Type": X_test["Facility_Type"], "User_Category": X_test["User_Category"],
                         "p_not_completed": p_not})
pred_tbl["Risk"] = pred_tbl["p_not_completed"].map(risk_level)


def explain_prediction(pipe, row, max_items=4, detail=True):
    pre_, tree_ = pipe.named_steps["pre"], pipe.named_steps["tree"]
    Xt = pre_.transform(row)
    dense = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    names = pre_.get_feature_names_out()
    out = []
    for node in tree_.decision_path(Xt).indices:
        f, thr = tree_.tree_.feature[node], tree_.tree_.threshold[node]
        if f < 0:
            continue
        n, val = names[f], dense[0, f]
        if n.startswith("cat__"):
            n = n[5:]
            feat = next(c for c in FEATURES_CAT if n.startswith(c + "_"))
            level = n[len(feat) + 1:]
            out.append(f"{feat} = {level}" if val > thr else f"{feat} is not {level}")
        else:
            out.append(f"{n[5:]} = {val:g} ({'≤' if val <= thr else '>'} {thr:.1f})" if detail else f"{n[5:]} = {val:g}")
    return out[:max_items]


top_rows = pred_tbl.sort_values("p_not_completed", ascending=False).head(6).copy()
top_rows["Why_top_factors"] = [", ".join(explain_prediction(clf_pipe, X_test.loc[[i]], 2, detail=False)) for i in top_rows.index]

OBS, PRED = "#2F6DB5", "#D9611E"


def frame_ax(ax, color, dashed=False):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(2.2); sp.set_linestyle("--" if dashed else "-")


def tag(ax, text, color):
    ax.set_title(text, loc="left", fontsize=9.5, fontweight="bold", color="white", backgroundcolor=color, pad=8)


def build_dashboard():
    fig = plt.figure(figsize=(17, 11.5))
    gs = fig.add_gridspec(3, 12, height_ratios=[0.85, 2.3, 2.5], hspace=0.62, wspace=1.6, top=0.93, bottom=0.07)
    fig.suptitle("Campus Facility Reservation — Administrator Dashboard (wireframe)", fontsize=16, fontweight="bold", y=0.985)

    n_total, n_known = len(df), len(known)
    n_comp = int((known["Reservation_Outcome"] == "Completed").sum())
    kpis = [("Total reservations", f"{n_total}", "unique transactions · OBSERVED", OBS, False),
            ("Completion rate", f"{n_comp / n_known:.1%}", f"{n_comp} of {n_known} with known outcome · OBSERVED", OBS, False),
            ("Cancellation + no-show rate", f"{(n_known - n_comp) / n_known:.1%}", f"{n_known - n_comp} of {n_known} with known outcome · OBSERVED", OBS, False),
            ("High-risk reservations (test set)", f"{int((pred_tbl['Risk'] == 'High').sum())}", f"of {len(pred_tbl)} scored · PREDICTED", PRED, True)]
    for i, (title, big, sub, colr, dashed) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i * 3:(i + 1) * 3]); frame_ax(ax, colr, dashed)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.80, title, ha="center", va="center", fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.45, big, ha="center", va="center", fontsize=24, fontweight="bold", color=colr, transform=ax.transAxes)
        ax.text(0.5, 0.12, sub, ha="center", va="center", fontsize=7.5, transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, 0:4]); frame_ax(ax, OBS)
    ax.bar(fac_counts.index, fac_counts.values, color=OBS)
    tag(ax, "OBSERVED — Reservations by facility type", OBS)
    ax.set_ylabel("Reservation count"); ax.tick_params(axis="x", rotation=30, labelsize=8)

    ax = fig.add_subplot(gs[1, 4:8]); frame_ax(ax, OBS)
    ax.barh(cat_tbl.index[::-1], cat_tbl["pct"][::-1], color=OBS)
    tag(ax, "OBSERVED — Cancellation / no-show % by user category", OBS)
    ax.set_xlabel("% Not Completed (of that category's reservations)"); ax.tick_params(axis="y", labelsize=8)

    ax = fig.add_subplot(gs[1, 8:12]); frame_ax(ax, OBS)
    ax.bar(x, ct_pct["Completed"], color="#59A14F", label="Completed")
    ax.bar(x, ct_pct["Not Completed"], bottom=ct_pct["Completed"], color="#E15759", label="Not Completed")
    ax.set_xticks(x); ax.set_xticklabels(LEAD_ORDER, rotation=20, fontsize=8)
    tag(ax, "OBSERVED — Outcome by lead-time category", OBS)
    ax.set_ylabel("% of reservations"); ax.legend(fontsize=7, frameon=False, loc="lower right")

    ax = fig.add_subplot(gs[2, 0:4]); frame_ax(ax, PRED, dashed=True)
    lv = pred_tbl["Risk"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    b = ax.bar(lv.index, lv.values, color=["#8CC084", "#F2C14E", "#D9534F"], hatch="//", edgecolor="white")
    ax.bar_label(b, fontsize=9)
    tag(ax, "PREDICTED — Risk level of scored reservations", PRED)
    ax.set_ylabel("Number of reservations")
    ax.set_xlabel(f"Predicted risk (Low <{RISK_LOW:.0%} · Medium {RISK_LOW:.0%}–{RISK_HIGH:.0%} · High ≥{RISK_HIGH:.0%})")

    ax = fig.add_subplot(gs[2, 4:12]); frame_ax(ax, PRED, dashed=True)
    ax.set_xticks([]); ax.set_yticks([])
    tag(ax, "PREDICTED — Highest-risk reservations (advisory; administrator decides)", PRED)
    cell = [[r.Reservation_ID, r.Facility_Type if isinstance(r.Facility_Type, str) else "n/a",
             r.User_Category if isinstance(r.User_Category, str) else "n/a",
             f"{r.p_not_completed:.2f} ({r.Risk})", r.Why_top_factors[:62]] for r in top_rows.itertuples()]
    tbl = ax.table(cellText=cell, colLabels=["Reservation", "Facility", "User category", "Risk score (level)", "Why (top factors)"],
                   colWidths=[0.12, 0.12, 0.16, 0.13, 0.47], cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.9)

    fig.text(0.5, 0.012, "Legend:  solid blue frame = OBSERVED fact counted from records   |   dashed orange frame = PREDICTED estimate from the Decision Tree (not a fact; shown with reasons; never auto-rejects).",
             ha="center", fontsize=9)
    return fig


with tabs[6]:
    st.header("Part VI — Decision Support and Critical Analysis (10 pts)")
    st.subheader("Task 11 (5 pts) — Facility administrator dashboard")
    st.markdown("""
**Design rule:** **observed facts** (counted from the records) use **solid blue** frames; **model predictions** use
**dashed orange** frames and the word *PREDICTED*. Predictions here are for the **held-out test reservations**, i.e. cases
the model never saw during training. The *risk score* is the class-weighted share of similar past bookings that were
Not Completed — it is **not a calibrated probability**, which is one more reason to present it as advisory.
""")
    dash_fig = build_dashboard()
    buf = io.BytesIO()
    dash_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    dash_png = buf.getvalue()
    show_fig(dash_fig)

    st.markdown("#### 🔎 Explore a scored reservation")
    ordered = pred_tbl.sort_values("p_not_completed", ascending=False)
    choice = st.selectbox("Reservation (sorted from highest to lowest predicted risk)", ordered["Reservation_ID"].tolist())
    sel = ordered[ordered["Reservation_ID"] == choice].iloc[0]
    sel_idx = ordered[ordered["Reservation_ID"] == choice].index[0]
    st.write(f"**PREDICTED** risk: **{sel['Risk']}** — risk score **{sel['p_not_completed']:.2f}**")
    st.write("Main factors: " + "; ".join(explain_prediction(clf_pipe, X_test.loc[[sel_idx]], 5)))

    st.subheader("Task 12 (5 pts) — Brief written answers")
    st.markdown("""
**(a) Why should a predicted high risk not automatically reject a reservation?**
A prediction is a probability learned from past behaviour, not a certain fact about this requester. Even a "High" label is wrong for some people (false positives), and automatic rejection would punish a legitimate user for something they have not done, with no way to be heard. The model is also only advisory-grade (Task 8), it learns historical patterns that may be unfair to a group (for example a user category), and rejecting a booking blocks a facility that could be used. The correct use is to let a **human administrator** decide, and to prefer lighter actions: a confirmation reminder, a deposit-free waitlist, or allowing overbooking of the slot.

**(b) How does poor data quality affect the model and the information system?**
*Model:* duplicates over-weight some behaviour, inconsistent spellings split one category into many (the tree sees "Comp Lab" and "Computer Lab" as different), and invalid or missing values distort the splits or force guesses, so accuracy drops and the importance ranking can mislead. *Information system:* statistics on the dashboard (counts, rates) become wrong, administrators lose trust, and decisions (room allocation, penalties) are made on false numbers. This is why the validation check at collection time (Task 3) and the cleaning rules (Task 4) matter, and why this app reports exactly how many records were removed or left blank.

**(c) What information should be displayed to explain a high-risk prediction?**
* The **risk level and probability**, labelled *PREDICTED*;
* the **main contributing factors** in plain language (the decision-tree path, shown below for a real example);
* the **observed historical rate** for the same group/lead time **with its sample size** (e.g. "x of n"), to separate fact from estimate;
* the **model's reliability** (test F1/recall, date last trained) and any **data-quality warnings** (e.g. "participants missing");
* the **suggested low-impact action** and a clear note that the administrator makes the final decision.
""")
    best = ordered.iloc[0]
    rid = ordered.index[0]
    factors = explain_prediction(clf_pipe, X_test.loc[[rid]], 5)
    grp = known[known["Facility_Type"] == best["Facility_Type"]]
    st.markdown(f"""### Example explanation card — {best['Reservation_ID']}
| Field | Value |
|---|---|
| **PREDICTED** risk | **{best['Risk']}** — risk score = **{best['p_not_completed']:.2f}** (0 = safe, 1 = very risky; not a calibrated probability) |
| Main factors (decision path) | {'; '.join(factors)} |
| **OBSERVED** history of the same facility type ({best['Facility_Type']}) | {int((grp['Reservation_Outcome'] == 'Not Completed').sum())} of {len(grp)} reservations were Not Completed ({(grp['Reservation_Outcome'] == 'Not Completed').mean():.1%}) |
| Model reliability | Test F1 = {f1:.2f}, recall = {rec:.2f} (n = {len(y_test)}); {'advisory use only' if advisory_ok else 'experimental only'} |
| Suggested action | Send a confirmation reminder; administrator decides — **no automatic rejection** |""")

# ---------------------------------------------------------------------
# DOWNLOADS
# ---------------------------------------------------------------------
with tabs[7]:
    st.header("⬇️ Deliverables")
    st.caption("Files the notebook wrote to disk are offered here as downloads instead.")
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button("facility_reservations.csv (raw)", csv_text.encode("utf-8"),
                           "facility_reservations.csv", "text/csv")
        st.download_button("facility_reservations_original_copy.csv", csv_text.encode("utf-8"),
                           "facility_reservations_original_copy.csv", "text/csv")
        st.download_button("facility_reservations_clean.csv", to_csv_bytes(df),
                           "facility_reservations_clean.csv", "text/csv")
    with dc2:
        st.download_button("facility_reservation_usecase.drawio", drawio_xml.encode("utf-8"),
                           "facility_reservation_usecase.drawio", "application/xml")
        st.download_button("facility_reservation_usecase.png", uc_png,
                           "facility_reservation_usecase.png", "image/png")
        st.download_button("admin_dashboard.png", dash_png, "admin_dashboard.png", "image/png")
