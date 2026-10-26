"""Part II — cleaning rules and derived variables (Tasks 4–5)."""
import re
import numpy as np
import pandas as pd

REQUIRED_COLS = ["Reservation_ID", "Request_Date", "Reservation_Date", "Lead_Time_Days", "Facility_Type",
                 "User_Category", "Purpose", "Participants", "Duration_Hours", "Start_Hour",
                 "Requires_Equipment", "Prior_Cancellations", "Usage_Status", "Actual_Attendance",
                 "Cancellation_Days_Before"]


LEAD_ORDER = ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"]
VALID_RANGES = {"Participants": (1, 500), "Duration_Hours": (0.5, 12), "Start_Hour": (6, 22),
                "Actual_Attendance": (0, 600), "Prior_Cancellations": (0, 20)}
MAPS = {
    "Facility_Type": {"classroom": "Classroom", "class room": "Classroom", "computer lab": "Computer Lab",
                      "comp lab": "Computer Lab", "auditorium": "Auditorium", "gymnasium": "Gymnasium",
                      "gym": "Gymnasium", "conference room": "Conference Room", "conf room": "Conference Room",
                      "sports field": "Sports Field", "field": "Sports Field"},
    "User_Category": {"student": "Student", "student organization": "Student Organization",
                      "student org": "Student Organization", "faculty": "Faculty",
                      "administrative staff": "Administrative Staff", "admin staff": "Administrative Staff",
                      "staff": "Administrative Staff", "external guest": "External Guest", "external": "External Guest"},
    "Purpose": {"academic class": "Academic Class", "meeting": "Meeting", "seminar/workshop": "Seminar/Workshop",
                "sports activity": "Sports Activity", "social event": "Social Event"},
    "Requires_Equipment": {"yes": "Yes", "y": "Yes", "no": "No", "n": "No"},
    "Usage_Status": {"used": "Used", "cancelled": "Cancelled", "canceled": "Cancelled",
                     "no show": "No-Show", "noshow": "No-Show"},
}


def norm_key(v):
    if pd.isna(v):
        return np.nan
    v = re.sub(r"[._\-]+", " ", str(v)).strip().lower()
    return re.sub(r"\s+", " ", v)


def clean_data(df_raw):
    df = df_raw.copy(deep=True)
    log = {}

    # FIX 1: standardise categories
    unmapped_total = 0
    for col, mapping in MAPS.items():
        keys = df[col].map(norm_key)
        fixed = keys.map(mapping)
        unmapped_total += int((keys.notna() & fixed.isna()).sum())
        df[col] = fixed.astype(object)
    log["category_values_standardised"] = {c: int(df_raw[c].nunique() - df[c].nunique()) for c in MAPS}
    log["distinct_after"] = {c: int(df[c].nunique()) for c in MAPS}
    log["unmapped_total"] = unmapped_total

    # FIX 2: dates
    for c in ["Request_Date", "Reservation_Date"]:
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")

    # FIX 3: duplicates
    n0 = len(df)
    df = df.drop_duplicates()
    log["exact_duplicates_removed"] = n0 - len(df)
    n1 = len(df)
    df = df.drop_duplicates(subset="Reservation_ID", keep="first")
    log["conflicting_id_duplicates_removed"] = n1 - len(df)
    df = df.reset_index(drop=True)

    # FIX 4: invalid numbers -> NaN
    log["invalid_set_to_missing"] = {}
    for col, (lo, hi) in VALID_RANGES.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        log["invalid_set_to_missing"][col] = int(bad.sum())
        df.loc[bad, col] = np.nan

    # FIX 5: Lead_Time_Days
    derived_lead = (df["Reservation_Date"] - df["Request_Date"]).dt.days
    bad_lead = df["Lead_Time_Days"].isna() | (df["Lead_Time_Days"] < 0)
    recoverable = bad_lead & derived_lead.notna() & (derived_lead >= 0)
    df.loc[recoverable, "Lead_Time_Days"] = derived_lead[recoverable]
    df.loc[bad_lead & ~recoverable, "Lead_Time_Days"] = np.nan
    log["lead_time_recomputed_from_dates"] = int(recoverable.sum())
    log["lead_time_left_missing"] = int(df["Lead_Time_Days"].isna().sum())

    # Missing counts after the fixes, before derived columns are added
    log["missing_after"] = df.isna().sum().to_dict()

    # Task 5 — derived variables
    df["Lead_Time_Category"] = pd.cut(df["Lead_Time_Days"], bins=[-np.inf, 2, 7, 14, np.inf],
                                      labels=LEAD_ORDER).astype(object)
    df["Reservation_Outcome"] = df["Usage_Status"].map({"Used": "Completed", "Cancelled": "Not Completed",
                                                        "No-Show": "Not Completed"})
    safe_den = df["Participants"].where(df["Participants"] > 0)
    df["Attendance_Ratio"] = (df["Actual_Attendance"] / safe_den).round(3)
    return df, log
