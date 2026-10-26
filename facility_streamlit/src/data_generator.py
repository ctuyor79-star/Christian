"""Section 0 — synthetic, intentionally messy facility reservation data."""
import numpy as np
import pandas as pd


def generate_facility_reservations(n=420, seed=42):
    """Create a synthetic, intentionally messy reservation log."""
    rng = np.random.default_rng(seed)

    facilities = ["Classroom", "Computer Lab", "Auditorium", "Gymnasium", "Conference Room", "Sports Field"]
    capacity = {"Classroom": 40, "Computer Lab": 35, "Auditorium": 500, "Gymnasium": 300,
                "Conference Room": 25, "Sports Field": 150}
    fac = rng.choice(facilities, size=n, p=[0.27, 0.21, 0.10, 0.12, 0.20, 0.10])
    users = rng.choice(["Student", "Student Organization", "Faculty", "Administrative Staff", "External Guest"],
                       size=n, p=[0.24, 0.29, 0.24, 0.15, 0.08])
    purpose_by_fac = {
        "Classroom":       (["Academic Class", "Seminar/Workshop", "Meeting"], [0.65, 0.20, 0.15]),
        "Computer Lab":    (["Academic Class", "Seminar/Workshop", "Meeting"], [0.70, 0.25, 0.05]),
        "Auditorium":      (["Seminar/Workshop", "Social Event", "Meeting"],   [0.45, 0.40, 0.15]),
        "Gymnasium":       (["Sports Activity", "Social Event", "Seminar/Workshop"], [0.65, 0.25, 0.10]),
        "Conference Room": (["Meeting", "Seminar/Workshop", "Academic Class"], [0.75, 0.15, 0.10]),
        "Sports Field":    (["Sports Activity", "Social Event"], [0.85, 0.15]),
    }
    purpose = np.array([rng.choice(purpose_by_fac[f][0], p=purpose_by_fac[f][1]) for f in fac])
    cap = np.array([capacity[f] for f in fac])
    participants = np.clip(rng.normal(0.6 * cap, 0.2 * cap), 5, cap).round().astype(int)
    duration = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0], size=n, p=[0.20, 0.15, 0.25, 0.15, 0.15, 0.10])
    start_hour = rng.integers(8, 19, size=n)

    bucket = rng.choice(4, size=n, p=[0.20, 0.30, 0.25, 0.25])
    lead = np.where(bucket == 0, rng.integers(0, 3, n),
           np.where(bucket == 1, rng.integers(3, 8, n),
           np.where(bucket == 2, rng.integers(8, 15, n), rng.integers(15, 46, n))))
    res_date = pd.Timestamp("2026-02-02") + pd.to_timedelta(rng.integers(0, 240, n), unit="D")
    req_date = res_date - pd.to_timedelta(lead, unit="D")
    is_weekend = np.asarray(res_date.dayofweek >= 5)

    lam = pd.Series(users).map({"Student": 1.0, "Student Organization": 1.2, "Faculty": 0.3,
                                "Administrative Staff": 0.3, "External Guest": 0.8}).to_numpy()
    prior_cancel = np.minimum(rng.poisson(lam), 6)
    equip = np.where(rng.random(n) < np.where(np.isin(purpose, ["Seminar/Workshop", "Social Event"]), 0.65, 0.30),
                     "Yes", "No")

    user_eff = pd.Series(users).map({"Student": 0.4, "Student Organization": 0.6, "Faculty": -0.6,
                                     "Administrative Staff": -0.4, "External Guest": 0.7}).to_numpy()
    purp_eff = pd.Series(purpose).map({"Academic Class": -0.6, "Meeting": -0.1, "Seminar/Workshop": 0.0,
                                       "Sports Activity": 0.2, "Social Event": 0.6}).to_numpy()
    lead_eff = np.select([lead <= 2, lead <= 7, lead <= 14], [-0.6, 0.0, 0.3], default=0.8)
    z = -2.0 + 1.6 * (user_eff + purp_eff + lead_eff) + 0.7 * prior_cancel + 0.5 * is_weekend + 0.3 * (duration >= 3)
    p_not = 1 / (1 + np.exp(-z))
    not_done = rng.random(n) < p_not
    kind = np.where(rng.random(n) < 0.55, "Cancelled", "No-Show")
    status = np.where(not_done, kind, "Used")

    cancel_days = np.array([rng.integers(0, l + 1) if s == "Cancelled" else np.nan for s, l in zip(status, lead)])
    att_used = np.maximum(1, (participants * np.clip(rng.normal(0.85, 0.12, n), 0.3, 1.10)).round()).astype(int)
    attendance = np.where(status == "Used", att_used, np.where(status == "No-Show", 0, np.nan))

    d = pd.DataFrame({
        "Reservation_ID": [f"FR-2026-{i:04d}" for i in range(1, n + 1)],
        "Request_Date": req_date.strftime("%Y-%m-%d"),
        "Reservation_Date": res_date.strftime("%Y-%m-%d"),
        "Lead_Time_Days": lead.astype(float),
        "Facility_Type": fac, "User_Category": users, "Purpose": purpose,
        "Participants": participants.astype(float), "Duration_Hours": duration,
        "Start_Hour": start_hour.astype(float), "Requires_Equipment": equip,
        "Prior_Cancellations": prior_cancel.astype(float),
        "Usage_Status": status, "Actual_Attendance": attendance,
        "Cancellation_Days_Before": cancel_days,
    })

    # ---------- Inject data-quality problems ----------
    def pick(k, pool=None):
        pool = np.arange(len(d)) if pool is None else np.asarray(pool)
        return rng.choice(pool, size=k, replace=False)

    used_idx = np.where(d["Usage_Status"] == "Used")[0]
    d.loc[pick(3), "Participants"] = -5
    d.loc[pick(3), "Participants"] = 0
    d.loc[pick(3), "Participants"] = 9999
    d.loc[pick(3), "Duration_Hours"] = 0
    d.loc[pick(2), "Duration_Hours"] = -2
    d.loc[pick(2), "Duration_Hours"] = 48
    d.loc[pick(3), "Start_Hour"] = 25
    d.loc[pick(2), "Prior_Cancellations"] = -1
    d.loc[pick(3, used_idx), "Actual_Attendance"] = -4
    d.loc[pick(5), "Lead_Time_Days"] = -3
    d.loc[pick(6), "Lead_Time_Days"] = np.nan
    both = pick(6)
    d.loc[both, ["Request_Date", "Lead_Time_Days"]] = np.nan
    d.loc[pick(4), "Participants"] = np.nan
    d.loc[pick(5), "Duration_Hours"] = np.nan
    d.loc[pick(4), "Start_Hour"] = np.nan
    d.loc[pick(5), "Prior_Cancellations"] = np.nan
    d.loc[pick(8, used_idx), "Actual_Attendance"] = np.nan

    variants = {
        "Facility_Type": {"Computer Lab": ["computer lab", "COMPUTER LAB", "Comp Lab", "Computer Lab "],
                          "Classroom": ["classroom", "Class Room", "CLASSROOM"],
                          "Auditorium": ["auditorium", "Auditorium "],
                          "Gymnasium": ["gym", "Gymnasium ", "GYMNASIUM"],
                          "Conference Room": ["conference room", "Conf. Room", "Conference  Room"],
                          "Sports Field": ["sports field", "Sports field", "Field"]},
        "User_Category": {"Student Organization": ["student org", "Student Org.", "STUDENT ORGANIZATION"],
                          "Administrative Staff": ["admin staff", "Staff", "Administrative staff"],
                          "Faculty": ["faculty", "FACULTY"], "Student": ["student", "STUDENT"],
                          "External Guest": ["external guest", "External"]},
        "Requires_Equipment": {"Yes": ["yes", "Y", "YES"], "No": ["no", "N", "NO"]},
        "Usage_Status": {"Used": ["used", "USED", "Used "],
                         "Cancelled": ["cancelled", "Canceled", "CANCELLED"],
                         "No-Show": ["No Show", "no-show", "NO_SHOW", "No-show"]},
    }
    for col, vmap in variants.items():
        d[col] = [str(rng.choice(vmap[v])) if (v in vmap and rng.random() < 0.18) else v for v in d[col]]

    d.loc[pick(8), "Facility_Type"] = np.nan
    d.loc[pick(8), "User_Category"] = np.nan
    d.loc[pick(5), "Purpose"] = np.nan
    d.loc[pick(5), "Requires_Equipment"] = np.nan
    d.loc[pick(6), "Usage_Status"] = np.nan

    dup_exact = d.iloc[pick(12)].copy()
    dup_near = d.iloc[pick(3)].copy()
    dup_near["Facility_Type"] = dup_near["Facility_Type"].apply(lambda v: v.upper() if isinstance(v, str) else v)
    d = pd.concat([d, dup_exact, dup_near]).sort_values("Reservation_ID", kind="stable").reset_index(drop=True)

    for c in ["Lead_Time_Days", "Participants", "Start_Hour", "Prior_Cancellations",
              "Actual_Attendance", "Cancellation_Days_Before"]:
        d[c] = d[c].round().astype("Int64")
    return d
