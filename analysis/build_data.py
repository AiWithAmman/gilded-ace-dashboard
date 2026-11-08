"""Build the data bundle behind dashboard.html.

Run from the repository root:  python analysis/build_data.py
Writes analysis/dashboard_data.json and injects it into dashboard.html.

Every number on the dashboard is produced here from the files in data/.
Conventions followed (see GUIDE.md §5): ratio of sums, active rows only,
scheduled_seconds from the calendar (DST), cents vs dollars, comp cost
counted once, carded-only scope for player tables.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
OUT_JSON = ROOT / "analysis" / "dashboard_data.json"
HTML = ROOT / "dashboard.html"


def rd(name: str) -> pd.DataFrame:
    df = pd.read_parquet(D / f"{name}.parquet")
    for c in df.columns:  # DECIMAL columns arrive as python Decimal objects
        if df[c].dtype == object and (c.endswith("_usd") or c in ("configured_rtp", "theo_cents", "carded_theo_cents", "rate")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def r(x, n=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), n)


# ---------------------------------------------------------------- load
pdv = rd("1_prepared_views/prepared_daily_view")
md = rd("2_machine_decision/machine_day")
gm = rd("2_machine_decision/game_math")
mach = rd("2_machine_decision/machines")
ops = rd("2_machine_decision/operations")
cal = rd("2_machine_decision/calendar_day")
prop = rd("1_prepared_views/property_day")
pp = rd("1_prepared_views/player_period")
ll = rd("3_guests/loyalty_ledger")
roster = rd("5_optional/shift_roster")
zt = rd("4_context/zone_traffic_hourly")
events = rd("4_context/events")
city = rd("4_context/real_sources/city_visitation_monthly")
actions = rd("2_machine_decision/action_catalog")
placements = rd("2_machine_decision/placements")

gm["var_mult"] = pd.to_numeric(gm["payout_multiplier_variance"])
gm["hit_p"] = pd.to_numeric(gm["hit_probability"])

# ---------------------------------------------------------------- machine-day base (active rows only)
a = pdv[pdv["active"]].copy()
a = a.merge(
    md[["machine_id", "gaming_date", "scheduled_seconds", "sum_bet_squared", "payout_positive_spins"]],
    on=["machine_id", "gaming_date"],
)
a = a.merge(gm[["config_id", "var_mult"]], on="config_id")
a["sched_h"] = a["scheduled_seconds"] / 3600.0
a["down_h"] = (a["sched_h"] - a["available_hours"]).clip(lower=0)
# Variance of house win for the day (dollars^2): sum(bet^2) * Var(payout multiplier)
a["hw_var"] = a["sum_bet_squared"] * a["var_mult"] / 1e4

# Location benchmark: expected theo per available hour for a machine's zone, month and weekday,
# scaled by its game's floor-wide pull. Fitted by alternating (raking) on all machines.
a["dow"] = pd.to_datetime(a["gaming_date"]).dt.day_of_week
cfg_f = pd.Series(1.0, index=a["config_id"].unique())
for _ in range(25):
    a["_cf"] = a["config_id"].map(cfg_f)
    zcell = a.groupby(["zone_id", "month", "dow"]).apply(
        lambda t: t["theo_usd"].sum() / (t["available_hours"] * t["_cf"]).sum(), include_groups=False
    )
    a["_zr"] = pd.MultiIndex.from_frame(a[["zone_id", "month", "dow"]]).map(zcell)
    cfg_f = a.groupby("config_id").apply(
        lambda t: t["theo_usd"].sum() / (t["available_hours"] * t["_zr"]).sum(), include_groups=False
    )
a["_cf"] = a["config_id"].map(cfg_f)
a["exp_theo_loc"] = a["available_hours"] * a["_zr"] * a["_cf"]  # what an average machine of this game earns here
# Same game, but at the floor-average location (zone effect removed)
floor_rate = a.groupby(["month", "dow"]).apply(
    lambda t: t["theo_usd"].sum() / (t["available_hours"] * t["_cf"]).sum(), include_groups=False
)
a["_fr"] = pd.MultiIndex.from_frame(a[["month", "dow"]]).map(floor_rate)
a["exp_theo_floor"] = a["available_hours"] * a["_fr"] * a["_cf"]
# Theo lost to downtime, valued at the machine's own location-expected rate
a["lost_theo"] = a["down_h"] * a["_zr"] * a["_cf"]
# second moments for a ratio-estimator CI on the machine index (theo / location-expected theo)
a["tt"] = a["theo_usd"] ** 2
a["te"] = a["theo_usd"] * a["exp_theo_loc"]
a["ee"] = a["exp_theo_loc"] ** 2

# ---------------------------------------------------------------- machine × month table (client re-aggregates by period)
mm = (
    a.groupby(["machine_id", "month"])
    .agg(
        days=("gaming_date", "count"),
        ci=("coin_in_usd", "sum"),
        hw=("house_win_usd", "sum"),
        theo=("theo_usd", "sum"),
        v=("hw_var", "sum"),
        sh=("sched_h", "sum"),
        ah=("available_hours", "sum"),
        oh=("occupied_hours", "sum"),
        f=("faults_opened", "sum"),
        el=("exp_theo_loc", "sum"),
        ef=("exp_theo_floor", "sum"),
        lt=("lost_theo", "sum"),
        sp=("paid_spins", "sum"),
        pps=("payout_positive_spins", "sum"),
        ctb=("contribution_after_supplied_costs_usd", "sum"),
        tt=("tt", "sum"),
        te=("te", "sum"),
        ee=("ee", "sum"),
        zone=("zone_id", "last"),
        bank=("bank_id", "last"),
    )
    .reset_index()
)

last = a.sort_values("gaming_date").groupby("machine_id").last()
first = a.sort_values("gaming_date").groupby("machine_id").first()
minfo = mach.set_index("machine_id")
machines_out = []
for mid, row in last.iterrows():
    cfg = gm.set_index("config_id").loc[row["config_id"]]
    machines_out.append(
        dict(
            id=mid,
            zone=row["zone_id"],
            bank=row["bank_id"],
            loc=row["location_id"],
            from_zone=first.loc[mid, "zone_id"],
            from_bank=first.loc[mid, "bank_id"],
            theme=row["theme"],
            cfg=row["config_id"],
            rtp=r(row["configured_rtp"], 4),
            vol=row["volatility_class"],
            hitp=r(cfg["hit_p"], 4),
            manu=row["manufacturer"],
            lease=row["lease_class"],
            install=str(minfo.loc[mid, "install_date"]),
            retire=None if pd.isna(minfo.loc[mid, "retire_date"]) else str(minfo.loc[mid, "retire_date"]),
        )
    )

mm_rows = [
    [m.machine_id, m.month, int(m.days), r(m.ci, 0), r(m.hw, 0), r(m.theo, 0), r(m.v, 0), r(m.sh, 1), r(m.ah, 1),
     r(m.oh, 1), int(m.f), r(m.el, 0), r(m.ef, 0), r(m.lt, 1), int(m.sp), int(m.pps), r(m.ctb, 0), r(m.tt, 0), r(m.te, 0), r(m.ee, 0), m.zone, m.bank]
    for m in mm.itertuples()
]
mm_cols = ["id", "month", "days", "ci", "hw", "theo", "v", "sh", "ah", "oh", "f", "el", "ef", "lt", "sp", "pps", "ctb", "tt", "te", "ee", "zone", "bank"]

zones = (
    pdv[pdv["zone_id"].notna()][["zone_id", "zone_name"]].drop_duplicates().set_index("zone_id")["zone_name"].to_dict()
)

# ---------------------------------------------------------------- natural experiment: machines that moved zones
moves = []
for mid, grp in a.sort_values("gaming_date").groupby("machine_id"):
    if grp["zone_id"].nunique() < 2:
        continue
    segs = []
    grp = grp.assign(seg=(grp["location_id"] != grp["location_id"].shift()).cumsum())
    for _, s in grp.groupby("seg"):
        segs.append(
            dict(
                zone=s["zone_id"].iloc[0],
                bank=s["bank_id"].iloc[0],
                start=str(s["gaming_date"].iloc[0]),
                end=str(s["gaming_date"].iloc[-1]),
                days=int(len(s)),
                theo_per_hr=r(s["theo_usd"].sum() / s["available_hours"].sum(), 2),
                occ=r(s["occupied_hours"].sum() / s["available_hours"].sum(), 3),
                zone_theo_per_hr=r((s["_zr"] * s["_cf"] * s["available_hours"]).sum() / s["available_hours"].sum(), 2),
            )
        )
    moves.append(dict(id=mid, theme=grp["theme"].iloc[0], segs=segs))

# ---------------------------------------------------------------- hit-rate check (leak screen)
hit = a.groupby("machine_id").agg(sp=("paid_spins", "sum"), pps=("payout_positive_spins", "sum"), cfg=("config_id", "last"))
hit = hit.join(gm.set_index("config_id")["hit_p"], on="cfg")
hit["hit"] = hit["pps"] / hit["sp"]
hit["hit_z"] = (hit["hit"] - hit["hit_p"]) / np.sqrt(hit["hit_p"] * (1 - hit["hit_p"]) / hit["sp"])
weekly_leak = {}
for mid in ["M-1158"]:
    t = md[(md["machine_id"] == mid) & md["active"]].copy()
    t["wk"] = pd.to_datetime(t["gaming_date"]).dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    w = t.groupby("wk").agg(sp=("paid_spins", "sum"), pps=("payout_positive_spins", "sum"),
                            ci=("coin_in_cents", "sum"), hw=("house_win_cents", "sum"))
    weekly_leak[mid] = [[k, r(v.pps / v.sp, 4), r(v.hw / v.ci, 4)] for k, v in w.iterrows()]

# ---------------------------------------------------------------- property / demand (Q2)
prop["gaming_date"] = prop["gaming_date"].astype(str)
cal["gaming_date"] = cal["gaming_date"].astype(str)
pr = prop.merge(cal[["gaming_date", "scheduled_seconds", "is_weekend", "federal_holiday"]], on="gaming_date")
days_out = []
for t in pr.itertuples():
    days_out.append(
        dict(
            d=t.gaming_date,
            wd=t.local_weekday,
            ev=t.events_active if isinstance(t.events_active, str) else "",
            hol=t.federal_holiday if isinstance(t.federal_holiday, str) else "",
            ci=r(t.coin_in_usd, 0),
            hw=r(t.house_win_usd, 0),
            theo=r(t.theo_usd, 0),
            ah=r(t.available_machine_hours, 1),
            oh=r(t.occupied_machine_hours, 1),
            pos=r(t.pos_cash_usd, 0),
            cogs=r(t.pos_cogs_usd, 0),
            comp=r(t.comp_cost_usd, 0),
            compfv=r(t.comp_face_value_usd, 0),
            faults=int(t.faults_opened),
            ent=None if pd.isna(t.total_zone_entries_observed) else int(t.total_zone_entries_observed),
            cov=r(t.counter_coverage_share, 3),
            sup=r(t.machine_supplied_costs_usd, 0),
            tech=r(t.tech_labor_cost_usd, 0),
            ctb=r(t.contribution_after_supplied_costs_usd, 0),
            card=r(t.carded_coin_in_share, 3),
            am=int(t.active_machines),
        )
    )

# zone × day theo/occupancy for demand-by-zone and Back Forty walkway closure
zd = a.groupby(["zone_id", "gaming_date"]).agg(theo=("theo_usd", "sum"), ah=("available_hours", "sum"), oh=("occupied_hours", "sum"),
                                               ci=("coin_in_usd", "sum")).reset_index()
zd["gaming_date"] = zd["gaming_date"].astype(str)
zone_day = {z: [[t.gaming_date, r(t.theo, 0), r(t.ah, 1), r(t.oh, 1), r(t.ci, 0)] for t in g.itertuples()] for z, g in zd.groupby("zone_id")}
# bank × day coin-in so the demand chart follows the bank drill-down
bd = a.groupby(["bank_id", "gaming_date"]).agg(ci=("coin_in_usd", "sum"), theo=("theo_usd", "sum")).reset_index()
bd["gaming_date"] = bd["gaming_date"].astype(str)
bank_day = {b: [[t.gaming_date, r(t.ci, 0), r(t.theo, 0)] for t in g.itertuples()] for b, g in bd.groupby("bank_id")}

ztc = zt[zt["counter_status"] == "ok"].copy()
ztc["gd"] = (ztc["hour_start_local"] - pd.Timedelta(hours=6)).dt.date.astype(str)
zone_entries = ztc.groupby(["zone_id", "gd"]).agg(e=("entries_count", "sum"), h=("coverage_minutes", "sum")).reset_index()
zone_entries["eph"] = zone_entries["e"] / (zone_entries["h"] / 60)  # entries per counter-hour (handles outages)
zone_ent = {z: [[t.gd, r(t.eph, 1)] for t in g.itertuples() if t.gd <= "2025-06-30"] for z, g in zone_entries.groupby("zone_id")}

# Walkway closure effect on Back Forty (difference-in-differences vs rest of floor)
closure = ("2025-03-03", "2025-04-06")
zd["period"] = np.where((zd["gaming_date"] >= closure[0]) & (zd["gaming_date"] <= closure[1]), "closed", "open")
did = zd.assign(b40=zd["zone_id"] == "Z-B40").groupby(["b40", "period"]).apply(
    lambda t: t["theo"].sum() / t["ah"].sum(), include_groups=False).unstack()
did_out = dict(
    b40_open=r(did.loc[True, "open"]), b40_closed=r(did.loc[True, "closed"]),
    rest_open=r(did.loc[False, "open"]), rest_closed=r(did.loc[False, "closed"]),
)
did_out["b40_change"] = r(did.loc[True, "closed"] / did.loc[True, "open"] - 1, 4)
did_out["rest_change"] = r(did.loc[False, "closed"] / did.loc[False, "open"] - 1, 4)
b40_hours_closed = a[(a["zone_id"] == "Z-B40") & (a["gaming_date"].astype(str).between(*closure))]["available_hours"].sum()
did_out["lost_theo"] = r((did.loc[True, "open"] * (1 + did_out["rest_change"]) - did.loc[True, "closed"]) * b40_hours_closed, 0)

city_out = [dict(m=t.month, visitors=int(t.visitor_volume), yoy=r(t.visitor_yoy_ratio, 4),
                 conv=int(t.convention_attendance), occ=r(t.hotel_occupancy_ratio, 4)) for t in city.itertuples()]
events_out = [dict(id=t.event_id, name=t.name, type=t.event_type, start=str(t.start_date), end=str(t.end_date),
                   announced=str(t.announced_date), scope=t.local_scope) for t in events.itertuples()]

# ---------------------------------------------------------------- downtime (Q3)
f = ops[ops["event_type"] == "fault_repair"].copy()
f["down_h"] = (f["available_at_utc"] - f["opened_at_utc"]).dt.total_seconds() / 3600
f["wait_h"] = (f["work_started_at_utc"] - f["opened_at_utc"]).dt.total_seconds() / 3600
f["fix_h"] = (f["available_at_utc"] - f["work_started_at_utc"]).dt.total_seconds() / 3600
f["gd"] = (f["opened_at_local"] - pd.Timedelta(hours=6)).dt.date.astype(str)
techs = roster[roster["role"] == "slot_tech"].sort_values("start_utc")
f = pd.merge_asof(f.sort_values("opened_at_utc"), techs[["start_utc", "end_utc", "shift", "planned_headcount", "actual_headcount"]],
                  left_on="opened_at_utc", right_on="start_utc", direction="backward")
f = f.merge(mach[["machine_id", "manufacturer"]], on="machine_id")
f["short"] = f["actual_headcount"] < f["planned_headcount"]
# value of an hour of downtime for that machine: its location-expected theo per available hour (overall)
rate = a.groupby("machine_id").apply(lambda t: t["exp_theo_loc"].sum() / t["available_hours"].sum(), include_groups=False)
f["rate"] = f["machine_id"].map(rate)
f["wait_cost"] = f["wait_h"] * f["rate"]
f["down_cost"] = f["down_h"] * f["rate"]
faults_out = [[t.gd, t.machine_id, t.manufacturer, t.shift, int(t.planned_headcount), int(t.actual_headcount),
               r(t.wait_h, 2), r(t.fix_h, 2), r(t.down_cost, 0), r(t.wait_cost, 0), int(t.opened_at_local.hour),
               r(t.direct_cost_cents / 100, 0)]
              for t in f.itertuples()]
faults_cols = ["d", "id", "manu", "shift", "planned", "actual", "wait", "fix", "cost", "waitcost", "hour", "repair"]

tech_out = [[str(t.gaming_date), t.shift, int(t.planned_headcount), int(t.actual_headcount)] for t in techs.itertuples()]
staff_effect = f.groupby(["shift", "short"]).agg(n=("wait_h", "size"), wait=("wait_h", "mean")).reset_index()
staff_effect_out = [dict(shift=t.shift, short=bool(t.short), n=int(t.n), wait=r(t.wait, 2)) for t in staff_effect.itertuples()]

# ---------------------------------------------------------------- players (Q4) and comps (Q5)
pp["tier"] = pp["opening_tier"]
pp["age"] = pp["age_band"].fillna("Unknown")
grp_cols = ["period_month", "tier", "home_region", "age"]
pg = pp.groupby(grp_cols).agg(
    n=("player_id", "size"), ci=("tracked_coin_in_usd", "sum"), pay=("tracked_payout_usd", "sum"),
    net=("tracked_net_usd", "sum"), theo=("theo_usd", "sum"), cc=("comp_cost_usd", "sum"),
    cfv=("comp_face_value_usd", "sum"), cash=("linked_pos_cash_usd", "sum"), cogs=("linked_pos_cogs_usd", "sum"),
    ctb=("tracked_contribution_usd", "sum"), ahead=("tracked_net_usd", lambda s: int((s > 0).sum())),
    played=("tracked_coin_in_usd", lambda s: int((s > 0).sum())), comps=("comps_redeemed", "sum"),
    pe=("points_earned", "sum"), prd=("points_redeemed", "sum"), hrs=("carded_play_hours", "sum"),
).reset_index()
pgroups = [[t.period_month, t.tier, t.home_region, t.age, int(t.n), r(t.ci, 0), r(t.pay, 0), r(t.net, 0), r(t.theo, 0),
            r(t.cc, 0), r(t.cfv, 0), r(t.cash, 0), r(t.cogs, 0), r(t.ctb, 0), t.ahead, t.played, int(t.comps),
            int(t.pe), int(t.prd), r(t.hrs, 0)] for t in pg.itertuples()]
pgroups_cols = ["m", "tier", "region", "age", "n", "ci", "pay", "net", "theo", "cc", "cfv", "cash", "cogs", "ctb",
                "ahead", "played", "comps", "pe", "prd", "hrs"]

# Concentration over the whole window: per-player totals → Lorenz points for theo and for net loss
tot = pp.groupby("player_id").agg(theo=("theo_usd", "sum"), net=("tracked_net_usd", "sum"), ci=("tracked_coin_in_usd", "sum"))
tot = tot[tot["ci"] > 0]


def lorenz(vals: pd.Series, pts=(0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)):
    v = np.sort(vals.values)[::-1]
    c = np.cumsum(v) / v.sum()
    return [[p, r(c[max(0, int(math.ceil(p * len(v))) - 1)], 4)] for p in pts]


losses = (-tot["net"]).clip(lower=0)
conc = dict(
    players=int(len(tot)),
    theo=lorenz(tot["theo"]),
    loss=lorenz(losses),
    ahead_share=r((tot["net"] > 0).mean(), 4),
    total_net=r(tot["net"].sum(), 0),
    total_theo=r(tot["theo"].sum(), 0),
    top1_net_loss=r(losses.sort_values(ascending=False).head(int(len(tot) * 0.01)).sum(), 0),
)
# distribution of six-month net result (histogram bins)
bins = [-1e9, -10000, -5000, -2000, -1000, -500, -200, -100, -50, 0, 50, 100, 500, 1000, 5000, 1e9]
labels = ["< −$10k", "−$10k…−5k", "−$5k…−2k", "−$2k…−1k", "−$1k…−500", "−$500…−200", "−$200…−100", "−$100…−50",
          "−$50…0", "$0…50", "$50…100", "$100…500", "$500…1k", "$1k…5k", "> $5k"]
hist = pd.cut(tot["net"], bins=bins, labels=labels, right=False).value_counts().reindex(labels)
conc["hist"] = [[k, int(v)] for k, v in hist.items()]

# Comp redemption rate (points per $ of comp face value), daily
red = ll[ll["event_type"] == "redeem"].copy()
red["gd"] = (red["event_at_utc"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
             - pd.Timedelta(hours=6)).dt.date.astype(str)
red["pts"] = -red["points_delta"]
red["fv"] = red["face_value_cents"] / 100
red["cost"] = red["economic_cost_cents"] / 100
rr = red.groupby("gd").agg(n=("pts", "size"), pts=("pts", "sum"), fv=("fv", "sum"), cost=("cost", "sum")).reset_index()
rr = rr[rr["gd"] <= "2025-06-30"]
comp_daily = [[t.gd, int(t.n), int(t.pts), r(t.fv, 0), r(t.cost, 0)] for t in rr.itertuples()]
aw = ll[ll["event_type"] == "award"].copy()
aw["gd"] = (aw["event_at_utc"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
            - pd.Timedelta(hours=6)).dt.date.astype(str)
award_daily = aw[aw["gd"] <= "2025-06-30"].groupby("gd")["points_delta"].sum()
comp_award = [[k, int(v)] for k, v in award_daily.items()]

# Before/after the May 1 comp change: same weekday-mix comparison of tracked theo per day & comp cost / theo
pdd = pd.DataFrame(days_out)
pdd["after"] = pdd["d"] >= "2025-05-01"
cmp_ = pdd.groupby("after").agg(days=("d", "size"), theo=("theo", "sum"), comp=("comp", "sum"), ci=("ci", "sum"),
                                card=("card", "mean"))
comp_change = dict(
    date="2025-05-01",
    before=dict(days=int(cmp_.loc[False, "days"]), comp_per_day=r(cmp_.loc[False, "comp"] / cmp_.loc[False, "days"], 0),
                reinv=r(cmp_.loc[False, "comp"] / cmp_.loc[False, "theo"], 5),
                ci_per_day=r(cmp_.loc[False, "ci"] / cmp_.loc[False, "days"], 0), card=r(cmp_.loc[False, "card"], 4)),
    after=dict(days=int(cmp_.loc[True, "days"]), comp_per_day=r(cmp_.loc[True, "comp"] / cmp_.loc[True, "days"], 0),
               reinv=r(cmp_.loc[True, "comp"] / cmp_.loc[True, "theo"], 5),
               ci_per_day=r(cmp_.loc[True, "ci"] / cmp_.loc[True, "days"], 0), card=r(cmp_.loc[True, "card"], 4)),
)
pts_before = red[(red["gd"] < "2025-05-01")]
pts_after = red[(red["gd"] >= "2025-05-01") & (red["gd"] <= "2025-06-30")]
comp_change["pts_per_usd_before"] = r(pts_before["pts"].sum() / pts_before["fv"].sum(), 2)
comp_change["pts_per_usd_after"] = r(pts_after["pts"].sum() / pts_after["fv"].sum(), 2)

# ---------------------------------------------------------------- forecast model (time-based holdout)
fc = pdd[["d", "wd", "ci"]].copy()
FORECAST_DATE = "2025-05-01"
fc["train"] = fc["d"] < FORECAST_DATE
# Event flag uses only events whose announcement predates the forecast date (no leakage:
# e.g. the Locals Appreciation Week was announced 2025-05-26 and is NOT flagged for the holdout).
known = events[events["announced_date"].astype(str) < FORECAST_DATE]
fc["ev"] = ""
for e in known.itertuples():
    mask = (fc["d"] >= str(e.start_date)) & (fc["d"] <= str(e.end_date))
    fc.loc[mask, "ev"] = e.name
tr = fc[fc["train"]]
mean_const = tr["ci"].mean()
wd_mean = tr.groupby("wd")["ci"].mean()
# Weekday × event model: event uplift ratio learned on training days only, applied to events
# that were announced before the forecast date (all events in events.csv were announced before May 1).
tr_ratio = tr["ci"] / tr["wd"].map(wd_mean)
ev_uplift = tr_ratio[tr["ev"] != ""].mean()
fc["p_const"] = mean_const
fc["p_wd"] = fc["wd"].map(wd_mean)
fc["p_wdev"] = fc["p_wd"] * np.where(fc["ev"] != "", ev_uplift, 1.0)
# Trend-adjusted: weekday mean × linear trend in weekly level fitted on training data
tr_idx = np.arange(len(tr))
slope, intercept = np.polyfit(tr_idx, tr_ratio.values, 1)
fc["p_wdtr"] = fc["p_wd"] * (intercept + slope * np.arange(len(fc)))
te = fc[~fc["train"]]
mae = {k: r((te[k] - te["ci"]).abs().mean(), 0) for k in ["p_const", "p_wd", "p_wdev", "p_wdtr"]}
mae_train = {k: r((tr.assign(**{k: fc.loc[fc["train"], k]})[k] - tr["ci"]).abs().mean(), 0) for k in ["p_const", "p_wd", "p_wdev", "p_wdtr"]}
mape = {k: r(((te[k] - te["ci"]).abs() / te["ci"]).mean(), 4) for k in ["p_const", "p_wd", "p_wdev", "p_wdtr"]}
model = dict(
    rows=[[t.d, t.wd, r(t.ci, 0), r(t.p_const, 0), r(t.p_wd, 0), r(t.p_wdev, 0), r(t.p_wdtr, 0), bool(t.train)] for t in fc.itertuples()],
    mae=mae, mae_train=mae_train, mape=mape, ev_uplift=r(ev_uplift, 4), wd_mean={k: r(v, 0) for k, v in wd_mean.items()},
    trend_per_day=r(slope, 6), const=r(mean_const, 0),
)
# Weekday share of predicted coin-in vs share of fault wait-hours (staffing decision)
f_wd = f.assign(wd=pd.to_datetime(f["gd"]).dt.strftime("%a")).groupby("wd").agg(n=("wait_h", "size"), wait=("wait_h", "sum"), waitcost=("wait_cost", "sum"))
model["fault_by_wd"] = {k: dict(n=int(v.n), wait=r(v.wait, 1), waitcost=r(v.waitcost, 0)) for k, v in f_wd.iterrows()}

# ---------------------------------------------------------------- bundle
bundle = dict(
    meta=dict(built_from="data/ (release 0.9.0-draft)", window=["2025-01-01", "2025-06-30"],
              months=sorted(a["month"].unique().tolist()), closure=list(closure)),
    zones=zones,
    game_factor={k: r(v, 3) for k, v in cfg_f.items()},
    themes=gm.set_index("config_id")["theme"].to_dict(),
    machines=machines_out,
    mm_cols=mm_cols,
    mm=mm_rows,
    hit={k: [r(v.hit, 4), r(v.hit_p, 4), r(v.hit_z, 2)] for k, v in hit.iterrows()},
    weekly_leak=weekly_leak,
    moves=moves,
    days=days_out,
    zone_day=zone_day,
    bank_day=bank_day,
    zone_ent=zone_ent,
    did=did_out,
    city=city_out,
    events=events_out,
    faults_cols=faults_cols,
    faults=faults_out,
    tech=tech_out,
    staff_effect=staff_effect_out,
    pgroups_cols=pgroups_cols,
    pgroups=pgroups,
    conc=conc,
    comp_daily=comp_daily,
    comp_award=comp_award,
    comp_change=comp_change,
    model=model,
    actions=[dict(id=t.action_type_id, name=t.name, unit=t.unit, cost=t.unit_cost_cents / 100,
                  daily=t.recurring_cost_cents_per_day / 100, lead=int(t.lead_days), down=int(t.downtime_minutes),
                  limit=int(t.quantity_limit), group=None if pd.isna(t.incompatibility_group) else t.incompatibility_group,
                  notes=t.notes) for t in actions.itertuples()],
)

OUT_JSON.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
print(f"wrote {OUT_JSON} ({OUT_JSON.stat().st_size/1024:.0f} KB)")

if HTML.exists():
    html = HTML.read_text(encoding="utf-8")
    start = html.index('<script id="data" type="application/json">') + len('<script id="data" type="application/json">')
    end = html.index("</script>", start)
    html = html[:start] + json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/") + html[end:]
    HTML.write_text(html, encoding="utf-8")
    print(f"injected into {HTML}")

# ---------------------------------------------------------------- console summary (for the write-up)
print("MAE", mae, "MAPE", mape, "train", mae_train, "ev_uplift", ev_uplift, "slope", slope)
print("DiD", did_out)
print("comp", comp_change)
print("conc", {k: v for k, v in conc.items() if k != "hist"})
print(staff_effect)
print("moves", json.dumps(moves, indent=1))
