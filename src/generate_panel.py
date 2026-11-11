"""Part B — SYNTHETIC policyholder x month panel with planted fraud patterns.

Everything here is invented. Assumptions are documented in
data/synthetic/ASSUMPTIONS.md; change a number here and update that file.

Run:  python -m src.generate_panel
Writes data/synthetic/{policyholders,policy_months,claims,ground_truth_planted}.csv
"""
import hashlib
import json

import numpy as np
import pandas as pd

from src import config

N_POLICYHOLDERS = 20_000
N_SHOPS = 400
REGIONS = np.array(["North", "South", "East", "West", "Central"])
REGION_P = [0.22, 0.26, 0.18, 0.24, 0.10]
REGION_MULT = np.array([1.00, 1.10, 0.95, 1.05, 0.90])
AGE_BANDS = np.array(["18-25", "26-35", "36-50", "51-65", "66+"])
AGE_P = [0.12, 0.28, 0.32, 0.20, 0.08]
AGE_MULT = np.array([1.40, 1.10, 0.90, 0.95, 1.10])
SEGMENTS = np.array(["Hatchback", "Sedan", "SUV"])
SEGMENT_P = [0.45, 0.25, 0.30]
SEGMENT_MULT = np.array([1.00, 0.90, 1.10])
TIERS = np.array(["Basic", "Standard", "Comprehensive"])
TIER_P = [0.25, 0.40, 0.35]
TIER_PREMIUM = np.array([600, 1100, 1900])          # INR / month before adjustments
TIER_CLAIM_MULT = np.array([0.6, 1.0, 1.2])

BASE_MONTHLY_CLAIM_P = 0.025
MONTHLY_LAPSE_P = 0.01
UPGRADE_P = 0.004            # x frailty: riskier people also upgrade more (drives FE vs RE gap)
DOWNGRADE_P = 0.003

CLAIM_TYPES = np.array(["Collision", "Theft", "Glass", "Natural", "ThirdParty"])
TYPE_P_BY_TIER = {
    0: [0.00, 0.00, 0.00, 0.00, 1.00],               # Basic = third-party liability only
    1: [0.55, 0.05, 0.20, 0.05, 0.15],
    2: [0.50, 0.08, 0.20, 0.07, 0.15],
}
TYPE_MEDIAN_INR = {"Collision": 28_000, "Theft": 120_000, "Glass": 9_000,
                   "Natural": 45_000, "ThirdParty": 35_000}
AMOUNT_SIGMA = 0.7

# Planted fraud: share of policyholders per pattern (disjoint sets)
P_EARLY, P_RISING, P_REPEAT, P_UNDER_LIMIT = 0.010, 0.004, 0.004, 0.0075
N_RINGS, RING_SIZE = 15, (4, 8)
UNPATTERNED_FRAUD_SHARE = 0.01   # share of ordinary claims that are (inflated) fraud
LABEL_CAPTURE_RATE = 0.85        # share of true fraud the historical SIU actually confirmed

# Data-quality "dirt" so the cleaning step does real work
DUP_ROW_SHARE, MISSING_AMOUNT_SHARE = 0.004, 0.002
REPORT_BEFORE_INCIDENT_SHARE, BEFORE_POLICY_START_SHARE = 0.001, 0.001
REGION_FORMAT_NOISE_SHARE = 0.02


def _amount(rng, claim_type, mult=1.0, size=None):
    med = np.vectorize(TYPE_MEDIAN_INR.get)(claim_type) if size is None else TYPE_MEDIAN_INR[claim_type]
    a = med * mult * rng.lognormal(0.0, AMOUNT_SIGMA, size=np.shape(claim_type) if size is None else size)
    return np.clip(np.round(a, -1), 1_000, 1_000_000)


def generate(seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    months = pd.date_range(config.PANEL_START, periods=config.PANEL_MONTHS, freq="MS")
    panel_end = months[-1] + pd.offsets.MonthEnd(0)
    n = N_POLICYHOLDERS
    ids = np.array([f"PH{i:06d}" for i in range(1, n + 1)])

    # ---------------- policyholders ----------------
    age_i = rng.choice(len(AGE_BANDS), n, p=AGE_P)
    reg_i = rng.choice(len(REGIONS), n, p=REGION_P)
    seg_i = rng.choice(len(SEGMENTS), n, p=SEGMENT_P)
    veh_age = rng.integers(0, 13, n)
    frailty = rng.gamma(2.0, 0.5, n)                          # unobserved claim propensity, mean 1
    start_offset = rng.integers(-36, 31, n)                   # month index relative to Jan 2023
    tier0 = rng.choice(3, n, p=TIER_P)
    contact = np.array([hashlib.sha1(f"{seed}-{i}".encode()).hexdigest()[:12] for i in range(n)])
    households = rng.choice(n, int(n * 0.03), replace=False)  # 3% share an address with someone else
    for a, b in households.reshape(-1, 2):
        contact[b] = contact[a]

    # shops belong to regions; popularity is skewed (a few big garages)
    shop_region = np.repeat(np.arange(len(REGIONS)), N_SHOPS // len(REGIONS))
    shop_ids = np.array([f"SH{i:04d}" for i in range(1, N_SHOPS + 1)])
    shop_pop = rng.pareto(1.5, N_SHOPS) + 1

    def pick_shop(region_idx):
        cand = np.flatnonzero(shop_region == region_idx)
        w = shop_pop[cand] / shop_pop[cand].sum()
        return shop_ids[rng.choice(cand, p=w)]

    # ---------------- choose planted fraudsters (disjoint) ----------------
    perm = rng.permutation(n)
    cuts = np.cumsum([int(n * P_EARLY), int(n * P_RISING), int(n * P_REPEAT), int(n * P_UNDER_LIMIT)])
    early_ids, rising_ids, repeat_ids, limit_ids = np.split(perm[: cuts[-1]], cuts[:-1])
    pool = perm[cuts[-1]:]
    ring_members = []
    pos = 0
    for r in range(N_RINGS):
        size = int(rng.integers(RING_SIZE[0], RING_SIZE[1] + 1))
        region = int(rng.integers(len(REGIONS)))
        members = [i for i in pool[pos:pos + 400] if reg_i[i] == region and start_offset[i] <= 20][:size]
        pos += 400
        ring_members.append((r + 1, region, np.array(members)))
    fraudster = np.zeros(n, bool)
    for arr in (early_ids, rising_ids, repeat_ids, limit_ids, *[m for _, _, m in ring_members]):
        fraudster[arr] = True
    # planted claims must be covered by the policy: own-damage patterns need Standard or better
    own_damage = np.concatenate([early_ids, rising_ids, repeat_ids, *[m for _, _, m in ring_members]])
    tier0[own_damage] = np.maximum(tier0[own_damage], 1)

    # P1: half new-policy claimers (start inside the panel), half upgrade-then-claim
    early_new, early_upg = early_ids[: len(early_ids) // 2], early_ids[len(early_ids) // 2:]
    start_offset[early_new] = rng.integers(1, 33, len(early_new))
    forced_upgrade_month = np.full(n, -1)
    forced_upgrade_month[early_upg] = rng.integers(4, 33, len(early_upg))
    tier0[early_upg] = np.minimum(tier0[early_upg], 1)
    start_offset[early_upg] = np.minimum(start_offset[early_upg], forced_upgrade_month[early_upg] - 4)

    start_day = rng.integers(0, 28, n)
    start_date = np.array([months[0] + pd.DateOffset(months=int(o)) + pd.Timedelta(days=int(d))
                           for o, d in zip(start_offset, start_day)])
    base_premium = TIER_PREMIUM[tier0] * SEGMENT_MULT[seg_i] * AGE_MULT[age_i] ** 0.5 * rng.normal(1, 0.08, n)

    # ---------------- monthly panel + ordinary claims ----------------
    tier = tier0.copy()
    premium = base_premium.copy()
    lapsed = np.zeros(n, bool)
    end_date = np.full(n, None, dtype=object)
    claim_mult = frailty * AGE_MULT[age_i] * REGION_MULT[reg_i] * SEGMENT_MULT[seg_i]
    panel_rows, claims = [], []
    rising_onset = np.full(n, 10_000)
    rising_onset[rising_ids] = rng.integers(8, 27, len(rising_ids))

    for m, mstart in enumerate(months):
        active = (start_offset <= m) & ~lapsed
        tenure = m - start_offset
        # coverage changes, effective on the 1st of the month
        if m > 0:
            # planted fraudsters keep their tier (except the forced upgrade) so planted claim types stay covered
            can = active & (tenure >= 1) & (forced_upgrade_month < 0) & ~fraudster
            up = can & (tier < 2) & (rng.random(n) < UPGRADE_P * frailty)
            down = can & ~up & (tier > 0) & (rng.random(n) < DOWNGRADE_P)
            forced = active & (forced_upgrade_month == m) & (tier < 2)
            for mask, step in ((up | forced, 1), (down, -1)):
                premium[mask] *= TIER_PREMIUM[np.clip(tier[mask] + step, 0, 2)] / TIER_PREMIUM[tier[mask]]
                tier[mask] += step
            # annual repricing on the policy anniversary month
            anniv = active & (tenure > 0) & (tenure % 12 == 0)
            premium[anniv] *= rng.normal(1.05, 0.03, anniv.sum())
        idx = np.flatnonzero(active)
        panel_rows.append(pd.DataFrame({
            "policyholder_id": ids[idx], "month": mstart.date().isoformat(),
            "coverage_tier": TIERS[tier[idx]], "monthly_premium_inr": np.round(premium[idx], 2)}))

        # ordinary claims (at most one per month)
        p = BASE_MONTHLY_CLAIM_P * claim_mult[idx] * TIER_CLAIM_MULT[tier[idx]]
        hit = idx[rng.random(len(idx)) < p]
        for i in hit:
            ctype = CLAIM_TYPES[rng.choice(5, p=TYPE_P_BY_TIER[int(tier[i])])]
            claims.append(dict(policyholder_id=ids[i], i=i, month=m, claim_type=ctype,
                               amount=float(_amount(rng, ctype, 1.15 if tier[i] == 2 else 1.0, 1)[0]),
                               pattern=None, ring_id=None))

        # P2 rising frequency: after onset, monthly claim probability ramps 0.05 -> 0.30
        ramp = idx[(rising_onset[idx] <= m)]
        pr = 0.05 + (0.25 * np.minimum(m - rising_onset[ramp], 10) / 10)
        for i in ramp[rng.random(len(ramp)) < pr]:
            ctype = rng.choice(["Glass", "Collision", "Natural"])
            claims.append(dict(policyholder_id=ids[i], i=i, month=m, claim_type=ctype,
                               amount=float(_amount(rng, ctype, 1.3, 1)[0]),
                               pattern="rising_frequency", ring_id=None))

        # lapses happen at month end; planted fraudsters are kept active
        lap = active & ~fraudster & (rng.random(n) < MONTHLY_LAPSE_P)
        lapsed |= lap
        for i in np.flatnonzero(lap):
            end_date[i] = (mstart + pd.offsets.MonthEnd(0)).date().isoformat()

    panel = pd.concat(panel_rows, ignore_index=True)

    def day_in_month(m):
        return months[m] + pd.Timedelta(days=int(rng.integers(0, months[m].days_in_month)))

    for c in claims:
        c["incident_date"] = day_in_month(c["month"])

    # P1 early claims: 3-60 days after policy start or after the forced upgrade
    for i in early_new:
        claims.append(dict(policyholder_id=ids[i], i=i, claim_type=rng.choice(["Collision", "Theft", "Natural"]),
                           incident_date=start_date[i] + pd.Timedelta(days=int(rng.integers(3, 61))),
                           amount=float(np.round(60_000 * rng.lognormal(0, 0.5), -1)),
                           pattern="early_claim_new_policy", ring_id=None))
    for i in early_upg:
        upg = months[forced_upgrade_month[i]]
        claims.append(dict(policyholder_id=ids[i], i=i, claim_type=rng.choice(["Collision", "Theft", "Natural"]),
                           incident_date=upg + pd.Timedelta(days=int(rng.integers(3, 61))),
                           amount=float(np.round(60_000 * rng.lognormal(0, 0.5), -1)),
                           pattern="early_claim_after_upgrade", ring_id=None))

    # P3 repeated near-identical claims: same type, same shop, amount within +-5%
    for i in repeat_ids:
        ctype = rng.choice(["Glass", "Collision"])
        base = float(_amount(rng, ctype, 1.0, 1)[0])
        shop = pick_shop(reg_i[i])
        d = months[0] + pd.DateOffset(months=int(max(start_offset[i], 0) + rng.integers(2, 12))) \
            + pd.Timedelta(days=int(rng.integers(0, 28)))
        for _ in range(int(rng.integers(3, 5))):
            claims.append(dict(policyholder_id=ids[i], i=i, claim_type=ctype, incident_date=d,
                               amount=float(np.round(base * rng.uniform(0.95, 1.05), -1)), shop=shop,
                               pattern="repeat_similar", ring_id=None))
            d = d + pd.Timedelta(days=int(rng.integers(30, 91)))

    # P4 amounts clustered just under the fast-track approval limit
    for i in limit_ids:
        for _ in range(int(rng.integers(1, 3))):
            m = int(rng.integers(max(start_offset[i], 0), config.PANEL_MONTHS))
            ctype = "ThirdParty" if tier0[i] == 0 else rng.choice(["Collision", "ThirdParty"])
            claims.append(dict(policyholder_id=ids[i], i=i, claim_type=ctype,
                               incident_date=day_in_month(m),
                               amount=float(np.round(rng.uniform(0.90, 0.998) * config.APPROVAL_LIMIT_INR, -1)),
                               pattern="under_approval_limit", ring_id=None))

    # P5 staged-accident rings: shared address hash, one garage, name each other as third party
    for ring_id, region, members in ring_members:
        ring_hash = hashlib.sha1(f"ring-{seed}-{ring_id}".encode()).hexdigest()[:12]
        contact[members] = ring_hash
        shop = pick_shop(region)
        # ring activity starts at a random month after every member's policy began
        t0 = months[0] + pd.DateOffset(months=int(rng.integers(max(start_offset[members].max(), 0) + 1, 34)))
        for j, i in enumerate(members):
            for _ in range(int(rng.integers(1, 3))):
                other = members[(j + 1 + int(rng.integers(0, len(members) - 1))) % len(members)]
                claims.append(dict(policyholder_id=ids[i], i=i, claim_type="Collision",
                                   incident_date=t0 + pd.Timedelta(days=int(rng.integers(0, 76))),
                                   amount=float(np.round(40_000 * rng.lognormal(0, 0.35), -1)),
                                   shop=shop, third_party_ref=ids[other], witness=True,
                                   pattern="staged_ring", ring_id=ring_id))

    cl = pd.DataFrame(claims)
    cl["incident_date"] = pd.to_datetime(cl["incident_date"])
    # unpatterned opportunistic fraud: a small share of ordinary claims, inflated
    ordinary = cl["pattern"].isna().to_numpy()
    unp = ordinary & (rng.random(len(cl)) < UNPATTERNED_FRAUD_SHARE)
    cl.loc[unp, "amount"] = np.round(cl.loc[unp, "amount"] * 1.5, -1)
    cl.loc[unp, "pattern"] = "unpatterned"
    is_fraud = cl["pattern"].notna().to_numpy()

    k = len(cl)
    delay = np.where(is_fraud, rng.exponential(9, k), rng.exponential(5, k)).astype(int)
    cl["report_date"] = cl["incident_date"] + pd.to_timedelta(delay, unit="D")
    # claims must fall inside the panel window and after the policy started
    keep = (cl["report_date"] <= panel_end) & (cl["incident_date"] >= pd.to_datetime(start_date[cl["i"]]))
    cl = cl[keep.to_numpy()].reset_index(drop=True)
    is_fraud = cl["pattern"].notna().to_numpy()
    k = len(cl)

    if "shop" not in cl:
        cl["shop"] = None
    needs_shop = cl["shop"].isna() & cl["claim_type"].ne("Theft")
    cl.loc[needs_shop, "shop"] = [pick_shop(reg_i[i]) for i in cl.loc[needs_shop, "i"]]
    if "third_party_ref" not in cl:
        cl["third_party_ref"] = None
    needs_tp = cl["third_party_ref"].isna() & cl["claim_type"].isin(["Collision", "ThirdParty"])
    tp = np.array([f"TP{x:07d}" for x in rng.integers(0, 10_000_000, needs_tp.sum())], dtype=object)
    between_customers = rng.random(needs_tp.sum()) < 0.02
    tp[between_customers] = ids[rng.integers(0, n, between_customers.sum())]
    cl.loc[needs_tp, "third_party_ref"] = tp
    witness_p = np.where(cl["witness"].eq(True).to_numpy(), 0.7,
                         np.where(is_fraud, 0.2, 0.3))
    cl["witness_present"] = np.where(rng.random(k) < witness_p, "Yes", "No")
    police_p = np.where(cl["claim_type"].isin(["Collision", "Theft"]), np.where(is_fraud, 0.35, 0.6), 0.1)
    cl["police_report_filed"] = np.where(rng.random(k) < police_p, "Yes", "No")
    cl["fraud_label"] = (is_fraud & (rng.random(k) < LABEL_CAPTURE_RATE)).astype(int)

    cl = cl.sort_values(["report_date", "policyholder_id", "incident_date", "amount"], kind="stable").reset_index(drop=True)
    cl["claim_id"] = [f"CL{x:07d}" for x in range(1, len(cl) + 1)]

    truth = cl.loc[cl["pattern"].notna(), ["claim_id", "policyholder_id", "pattern", "ring_id", "fraud_label"]].copy()
    truth["ring_id"] = truth["ring_id"].astype("Int64")
    truth = truth.rename(columns={"fraud_label": "labelled_by_siu"})

    out = pd.DataFrame({
        "claim_id": cl["claim_id"], "policyholder_id": cl["policyholder_id"],
        "incident_date": cl["incident_date"].dt.date.astype(str), "report_date": cl["report_date"].dt.date.astype(str),
        "claim_type": cl["claim_type"], "claim_amount_inr": cl["amount"].map(lambda v: f"{v:.0f}"),
        "repair_shop_id": cl["shop"].fillna(""), "third_party_ref": cl["third_party_ref"].fillna(""),
        "police_report_filed": cl["police_report_filed"], "witness_present": cl["witness_present"],
        "fraud_label": cl["fraud_label"],
    })

    # ---------------- data-quality dirt ----------------
    r = rng.random(len(out))
    out.loc[r < MISSING_AMOUNT_SHARE, "claim_amount_inr"] = ""
    swap = (r >= MISSING_AMOUNT_SHARE) & (r < MISSING_AMOUNT_SHARE + REPORT_BEFORE_INCIDENT_SHARE)
    out.loc[swap, ["incident_date", "report_date"]] = out.loc[swap, ["report_date", "incident_date"]].to_numpy()
    pre = (r >= 0.5) & (r < 0.5 + BEFORE_POLICY_START_SHARE)
    out.loc[pre, "incident_date"] = (pd.to_datetime(out.loc[pre, "incident_date"]) - pd.DateOffset(years=4)).dt.date.astype(str)
    dups = out.sample(frac=DUP_ROW_SHARE, random_state=seed)
    out = pd.concat([out, dups]).sort_values(["report_date", "claim_id"], kind="stable").reset_index(drop=True)

    region_txt = REGIONS[reg_i].astype(object)
    noisy = rng.random(n) < REGION_FORMAT_NOISE_SHARE
    region_txt[noisy] = [s.lower() if rng.random() < 0.5 else s + " " for s in region_txt[noisy]]
    ph = pd.DataFrame({
        "policyholder_id": ids, "age_band": AGE_BANDS[age_i], "region": region_txt,
        "vehicle_segment": SEGMENTS[seg_i], "vehicle_age_years": veh_age, "contact_hash": contact,
        "policy_start_date": [d.date().isoformat() for d in start_date],
        "policy_end_date": [e or "" for e in end_date],
    })
    return {"policyholders": ph, "policy_months": panel, "claims": out, "ground_truth_planted": truth}


def main():
    config.DATA_SYN.mkdir(parents=True, exist_ok=True)
    tables = generate()
    summary = {}
    for name, df in tables.items():
        path = config.DATA_SYN / f"{name}.csv"
        df.to_csv(path, index=False, lineterminator="\n")
        summary[name] = {"rows": len(df), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"{name:22s} {len(df):>8,d} rows -> {path.name}")
    gt = tables["ground_truth_planted"]
    summary["planted_claims_by_pattern"] = gt["pattern"].value_counts().sort_index().to_dict()
    summary["seed"] = config.SEED
    (config.DATA_SYN / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["planted_claims_by_pattern"], indent=2))


if __name__ == "__main__":
    main()
