# Part B: synthetic panel assumptions

**Everything in Part B is invented.** These numbers were chosen to be plausible and to exercise the methods. They are not estimates of any real insurer's book. The source of truth is the constants at the top of [`src/generate_panel.py`](../../src/generate_panel.py), with seed `20261111` in `src/config.py`.

## Book of business
| Item | Assumption |
|---|---|
| Policyholders | 20,000 |
| Panel | 36 months, Jan 2023 – Dec 2025, one row per active policyholder-month |
| Policy start | Uniform month offset from −36 to +30 relative to Jan 2023, so some policies start before the panel |
| Lapse | 1% per month (planted fraudsters never lapse, so their patterns stay observable) |
| Age band | 18–25 / 26–35 / 36–50 / 51–65 / 66+ in shares 12 / 28 / 32 / 20 / 8% |
| Region | North / South / East / West / Central in shares 22 / 26 / 18 / 24 / 10% |
| Vehicle | Hatchback / Sedan / SUV in shares 45 / 25 / 30%, vehicle age 0–12 |
| Coverage tier | Basic (third-party only) / Standard / Comprehensive in shares 25 / 40 / 35%; base premium ₹600 / 1,100 / 1,900 per month; annual repricing of about +5% |
| Coverage changes | Upgrade 0.4% × frailty per month; downgrade 0.3% per month |
| Shared addresses | 3% of policyholders are in two-person households sharing a contact hash |
| Repair shops | 400, split evenly across regions, with Pareto-skewed popularity |

## Honest claims
- **Claim probability:** 2.5% per month × frailty × age, region and vehicle multipliers × tier multiplier (Basic 0.6, Standard 1.0, Comprehensive 1.2), with at most one honest claim a month.
- **Frailty:** an unobserved claim propensity drawn from Gamma(2, 0.5), mean 1. It also raises the chance of upgrading, which is what separates the fixed-effects and random-effects results.
- **Claim types by tier:** Basic covers third-party claims only. Standard and Comprehensive claims are split across Collision, Theft, Glass, Natural and ThirdParty.
- **Amounts:** lognormal with σ = 0.7 around a median per type: Collision ₹28k, Theft ₹120k, Glass ₹9k, Natural ₹45k, ThirdParty ₹35k. Comprehensive claims are ×1.15. Amounts are clipped to ₹1k–₹10 lakh and rounded to ₹10.
- **Decoys:** many honest claims fall naturally just under ₹50k or soon after the policy starts.

## Planted fraud (disjoint groups of policyholders)
| Pattern | Who | Mechanism |
|---|---|---|
| `early_claim_new_policy` | 100 policyholders | One claim 3–60 days after the policy starts, around ₹60k |
| `early_claim_after_upgrade` | 100 policyholders | A forced upgrade, then a claim 3–60 days later, around ₹60k |
| `rising_frequency` | 0.4% of policyholders | After an onset month, claim probability ramps from 5% to 30% a month over 10 months, with amounts ×1.3 |
| `repeat_similar` | 0.4% of policyholders | 3–4 claims of the same type at the same shop, amounts within ±5%, 30–90 days apart |
| `under_approval_limit` | 0.75% of policyholders | 1–2 claims at 90–99.8% of the ₹50,000 fast-track limit |
| `staged_ring` | 15 rings of 4–8 policyholders | Shared contact hash, one shop, each other named as the third party, claims within a 75-day window, witness present 70% of the time |
| `unpatterned` | 1% of ordinary claims | Inflated ×1.5, with no other pattern |

Additional assumptions:
- **Planted claims always match the policy's cover**, because own-damage patterns need Standard or better. An earlier version broke this rule, and profile check [11] guards against it.
- **Assumed behavioural differences for fraud, which the model can learn:**
  - reporting delay averages 9 days for fraud vs 5 for honest claims
  - a police report is filed for 35% of fraud vs 60% of honest Collision and Theft claims
- **Historical labels:** the past SIU confirmed 85% of true fraud (`fraud_label`). The other 15% looks like a legitimate claim in the training data.

## Planted data-quality problems
| Problem | Rate |
|---|---|
| Exact duplicate claim rows | 0.4% |
| Blank claim amount | 0.2% |
| Report and incident dates swapped | 0.1% |
| Incident date shifted 4 years earlier (before cover) | 0.1% |
| Region with stray case or trailing space | 2% of policyholders |
