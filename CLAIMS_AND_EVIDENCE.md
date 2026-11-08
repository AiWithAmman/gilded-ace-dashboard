# Claims & Evidence appendix

Every number is rebuilt from `data/` by `python analysis/build_data.py`; the dashboard (`dashboard.html`) shows the same claims, numbered, in its appendix section. Window: gaming days 2025-01-01 to 2025-06-30, active machine-days only.

| # | Claim | Table / query | Result | Alternative tested | Limitation |
|---|---|---|---|---|---|
| 1 | 9 of the manager's bottom 10 sit in Back Forty, where the same game earns ~34% less theo per hour | `prepared_daily_view` → multiplicative fit of theo per available hour = zone × month × weekday rate × game factor | Location factor 0.65–0.68 for Back Forty machines | Fewer players choose these specific titles → machine index vs same-game peers in-zone ≈ 1.0 | Benchmark fitted on the same data; zones with few machines of a game are noisier |
| 2 | None of the bottom 10 is truly weaker than its in-zone peers | Machine index = Σ theo ÷ Σ location-expected theo, ratio-estimator 90% interval | Every bottom-10 upper bound ≥ 0.90; the only machine below, M-1184 (0.80), is not on the list | Short windows (M-5076, M-1118) → wide intervals shown | Theo-based: assumes RTP settings are correct (see #3) |
| 3 | M-1158 likely has a wrong paytable or setting | `machine_day.payout_positive_spins ÷ paid_spins` vs `game_math.hit_probability`, binomial z | 38.6% vs 31.5%, z ≈ 124, stable every week; luck z −4.6; $36K below theo | Meter or reporting error → the inspection resolves it; luck alone cannot produce z = 124 | A lead to inspect, not proof of wrongdoing |
| 4 | Luck is spread close to chance; only one machine is an outlier | (house win − theo) ÷ √(`machine_day.sum_bet_squared` × `game_math.payout_multiplier_variance`) | 24 of 310 machines beyond \|z\| 1.96 vs ~16 expected; only M-1158 passes the multiple-testing cut-off (−3.6) | Variance may understate jackpots → checked by volatility: low-volatility games 3.8% beyond 1.96 (chance), medium to very-high 9–11% | Normal approximation is weak for heavy-tailed games, so \|z\| between 2 and 3 is weak evidence |
| 5 | The location effect is causal, not just correlation | Natural experiment: M-1152 moved Back Forty → Main Entrance on 2025-02-04 (`prepared_daily_view.location_id`) | Theo/hr $14.3 → $29.4, same machine and game | Seasonality → the floor-wide rate fell 9% Feb–Jun while M-1152 doubled; two control moves between busy zones changed −2% and −13% | One machine; the displaced slot's previous occupant isn't observed |
| 6 | Back Forty depends on foot traffic from the entrance | Zone × day theo/hr, walkway closure 2025-03-03 to 04-06 (`floor_walkways`), difference-in-differences | Back Forty −25% vs rest +6% (≈ −30% relative), ≈ $152K theo lost | March events lifted the rest → measured relative to the rest of the floor | Single closure episode |
| 7 | Downtime cost ≈ 0.4% of theo; worst on the grave shift or when short-staffed | `machine_day` available vs `calendar_day.scheduled_seconds`; `operations` × `shift_roster` | 5,022 down hours, $93.8K lost theo; wait 0.6–0.7 h with 2 techs, ~2 h with 1 tech or short-staffed; Ridgeline faults 2.9× the others | Night faults may simply be harder → fix time (work start → available) is 2.9–3.0 h on every shift, so the gap is waiting | Lost theo is an upper bound (players move machines) |
| 8 | A weekday model beats the constant baseline by ~70% on a time-based holdout | `property_day` + `events` announced before 2025-05-01; train Jan–Apr, test May–Jun | MAE $68.7K (weekday), $63.8K (weekday × event) vs $218.3K (constant) | Trend model → overfit, rejected (MAE $98.3K) | Two-month holdout; July (holiday) may differ |
| 9 | The comp redemption rate was cut ~56% on May 1 with no visible play lift | `loyalty_ledger` points ÷ face value; `property_day` | 13.1 → 5.8 pts per $; comp cost/day +7.6%; coin-in/day −2.1% | Seasonal decline (LVCVA visitation −1% to −12% YoY) → hence a controlled pilot | Before/after only; no control group |
| 10 | Player results are highly concentrated | `player_period` summed per player | Top 1% = 20% of theo and 18% of net losses; 16% of players ahead | Carded play only; uncarded players excluded | Not a target list; responsible-gaming use only |

## Priced actions (July 1–30, 2025)

Cost = quantity × (unit cost + daily cost × max(0, 30 − lead days)). Benefit = 30-day theo after the lead time.

| Action | Qty / target | Cost | Est. benefit (range) | Evidence | Measurement |
|---|---|---:|---|---|---|
| A-INSPECT | 1 × M-1158 | $1,200 | ~$4.1K ($2.3K–$5.5K; P(wrong setting) 40–95%) | #3 | Weekly hit rate back to 31.5% ± 0.5 pt; hold vs theo for 4 weeks |
| A-WAYFINDING | 1 × Back Forty | $7,500 | ~$13.2K ($2.6K–$26.3K; lift 1–10%) | #5, #6 | DiD: Back Forty theo/hr and entries vs the other zones, Jul 11–30 vs June, 90% CI |
| A-COMP-PILOT | 1, 50/50 by player_id hash | $4,000 (no added comp cost: the test arm returns to the stricter pre-May rate) | Information; reverting saves ~$28K/yr if no play lift | #9 | Theo per player, visits, comp cost ÷ theo; revert if test is within −2% of control |
| Keep | — | $87,300 | — | — | — |

Manager's plan for comparison: lease 10 replacements = 10 × ($2,500 + 16 × $55) = $33,800 for ~$1.6K of July theo (claims #1–#2).

## How to open and reproduce

- Open `dashboard.html` in any modern browser. It needs no server; fonts load from Google Fonts when online and fall back to system fonts offline.
- Rebuild: `python -m pip install pandas pyarrow`, then `python analysis/build_data.py` from the repository root (Python 3.11+). This rewrites `analysis/dashboard_data.json` and the embedded data block in `dashboard.html`.

## AI-use disclosure

The analysis pipeline (`analysis/build_data.py`), the dashboard (HTML/SVG/JS, including the visual design), this appendix and the write-up were drafted with Claude (Anthropic) via Claude Code. Claude wrote the pandas code and the charts, and drafted the text. Every number is computed from the provided data by the script.

The walkthrough video (https://youtu.be/Cg1c1aeTQMA) was also produced with Claude: an automated screen recording of the dashboard with captions and animations. I chose not to use my own voice publicly, so the narration uses a synthetic text-to-speech voice (Windows "Microsoft David"), not a recording of a real person.
