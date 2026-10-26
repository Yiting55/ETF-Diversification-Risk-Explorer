# Out-of-sample validation

Pre-crisis window: 5 years of daily returns ending the day before each crisis. Crisis windows: S&P 500 peak to trough (COVID 2020-02-19 to 2020-03-23; 2022-01-03 to 2022-10-12). Worst fall is buy-and-hold over the crisis window. Seed 7.

## Test 1: does a higher pre-crisis score mean a smaller crisis fall?

The same 1000 random portfolios in both crises, 3-8 ETFs from a 30-ETF universe, random weights.

| Crisis | Spearman(score, fall) | Same, controlling for pre-crisis volatility | Avg fall, lowest-score fifth | Avg fall, highest-score fifth |
|---|---|---|---|---|
| covid | +0.60 | -0.10 | -33.1% | -20.1% |
| 2022 | +0.40 | +0.09 | -24.2% | -19.3% |

Positive correlation = portfolios with more equivalent bets before the crisis fell less.

## Test 2: did the What-if top pick, chosen before the crisis, soften the fall?

The same 500 equity-heavy portfolios in both crises (half also hold a bond fund). For each, the candidate with the most added bets at 10%, using pre-crisis data only.

| Crisis | Top pick softened the fall | Avg change in fall, top pick | Avg change in fall, all candidates | Most common top picks |
|---|---|---|---|---|
| covid | 100% | +4.49 pts | +1.64 pts | TLT 98%, DBC 1%, GLD 1% |
| 2022 | 17% | -0.39 pts | +0.42 pts | TLT 100% |

Change in fall > 0 means a shallower fall than the original portfolio.

### Each candidate's average effect

| Candidate | COVID: avg change in fall | 2022: avg change in fall | Softened both (share of portfolios) |
|---|---|---|---|
| IEF | +3.92 pts | +0.80 pts | 100% |
| TLT | +4.56 pts | -0.39 pts | 17% |
| GLD | +2.91 pts | +1.20 pts | 99% |
| VNQ | -0.86 pts | -0.65 pts | 3% |
| DBC | +0.73 pts | +2.78 pts | 86% |
| EEM | +0.24 pts | -0.46 pts | 13% |
| VXUS | -0.02 pts | -0.38 pts | 10% |

## Test 3: did the best COVID cushion also help in 2022?

Across 500 portfolios, the candidate that cushioned COVID best was most often TLT (91%). Holding that same pick in 2022 softened the fall in 22% of portfolios, with an average change of -0.30 pts.
