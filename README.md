# SupeRR Selector Season 2 · Round 1: Who Delivers When It Matters?

An analytical framework and interactive scouting dashboard identifying the most reliable T20 batters and bowlers under high-pressure match states to inform **Rajasthan Royals' IPL 2027 Auction Strategy**.

---

## Executive Summary
* **No Separate "Clutch" Trait:** High-pressure outperformance largely reflects elite baseline quality in high-leverage contexts rather than lucky streaks.
* **High-Pressure Concentration:** **19.5%** of deliveries exceed the High-Pressure threshold (`PI ≥ 1.5`), producing a **6.0× larger** average win-probability swing than ordinary balls.
* **Top Auction & Retention Picks:**
  * **Batters:** **Suryakumar Yadav** (steadiest: Top-10 in 73% of bootstrap resamples), **Tim David** & **Andre Russell** (death overs), **Heinrich Klaasen** (middle overs), and **Phil Salt** (powerplay). *(**Vaibhav Sooryavanshi** ranks #1 overall on 137 pressure balls and is flagged as an existing RR retention asset.)*
  * **Bowlers:** **Jasprit Bumrah** (#1 across every threshold), **Sunil Narine** & **Kuldeep Yadav** (middle-over squeeze), and **Matheesha Pathirana** & **Nathan Ellis** (death).

---

## Top-10 Consensus Rankings

| Rank | Top-10 Batters | PV / 100 Balls | Top-10 Bowlers | PV / 100 Balls |
| :--- | :--- | :---: | :--- | :---: |
| **1** | Vaibhav Sooryavanshi *(RR)* | +28.6 | Jasprit Bumrah | +19.0 |
| **2** | Abhishek Sharma | +21.5 | Sunil Narine | +15.2 |
| **3** | Suryakumar Yadav | +22.4 | Matheesha Pathirana | +13.4 |
| **4** | Tim David | +24.9 | Sufiyan Muqeem *(Analytical)* | +12.3 |
| **5** | Heinrich Klaasen | +30.5 | Josh Hazlewood | +8.7 |
| **6** | Andre Russell | +19.2 | Nathan Ellis | +12.8 |
| **7** | Phil Salt | +13.9 | Blessing Muzarabani *(T20I-Heavy)* | +13.7 |
| **8** | Rajat Patidar | +20.0 | Kuldeep Yadav | +15.5 |
| **9** | Jos Buttler | +19.9 | Mustafizur Rahman | +11.0 |
| **10** | Glenn Phillips *(T20I-Heavy)* | +13.2 | Bhuvneshwar Kumar | +11.6 |

---

## Methodology Pipeline
1. **Data Cleaning (`code/prep.py`):** Filtered 818 completed, non-DLS matches (357 IPL, 461 T20I; 195,124 deliveries) from 2022 onward and decoded extras codes.
2. **Win-Probability & Pressure Index (`code/model.py`):** Out-of-fold gradient-boosted win-probability models (`0.875` 2nd-innings AUC; `0.690` 1st-innings AUC) computing expected `|ΔWP|` per ball, scaled to `1.0` average PI with a `+25%` knockout boost.
3. **Context-Adjusted Valuation (`code/gbm.py`, `code/rate3.py`, `code/rate5.py`):** Expected runs and dismissal probabilities conditioned on match state, pace/spin, and opponent quality, with separate empirical-Bayes shrinkage for runs and wickets plus a consistency score.
4. **Consensus & Validation (`code/robust.py`, `code/consensus.py`):** Qualified 169 batters and 170 bowlers (`≥50` pressure balls, `≥12` pressure matches, `≥150` total balls) and averaged ranks across `PI ≥ 1.25`, `1.50`, `2.00`, and 300 match-level bootstrap resamples.

---

## Repository Structure
* `index.html` — Interactive Rajasthan Royals Pressure Analytics Dashboard.
* `report/` — Full Round 1 Word submission (`.docx`) and PDF methodology report (`.pdf`).
* `data/` — Complete rankings and bootstrap metrics CSV for all 339 qualified players.
* `code/` — Reproducible Python modeling and visualization scripts.

---

**Author:** Varsha Digambar Patil · B.Tech IT, NMIMS Shirpur
