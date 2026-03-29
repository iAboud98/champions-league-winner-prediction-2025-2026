# UEFA Champions League Winner Prediction (2025-2026)

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-0A192F)
![Status](https://img.shields.io/badge/Status-Active-success)
![Domain](https://img.shields.io/badge/Domain-Football%20Analytics-1D3557)

---

## 🚀 Project Overview
This project builds a full match-outcome prediction pipeline for football, then uses that trained model to simulate UEFA Champions League knockout matchups.

The workflow is implemented in a single Jupyter notebook and includes:

- historical data cleaning and validation,
- advanced feature engineering (including rolling last-5 team features),
- multi-model training and evaluation,
- scenario-based tournament simulation on custom team fixtures.

The target is binary: predict whether the home team wins.

---

## 🧠 Problem Statement
Predict match outcomes as a binary classification task:

- `ftresult = 1` -> home team win
- `ftresult = 0` -> home team does not win (draw or away win)

After model training, the project creates synthetic head-to-head rows (`team1` vs `team2`) and applies the model to estimate winners in knockout-style pairings.

---

## 📊 Dataset Description

### Historical Dataset
- File: `data/Matches.csv`
- Scope used in notebook: matches after `2020-01-01`

### Tournament Scenario Dataset
- File: `data/ucl_round_8.csv`
- Purpose: custom UCL fixture simulation after training

### Core Raw Columns

| Category | Columns |
|---|---|
| Match Meta | Division, MatchDate, MatchTime, HomeTeam, AwayTeam |
| Team Strength | HomeElo, AwayElo |
| Form | Form3Home, Form5Home, Form3Away, Form5Away |
| Full-Time Outcome | FTHome, FTAway, FTResult |
| Half-Time Outcome | HTHome, HTAway, HTResult |
| Match Stats | HomeShots, AwayShots, HomeTarget, AwayTarget, HomeFouls, AwayFouls, HomeCorners, AwayCorners, HomeYellow, AwayYellow, HomeRed, AwayRed |
| Odds | OddHome, OddDraw, OddAway, MaxHome, MaxDraw, MaxAway |
| Market Lines | Over25, Under25, MaxOver25, MaxUnder25, HandiSize, HandiHome, HandiAway |
| Extra Bookmaker Fields | C_LTH, C_LTA, C_VHD, C_VAD, C_HTB, C_PHB |

---

## 🏗️ Pipeline Overview

### 1. Data Loading
- Import libraries (`pandas`, `numpy`, plotting, sklearn, xgboost).
- Load historical matches from `data/Matches.csv`.
- Inspect schema and shape repeatedly during pipeline transitions.

### 2. Data Preprocessing

#### Column and date normalization
- Filter rows to `MatchDate > 2020-1-1`.
- Normalize all column names: lowercase, stripped, no spaces.
- Convert `matchdate` to datetime.
- Drop `matchtime`.

#### Label consistency checks
- Validate `FTResult` against `FTHome`/`FTAway`.
- Validate `HTResult` against `HTHome`/`HTAway`.

#### Label encoding
- Convert `ftresult` to binary (`H` -> 1 else 0).
- Convert `htresult` to binary (`H` -> 1 else 0).

#### Odds transformation
3-way odds are transformed into two-way style features:

- `oddwin = oddhome`
- `oddloss = 1 / (1/odddraw + 1/oddaway)`
- `maxwin = maxhome`
- `maxloss = 1 / (1/maxdraw + 1/maxaway)`

Then drop raw 3-way fields:

- `oddhome`, `odddraw`, `oddaway`, `maxhome`, `maxdraw`, `maxaway`

#### Column cleanup
- Drop form short windows: `form3home`, `form3away`
- Drop additional sparse/noisy columns:
  - `c_lth`, `c_lta`, `c_vhd`, `c_vad`, `c_htb`, `c_phb`

#### Missing value strategy

Elo:
- Build stacked team-Elo table (home + away).
- Fill missing `homeelo`/`awayelo` by team mean, then global mean.

Match event stats (shots, target, fouls, corners, cards):
- Fill by team median in home/away context.
- Remaining nulls filled by global median.

Market features (`over25`, `under25`, `maxover25`, `maxunder25`, `handisize`, `handihome`, `handiaway`):
- Derive `season` from date with July rollover.
- Fill by `(division, season)` median.
- Backfill with global median.

Half-time score reconstruction:
- `hthome` missing -> `htresult`
- `htaway` missing -> `1 - htresult`

### 3. Feature Engineering

#### Differential features
Create directional matchups as home-minus-away:

- `elodiff`
- `ftgoalsdiff`, `htgoalsdiff`
- `shotsdiff`, `targetdiff`
- `foulsdiff`, `cornersdiff`
- `reddiff`, `yellowdiff`
- `odddiff`, `maxodddif`
- `diff25`, `diffmax25`
- `handidiff`

Then drop many source columns to avoid redundancy.

#### Outlier diagnostics and clipping
- IQR diagnostics run across numerics.
- Upper clipping applied to `oddwin`, `oddloss`, `maxwin`, `maxloss` using `Q3 + 2.5*IQR`.

#### Rolling temporal features (last-5)
This is the most important modeling block:

1. Select base engineered columns.
2. Convert match data into team-centric rows for both home and away sides.
3. Negate away-side values so both sides are represented consistently.
4. Sort by team and date.
5. Compute `shift(1).rolling(5).mean()` per team.
6. Reconstruct match rows as home-rolling minus away-rolling.

Generated rolling features include:

- `ftgoalsdifflast5`, `htgoalsdifflast5`, `shotsdifflast5`, `targetdifflast5`, `foulsdifflast5`, `cornersdifflast5`, `yellowdifflast5`, `reddifflast5`, `htresultlast5`, `odddifflast5`, `maxodddiflast5`, `diff25last5`, `diffmax25last5`, `handidifflast5`

Data reduction after rolling null removal:

- Before dropna: `(56161, 22)`
- After dropna: `(53219, 22)`

### 4. Modeling

#### Temporal split
- Train (2020-2022): `(26501, 22)`
- Validation (2023): `(12169, 22)`
- Test (2024-2025): `(14549, 22)`

Excluded from predictors:

- `ftresult`, `division`, `matchdate`, `hometeam`, `awayteam`, `season`

#### Trained models

| Model | Configuration |
|---|---|
| Logistic Regression | `LogisticRegression(random_state=42, max_iter=1000)` |
| Random Forest | `RandomForestClassifier(random_state=42, n_estimators=1000)` |
| XGBoost | `XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')` |

### 5. Evaluation
Validation accuracy:

- Logistic Regression: `0.6290`
- Random Forest: `0.6131`
- XGBoost: `0.6152`

Confusion matrices are generated for all three.

### 6. Simulation (Knockout-style)

Scenario workflow:

1. Load `data/ucl_round_8.csv`.
2. Re-apply engineered feature logic to scenario data.
3. Aggregate team rows by 5-row blocks into team-level last-5 style representation.
4. Insert team order and map manual Elo values.
5. Build synthetic matchups as `team1 - team2` feature vectors.
6. Align scenario rows to training predictors.
7. Predict winners using Random Forest probabilities and class outputs.

---

## 🧪 Model Performance

### Precision Summary (Primary Metric)

| Model | Precision |
|---|---:|
| Logistic Regression | 0.639 |
| XGBoost | 0.602 |
| Random Forest | 0.596 |

### Accuracy Summary

| Model | Validation Accuracy |
|---|---:|
| Logistic Regression | 0.6290 |
| Random Forest | 0.6131 |
| XGBoost | 0.6152 |

### Confusion Matrix Summary

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 5528 | 1203 | 3312 | 2126 |
| Random Forest | 5203 | 1528 | 3180 | 2258 |
| XGBoost | 5264 | 1467 | 3216 | 2222 |

---

## ⚔️ Predictions

### Quarterfinal simulation output

| team1 | team2 | team1_win_prob | predicted_result |
|---|---|---:|---|
| Barcelona | Atletico | 0.444 | Atletico |
| Arsenal | Sporting | 0.532 | Arsenal |
| PSG | Liverpool | 0.334 | Liverpool |
| Real Madrid | Bayern | 0.479 | Bayern |

### Semifinal simulation output

| team1 | team2 | team1_win_prob | predicted_result |
|---|---|---:|---|
| Atletico | Arsenal | 0.434 | Arsenal |
| Liverpool | Bayern | 0.410 | Bayern |

### Final simulation output

| team1 | team2 | team1_win_prob | predicted_result |
|---|---|---:|---|
| Arsenal | Bayern | 0.455 | Bayern |

---

## 📂 Project Structure

```text
.
├── main.ipynb
├── README.md
├── requirements.txt
├── steps.txt
└── data
    ├── Matches.csv
    └── ucl_round_8.csv
```

---

## ⚙️ How to Run

### 1) Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the Notebook
Open `main.ipynb` and execute cells sequentially from top to bottom.

Recommended order:

1. Data loading and cleaning
2. Feature engineering and rolling stats
3. Model training and validation
4. UCL simulation and custom match predictions

### 3) Expected Inputs and Outputs

Inputs:

- `data/Matches.csv`
- `data/ucl_round_8.csv`

Key in-memory outputs:

- Models: `lr_model`, `rf_model`, `xgb_model`
- Engineered data: `df`
- Scenario dataframes: `ucl_df`, `h2h_df`, `h2h_df_new`, `h2h_one_row_df`
- Prediction outputs: `predictions_df`, `predictions_new_df`, `prediction_one_row_df`

---

## 📦 Dependencies

From `requirements.txt`:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

---

## ⚠️ Assumptions and Limitations

- Binary target collapses draw and away win into one class (`0`), which may lose tactical nuance.
- Manual Elo mapping and manual fixture selection are used in simulation.
- Team-name mismatches can break lookup/mapping logic.
- No explicit calibration step for predicted probabilities.
- No hyperparameter search or cross-validation pipeline in this version.
- Potential leakage risk is low for rolling features because `shift(1)` is used, but careful chronological execution remains required.

---

## 🔮 Future Improvements

- Move to multiclass outcome prediction (home win / draw / away win).
- Add hyperparameter optimization and robust validation strategy.
- Add probability calibration and reliability plots.
- Add stricter team-name normalization and data contracts for simulation files.
- Export trained model artifacts and prediction reports.
- Refactor notebook into modular Python package with tests and CI/CD.

---

## ✅ Final Note
This project combines football domain logic, temporal feature engineering, and practical simulation to produce actionable matchup predictions for knockout scenarios in a reproducible notebook workflow.
