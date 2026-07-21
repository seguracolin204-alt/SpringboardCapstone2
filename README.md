# Predicting Residential Solar Adoption Across U.S. Counties

![cover](reports/figures/08_predicted_vs_actual.png)

*U.S. residential solar has grown from about 30,000 installations a year in 2010 to more than 700,000 today, but that growth is deeply uneven across the country. Two counties with nearly identical sunshine, income, and home ownership can differ by an order of magnitude in installations per capita. This project builds a county-level regression model that predicts residential PV adoption from publicly available data and identifies which features actually drive it — so installers can find underserved markets, utilities can forecast distributed solar on their feeders, and state energy offices can evaluate whether their incentive programs are working.*

## 1. Data

The project combines three public data sources at the U.S. county level.

> - [LBNL Tracking the Sun](https://emp.lbl.gov/tracking-the-sun) — installation-level data on 3M+ U.S. distributed PV systems (system size, install date, cost, location). This is the primary source for the modeling target.
> - [U.S. Census American Community Survey](https://www.census.gov/programs-surveys/acs/data.html) — county-level median household income, housing units, owner-occupied units, and median year housing was built.
> - [EIA Open Data](https://www.eia.gov/opendata/) — state-level residential retail electricity prices.

Because the raw Tracking the Sun release is several gigabytes, the repository ships with schema-faithful synthetic sample CSVs so the pipeline runs end-to-end without downloads. The notebooks execute unchanged on the real data — you only need to swap in the real files.

## 2. Method

This is a **regression** problem: the target (`installs_per_1k_oo_hh`) is continuous. The modeling target is `log1p(installs_per_1k_oo_hh)` since the raw target is heavily right-skewed (skewness = 2.4), then predictions are transformed back to the raw scale for interpretable metrics.

Four regressors were compared across three algorithm families:

1. **Linear Regression** (baseline, no regularization)
2. **Ridge Regression** (L2 regularization, hyperparameter `alpha` tuned via 5-fold GridSearchCV)
3. **Random Forest Regressor** (bagged trees, `n_estimators`/`max_depth`/`min_samples_leaf` tuned)
4. **Gradient Boosting Regressor** (boosted trees, `n_estimators`/`max_depth`/`learning_rate` tuned)

The final model was selected on cross-validated **RMSE** on the raw target scale. RMSE was chosen over MAE because large prediction errors are disproportionately costly for real stakeholders — an installer would rather see a 10% error in each of 10 counties than a 100% error in one.

## 3. Data Wrangling

[Data Wrangling Notebook](01_data_wrangling.ipynb)

The wrangling step reduced 12,724 raw installation rows to 50 clean county rows with a modeling target ready to use. Key cleaning decisions:

- **County FIPS was forced to string on read** so leading zeros (California `06xxx`, Colorado `08xxx`, Connecticut `09xxx`) survived the load. Losing them would have caused silent join failures with Census data.
- **Residential filter** applied to Tracking the Sun before aggregation — commercial systems misclassified as residential would have inflated the per-capita target.
- **System size outliers** dropped in two directions: non-positive values (data entry errors) and above 30 kW (commercial). Cost outliers (zero or over $200k) were converted to `NaN` rather than dropped, because cost is a feature — the rest of the row is still usable.
- **Census sentinel values** (`-666666666`) replaced with `NaN` before any arithmetic. Left in place they would have made downstream means and medians nonsensical.
- **Left join Census → Tracking the Sun** so counties with zero installations were kept as zero-adoption observations rather than dropped.

## 4. EDA

[Exploratory Data Analysis Notebook](02_eda.ipynb)

The EDA surfaced findings that reshaped the modeling plan.

**Target distribution.** Strong right skew (skewness = 2.4), which motivated the log transform for linear models.

![target distribution](reports/figures/01_target_distribution.png)

**Feature relationships with target.** Contrary to the project proposal's expectation, the strongest single correlate is not electricity price or income — it is `population`, with a large **negative** Pearson r of −0.69. Bigger counties have lower per-capita adoption, likely because urban counties skew multifamily/renter-occupied. This finding shaped feature selection for the modeling step.

![correlation heatmap](reports/figures/04_correlation_heatmap.png)

**State-level variation.** The boxplot shows visible state differences in adoption, with California, Arizona, and Massachusetts at the top and Michigan, Ohio, and Pennsylvania at the bottom. The ANOVA test could not formally detect the effect at this sample size (n=50 across 21 states), but the visual pattern is consistent with what state-level policy differences would produce.

![state boxplot](reports/figures/05_state_boxplot.png)

## 5. Pre-processing

[Pre-processing Notebook](03_preprocessing.ipynb)

The pre-processing step produced the model-ready train/test split:

- **Median imputation** with binary missing-indicator flags for `median_household_income` and `median_year_built` (3 rows total, all Census-suppressed).
- **Engineered features:** `log_population`, `housing_age`, `log1p_target`.
- **One-hot encoded** the `state` categorical with `drop_first=True`, adding 20 dummy columns.
- **80/20 train/test split** with `random_state=42` (40 training / 10 testing counties on the synthetic sample).
- **StandardScaler fit on the training set only**, then applied to both partitions. Fitting on the full dataset before splitting would leak test statistics into training.

The output is four CSVs (`X_train_scaled.csv`, `X_test_scaled.csv`, `y_train.csv`, `y_test.csv`) that feed directly into the modeling notebook.

## 6. Algorithms & Machine Learning

[Modeling Notebook](04_modeling.ipynb)

All four models were evaluated with the same 5-fold CV harness on the training set, then compared on RMSE, MAE, and R² measured on the raw target scale.

![model comparison](reports/figures/06_model_comparison.png)

| Model | CV RMSE | CV R² | Notes |
|---|---|---|---|
| Linear Regression | 1.04 ± 0.59 | -6.6 | Baseline; overfits severely at p ≈ n |
| Ridge (α = 10) | 0.80 ± 0.65 | -0.05 | Regularization tames OLS instability |
| Random Forest (tuned) | 0.73 ± 0.58 | +0.19 | Captures nonlinearity via bagged trees |
| **Gradient Boosting (tuned)** | **0.63 ± 0.48** | **+0.28** | **Selected as final model** |

**WINNER: Gradient Boosting Regressor**
Hyperparameters: `n_estimators=300`, `max_depth=2`, `learning_rate=0.1`

> The plain OLS baseline overfits catastrophically — 28 features and 40 training rows is exactly the regime OLS was not designed for. Ridge regularization stabilizes it. Both tree ensembles beat the linear models, and Gradient Boosting wins by a comfortable margin. This is the model architecture the project proposal named for the final version.

## 7. Feature Importance

Once selected, the Gradient Boosting model was fit on the full training set and its native feature importances extracted. `log_population` dominates (80% importance), followed by `residential_price_cents_kwh` and a California state indicator.

![feature importance](reports/figures/07_feature_importance.png)

This importance ordering is consistent with the EDA correlations — population showed the strongest bivariate relationship with the target, and electricity price was the strongest continuous economic feature. That consistency across two different analyses adds confidence that the model is picking up real signal rather than fitting noise.

## 8. Predictions on the Held-out Test Set

The 10 test-set counties were touched only once, at the final evaluation step, to produce an honest generalization estimate.

**Test set: RMSE = 0.920, R² = +0.757**

![predicted vs actual](reports/figures/08_predicted_vs_actual.png)

Points cluster along the 45° reference line, with the model slightly under-predicting at the high end — a common pattern for tree ensembles, which regress toward the training mean. The test R² is higher than the CV R² because the test partition happened to include the widest-spread counties (0.16 to 5.89 installs per 1,000 households), and R² is easier to achieve when the target has more variance to explain.

## 9. Recommendations

The three concrete recommendations the modeling supports:

1. **For solar installers:** The counties where actual adoption is running *below* the model's prediction are the most commercially interesting — those are underserved markets where the demographic and economic fundamentals should support solar but the installer channel hasn't fully covered. The model's residuals produce a rankable list of those markets.

2. **For utilities:** Population is the single most predictive feature, and it's negatively correlated with per-capita adoption. That's an actionable planning signal — the highest-growth distributed-solar territories are unlikely to be the utility's densest urban service areas. Feeder-level planning should prioritize mid-sized county service areas with elevated electricity prices.

3. **For state energy offices:** Median household income showed essentially no linear relationship with adoption in this analysis (r ≈ +0.06). If that finding holds on the full dataset, income-targeted rebate programs are targeting the wrong variable. Electricity price and housing stock characteristics appear to be stronger levers.

## 10. Limitations & Future Improvements

- **Sample size.** The synthetic sample is 50 counties. The real dataset is ~3,100. Every effect size in this report has wider confidence intervals than what the real dataset will produce. The pipeline is production-ready; the numbers are directional.
- **Additional features to add.** [NREL NSRDB](https://nsrdb.nrel.gov/) irradiance data at the county level, and [DSIRE](https://www.dsireusa.org/) policy indicators (net metering strength, SREC market presence, rebate availability) were deferred to a follow-up feature-engineering pass. Both are named in the original project proposal.
- **Cross-sectional vs. panel.** This model uses a single most-recent-year snapshot. A panel model using all years of EIA prices and DSIRE policy history would let us study within-county change and produce cleaner causal claims about incentive effectiveness.
- **Housing tenure.** Population's role as the top predictor is probably a proxy for housing tenure and building type (single-family vs. multifamily). An explicit tenure feature from ACS would separate these effects.

## 11. Reports & Reference

- [Final Project Report (PDF)](reports/Capstone_Two_Final_Report.pdf)
- [Slide Deck (PPTX)](reports/Capstone_Two_Slide_Deck.pptx)
- [Model Metrics File](reports/Model_Metrics.txt)

## 12. Repository Structure

```
capstone_two_solar/
├── README.md                            <- this file
├── requirements.txt                     <- Python dependencies for reproducibility
├── 01_data_wrangling.ipynb              <- Step 2: cleaning and merging
├── 02_eda.ipynb                         <- Step 3: exploratory data analysis
├── 03_preprocessing.ipynb               <- Step 4: feature engineering, scaling, split
├── 04_modeling.ipynb                    <- Step 5: model comparison and selection
├── tracking_the_sun_sample.csv          <- raw input: installation-level PV data
├── acs_county_housing_income.csv        <- raw input: Census demographics
├── eia_state_electricity_prices.csv     <- raw input: state electricity prices
├── county_solar_adoption.csv            <- Step 2 output: cleaned county-level dataset
├── X_train_scaled.csv                   <- Step 4 output: scaled training features
├── X_test_scaled.csv                    <- Step 4 output: scaled test features
├── y_train.csv                          <- Step 4 output: training target (raw + log)
├── y_test.csv                           <- Step 4 output: test target (raw + log)
├── final_model.joblib                   <- Step 5 output: serialized final model
├── generate_synthetic_raw.py            <- reproduces the sample data from scratch
└── reports/
    ├── Capstone_Two_Final_Report.pdf
    ├── Capstone_Two_Slide_Deck.pptx
    ├── Model_Metrics.txt
    └── figures/                         <- charts used in the report and README
```

## 13. How to run

```bash
pip install -r requirements.txt
jupyter notebook
```

Then open and run the notebooks in order (01 → 04). Every notebook expects the data files in the same directory as the notebook itself, so no path configuration is needed.

## 14. Credits

Thanks to my Springboard mentor for guidance across the six-step Data Science Method, and to the LBNL, U.S. Census Bureau, and EIA teams that maintain the public data sources this project relies on.
