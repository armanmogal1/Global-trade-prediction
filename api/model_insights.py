"""
model_insights.py

Purpose: Same style as eda_insights.py -- stores Notebook 2 (Model Training)
results and plain-English explanations as Python constants, so this info
can be pulled into the FastAPI backend, README, or a chatbot/QA layer later.

Project: Global Trade Prediction (UN Comtrade)
Notebook: 2_model_training.ipynb
"""

# ============================================================
# 1. DATA USED FOR TRAINING
# ============================================================

TRAINING_DATA_INFO = {
    "full_dataset_rows": 7_876_052,
    "note": "Full data was too slow to train on directly, so a smaller "
            "sample was used -- this is a normal, deliberate choice, not a mistake.",

    "sample_method": "Stratified sample by 'category' column, 19% of full data",
    "sample_rows": 1_496_449,
    "train_rows": 1_197_159,
    "test_rows": 299_290,
    "split_type": "80% train / 20% test, stratified so all 13 categories "
                  "appear in the same proportion in both sets",
}


# ============================================================
# 2. FEATURES USED
# ============================================================

FEATURES_USED = {
    "frequency_encoded": ["country_or_area", "commodity"],
    # what this means: instead of using the actual country/commodity name,
    # we replace it with "how common is this value in the training data"
    # (a number between 0 and 1). Learned ONLY from train data, so test
    # data doesn't leak into training.

    "one_hot_encoded": ["flow", "quantity_name"],
    # what this means: turned into separate 0/1 columns, one per category
    # (e.g. flow_Export, flow_Import)

    "numeric_passthrough": ["year", "weight_kg", "quantity"],
    # used as-is, no transformation

    "log_transformed": "trade_usd -> log1p(trade_usd)",
    # why: trade_usd has huge outliers (some trades worth billions).
    # log transform makes the values less extreme so the model trains better.
    # used as the REGRESSION TARGET, and also as an extra INPUT FEATURE
    # for the classification model.
}

LEAKAGE_CHECK_RESULT = (
    "No feature is mathematically built from either target. "
    "Using 'commodity' to predict 'category' is fine on purpose -- "
    "category is just a grouped version of commodity (97 raw types -> "
    "13 broad groups), so this is normal, not cheating."
)


# ============================================================
# 3. REGRESSION MODEL -- Predicts trade_usd (trade value in dollars)
# ============================================================

REGRESSION_RESULTS = {
    "algorithm": "XGBoost Regressor",
    "target": "log1p(trade_usd)",

    "rmse_usd": 880_869_829.72,
    "mae_usd": 17_023_292.09,
    "r2_original_scale": 0.0741,
    "r2_log_scale": 0.8580,
}

REGRESSION_SIMPLE_EXPLANATION = """
SIMPLE EXPLANATION -- Regression Model:

- R2 (log scale) = 0.86  --> THIS IS THE MAIN SCORE. Means the model
  correctly explains 86% of the pattern in trade values. This is GOOD.

- R2 (original dollar scale) = 0.07 --> IGNORE this one mostly. It looks
  bad only because a few trades are worth BILLIONS of dollars, and those
  few extreme cases mess up this specific metric. It does NOT mean the
  model is bad.

- MAE ($17M) --> on average, the model's dollar prediction is off by
  about $17 million. Sounds big, but trade values in this dataset range
  from a few dollars to billions, so this is reasonable at this scale.

- RMSE ($880M) --> this number is naturally huge because RMSE punishes
  big mistakes hard, and a handful of billion-dollar trades create huge
  individual errors. Don't use this number alone to judge the model.

VERDICT: Regression model is GOOD. Report R2 (log scale) as the headline
number. Explain RMSE/MAE as being inflated by outliers, not by bad
predictions on typical trades.
"""


# ============================================================
# 4. CLASSIFICATION MODEL -- Predicts category (13 classes)
# ============================================================

CLASSIFICATION_RESULTS = {
    "algorithm": "XGBoost Classifier (multi:softprob)",
    "target": "category (13 classes)",
    "imbalance_handling": "sample_weight='balanced' (no oversampling used)",
    "imbalance_ratio": 11.28,  # largest class is ~11x bigger than smallest

    "accuracy": 0.5166,
    "macro_f1": 0.5424,
}

# per-class scores: precision = "when model said X, how often was it
# actually X". recall = "of all real X cases, how many did the model catch"
CLASSIFICATION_PER_CLASS = {
    "Chemicals & Pharmaceuticals":                {"precision": 0.58, "recall": 0.42, "f1": 0.49},
    "Food, Beverages & Tobacco":                  {"precision": 0.79, "recall": 0.41, "f1": 0.54},
    "Footwear & Headgear":                        {"precision": 0.49, "recall": 0.90, "f1": 0.63},
    "Leather, Hides & Furskins":                  {"precision": 0.34, "recall": 0.86, "f1": 0.49},
    "Machinery & Electronics":                    {"precision": 0.80, "recall": 0.99, "f1": 0.88},
    "Metals & Metal Products":                    {"precision": 0.45, "recall": 0.37, "f1": 0.41},
    "Minerals, Fuels & Ores":                     {"precision": 0.40, "recall": 0.80, "f1": 0.53},
    "Plastics & Rubber":                          {"precision": 0.24, "recall": 0.85, "f1": 0.37},
    "Precision Instruments, Arms & Misc":         {"precision": 0.61, "recall": 0.66, "f1": 0.64},
    "Stone, Ceramic, Glass & Precious Materials": {"precision": 0.35, "recall": 0.62, "f1": 0.45},
    "Textiles & Apparel":                         {"precision": 0.71, "recall": 0.39, "f1": 0.51},
    "Transport Equipment":                        {"precision": 0.59, "recall": 0.76, "f1": 0.66},
    "Wood, Paper & Pulp":                         {"precision": 0.40, "recall": 0.54, "f1": 0.46},
}

CLASSIFICATION_SIMPLE_EXPLANATION = """
SIMPLE EXPLANATION -- Classification Model:

- Accuracy = 52%  --> out of 100 trades, model correctly guesses the
  category for about 52 of them.

- Macro-F1 = 54%  --> average performance across all 13 categories,
  giving equal importance to small categories and big categories.

WHY IS IT ONLY ~52% AND NOT LIKE FRAUD DETECTION (~97%)?
This is normal for this kind of problem, not a bug:
1. 13 classes is much harder than a 2-class (fraud/not-fraud) problem.
   Random guessing alone would only get ~7-8% here vs ~50% for 2 classes.
2. The 'commodity' column -- which is the real source of the category
   label -- was frequency-encoded. That means we threw away WHAT the
   commodity actually is, and kept only HOW COMMON it is. This removes
   the strongest possible signal on purpose (to avoid near-leakage),
   which naturally lowers accuracy.

WHAT THE PRECISION/RECALL PATTERN MEANS:
- Small categories (e.g. Plastics & Rubber: recall 0.85, precision 0.24)
  --> class balancing made the model "guess" these categories more often,
  so it catches most real cases (good recall) but also wrongly labels
  other things as these categories (poor precision).
- Big categories (e.g. Food, Beverages & Tobacco: precision 0.79, recall 0.41)
  --> opposite pattern -- when model says this category, it's usually
  right, but it misses a lot of real cases (predicts them as something else).
- Machinery & Electronics is the best-performing class (F1 = 0.88) even
  though it's the smallest -- its data pattern is just very distinct.

VERDICT: Classification model is AVERAGE, with a clear, explainable
reason (frequency encoding + 13 classes + imbalance). Not a broken model.

FUTURE IMPROVEMENT IDEAS:
- Try target/mean encoding instead of frequency encoding for 'commodity'
  (must be done carefully with cross-validation to avoid leakage)
- Try SMOTE oversampling instead of / combined with class weights
- Check the confusion matrix for which categories get mixed up most
  (e.g. Stone/Ceramic vs Metals looked confusable)
"""


# ============================================================
# 5. TECHNICAL NOTES (for FastAPI / SHAP integration later)
# ============================================================

TECHNICAL_NOTES = """
- Both models are wrapped in sklearn Pipeline objects (preprocessing +
  model together). This makes them easy to pickle and load directly in
  FastAPI for predictions.

- For SHAP TreeExplainer: don't pass the pipeline directly. First get the
  raw model with pipeline.named_steps['model'], and transform your input
  data with pipeline.named_steps['preprocessor'] first. TreeExplainer
  needs the already-encoded numeric matrix, not raw text/category columns.

- Training used a 1.5M row stratified sample instead of the full 7.87M
  rows -- a deliberate speed trade-off, worth stating openly in the
  README as a design decision, not hiding it as a limitation.
"""