import json
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:                                   # run from project root: uvicorn api.main:app
    from api.encoders import FrequencyEncoder
except ImportError:                    # run from inside api/: uvicorn main:app
    from encoders import FrequencyEncoder

# CRITICAL: the notebook defined FrequencyEncoder in __main__, so the pickles
# reference '__main__.FrequencyEncoder'. Register it so pickle.load can find it.
sys.modules["__main__"].FrequencyEncoder = FrequencyEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# Schemas
# ============================================================

class TradeInput(BaseModel):
    country_or_area: str = Field(default="India", description="Reporting country")
    commodity: str = Field(
        default="Petroleum oils, oils from bituminous minerals, crude",
        description="Commodity name (as in the dataset)",
    )
    flow: str = Field(default="Export", description="Export, Import, Re-Export or Re-Import")
    quantity_name: str = Field(default="Weight in kilograms", description="Unit of quantity")
    year: int = Field(default=2016, ge=1988, le=2016)
    weight_kg: float = Field(default=1000.0, ge=0)
    quantity: float = Field(default=1000.0, ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "country_or_area": "India",
                "commodity": "Petroleum oils, oils from bituminous minerals, crude",
                "flow": "Import",
                "quantity_name": "Weight in kilograms",
                "year": 2016,
                "weight_kg": 5000000.0,
                "quantity": 5000000.0,
            }
        }
    }


class CategoryProb(BaseModel):
    category: str
    probability: float


class PredictionOutput(BaseModel):
    predicted_trade_usd: float
    predicted_category: str            # ML model's prediction (commodity name NOT used)
    category_confidence: float         # real model probability
    top_categories: List[CategoryProb]
    commodity_category: Optional[str] = None   # reference category looked up from the commodity


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class ExplainOutput(BaseModel):
    predicted_trade_usd: float
    predicted_category: str
    regression_base_value: float
    regression_top_features: List[FeatureContribution]
    classification_base_value: float
    classification_top_features: List[FeatureContribution]
    insight_summary: str


# ============================================================
# App + static data
# ============================================================

app = FastAPI(title="Global Trade Prediction API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TOP_15_COMMODITIES = [
    {"commodity": "Petroleum oils, oils from bituminous minerals, crude", "trade_value_billion_usd": 18866},
    {"commodity": "Oils petroleum, bituminous, distillates, except crude", "trade_value_billion_usd": 9883},
    {"commodity": "Medicaments nes, in dosage", "trade_value_billion_usd": 5234},
    {"commodity": "Fixed wing aircraft, unladen weight > 15,000 kg", "trade_value_billion_usd": 3917},
    {"commodity": "Natural gas in gaseous state", "trade_value_billion_usd": 2992},
    {"commodity": "Gold in unwrought forms non-monetary", "trade_value_billion_usd": 2450},
    {"commodity": "Aircraft parts nes", "trade_value_billion_usd": 2373},
    {"commodity": "Iron ore, concentrate, not iron pyrites, unagglomerate", "trade_value_billion_usd": 2225},
    {"commodity": "Commodities not specified according to kind", "trade_value_billion_usd": 1912},
    {"commodity": "Natural gas, liquefied", "trade_value_billion_usd": 1854},
    {"commodity": "Bituminous coal, not agglomerated", "trade_value_billion_usd": 1695},
    {"commodity": "Automobiles, spark ignition engine of 1500-3000 cc", "trade_value_billion_usd": 1565},
    {"commodity": "Copper ores and concentrates", "trade_value_billion_usd": 1235},
    {"commodity": "Wheat except durum wheat, and meslin", "trade_value_billion_usd": 1152},
    {"commodity": "Antisera and other blood fractions", "trade_value_billion_usd": 1114},
]


# ---------- Fixed category for the 15 dropdown commodities ----------
# Category is determined by the commodity, so for these known commodities the
# category is shown directly. Takes priority over commodity_category.json.
# To change one, just edit the category on its line.
COMMODITY_CATEGORY_OVERRIDE = {
    "Petroleum oils, oils from bituminous minerals, crude": "Minerals & Metals",
    "Oils petroleum, bituminous, distillates, except crude": "Minerals & Metals",
    "Medicaments nes, in dosage": "Chemicals & Plastics",
    "Fixed wing aircraft, unladen weight > 15,000 kg": "Transport Equipment",
    "Natural gas in gaseous state": "Minerals & Metals",
    "Gold in unwrought forms non-monetary": "Minerals & Metals",
    "Aircraft parts nes": "Transport Equipment",
    "Iron ore, concentrate, not iron pyrites, unagglomerate": "Minerals & Metals",
    "Commodities not specified according to kind": "Precision & Misc",
    "Natural gas, liquefied": "Minerals & Metals",
    "Bituminous coal, not agglomerated": "Minerals & Metals",
    "Automobiles, spark ignition engine of 1500-3000 cc": "Transport Equipment",
    "Copper ores and concentrates": "Minerals & Metals",
    "Wheat except durum wheat, and meslin": "Food & Agri",
    "Antisera and other blood fractions": "Chemicals & Plastics",
}

# ============================================================
# Load models
# ============================================================

try:
    with open(MODEL_DIR / "regressor.pkl", "rb") as f:
        reg_pipeline = pickle.load(f)
    with open(MODEL_DIR / "classifier.pkl", "rb") as f:
        clf_pipeline = pickle.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
except FileNotFoundError as e:
    raise RuntimeError(f"Model file not found: {e}. Run the save cell in models.ipynb first.")
except Exception as e:
    raise RuntimeError(
        f"Could not load models: {e}. Library versions in requirements.txt must "
        f"match the environment where models.ipynb was run."
    )

# Reference lookup: commodity -> category. Category is derived from the commodity
# code, so this is a deterministic reference shown separately from the ML prediction.
# First run: built from dataset/trade_cleaned.parquet and cached in
# api/commodity_category.json (commit that json so deployment doesn't need the parquet).
LOOKUP_JSON = BASE_DIR / "api" / "commodity_category.json"
PARQUET_PATH = BASE_DIR / "dataset" / "trade_cleaned.parquet"


def load_commodity_lookup() -> dict:
    if LOOKUP_JSON.exists():
        with open(LOOKUP_JSON, encoding="utf-8") as f:
            return json.load(f)
    if PARQUET_PATH.exists():
        try:
            import pyarrow.parquet as pq
            df = pq.read_table(
                PARQUET_PATH, columns=["commodity", "category_final"],
                read_dictionary=["commodity", "category_final"],
            ).to_pandas()
            counts = (df.groupby(["commodity", "category_final"], observed=True)
                        .size().reset_index(name="n"))
            counts = counts.sort_values("n").drop_duplicates("commodity", keep="last")
            mapping = {str(c): str(k) for c, k in zip(counts["commodity"], counts["category_final"])}
            with open(LOOKUP_JSON, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False)
            return mapping
        except Exception as e:
            print(f"[lookup] could not build commodity lookup from parquet: {e}")
            return {}
    return {}


COMMODITY_CATEGORY = load_commodity_lookup()
if COMMODITY_CATEGORY:
    print(f"[lookup] commodity->category loaded: {len(COMMODITY_CATEGORY)} commodities")
else:
    print("[lookup] WARNING: lookup is empty, so the Category card will show the ML model's guess. "
          "Check that dataset/trade_cleaned.parquet has 'commodity' and 'category_final' columns.")

# Pull the raw estimators out of the pipelines (SHAP needs the booster, not the pipeline)
reg_preprocessor = reg_pipeline.named_steps["preprocessor"]
reg_model = reg_pipeline.named_steps["model"]
clf_preprocessor = clf_pipeline.named_steps["preprocessor"]
clf_model = clf_pipeline.named_steps["model"]   # SMOTE step is training-only, ignored here


# ============================================================
# Feature building (must mirror models.ipynb exactly)
# ============================================================

BASE_COLS = ["country_or_area", "commodity", "flow", "quantity_name",
             "year", "weight_kg", "quantity"]


def build_reg_df(d: TradeInput) -> pd.DataFrame:
    return pd.DataFrame([{c: getattr(d, c) for c in BASE_COLS}])


def build_clf_df(d: TradeInput, trade_usd: float) -> pd.DataFrame:
    """Classifier needs trade_usd_log + 3 ratio features that were engineered
    from trade_usd / weight_kg / quantity in the EDA notebook."""
    df = build_reg_df(d)
    w, q = d.weight_kg, d.quantity
    df["trade_usd_log"] = np.log1p(trade_usd)
    df["unit_price"] = trade_usd / q if q > 0 else 0.0
    df["price_per_kg"] = trade_usd / w if w > 0 else 0.0
    df["qty_per_kg"] = q / w if w > 0 else 0.0
    return df


def run_predictions(d: TradeInput):
    reg_df = build_reg_df(d)
    pred_log = float(np.clip(reg_pipeline.predict(reg_df)[0], 0, 40))
    pred_usd = max(float(np.expm1(pred_log)), 0.0)

    clf_df = build_clf_df(d, pred_usd)
    proba = clf_pipeline.predict_proba(clf_df)[0]
    cls = int(np.argmax(proba))
    category = str(label_encoder.inverse_transform([cls])[0])
    return reg_df, clf_df, pred_usd, cls, category, float(proba[cls]), proba


# ============================================================
# SHAP helpers (lazy: explainers are built on first /explain call,
# so startup stays fast and /predict never pays the SHAP memory cost)
# ============================================================

_name_cache = {}


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=float)


def tree_shap(model, X, class_idx=None):
    """SHAP values (1-D) + base value for a single row, using XGBoost's built-in
    TreeSHAP (pred_contribs). Same algorithm and values as shap.TreeExplainer,
    without the heavy shap/numba dependency or building an explainer object."""
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    if class_idx is None:
        return contribs[0, :-1], float(contribs[0, -1])
    return contribs[0, class_idx, :-1], float(contribs[0, class_idx, -1])


def feature_names(key, preprocessor, n_features):
    if key not in _name_cache:
        try:
            names = [str(n) for n in preprocessor.get_feature_names_out()]
            if len(names) != n_features:
                raise ValueError("name/feature count mismatch")
        except Exception:
            names = [f"feature_{i}" for i in range(n_features)]
        _name_cache[key] = names
    return _name_cache[key]


def top_features(values, names, k=5):
    idx = np.argsort(np.abs(values))[::-1][:k]
    return [
        FeatureContribution(feature=names[i], shap_value=round(float(values[i]), 4))
        for i in idx
    ]


# ============================================================
# Plain-English insight summary
# ============================================================

READABLE = {
    "commodity": "the commodity type",
    "country_or_area": "the reporting country",
    "year": "the trade year",
    "weight_kg": "the shipment weight",
    "quantity": "the quantity shipped",
    "trade_usd_log": "the estimated trade value",
    "unit_price": "the price per unit",
    "price_per_kg": "the price per kilogram",
    "qty_per_kg": "the number of units per kilogram",
}


def humanize_feature(raw_name: str) -> str:
    name = raw_name.split("__", 1)[-1]
    if name.endswith("_freq"):
        name = name[: -len("_freq")]
    if name in READABLE:
        return READABLE[name]
    if name.startswith("flow_"):
        return f"the trade flow ({name[len('flow_'):]})"
    if name.startswith("quantity_name_"):
        return f"the unit type ({name[len('quantity_name_'):]})"
    return name.replace("_", " ")


def generate_insight_summary(pred_usd, category, confidence, reg_features):
    if pred_usd < 1_000:
        band = "very low"
    elif pred_usd < 10_000:
        band = "low"
    elif pred_usd < 100_000:
        band = "moderate"
    elif pred_usd < 1_000_000:
        band = "high"
    else:
        band = "very high"

    if confidence >= 0.85:
        conf_desc = "highly confident"
    elif confidence >= 0.6:
        conf_desc = "fairly confident"
    else:
        conf_desc = "less certain"

    up = next((f for f in reg_features if f.shap_value > 0), None)
    down = next((f for f in reg_features if f.shap_value < 0), None)

    parts = [f"This trade record is predicted to be worth about ${pred_usd:,.0f}, a {band} trade value."]
    if up:
        parts.append(f"The biggest factor pushing this value up was {humanize_feature(up.feature)}.")
    if down:
        parts.append(f"On the other hand, {humanize_feature(down.feature)} pulled the value down.")
    parts.append(
        f"Judging only by trade characteristics (not the commodity name), the model is "
        f"{conf_desc} ({confidence * 100:.1f}%) that this trade falls under '{category}'."
    )
    return " ".join(parts)


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    return {"status": "API running",
            "endpoints": ["/predict", "/explain", "/commodities", "/health"]}


@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": True}


@app.get("/commodities")
def get_commodities():
    return {"commodities": TOP_15_COMMODITIES}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: TradeInput):
    try:
        _, _, pred_usd, _, category, conf, proba = run_predictions(input_data)
        top_idx = np.argsort(proba)[::-1][:3]
        top_categories = [
            CategoryProb(
                category=str(label_encoder.inverse_transform([int(i)])[0]),
                probability=round(float(proba[i]), 4),
            )
            for i in top_idx
        ]
        return PredictionOutput(
            predicted_trade_usd=round(pred_usd, 2),
            predicted_category=category,
            category_confidence=round(conf, 4),
            top_categories=top_categories,
            commodity_category=COMMODITY_CATEGORY_OVERRIDE.get(input_data.commodity)
            or COMMODITY_CATEGORY.get(input_data.commodity),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/explain", response_model=ExplainOutput)
def explain(input_data: TradeInput):
    try:
        reg_df, clf_df, pred_usd, cls, category, conf, _ = run_predictions(input_data)

        # ---- Regression SHAP (log scale) ----
        reg_X = _dense(reg_preprocessor.transform(reg_df))
        reg_vals, reg_base = tree_shap(reg_model, reg_X)
        reg_names = feature_names("reg", reg_preprocessor, reg_X.shape[1])
        reg_top = top_features(reg_vals, reg_names)

        # ---- Classification SHAP (for the predicted class) ----
        clf_X = _dense(clf_preprocessor.transform(clf_df))
        clf_vals, clf_base = tree_shap(clf_model, clf_X, class_idx=cls)
        clf_names = feature_names("clf", clf_preprocessor, clf_X.shape[1])
        clf_top = top_features(clf_vals, clf_names)

        return ExplainOutput(
            predicted_trade_usd=round(pred_usd, 2),
            predicted_category=category,
            regression_base_value=round(float(np.expm1(reg_base)), 2),
            regression_top_features=reg_top,
            classification_base_value=round(clf_base, 4),
            classification_top_features=clf_top,
            insight_summary=generate_insight_summary(pred_usd, category, conf, reg_top),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")