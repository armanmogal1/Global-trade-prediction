"""
eda_insights.py

Stores complete EDA insights and results from the Global Trade
Prediction project as structured Python data. Meant to be imported
by the FastAPI backend / future chatbot so it can answer questions
about the dataset using these precomputed insights instead of
recomputing them from the full dataset every time.

Dataset: Global Commodity Trade Statistics
Source: UN Comtrade database, published on Kaggle by United Nations
(unitednations/global-commodity-trade-statistics)
Coverage: Year 1988 to 2016
"""

# ----------------------------------------------------------------
# 1. Dataset overview
# ----------------------------------------------------------------
DATASET_OVERVIEW = {
    "source": "UN Comtrade database, published on Kaggle by United Nations "
               "(unitednations/global-commodity-trade-statistics)",
    "year_range": "1988 to 2016",
    "original_row_count": 8216686,  # after removing TOTAL rows, before dropna
    "cleaned_row_count": 7876052,   # after dropna on weight_kg/quantity
    "final_columns": [
        "country_or_area", "year", "commodity", "flow", "trade_usd",
        "weight_kg", "quantity_name", "quantity", "category"
    ],
    "cleaning_steps": [
        "Removed 'TOTAL' rows (comm_code = 'TOTAL', commodity = 'ALL COMMODITIES'), "
        "9185 rows. These were aggregate summary rows per country-year-flow, not "
        "real transactions, and would have caused double counting.",
        "No duplicate rows found (0 duplicates).",
        "No negative or zero values found in trade_usd.",
        "Dropped rows with missing weight_kg (1.45%) and quantity (3.60%).",
        "Dropped comm_code column - confirmed 1-to-1 duplicate of commodity "
        "column (every comm_code mapped to exactly 1 unique commodity name, "
        "max nunique = 1.0).",
    ],
    "trade_usd_stats": {
        "count": 8216686,
        "mean": 24020130,       # ~2.4 crore USD
        "std": 741426200,
        "min": 1,
        "25%": 11031,
        "50%": 152584.5,
        "75%": 1878543,
        "max": 436949200000,    # ~437 billion USD
    },
}

# ----------------------------------------------------------------
# 2. Top 15 countries by total trade value (Billion USD)
# ----------------------------------------------------------------
TOP_15_COUNTRIES = [
    {"country": "EU-28", "trade_value_billion_usd": 23602},
    {"country": "China", "trade_value_billion_usd": 16578},
    {"country": "Germany", "trade_value_billion_usd": 13153},
    {"country": "Belgium", "trade_value_billion_usd": 9862},
    {"country": "Canada", "trade_value_billion_usd": 9855},
    {"country": "France", "trade_value_billion_usd": 8565},
    {"country": "Japan", "trade_value_billion_usd": 7684},
    {"country": "China, Hong Kong SAR", "trade_value_billion_usd": 6358},
    {"country": "Australia", "trade_value_billion_usd": 6033},
    {"country": "Italy", "trade_value_billion_usd": 5308},
    {"country": "Brazil", "trade_value_billion_usd": 4423},
    {"country": "USA", "trade_value_billion_usd": 4220},
    {"country": "India", "trade_value_billion_usd": 4118},
    {"country": "Austria", "trade_value_billion_usd": 3712},
    {"country": "Indonesia", "trade_value_billion_usd": 2107},
]

TOP_COUNTRIES_INSIGHT = (
    "EU-28 (European Union, a bloc of 28 member countries before Brexit) is the "
    "largest entity in the dataset by total trade value (~23.6 trillion USD), "
    "but it is a combined bloc, not a single country - worth flagging clearly "
    "in reporting since it can make the ranking look misleading otherwise. "
    "China is the largest individual country (~16.6 trillion USD), reflecting "
    "its role as a global manufacturing and export hub. Germany, Belgium, and "
    "Canada follow next. India ranks 13th (~4.1 trillion USD) among the top 15."
)

# ----------------------------------------------------------------
# 3. Top 15 commodities by total trade value (Billion USD)
# ----------------------------------------------------------------
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

TOP_COMMODITIES_INSIGHT = (
    "Crude petroleum oil is the single biggest traded commodity in the dataset "
    "(~18.9 trillion USD), followed by refined/distilled petroleum oil "
    "(~9.9 trillion USD). Together these two oil-related commodities alone "
    "outweigh every other commodity by a large margin, showing oil trade is "
    "the single strongest driver of global trade value. Medicines (~5.2 "
    "trillion USD) and large aircraft (~3.9 trillion USD) are the next "
    "highest-value commodities, followed by natural gas, gold, and aircraft "
    "parts. Iron ore, coal, automobiles, copper, and wheat also appear in the "
    "top 15, reflecting a mix of energy, industrial raw material, and "
    "manufactured goods trade."
)

# ----------------------------------------------------------------
# 4. Year-wise trade value trend (Billion USD, approx from chart)
# ----------------------------------------------------------------
YEAR_TREND_APPROX = {
    1988: 500, 1989: 850, 1990: 950, 1991: 1100, 1992: 1350,
    1993: 1700, 1994: 2250, 1995: 2900, 1996: 3100, 1997: 3250,
    1998: 3150, 1999: 3200, 2000: 3750, 2001: 3750, 2002: 3950,
    2003: 4650, 2004: 5650, 2005: 6550, 2006: 7700, 2007: 8800,
    2008: 10600, 2009: 8400, 2010: 10300, 2011: 13000, 2012: 12400,
    2013: 13100, 2014: 13000, 2015: 10800, 2016: 8100,
}

YEAR_TREND_INSIGHT = (
    "Global trade value grew steadily from 1988 (~500 billion USD) to 2008 "
    "(~10.6 trillion USD). A sharp dip appears in 2009 (down to ~8.4 trillion "
    "USD), matching the real-world 2008 global financial crisis, confirming "
    "the data reflects actual economic events correctly. Trade recovered and "
    "peaked around 2011-2014 (~13 trillion USD). A drop is visible in 2016 "
    "(~8.1 trillion USD) - this needs verification, since 2016 may only have "
    "partial-year data in this dataset rather than a genuine trade decline. "
    "This should be checked and mentioned as a limitation if 2016 is incomplete."
)

# ----------------------------------------------------------------
# 5. Flow-wise split
# ----------------------------------------------------------------
FLOW_SPLIT = {
    "Import": {"percent": 58.9, "row_count": 4848524},
    "Export": {"percent": 34.8, "row_count": 2856309},
    "Re-Export": {"percent": 4.4, "row_count": 367263},
    "Re-Import": {"percent": 1.8, "row_count": 153775},
}

FLOW_SPLIT_INSIGHT = (
    "Import records make up the largest share of the dataset (58.9%, "
    "4,848,524 rows). Export records are 34.8% (2,856,309 rows). Re-Export "
    "(4.4%) and Re-Import (1.8%) together make up a small share and are "
    "minority/imbalanced classes if flow is ever used as a prediction target. "
    "More countries report their imports than their exports to UN Comtrade, "
    "which is why import records dominate the data."
)

# ----------------------------------------------------------------
# 6. trade_usd distribution
# ----------------------------------------------------------------
TRADE_USD_DISTRIBUTION_INSIGHT = (
    "The raw trade_usd column is extremely right-skewed (min=1 USD, "
    "max=~437 billion USD, mean ~24 million USD vs median ~152,585 USD). Most "
    "transactions have small values, but a few transactions have extremely "
    "large values, which stretches the scale and makes the raw distribution "
    "histogram look like a single tall bar near zero. After applying a log "
    "transformation (log(1 + trade_usd)), the distribution becomes close to a "
    "normal (bell-shaped) curve centered around log value ~11-12. This "
    "confirms the regression model should be trained to predict "
    "log(trade_usd) instead of raw trade_usd, and predictions should be "
    "converted back to the original scale (using exponential/expm1) before "
    "reporting final error metrics like RMSE and MAE."
)

# ----------------------------------------------------------------
# 7. Category grouping (classification target) - row counts
# ----------------------------------------------------------------
CATEGORY_DISTRIBUTION = [
    {"category": "Food, Beverages & Tobacco", "row_count": 1977773},
    {"category": "Textiles & Apparel", "row_count": 1282684},
    {"category": "Chemicals & Pharmaceuticals", "row_count": 987123},
    {"category": "Metals & Metal Products", "row_count": 862194},
    {"category": "Precision Instruments, Arms & Misc", "row_count": 593295},
    {"category": "Wood, Paper & Pulp", "row_count": 412838},
    {"category": "Stone, Ceramic, Glass & Precious Materials", "row_count": 380028},
    {"category": "Minerals, Fuels & Ores", "row_count": 281028},
    {"category": "Transport Equipment", "row_count": 267560},
    {"category": "Leather, Hides & Furskins", "row_count": 232940},
    {"category": "Footwear & Headgear", "row_count": 225027},
    {"category": "Plastics & Rubber", "row_count": 198248},
    {"category": "Machinery & Electronics", "row_count": 175314},
]

CATEGORY_GROUPS = [c["category"] for c in CATEGORY_DISTRIBUTION]

CATEGORY_GROUPING_INSIGHT = (
    "The original dataset had 97 detailed commodity categories (HS chapter "
    "level, e.g. '16_meat_fish_and_seafood_food_preparations_nes'). This was "
    "too many classes for a clean classification target, and many classes had "
    "very few rows (e.g. '99_commodities_not_specified_according_to_kind' had "
    "only ~1705-7024 rows). These 97 categories were grouped into 13 broader, "
    "business-friendly categories following the standard HS (Harmonized "
    "System) trade classification structure used internationally. This "
    "grouped 'category' column replaced the original category column entirely "
    "and is used as the classification target. 'Food, Beverages & Tobacco' "
    "has the most rows (1,977,773), followed by 'Textiles & Apparel' "
    "(1,282,684) and 'Chemicals & Pharmaceuticals' (987,123). 'Machinery & "
    "Electronics' has the fewest rows (175,314). This is a moderate class "
    "imbalance (roughly 11x between largest and smallest class) and should be "
    "handled during model training, e.g. using class weights or evaluating "
    "with macro-F1 instead of plain accuracy."
)

# ----------------------------------------------------------------
# 8. Average trade value per category (Million USD per transaction)
# ----------------------------------------------------------------
AVG_TRADE_VALUE_BY_CATEGORY = [
    {"category": "Minerals, Fuels & Ores", "avg_trade_value_million_usd": 158},
    {"category": "Transport Equipment", "avg_trade_value_million_usd": 67},
    {"category": "Stone, Ceramic, Glass & Precious Materials", "avg_trade_value_million_usd": 26},
    {"category": "Chemicals & Pharmaceuticals", "avg_trade_value_million_usd": 21},
    {"category": "Precision Instruments, Arms & Misc", "avg_trade_value_million_usd": 16},
    {"category": "Food, Beverages & Tobacco", "avg_trade_value_million_usd": 16},
    {"category": "Footwear & Headgear", "avg_trade_value_million_usd": 15},
    {"category": "Leather, Hides & Furskins", "avg_trade_value_million_usd": 14},
    {"category": "Wood, Paper & Pulp", "avg_trade_value_million_usd": 14},
    {"category": "Metals & Metal Products", "avg_trade_value_million_usd": 13},
    {"category": "Plastics & Rubber", "avg_trade_value_million_usd": 13},
    {"category": "Machinery & Electronics", "avg_trade_value_million_usd": 9},
    {"category": "Textiles & Apparel", "avg_trade_value_million_usd": 6},
]

AVG_TRADE_VALUE_INSIGHT = (
    "This view is different from the row-count chart above: it shows the "
    "average value of a single transaction, not how many transactions exist. "
    "'Minerals, Fuels & Ores' has by far the highest average transaction value "
    "(~158 million USD per transaction), even though it does not have the "
    
    "most rows (ranked 8th by row count). This is because oil, gas, and ore "
    "shipments involve very large values per trade, even though there are "
    "fewer trade records overall. 'Transport Equipment' (ships, aircraft, "
    "vehicles) has the second highest average value per transaction "
    "(~67 million USD), which makes sense since these are expensive, "
    "low-frequency goods. 'Food, Beverages & Tobacco' has the highest row "
    "count but only a mid-level average value (~16 million USD) - a "
    "high-volume, comparatively lower-value-per-trade category. 'Textiles & "
    "Apparel' has the lowest average value per transaction (~6 million USD) "
    "despite having the second highest row count - confirming it is also a "
    "high-volume, low-value-per-trade category. Business takeaway: row count "
    "and average trade value tell two different stories - some categories are "
    "traded very frequently but in small amounts (food, textiles), while "
    "others are traded rarely but in huge amounts (fuel, transport equipment)."
)

# ----------------------------------------------------------------
# 9. Import vs export trend over years
# ----------------------------------------------------------------
IMPORT_EXPORT_TREND_INSIGHT = (
    "Import values are consistently higher than export values across almost "
    "all years from 1988 to 2016. The gap between import and export value "
    "widens after 2010, with import reaching roughly 6.9-7 trillion USD at "
    "peak (2013-2014) versus export around 6 trillion USD in the same period. "
    "Both import and export values dip together in 2009 (matching the "
    "financial crisis) and again in 2016 (matching the overall year-wise dip "
    "seen in the combined trend), supporting that the 2016 drop is a "
    "consistent pattern across both flow types - though the possibility of "
    "partial/incomplete 2016 data should still be verified before modeling."
)

# ----------------------------------------------------------------
# 10. Feature engineering decisions made during EDA
# ----------------------------------------------------------------
FEATURE_DECISIONS = {
    "comm_code": "Dropped. Verified 1-to-1 duplicate of the commodity column "
                 "(df.groupby('comm_code')['commodity'].nunique() had max=1.0 "
                 "across all 5045 unique codes).",
    "commodity": "Kept as-is in EDA notebook (5031 unique values, too many to "
                 "one-hot encode directly). To be converted to a numeric "
                 "feature using frequency encoding in the model training "
                 "notebook (not in EDA, since EDA is only for "
                 "cleaning/understanding, not feature engineering). The "
                 "original text column will also be kept alongside the "
                 "encoded version for SHAP explainability and error analysis.",
    "quantity_name": "Has 12 different measurement units: Weight in "
                      "kilograms (6,617,980 rows - dominant), Number of "
                      "items (709,321), No Quantity (495,991), Volume in "
                      "litres (144,599), Area in square metres (135,912), "
                      "Number of pairs (67,929), Length in metres (18,225), "
                      "Volume in cubic meters (17,775), Weight in carats "
                      "(5,237), Thousands of items (1,386), Electrical "
                      "energy in thousands of kilowatt-hours (1,264), Number "
                      "of packages (1,067). Because of this, the 'quantity' "
                      "column cannot be used directly as a plain number - it "
                      "needs to be paired with quantity_name (one-hot "
                      "encoding) during feature engineering, otherwise the "
                      "model would wrongly compare quantities measured in "
                      "different units.",
    "trade_usd": "Log-transformed (log1p) for the regression target due to "
                 "heavy right skew (see trade_usd_distribution insight).",
    "category": "Original 97 detailed categories replaced entirely by 13 "
                "broad HS-based groups for the classification target (see "
                "category_grouping insight).",
    "weight_kg_and_quantity_nulls": "Rows with missing weight_kg (1.45%) and "
                                     "quantity (3.60%) were dropped since "
                                     "these are used as model features.",
}

# ----------------------------------------------------------------
# Convenience: all insights in one dict (for chatbot lookup)
# ----------------------------------------------------------------
ALL_INSIGHTS = {
    "dataset_overview": DATASET_OVERVIEW,
    "top_15_countries": TOP_15_COUNTRIES,
    "top_countries_insight": TOP_COUNTRIES_INSIGHT,
    "top_15_commodities": TOP_15_COMMODITIES,
    "top_commodities_insight": TOP_COMMODITIES_INSIGHT,
    "year_trend_approx": YEAR_TREND_APPROX,
    "year_trend_insight": YEAR_TREND_INSIGHT,
    "flow_split": FLOW_SPLIT,
    "flow_split_insight": FLOW_SPLIT_INSIGHT,
    "trade_usd_distribution_insight": TRADE_USD_DISTRIBUTION_INSIGHT,
    "category_distribution": CATEGORY_DISTRIBUTION,
    "category_groups": CATEGORY_GROUPS,
    "category_grouping_insight": CATEGORY_GROUPING_INSIGHT,
    "avg_trade_value_by_category": AVG_TRADE_VALUE_BY_CATEGORY,
    "avg_trade_value_insight": AVG_TRADE_VALUE_INSIGHT,
    "import_export_trend_insight": IMPORT_EXPORT_TREND_INSIGHT,
    "feature_decisions": FEATURE_DECISIONS,
}


if __name__ == "__main__":
    # quick manual check when running this file directly
    for key, value in ALL_INSIGHTS.items():
        print(f"\n--- {key} ---")
        print(value)