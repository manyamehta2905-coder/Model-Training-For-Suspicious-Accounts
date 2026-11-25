
import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

# 1. Load trained Developing Chain model
print("[1/4] Loading model...")

model_path = os.path.join(
    C.MODELS_DIR, "developing_chain.joblib"
)

bundle = joblib.load(model_path)

model = bundle["model"]
features = bundle["numeric"] + bundle["categorical"]

# 2. Load test data
print("[2/4] Loading chain test data...")

df = pd.read_csv(C.CHAIN_SNAP_CSV, low_memory=False)

test = df[
    (df["split"] == "test") &
    (df["chain_id"].notna())
].copy()

X_test = test[features]

# Limit explanation data for faster execution
if len(X_test) > 300:
    X_test = X_test.sample(300, random_state=42)

# 3. Apply training pipeline preprocessing
print("[3/4] Computing SHAP values...")

preprocessor = model.named_steps["pre"]
classifier = model.named_steps["clf"]

X_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_transformed = pd.DataFrame(
    X_transformed,
    columns=feature_names
)

explainer = shap.TreeExplainer(classifier)
shap_exp = explainer(X_transformed)

# 4. Save SHAP outputs
os.makedirs(C.REPORTS_DIR, exist_ok=True)

# Global feature impact
shap.plots.beeswarm(shap_exp, max_display=14, show=False)

plt.savefig(
    os.path.join(C.REPORTS_DIR, "chain_shap_summary.png"),
    bbox_inches="tight",
    dpi=150
)
plt.close()

# Global feature importance
shap.plots.bar(shap_exp, max_display=14, show=False)

plt.savefig(
    os.path.join(C.REPORTS_DIR, "chain_shap_importance.png"),
    bbox_inches="tight",
    dpi=150
)
plt.close()

# Individual chain explanation
shap.plots.waterfall(shap_exp[0], max_display=14, show=False)

plt.savefig(
    os.path.join(C.REPORTS_DIR, "chain_shap_individual.png"),
    bbox_inches="tight",
    dpi=150
)
plt.close()

# Numerical feature importance
importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": np.abs(shap_exp.values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

importance.to_csv(
    os.path.join(C.REPORTS_DIR, "chain_shap_importance.csv"),
    index=False
)

print("\nSHAP analysis completed successfully!")
print(importance.head(10).to_string(index=False))
print("\nFiles saved in reports/")
