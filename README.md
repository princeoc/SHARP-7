# SHARP-7

## Method

- Patient-grouped, stratified five-fold development CV; equal-weight calibrated HistGBM/XGBoost backbone.
- Lower threshold: largest OOF cutoff achieving sensitivity >= 0.75. Upper threshold: smallest OOF cutoff achieving specificity >= 0.85.
- Below the lower threshold, both TabPFN and GBM must be positive to rescue a case. The upper threshold only stratifies positives. If upper <= lower, the gray zone is empty.
- External probabilities are averaged across five fitted pipelines **per model**, then fused/classified using locked thresholds. External labels are never supplied to `fit`.

**Implementation choices requiring confirmation:** sigmoid calibration with five-fold patient-grouped inner OOF predictions (`ensemble=False`); training-fold median imputation for continuous features and mode imputation for binary features. These details were not specified in the supplied Supplementary Methods. HistGBM/XGBoost/GBM parameters follow the supplied nine-model parameter table; they are not verified hybrid-model settings. TabPFN uses the installed version's defaults and full outer training folds, not the older script's subsampled context. Record the TabPFN checkpoint and software environment used in the final analysis.

TabPFN and GBM threshold rules are **required inputs**, because the document does not specify them. Each accepts `("sensitivity", target)`, `("specificity", target)` or `("fixed", verified_probability)`. No 80% constraint or 0.5 cutoff is assumed. Targets must come from the development analysis plan, never external validation performance. A specificity target of 1 may yield a cutoff just above 1 (no positive predictions).

## Use

Python 3.11+; `pip install -r requirements.txt`. TabPFN requires its official pretrained checkpoint, any applicable model access authorization, and suitable hardware. CUDA is recommended for the full development cohort. No smaller substitute is silently used.

```python
import json
import pandas as pd
from sharp7 import SHARP7, FEATURES

train = pd.read_csv("development.csv")
external = pd.read_csv("external.csv")
rules = json.load(open("rescue_rules.json", encoding="utf-8"))
model = SHARP7(rules["sentinel"], rules["confirmatory"])
model.fit(train[FEATURES], train["hypoxemia"], train["patient_id"])
model.predict(external[FEATURES]).to_csv("external_predictions.csv", index=False)
with open("thresholds.json", "w", encoding="utf-8") as f:
    json.dump(model.thresholds_, f, indent=2)
```

Create `rescue_rules.json` with keys `sentinel` and `confirmatory`, each holding a two-element array `[rule_name, numeric_target]` based on your confirmed protocol. There is intentionally no fabricated default. Use the same locked model for all three external cohorts. Archive fitted pipelines and their runtime environment privately for reuse; do not place patient-containing artifacts on GitHub.

Input feature names and order are in `FEATURES`. The first three are numeric; the last four are 0/1 (1 = yes). Use the same neck-to-height units across cohorts, oxygen saturation in percent, and outcome 1 = SpO2 <92% for >10 seconds. Patient identifiers must be globally consistent; all records from a patient stay together. Missing predictors are allowed, missing outcomes/identifiers are not. Supply y/groups in the same row order as X. Features are treated as fixed; feature-selection code is outside this module.

`predict` returns each model's probability, the continuous backbone `risk_score`, the final binary `prediction` and `stratum`. AUROC/AUPRC/Brier scores evaluate `risk_score`; rescue changes binary classifications, not this continuous score. Rescue alone cannot improve its AUPRC. `oof_` is used to choose final operating thresholds; classification performance evaluated on those same OOF rows is not an unbiased internal validation of threshold selection. Use an additional outer evaluation loop for that purpose.

Test status: synthetic integration/logic checks only; no full four-model clinical training or reproduction of published results has been performed. Dependency ranges are compatibility requirements, not an original-analysis lockfile; record exact versions after a full run (`pip freeze`).

API references: [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html), [TabPFN](https://github.com/PriorLabs/TabPFN).
