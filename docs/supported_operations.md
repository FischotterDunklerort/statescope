# Supported operations

The contracts target scikit-learn 1.8.0. Table 3 and Section 4.2 of the manuscript report support at the level of a configuration and intended use. The 61 operation profiles exercise fitted-source checks under a specific schedule. Exposure was confirmed on 46 profiles in each exposed schedule, and isolation on 44 in the clean schedule. The counts reflect the different evidence requirements for exposure and isolation.

| Operation or component | Reviewed boundary |
|---|---|
| SGDClassifier | Binary float64, L2, integer seed, fitted intercept, no averaging, warm start or early stopping for the main path-level coefficient profile. The operation sweep specifies the reviewed losses, weight cases and schedules. |
| SGDRegressor | Reviewed single-target squared-error paths and purpose checks. |
| GradientBoostingClassifier | Verified tree growth and retention. Zero-growth class reconstruction leaves a required source link unresolved. |
| LogisticRegression | Whole-record and selected-state checks for reviewed profiles. |
| RandomForestClassifier | The reviewed local path requires unweighted binary growth, disabled bootstrap and a fixed integer seed. |
| StandardScaler and TfidfVectorizer | The feature producer, its observed output and the consuming operation must be linked. The TF–IDF profile covers the reviewed unchanged float64 sparse output. |
| HashingVectorizer | The reviewed stateless transform checks ordered input identities and its actual sparse output. |
| Nested Taxi date callback | A reviewed row-preservation contract applies to the specified callback and pipeline. |

The [evidence map](../reproducibility/evidence/PAPER_TO_ARTIFACT_MAP.md) lists the operation inventory, purpose controls and representative native histories.
