# Drone-Assisted Disaster Data Engineering & Rescue Analytics

## Project title
Drone-Assisted Disaster Data Engineering & Rescue Analytics

## Abstract
This project develops a locally runnable data engineering and analytics application for drone-assisted disaster-response operations. It processes a real operational dataset containing flight telemetry, disaster context, injury signals, and rescue-priority indicators. The application demonstrates the full lifecycle of data ingestion, validation, preprocessing, feature engineering, analytics, and machine-learning-based severity modeling. The project is designed for academic demonstration, faculty evaluation, and practical understanding of how data engineering can support rescue prioritization and disaster-response planning.

## Problem statement
During natural disasters such as floods, fires, and structural collapses, timely assessment of affected zones and individuals is essential for safe and efficient rescue operations. This project addresses that need by building a data pipeline that ingests drone sensor data, validates it, cleans it, engineers useful operational features, analyzes patterns, and supports injury-severity prediction with transparent evaluation. The system is intended to aid response teams in understanding risk patterns and prioritization logic without claiming clinical or operational authority.

## Objectives
- Inspect and validate the real dataset schema before coding.
- Build a modular Python data pipeline for ingestion, validation, preprocessing, and transformation.
- Create a professional Streamlit dashboard for disaster-response analytics.
- Analyze disaster type, injury severity, operational performance, and rescue-priority trends.
- Train and evaluate suitable ML models for injury classification while avoiding target leakage.
- Produce downloadable reports and maintain reproducible processing steps.

## Dataset description
The project uses the dataset located in the workspace root:

- Autonomous_Medical_Aid_Dataset.csv

This file contains 44,569 records and 46 columns. It includes drone telemetry, environmental conditions, hazard indicators, individual injury markers, and rescue-priority fields. The dataset is treated as an operational analytics dataset and not as a clinical source. The actual schema was inspected before implementation, and the code uses the fields present in the file rather than assumptions from a documentation description.

## System architecture
The application follows a modular architecture:

- Data ingestion: CSV/Excel upload and local default dataset loading.
- Validation: duplicate detection, missing-value reporting, invalid coordinate checks, timestamp validation.
- Preprocessing: duplicate removal, type correction, null handling, and derived features.
- Analytics: report generation, time-series analysis, distribution analysis, and interactive charts.
- Modeling: ML pipeline with train/test split, baseline models, and performance evaluation.
- Presentation: Streamlit dashboard pages for operational monitoring and reporting.

## Dashboard interface
The Streamlit interface uses a restrained Fastship-inspired neutral light theme and reusable components in `components/ui.py`. It includes a date-filtered historical-data overview, an ingestion and quality workbench, filterable cross-category analytics, a bounded interactive coordinate view, leakage-audited model comparison with a single-record demonstration, and a reporting/export workspace. Category values are shown as source codes because the supplied file contains no descriptive codebook. The dataset is not a live drone or dispatch feed.

## Data flow diagram
The pipeline works as follows:

1. Input dataset is loaded from the local file or uploaded file.
2. Schema and data quality checks are performed.
3. Data is cleaned and transformed into a consistent dataset.
4. Derived features are created for risk and operational analysis.
5. Cleaned analytical data and engineered features are saved as separate files in `data/processed`; completed stage timings and output sizes are reported by the Workbench.
6. Visual analytics and ML workflows operate on the processed data.
7. Downloadable outputs are saved to the reports folder.

## Data engineering pipeline explanation
### Stage 1: Data ingestion
The application supports the default local CSV file and user-uploaded CSV/Excel files. It records the source file and handles file-format and file-not-found errors gracefully. The Workbench reports completed ingestion, validation, normalization, profiling, feature-engineering, and persistence stages with measured durations, file sizes, row counts, and execution details. Invalid values are preserved or coerced to missing rather than counted as rejected rows.

### Stage 2: Data validation
The validation layer checks:
- row and column counts
- duplicate records
- missing values
- timestamp validity
- coordinate validity
- basic numeric range conditions

### Stage 3: Data preprocessing
The preprocessing layer removes exact duplicate rows, parses timestamps, and normalizes coordinate and numeric field types. It preserves missing and invalid observations as missing rather than silently imputing or replacing coordinates. The original source file remains unchanged; processed output is stored separately.

### Stage 4: Feature engineering
The reusable feature-preparation function derives:
- hour, day, and month extraction from Timestamp
- night-operation flag
- battery-risk flag
- hazard-risk score
- resource-pressure indicator
- visibility-risk flag

Weather categories are numeric codes without a codebook, so the pipeline does not infer a weather-risk label from those values.

### Stage 5: Transformation
The cleaned analytical dataset and derived feature dataset are saved separately under `data/processed`. The Workbench provides downloads for both. The raw uploaded file is preserved under `data/raw`; invalid and missing values are surfaced in quality reports and not silently imputed.

### Stage 6: Analytics
The application includes plots and summary statistics for disaster distribution, injury severity, operational timing, and location trends.

### Stage 7: ML
The modeling workflow uses numeric feature columns while excluding direct target and leakage-prone fields. The dataset supports classification tasks such as Injury_Severity and Injury_Detection. Models are compared using train/test split and evaluation metrics.

## EDA summary
From the actual dataset inspection:
- 44,569 rows are present.
- No duplicate rows were found.
- No missing values were found in the dataset.
- Timestamp values span from 2019-10-01 to 2024-10-31.
- Disaster categories are encoded numerically, and the file contains operational labels that can be analyzed as categorical variables.
- The dataset combines environmental, operational, and medical-risk indicators suitable for a disaster-response analytics dashboard.

## Model methodology and target-leakage audit
The modeling pipeline evaluates `Injury_Severity` or `Injury_Detection` using a stratified 25% holdout split (random state 42). Imputation, one-hot encoding of documented source-coded categorical columns, scaling for Logistic Regression, and model fitting are composed in scikit-learn pipelines and fit only on the training partition. Accuracy, balanced accuracy, macro-F1, per-class precision/recall, confusion matrices, and Random Forest feature importance are reported.

Two feature sets are compared:

- **Baseline:** available numeric predictors under the existing downstream-field exclusions, with the selected target always removed.
- **Leakage-aware:** additionally excludes injury-specific assessments (`Injury_Type`, `Blood_Loss_Severity`, `Pain_Level_Score`, respiration rate, and heart rate) because their collection timing relative to severity assignment is undocumented. This is a conservative point-in-time policy, not a claim that the observed data proves these fields are derived from severity.

`Recommended_Action`, `Rescue_Priority_Score`, both injury labels, and other pre-existing downstream/target fields are excluded from the baseline as well. A target relationship audit reports Cramer's V for coded categories and Eta for numeric candidate fields; those descriptive associations do not select model features.

The models are:

- Logistic Regression
- Random Forest Classifier

### Leakage bug discovered
The former `prepare_model_features` function removed the selected target from its exclusion set. This directly placed `Injury_Severity` itself among its own predictors and caused both classifiers to report 100% accuracy and macro-F1. The target is now unconditionally excluded and tests assert it is absent from both feature sets.

### Evaluation on the supplied dataset
Results below were computed from the 44,569-row local CSV using the stratified random split described above. Rounded values are presented; outputs in the application and downloadable report are computed from the selected dataset.

| Feature set | Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Baseline | Logistic Regression | 0.314 | 0.328 | 0.287 |
| Baseline | Random Forest | 0.599 | 0.333 | 0.250 |
| Leakage-aware | Logistic Regression | 0.306 | 0.326 | 0.282 |
| Leakage-aware | Random Forest | 0.599 | 0.333 | 0.250 |

The Random Forest predicts only the majority class on this split (class 0); its 0.599 accuracy therefore hides zero recall for classes 1 and 2. Balanced accuracy and the per-class report expose this failure. The evaluation does not support a deployment or clinical-readiness claim.

The candidate-feature audit found no candidate field to be an exact copy of `Injury_Severity`. Observed association values were small on this dataset: `Injury_Type` Cramer's V 0.0087; `Blood_Loss_Severity` Cramer's V 0.0071; pain, respiration, and heart-rate Eta values 0.0034, 0.0067, and 0.0068 respectively; `Recommended_Action` Cramer's V 0.0115; `Rescue_Priority_Score` Eta 0.0024; and `Injury_Detection` Cramer's V 0.0033. These associations do not establish causal or temporal relationships. The 100% result came specifically from the target column itself being present in the old feature matrix.

## Application screenshots
Use the following instructions for demonstration:

1. Open the project folder in VS Code.
2. Activate the configured Python environment.
3. Run the app with the command shown below.
4. Navigate through the pages for Command Center, Data Workbench, Disaster Analytics, Geospatial Intelligence, ML Analytics, and Reports.
5. Demonstrate dataset upload and the cleaned output download.
6. Show the ML evaluation and explain the modeling constraints and limitations.

## Limitations
- The dataset appears to be structured operational data rather than real-world clinical records.
- Source category codes have no accompanying descriptive dictionary; the interface does not invent code meanings.
- The project does not claim clinical accuracy or emergency-response authority.
- Geospatial points are validated and sampled for display but are not independently verified incident locations.
- A stratified random holdout may overstate generalization across time or missions; a chronologically or mission-grouped external evaluation is future work.
- The measurement timing and data-generation process are not documented. Leakage-aware exclusions are deliberately conservative.

## Conclusion
This project demonstrates a realistic data engineering workflow for disaster-response analytics. It covers the end-to-end lifecycle from raw dataset ingestion to transformation, visualization, and model evaluation. It is suitable for academic demonstrations and exhibits practical thinking in operational data processing and decision-support modeling.

## Future scope
- Add a richer geospatial dashboard with Folium-based mapping.
- Integrate the supplementary AIDER image dataset as optional enrichment.
- Add alerting rules for battery-risk and safety thresholds.
- Extend the ML system to rescue-priority estimation with careful feature design.
- Expand the reporting layer to produce PDF summaries.

## References
This project uses the actual local CSV dataset supplied in the workspace. No fabricated references or unsupported claims are included.

