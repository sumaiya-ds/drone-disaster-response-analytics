from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT_DIR / 'Autonomous_Medical_Aid_Dataset.csv'
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
REPORTS_DIR = ROOT_DIR / 'reports'
MODELS_DIR = ROOT_DIR / 'models'

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CLEANED_DATA = PROCESSED_DATA_DIR / 'cleaned_disaster_dataset.csv'
DEFAULT_FEATURE_DATA = PROCESSED_DATA_DIR / 'engineered_disaster_features.csv'
DEFAULT_QUALITY_REPORT = REPORTS_DIR / 'data_quality_report.csv'
DEFAULT_ANALYTICS_REPORT = REPORTS_DIR / 'analytics_summary.csv'
DEFAULT_ML_REPORT = REPORTS_DIR / 'ml_evaluation_report.csv'
