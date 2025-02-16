from dataclasses import dataclass, field
from pathlib import Path
import mlflow
from typing import List

@dataclass
class XGBoostFeatureSelectionConfig:
    params: dict = field(default_factory=lambda: {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 3,  # Smaller depth for feature selection
        'learning_rate': 0.05,  # Lower learning rate
        'n_estimators': 50,  # Fewer trees for faster selection
        'subsample': 0.8,  # Use subset of data
        'colsample_bytree': 0.8,  # Use subset of features
        'min_child_weight': 5,  # More conservative splits
        'random_state': 42
    })

@dataclass
class FeatureSelectionConfig:
    n_features: int = 10
    #selection_method: str = "importance"  # or "sequential"
    random_state: int = 42
    cv_splits: int = 3
    cv_test_size: int = 365  # For time series CV
    xgboost_params: XGBoostFeatureSelectionConfig = field(
        default_factory=XGBoostFeatureSelectionConfig
    )

@dataclass
class ModelConfig:
    target_column: str = "price"
    test_year: int = 2023  # Year to use for testing
    year_column: str = "year"  # Column containing the year
    id_columns: list = field(default_factory=lambda: ["txn_id", "feature_id", "property_id"])
    date_column: str = "date"  # Column containing the date
    random_state: int = 42
    use_robust_scaler: bool = True  # Set to True to use RobustScaler, False for StandardScaler
    exclude_from_scaling: List[str] = field(default_factory=lambda: [
        'year', 'month'
    ])

@dataclass
class OptunaConfig:
    n_trials: int = 100
    timeout: int = 1800  # 30 minutes instead of 1 hour
    study_name: str = "xgboost_optimization"
    random_state: int = 42

@dataclass
class MLFlowConfig:
    tracking_uri: str = "local"  # "local" or "azure"
    experiment_name: str = "house_price_prediction"

    def setup_tracking(self):
        if self.tracking_uri == "azure":
            mlflow.set_tracking_uri("azureml://")
        else:
            mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment(self.experiment_name)

@dataclass
class PathConfig:
    data_dir: Path = Path("./data")
    raw_data: Path = Path("./data/raw/avm_modeling_dataset.xlsx")
    interim_dir: Path = Path("./data/interim")
    model_dir: Path = Path("./models")

    def setup_dirs(self):
        """Create all necessary directories"""
        self.data_dir.mkdir(exist_ok=True)
        self.interim_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)

@dataclass
class PreprocessingConfig:
    numeric_impute_strategy: str = 'median'
    categorical_impute_strategy: str = 'constant'
    categorical_impute_value: str = 'missing'
    scale_numeric: bool = True 