import mlflow
import joblib
from pathlib import Path
from src.config import ModelConfig, FeatureSelectionConfig, MLFlowConfig, PathConfig
from src.data.data_loader import DataLoader
from src.features.feature_selector import FeatureSelector

def main():
    # Initialize configs
    path_config = PathConfig()
    path_config.setup_dirs()
    
    mlflow_config = MLFlowConfig(tracking_uri="local", experiment_name="feature_selection")
    model_config = ModelConfig()
    feature_config = FeatureSelectionConfig()

    # Setup MLflow tracking
    mlflow_config.setup_tracking()

    with mlflow.start_run():
        # Load and split data
        data_loader = DataLoader(model_config)
        X_train, X_test, y_train, y_test, train_years = data_loader.load_data(path_config.raw_data)
        
        # Feature selection
        feature_selector = FeatureSelector(feature_config)
        X_train_selected, X_test_selected, selected_features = feature_selector.select_features(
            X_train, y_train, X_test
        )
        
        # Save selected features and transformed data
        joblib.dump(selected_features, path_config.interim_dir / "selected_features.joblib")
        joblib.dump({
            "X_train": X_train_selected,
            "X_test": X_test_selected,
            "y_train": y_train,
            "y_test": y_test,
            "train_years": train_years
        }, path_config.interim_dir / "processed_data.joblib")

if __name__ == "__main__":
    main() 