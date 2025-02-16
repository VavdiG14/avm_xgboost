import mlflow
import joblib
from pathlib import Path
from src.config import ModelConfig, OptunaConfig, MLFlowConfig, PathConfig
from src.models.optimizer import OptunaOptimizer
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

def main():
    # Initialize configs
    path_config = PathConfig()
    path_config.setup_dirs()
    
    mlflow_config = MLFlowConfig(tracking_uri="local", experiment_name="model_optimization")
    model_config = ModelConfig()
    optuna_config = OptunaConfig()

    # Setup MLflow tracking
    mlflow_config.setup_tracking()

    # Load processed data
    processed_data = joblib.load(path_config.interim_dir / "processed_data.joblib")
    selected_features = joblib.load(path_config.interim_dir / "selected_features.joblib")

    X_train = processed_data["X_train"]
    y_train = processed_data["y_train"]
    X_test = processed_data["X_test"]
    y_test = processed_data["y_test"]
    train_years = processed_data["train_years"]

    with mlflow.start_run():
        # Log selected features
        mlflow.log_param("selected_features", selected_features.tolist())
        
        # Hyperparameter optimization with time-based CV
        optimizer = OptunaOptimizer(optuna_config, X_train, y_train, train_years)
        best_params = optimizer.optimize()
        
        # Train final model on all training data
        final_model = xgb.XGBRegressor(**best_params, random_state=model_config.random_state)
        final_model.fit(X_train, y_train)
        
        # Evaluate on test year (2024)
        test_pred = final_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        mlflow.log_metric("test_rmse", test_rmse)
        
        # Log and save model
        mlflow.xgboost.log_model(final_model, "model")
        model_dir = Path("./models")
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, path_config.model_dir / "final_model.joblib")

if __name__ == "__main__":
    main() 