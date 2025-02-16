import mlflow
from src.config import ModelConfig, FeatureSelectionConfig, OptunaConfig, MLFlowConfig
from src.data.data_loader import DataLoader
from src.features.feature_selector import FeatureSelector
from src.models.optimizer import OptunaOptimizer
import xgboost as xgb

def main():
    # Initialize configs
    mlflow_config = MLFlowConfig(tracking_uri="local")
    model_config = ModelConfig()
    feature_config = FeatureSelectionConfig()
    optuna_config = OptunaConfig()

    # Setup MLflow tracking
    mlflow_config.setup_tracking()

    with mlflow.start_run():
        # Load and split data
        data_loader = DataLoader(model_config)
        X_train, X_test, y_train, y_test = data_loader.load_data("./data/house_prices.csv")
        
        # Feature selection
        feature_selector = FeatureSelector(feature_config)
        X_train_selected, X_test_selected, selected_features = feature_selector.select_features(
            X_train, y_train, X_test
        )
        
        # Hyperparameter optimization
        optimizer = OptunaOptimizer(optuna_config, X_train_selected, y_train)
        best_params = optimizer.optimize()
        
        # Train final model
        final_model = xgb.XGBRegressor(**best_params, random_state=model_config.random_state)
        final_model.fit(X_train_selected, y_train)
        
        # Log model
        mlflow.xgboost.log_model(final_model, "model")

if __name__ == "__main__":
    main() 