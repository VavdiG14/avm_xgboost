import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

class XGBoostAnalyzer:
    def __init__(self, config=None, params=None):
        """
        Initialize with default XGBoost parameters if no config provided
        """
        self.default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
        # Use feature selection specific params if provided
        if config and hasattr(config, 'xgboost_params'):
            self.default_params = config.xgboost_params.params
        elif params:
            self.default_params = params
            
        self.config = config
        self.model = None
        self.feature_importance = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model with early stopping if validation set provided"""
        params = self.default_params
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_train, y_train)
            
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def evaluate(self, X, y, dataset_name=""):
        """Evaluate model performance"""
        predictions = self.model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        
        metrics = {
            f'{dataset_name}_rmse': rmse,
            f'{dataset_name}_r2': r2
        }
        
        # Log metrics to MLflow
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            
        return metrics
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        plt.figure(figsize=(12, 6))
        n_features = min(top_n, len(self.feature_importance))
        
        sns.barplot(
            data=self.feature_importance.head(n_features),
            x='importance',
            y='feature'
        )
        plt.title(f'Top {n_features} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save plot to MLflow
        plt.savefig('img/feature_importance.png')
        mlflow.log_artifact('img/feature_importance.png')
        plt.close()
    
    def plot_predictions(self, X, y, dataset_name=""):
        """Plot actual vs predicted values"""
        predictions = self.model.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted ({dataset_name})')
        plt.tight_layout()
        
        # Save plot to MLflow
        plt.savefig(f'predictions_{dataset_name}.png')
        mlflow.log_artifact(f'predictions_{dataset_name}.png')
        plt.close()
    
    def plot_learning_curves(self):
        """Plot learning curves from training history"""
        if not hasattr(self.model, 'evals_result'):
            print("No evaluation results found. Train with validation set to see learning curves.")
            return
            
        results = self.model.evals_result()
        
        plt.figure(figsize=(10, 6))
        plt.plot(results['validation_0']['rmse'], label='train')
        plt.plot(results['validation_1']['rmse'], label='validation')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title('Learning Curves')
        plt.legend()
        plt.tight_layout()
        
        # Save plot to MLflow
        plt.savefig('learning_curves.png')
        mlflow.log_artifact('learning_curves.png')
        plt.close()
    
    def get_base_model(self):
        """Return a fresh XGBoost model with current parameters"""
        params = self.default_params
        return xgb.XGBRegressor(**params) 