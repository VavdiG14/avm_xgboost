# House Price Prediction with XGBoost

A machine learning project for predicting house prices using XGBoost with feature selection and hyperparameter optimization.

## Project Structure

The project is organized into the following directories:

- `data/`: Contains the dataset and related files.
- `notebooks/`: Jupyter notebooks for data exploration, model training, and evaluation.
- `src/`: Source code for the project.
    
## Features

### Data Preprocessing
- Robust scaling of numeric features
- Option to exclude specific features from scaling
- Label encoding for categorical variables

### Feature Selection
- Feature importance based selection
- Sequential feature selection
- Time-based cross-validation

### Model Optimization
- Hyperparameter optimization using Optuna
- Time-based cross-validation
- MLflow experiment tracking

### Model Analysis
- Performance metrics (RMSE, R2, MAE)
- Feature importance visualization
- Prediction analysis
- Learning curves



## MLflow Tracking

The project uses MLflow to track:
- Feature selection results
- Model parameters
- Performance metrics
- Visualizations
- Model artifacts

View experiments:

```bash
mlflow ui
```
