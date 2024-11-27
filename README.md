# Predicting Airbnb Host Status

This was a machine learning project aimed at predicting the **superhost status** of Airbnb hosts based on listing attributes. The project explores multiple classification techniques, including baseline, advanced boosting, and ensemble models, while emphasizing hyperparameter tuning strategies to optimize performance.

## Dataset

The dataset was sourced from the [Kaggle competition](https://www.kaggle.com/competitions/classification-sp24-sec21-airbnb-host-status/leaderboard). Key features in the dataset include:
- **Host information** (e.g., response rate, number of listings)
- **Property details** (e.g., location, reviews)
- **Listing activity** (e.g., number of days active, booking success rate)

The target variable is the binary classification of whether a host is a superhost (`1`) or not (`0`). Data was split into training (`train_classification.csv`) and testing (`test_classification.csv`) sets.

## Libraries Used

- **Core Libraries**: `pandas`, `numpy`
- **Data Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`

## Modeling Process

### 1. **Data Preprocessing**
   - Imputed missing values and encoded categorical variables.
   - Conducted exploratory data analysis (EDA) to understand feature importance.
   - Applied log-transformation to skewed predictors, which improved model performance significantly.
   - Ensured consistency between training and test datasets by aligning feature names after transformations.

### 2. **Baseline Models**
   - Implemented **K-Nearest Neighbors (KNN)** as a simple baseline to establish a starting point for model evaluation.
   - Utilized **Random Forests**, which outperformed KNN due to its ability to handle non-linear decision boundaries.

### 3. **Boosting Models**
   - **Gradient Boosting**: Established an advanced baseline for boosting models.
   - **XGBoost**: Tuned using `GridSearchCV` for fine-grained parameter optimization.
   - **LightGBM**: Leveraged for its efficiency and speed on large datasets.
   - **CatBoost**: Achieved the highest standalone accuracy by directly handling categorical features.

   ### Reflections on Boosting
   - Initial tuning efforts with XGBoost and Gradient Boosting yielded limited improvements in accuracy.
   - A breakthrough came with **CatBoost**, which achieved over 93% accuracy after hyperparameter tuning using `RandomizedSearchCV`.

### 4. **Ensemble Learning**
   - Combined base models using **soft voting ensembles**, resulting in the highest performance.
   - Final ensemble included:
     - Untuned **CatBoost**
     - Tuned **XGBoost**
     - Tuned **Gradient Boosting**

   ### Reflections on Ensembling
   - The ensemble reached a Kaggle accuracy of **93.5%**, outperforming all individual models.
   - Trial-and-error with different base models and preprocessing steps, including log-transformation, proved critical to success.

### 5. **Hyperparameter Tuning**
   - **Methods Used**:
     - **GridSearchCV**: Applied to XGBoost for fine-grained tuning.
     - **RandomizedSearchCV**: Used for CatBoost and Gradient Boosting to explore larger hyperparameter spaces efficiently.
   - **Challenges**:
     - Tuning times were significant, especially for boosting models (e.g., CatBoost took ~2 hours and 40 minutes overnight).
     - Early attempts plateaued at ~93.2% accuracy, leading to a detailed review of preprocessing steps, which ultimately unlocked further gains.

### 6. **Evaluation Metrics**
   - Metrics such as **accuracy**, **precision**, **recall**, and **F1-score** were used to compare models.

## Key Findings

- **Baseline Models**: Random Forests outperformed KNN by capturing complex relationships.
- **Boosting Insights**:
  - CatBoost excelled in handling categorical data and required minimal preprocessing.
  - LightGBM and XGBoost offered competitive performance with faster training times.
- **Ensembling Success**:
  - A soft voting ensemble combining untuned CatBoost, tuned XGBoost, and Gradient Boosting achieved the best results.

## Next Steps

- Explore additional features (e.g., reviews, location data) to improve classification accuracy.
- Experiment with advanced ensembling techniques, such as stacking and blending.
- Optimize boosting models using Bayesian optimization for further tuning efficiency.

## Acknowledgements

- [Kaggle Competition](https://www.kaggle.com/competitions/classification-sp24-sec21-airbnb-host-status/leaderboard)
- [NUStat Course Notes](https://nustat.github.io/STAT303-3-class-notes/)
- Professor Arvind Krishna
