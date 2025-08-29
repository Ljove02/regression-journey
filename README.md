# 🚀 Regression Journey

**Quick View:** [Student Performance](student_performance.ipynb) | [Falcon 9 Landing](falcon9.ipynb)

A comprehensive exploration of machine learning regression techniques through two real-world case studies: predicting student academic performance and SpaceX Falcon 9 landing outcomes.

## Projects Overview

### 1. Student Performance Prediction (Linear Regression)

Predicting student performance index using academic and lifestyle factors.

**Key Features:**

- **Algorithm**: Linear Regression with Polynomial Features (degree 2)
- **Performance**: 98.87% R² accuracy
- **Features**: Study hours, previous scores, extracurricular activities, sleep patterns
- **Techniques**: Feature standardization, polynomial expansion, interactive hyperparameter tuning

### 2. Falcon 9 Landing Prediction (Logistic Regression)

Machine learning model to predict SpaceX Falcon 9 first-stage landing success.

**Key Features:**

- **Algorithm**: Logistic Regression with Polynomial Features (degree 3) + L2 Regularization
- **Performance**: 89% Precision/Recall/F1-Score, AUC = 0.84
- **Features**: Payload mass, orbit type, technical specs (Block, GridFins, Legs), flight history
- **Techniques**: One-hot encoding, feature engineering, ROC analysis

## Technical Implementation

- **Custom ML Library**: Built using custom `kockice` library for deeper algorithmic understanding
- **Interactive Notebooks**: Marimo notebooks with real-time parameter tuning
- **From Scratch**: Manual implementation of gradient descent, loss functions, and evaluation metrics
- **Professional Workflow**: Complete ML pipeline from EDA to model evaluation

## Results Summary

| Project             | Algorithm           | Accuracy     | Key Insight                                                                    |
| ------------------- | ------------------- | ------------ | ------------------------------------------------------------------------------ |
| Student Performance | Linear Regression   | 98.87% R²    | Linear relationships dominate - polynomial features didn't improve performance |
| Falcon 9 Landings   | Logistic Regression | 89% F1-Score | GridFins (0.64) and Legs (0.67) are strongest predictors of landing success    |

### Interactive Notebooks

Choose your preferred format:

- **Jupyter** (GitHub viewable): Links above ☝️
- **Marimo** (interactive): Commands below ⬇️

## Setup & Usage

### Prerequisites

```bash
pip install marimo pandas numpy matplotlib seaborn plotly scikit-learn kagglehub
```

### Running the Notebooks

```bash
# Student Performance Analysis
marimo run student_performance_linear_regression.py

# Falcon 9 Landing Prediction
marimo run falcon9_logistic_regression.py
```

## Project Structure

```
regression-journey/
├── falcon9_logistic_regression.py       # SpaceX landing prediction (Marimo)
├── student_performance_linear_regression.py  # Student performance analysis (Marimo)
├── falcon9.ipynb                        # SpaceX analysis (Jupyter - GitHub viewable)
├── student_performance.ipynb           # Student performance (Jupyter - GitHub viewable)
├── data/                               # Dataset files
├── images/                             # Visualizations and assets
├── kockice.py                          # Custom ML library
└── README.md                           # Project documentation
```

## Key Learning Outcomes

- **Feature Engineering**: Polynomial expansion, one-hot encoding, standardization
- **Model Evaluation**: ROC curves, confusion matrices, precision/recall analysis
- **Regularization**: L2 penalty to prevent overfitting
- **Domain Knowledge**: Understanding real-world constraints (SpaceX landing procedures)
- **Interactive ML**: Building responsive parameter tuning interfaces

## Visualizations

Both projects include comprehensive visualizations:

- Correlation matrices and heatmaps
- Loss curves and convergence analysis
- ROC curves and performance metrics
- Interactive scatter plots and distribution charts

## Technical Highlights

- **Custom implementations** of gradient descent, BCE loss, MSE loss
- **Stratified sampling** for balanced train/test splits
- **Polynomial feature engineering** while avoiding redundant OHE combinations
- **Professional evaluation** using multiple metrics beyond accuracy

## Future Enhancements

- Cross-validation implementation
- Comparison with ensemble methods (Random Forest, XGBoost)
- Feature importance analysis
- Model deployment as REST API
- Deep learning approaches for comparison

## Dataset Attribution

**Falcon 9 Dataset:** IBM Data Science Capstone, licensed under GPL-3.0  
Source: [IBM-Data-Science-Capstone-SpaceX](https://github.com/chuksoo/IBM-Data-Science-Capstone-SpaceX/tree/main?tab=readme-ov-file)

---
