# Logistic Regression Practical Implementation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, beginner-friendly, and production-style implementation of Logistic Regression using Python and Scikit-Learn. This project demonstrates model training, evaluation, and deployment workflow using a Jupyter notebook and a Python training script.

---

## 💡 Overview

This project implements a *binary Logistic Regression classifier* using the scikit-learn library. It specifically uses the *Iris dataset* but is filtered to classify only two species (*versicolor* and *virginica*) to create a binary classification task. The primary goal is to provide a reproducible, end-to-end example that covers:

- Data loading and exploratory data analysis (EDA)
- Data preprocessing and feature engineering
- Model training with hyperparameter tuning
- Model evaluation with detailed metrics
- Model persistence for deployment

---

## 📁 Project Structure


.
├── notebooks/                       
│   └── Logistic_Regression_Practical_Implementation.ipynb
├── src/                             
│   └── train.py
├── data/                            
│   └── sample_data.csv
├── models/                          
├── figures/                         
├── reports/                         
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE


### Directory Details

- *notebooks/*: Contains the Jupyter notebook with step-by-step implementation
- *src/*: Python scripts for training and deployment
- *data/*: Dataset storage (placeholder CSV included)
- *models/*: Saved trained models (.pkl files)
- *figures/*: Generated visualizations and plots
- *reports/*: Model evaluation reports and documentation

---

## 🧠 Key Learnings from the Notebook

The accompanying Jupyter notebook, Logistic_Regression_Practical_Implementation.ipynb, demonstrates the full workflow:

### 1. *Data Loading & Initial EDA*
- Loads the Iris dataset using seaborn
- Performs initial data quality checks (null values, data types, distribution)

### 2. *Data Preparation (Binary Classification)*
- Filters the dataset to include only two species: *versicolor* and *virginica*
- Ensures balanced or imbalanced class representation

### 3. *Target Encoding*
- Maps categorical species to numerical labels:
  - versicolor → 0
  - virginica → 1

### 4. *Feature/Target Split*
- Separates independent features (X) and dependent target (y)
- Ensures proper data structure for modeling

### 5. *Train-Test Split*
- Splits data with test_size=0.25 (75% training, 25% testing)
- Maintains reproducibility with random state

### 6. *Hyperparameter Tuning (GridSearchCV)*
- Uses GridSearchCV to find optimal hyperparameters:
  - *penalty*: l1, l2, elasticnet
  - *C* (regularization strength): Various values
  - *max_iter*: Maximum iterations for convergence

*Note*: The output shows that the default solver (lbfgs) is incompatible with l1 and elasticnet penalties, leading to FitFailedWarning. The final best parameters select the compatible 'penalty': 'l2'.

### 7. *Model Evaluation*
- *Best cross-validation score*: ~0.973
- *Final test accuracy*: 0.92
- Comprehensive classification report with precision, recall, and F1-score

---

## 🧩 Features

✅ Logistic Regression implementation using scikit-learn  
✅ Train-test split and cross-validation  
✅ Hyperparameter tuning with GridSearchCV  
✅ *Detailed Metrics*: Accuracy, precision, recall, F1-score  
✅ Model serialization using joblib  
✅ Easily replaceable dataset (data/sample_data.csv)  
✅ Command-line interface (CLI) for training  
✅ Production-ready code structure  
✅ Extensible for visualization and reporting  

---

## ⚙ Installation

### 1. Clone the repository

bash
git clone https://github.com/your-username/logistic-regression-practical.git
cd logistic-regression-practical


### 2. Create a virtual environment

bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


### 3. Install dependencies

bash
pip install -r requirements.txt


### Requirements

The project requires the following Python packages:


numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
joblib>=1.0.0


---

## 🧪 Usage

### ▶ Run the Notebook

Launch the Jupyter notebook to walk through the step-by-step implementation, data exploration, and model tuning:

bash
jupyter notebook notebooks/Logistic_Regression_Practical_Implementation.ipynb


### 💻 Run the Training Script

The project includes a reusable command-line interface (CLI) training script:

bash
python src/train.py --data data/sample_data.csv --target target


#### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| --data | Path to your CSV file | data/sample_data.csv |
| --target | Name of the target column | target |
| --test-size | Test split size (0.0 to 1.0) | 0.25 |
| --model-path | Output model path | models/logistic_regression.pkl |

#### Example with Custom Test Size

bash
python src/train.py --data data/sample_data.csv --target target --test-size 0.3


---

## 📈 Example Output

### GridSearchCV Results

The GridSearchCV found the following optimal hyperparameters:

python
{'C': 1, 'max_iter': 100, 'penalty': 'l2'}


### Model Evaluation on Test Set


Accuracy: 0.92

Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.93      0.93        14
           1       0.91      0.91      0.91        11

    accuracy                           0.92        25
   macro avg       0.92      0.92      0.92        25
weighted avg       0.92      0.92      0.92        25


---

## 🎯 Model Performance Interpretation

- *Accuracy (92%)*: The model correctly classifies 92% of the test samples
- *Precision (0.93 for class 0, 0.91 for class 1)*: High precision indicates low false positive rate
- *Recall (0.93 for class 0, 0.91 for class 1)*: High recall indicates low false negative rate
- *F1-Score (0.93 for class 0, 0.91 for class 1)*: Balanced performance across both classes

---

## 🔧 Customization

### Using Your Own Dataset

1. Place your CSV file in the data/ directory
2. Ensure your dataset has:
   - Feature columns (independent variables)
   - Target column (dependent variable for binary classification)
3. Run the training script:

bash
python src/train.py --data data/your_dataset.csv --target your_target_column


### Modifying Hyperparameters

Edit the param_grid in the notebook or training script:

python
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'max_iter': [100, 200, 500],
    'solver': ['liblinear', 'saga']  # Compatible with all penalties
}


---

## 📚 Resources

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Logistic Regression Theory](https://en.wikipedia.org/wiki/Logistic_regression)
- [GridSearchCV Guide](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

---

## 🧾 License

This project is licensed under the *MIT License*. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. *Fork* the repository
2. Create a new branch: git checkout -b feature-name
3. *Commit* your changes: git commit -m "Add feature"
4. *Push* to your branch: git push origin feature-name
5. Open a *Pull Request*

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 🧑‍💻 Author

*Your Name*  
📧 your.email@example.com  
🌐 [LinkedIn](https://linkedin.com/in/yourprofile) | [Portfolio](https://yourwebsite.com) | [GitHub](https://github.com/yourusername)

---

## ⭐ Support

If you find this project helpful, please:

- ⭐ *Star* the repository
- 🍴 *Fork* it for your own projects
- 📢 *Share* it with others
- 🐛 *Report* issues or bugs
- 💡 *Suggest* new features

---

## 🔖 Tags

machine-learning logistic-regression python scikit-learn classification data-science jupyter-notebook hyperparameter-tuning binary-classification model-evaluation

---

## 📝 Changelog

### Version 1.0.0 (Initial Release)
- Basic logistic regression implementation
- Jupyter notebook with full workflow
- CLI training script
- GridSearchCV hyperparameter tuning
- Comprehensive evaluation metrics

---

## 🚀 Future Enhancements

- [ ] Add feature importance visualization
- [ ] Implement cross-validation with multiple metrics
- [ ] Add ROC curve and AUC score analysis
- [ ] Support for multiclass classification
- [ ] Web API deployment using Flask/FastAPI
- [ ] Docker containerization
- [ ] CI/CD pipeline integration
- [ ] Model monitoring and logging

---

*Made with ❤ for the Data Science Community*
