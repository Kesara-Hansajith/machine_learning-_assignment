# Income Prediction Model

[Income Prediction Model](https://mlassignmentkdu.streamlit.app/) analyzes the Adult Census Income dataset to predict whether an individual's income exceeds $50K per year based on demographic and employment data.

## 📊 Dataset
Used the [Adult Census Income dataset](https://www.kaggle.com/datasets/uciml/adult-census-income) from Kaggle, which contains census data with the following:

- **Features**: age, workclass, education, marital status, occupation, and more
- **Target variable**: income (>50K or ≤50K)
- **Records**: 32,561 training instances

## 🧠 Project Workflow

### 1. Data Exploration & EDA
- Analyzed distributions of numerical features (age, hours-per-week, etc.)
- Examined relationships between categorical variables and income
- Identified key patterns in education, occupation, and demographic factors
- Visualized correlations between features

### 2. Data Preprocessing
- Handled missing values (marked as '?')
- Encoded categorical variables using Label Encoding
- Scaled numerical features to improve model performance
- Split data into training and testing sets (80/20)

### 3. Model Fitting
We evaluated several machine learning algorithms:
- Logistic Regression (baseline)
- Decision Trees
- Random Forest
- Support Vector Machines
- Neural Networks

### 4. Model Evaluation
- Compared models using accuracy, precision, recall, and F1-score
- Generated ROC curves and confusion matrices
- Performed cross-validation to ensure model stability
- **Best Model**: Random Forest (87.3% accuracy)

### 5. Model Deployment
- Created a Streamlit web application for interactive predictions
- Implemented proper encoding of categorical features
- Designed user-friendly interface for input and results display

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, streamlit, matplotlib, seaborn

### Installation
1. Clone this repository
```bash
git clone https://github.com/yourusername/income-prediction.git
cd income-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app
```bash
streamlit run app.py
```

## 📁 Project Structure
```
income-prediction/
├── data/
│   └── adult.csv
├── notebooks/
│   ├── ML_Assignment.ipynb
├── models/
│   └── random_forest_income_prediction_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

## 📊 Key Findings
- Education level has the strongest correlation with income
- Gender and marital status significantly impact income predictions
- Working hours show moderate correlation with income level
- Occupation types like Executive-Managerial and Professional-Specialty are strong predictors of higher income

## 🖥️ Application Demo
Try this: https://mlassignmentkdu.streamlit.app/

![image](https://github.com/user-attachments/assets/b13e2c31-f1f8-4aaf-ba89-fee6f323508a)

app_scamlit application allows users to:
- Input their demographic and employment information
- Receive instant income predictions
- View probeblity of earning


## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
