# Titanic-Survivel-


This project predicts the survival of passengers aboard the Titanic using various classification algorithms. The dataset is derived from the famous [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data), and several machine learning models are applied to predict whether a passenger survived or not.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Algorithms Used](#algorithms-used)
- [Data Preprocessing](#data-preprocessing)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [How to Run the Project](#how-to-run-the-project)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Project Overview
The sinking of the Titanic is one of the most infamous shipwrecks in history. This project aims to predict the survival of passengers using features such as age, gender, class, and fare. The main objective is to apply various classification algorithms and compare their performance.

## Dataset
The dataset used for this project is the [Titanic Dataset](https://www.kaggle.com/c/titanic/data) provided by Kaggle. It contains the following features:
- `PassengerId`: ID of the passenger.
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- `Name`: Name of the passenger.
- `Sex`: Gender of the passenger.
- `Age`: Age of the passenger.
- `SibSp`: Number of siblings or spouses aboard.
- `Parch`: Number of parents or children aboard.
- `Ticket`: Ticket number.
- `Fare`: Passenger fare.
- `Cabin`: Cabin number (if available).
- `Embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

The target variable is `Survived`, where 0 indicates the passenger did not survive, and 1 indicates the passenger survived.

## Algorithms Used
The following classification algorithms were applied:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Naive Bayes
7. Gradient Boosting
8. XGBoost

## Data Preprocessing
- **Handling Missing Values**: Missing values in `Age` and `Cabin` were filled using appropriate strategies.
- **Feature Encoding**: Categorical features like `Sex` and `Embarked` were encoded using label encoding or one-hot encoding.
- **Feature Scaling**: Numerical features such as `Age` and `Fare` were scaled to improve model performance.
- **Splitting the Dataset**: The dataset was split into training and testing sets in an 80/20 ratio.

## Evaluation Metrics
To evaluate the models, the following metrics were used:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

## Results
- Logistic Regression: Accuracy - 80.1%
- Decision Tree: Accuracy - 77.8%
- Random Forest: Accuracy - 83.7%
- SVM: Accuracy - 82.6%
- KNN: Accuracy - 79.4%
- Naive Bayes: Accuracy - 78.9%
- Gradient Boosting: Accuracy - 85.1%
- XGBoost: Accuracy - 86.2%

Among these, XGBoost provided the highest accuracy and best overall performance based on other evaluation metrics.

## How to Run the Project
1. Clone the repository:
   ```
   git clone https://github.com/your-username/titanic-survival-prediction.git
   ```
2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
3. Run the notebook or script:
   ```
   jupyter notebook titanic_survival_prediction.ipynb
   ```
   OR
   ```
   python titanic_survival_prediction.py
   ```

## Conclusion
In this project, various classification algorithms were used to predict the survival of passengers aboard the Titanic. XGBoost was the most accurate model, achieving an accuracy of 86.2%. The project highlights the importance of feature engineering and model selection in building predictive models.

## Acknowledgments
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data) for providing the data.
- The open-source Python libraries used in the project.
