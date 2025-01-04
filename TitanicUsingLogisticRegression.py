import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def TitanicLogistic():
    titanic_data = pd.read_csv('TitanicDataset.csv')

    print("First 5 entries from loaded dataset")
    print(titanic_data.head())

    print("Number of passengers: " + str(len(titanic_data)))

    print("Visualization: Survived and non-survived passengers")
    figure()
    target = "Survived"
    countplot(data=titanic_data, x=target).set_title("Survived and non-survived passengers")
    show()

    print("Visualization: Survived and non-survived passengers based on gender")
    figure()
    countplot(data=titanic_data, x=target, hue="Sex").set_title("Survived and non-survived passengers based on Gender")
    show()

    print("Visualization: Survived and non-survived passengers based on passenger class")
    figure()
    countplot(data=titanic_data, x=target, hue="Pclass").set_title("Survived and non-survived passengers based on Passenger class")
    show()

    print("Visualization: Survived and non-survived passengers based on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Survived and non-survived passengers based on Age")
    show()

    print("Visualization: Survived and non-survived passengers based on the fare")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Survived and non-survived passengers based on fare")
    show()

    # Dropping irrelevant columns
    if 'zero' in titanic_data.columns:
        titanic_data.drop("zero", axis=1, inplace=True)

    print("First 5 entries from loaded dataset after removing zero column (if it exists)")
    print(titanic_data.head())

    print("Values of Sex column")
    print(pd.get_dummies(titanic_data["Sex"]))

    print("Values of Sex column after removing one field")
    Sex = pd.get_dummies(titanic_data["Sex"], drop_first=True)
    print(Sex.head())

    print("Values of Pclass column after removing one field")
    Pclass = pd.get_dummies(titanic_data["Pclass"], drop_first=True)
    print(Pclass.head())

    print("Value of dataset after concatenating new columns")
    titanic_data = pd.concat([titanic_data, Sex, Pclass], axis=1)
    print(titanic_data.head())

    # Check which columns exist before attempting to drop them
    columns_to_drop = ["Sex", "SibSp", "Parch", "Embarked", "Pclass"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in titanic_data.columns]

    print("Value of dataset after dropping unused columns")
    titanic_data.drop(existing_columns_to_drop, axis=1, inplace=True)
    print(titanic_data.head())

    # Handling missing values
    titanic_data = titanic_data.dropna()

    # Ensure all column names are strings
    titanic_data.columns = titanic_data.columns.astype(str)

    x = titanic_data.drop("Survived", axis=1)
    y = titanic_data["Survived"]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5)

    logmodel = LogisticRegression(max_iter=200)  # Increase max_iter for convergence

    logmodel.fit(xtrain, ytrain)

    prediction = logmodel.predict(xtest)

    print("Classification report of Logistic Regression: ")
    print(classification_report(ytest, prediction))

    print("Confusion matrix of Logistic Regression: ")
    print(confusion_matrix(ytest, prediction))

    print("Accuracy of Logistic Regression: ")
    print(accuracy_score(ytest, prediction))

def main():
    print("----- Titanic Logistic Regression Analysis -----")
    print("Supervised machine learning")
    print("Logistic Regression on Titanic dataset")

    TitanicLogistic()

if __name__ == "__main__":
    main()
