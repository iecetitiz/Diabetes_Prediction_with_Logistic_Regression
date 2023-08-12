# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 09:09:31 2023

@author: iecet
"""

######################################################
# Diabetes Prediction with Logistic Regression
######################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics


from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report,RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


#creates limits for the outlier values
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

#checks whether a column contains an outlier value or not
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

#replaces with outlier values with thresholds
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


######################################################
# Exploratory Data Analysis
######################################################

df = pd.read_csv("C:/datasets/machine_learning/diabetes.csv")

df["Outcome"].value_counts()

df.columns

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.25, 0.50, 0.95, 0.99]).T)


def plot_numerical_col(df, numerical_col):
    df[numerical_col].hist(bins = 20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


cols = [col for col in df.columns if "Outcome" not in col]


for col in cols:
    plot_numerical_col(df, col)
    
    

##########################
# Target vs Features
##########################

df.groupby("Outcome").agg({"Pregnancies": "mean"})

def target_summary_with_numeric_features(dataframe, target, numeric_features):
    print(dataframe.groupby(target).agg({numeric_features: "mean"}), end="\n\n\n\n")


for col in cols:
    target_summary_with_numeric_features(df, "Outcome", col)



######################################################
# Data Preprocessing (in a simple way and surface level)
######################################################

#checks columns one by one for the outliers
for col in cols:
    print(col, check_outlier(df, col))

#only Inslulin column has outlier values
replace_with_thresholds(df, "Insulin")

#Scaling with RobustScaler
for col in cols:
    df[col] = RobustScaler().fit_transform(df[col])



######################################################
# Model & Prediction
######################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)

log_model = LogisticRegression().fit(X, y)

#bias
b = log_model.intercept_

#weights
w = log_model.coef_

#Making a prediction with all dataset
y_pred = log_model.predict(X)


######################################################
# Model Evaluation
######################################################



def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))


######################################################
# Model Validation: Holdout
######################################################


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

# =============================================================================
# Plot the ROC Curve
# =============================================================================

y_pred_proba = log_model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# =============================================================================
# Plot the ROC Curve with AUC
# =============================================================================

#define metrics
y_pred_proba = log_model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


