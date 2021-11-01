"""
Project Predict Custumer Churn, Udacity ML DevOps course.
Author: Vanessa Urdaneta
Date: 24/10/2021
"""

# Needed libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_roc_curve

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(data_path):
    """
    returns dataframe for the csv found at pth
    input:
            data_path: a path to the csv
    output:
            df: pandas dataframe
    """
    dataframe = pd.read_csv(data_path)
    dataframe["Churn"] = dataframe.Attrition_Flag.apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return dataframe


def perform_eda(df_churn):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
    output:
            None
    """
    column_names = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans",
        "Heatmap"]
    for column_name in column_names:
        plt.figure(figsize=(20, 10))
        if column_name == "Churn":
            df_churn.Churn.hist()
        elif column_name == "Customer_Age":
            df_churn.Customer_Age.hist()
        elif column_name == "Marital_Status":
            df_churn.Marital_Status.value_counts("normalize").plot(kind="bar")
        elif column_name == "Total_Trans":
            sns.displot(df_churn.Total_Trans_Ct)
        elif column_name == "Heatmap":
            sns.heatmap(
                df_churn.corr(),
                annot=False,
                cmap="Dark2_r",
                linewidths=2)
        plt.savefig("images/eda/%s.jpg" % column_name)
        plt.close()


def encoder_helper(df_churn, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
    output:
            df: pandas dataframe with new columns for
    '''

    for category_name in category_lst:
        category_lst = []
        category_groups = df_churn.groupby(category_name).mean()["Churn"]
        for val in df_churn[category_name]:
            category_lst.append(category_groups.loc[val])
        df_churn["%s_%s" % (category_name, "Churn")] = category_lst
    return df_churn


def perform_feature_engineering(df_churn):
    """
    input:
              dataframe: pandas dataframe
    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    y_churn = df_churn["Churn"]
    x_churn = pd.DataFrame()
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn"]

    x_churn[keep_cols] = df_churn[keep_cols]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_churn, y_churn, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    output:
             None
    """

    # Random Forest
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("images/results/%s.jpg" % 'Random_Forest')
    plt.close()

    # Logistic Regression
    plt.rc('figure', figsize=(7, 7))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("images/results/%s.jpg" % 'Logistic_Regression')
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
             None
    """
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 15))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=45)
    plt.savefig("images/%s/Feature_Importance.jpg" % output_pth)
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=1000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"]
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(cv_rfc, x_test, "results")

    joblib.dump(cv_rfc.best_estimator_, "models/rfc_model.pkl")
    joblib.dump(lrc, "models/logistic_model.pkl")

    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    roc_curves_models(lr_model, rfc_model, x_test, y_test)


def roc_curves_models(lr_model, rfc_model, x_test, y_test):
    """
    Save ROC curves of the two used models
    input:
              lr_model: logistic regression model
              rfc_model: random forest model
              x_test: x testing data
              y_test: y testing data
    output:
              None
    """

    # ROC curves ----------------------------
    # logistic:
    plt.figure(figsize=(15, 8))
    plt.title("Logistic Model ROC Curve")
    ax = plt.gca()
    lrc_plot = plot_roc_curve(lr_model, x_test, y_test, ax=ax)
    plt.savefig("images/results/ROC_curve_Logistic.jpg")
    plt.close()

    # random forest
    plt.figure(figsize=(15, 8))
    plt.title("Random Forest Model ROC Curve")
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, x_test, y_test, ax=ax, alpha=0.8)
    plt.savefig("images/results/ROC_curve_RandomF.jpg")
    plt.close()

    # comparison:
    plt.figure(figsize=(15, 8))
    plt.title("Logistic Model & Random Forest ROC Curve comparison")
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, x_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("images/results/ROC_curve_Comparison.jpg")
    plt.close()


if __name__ == "__main__":
    # 1 Import data
    data_df = import_data("data/bank_data.csv")
    # 2 Perform EDA
    perform_eda(data_df)
    # 3 Encode categorical features
    data_encoded = encoder_helper(data_df,
                                  ["Gender",
                                      "Education_Level",
                                      "Marital_Status",
                                      "Income_Category",
                                      "Card_Category"])
    # 4 Sets for train and test
    x_train_, x_test_, y_train_, y_test_ = perform_feature_engineering(
        data_encoded)
    # 5 Train Models
    train_models(x_train_, x_test_, y_train_, y_test_)
