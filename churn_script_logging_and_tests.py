"""
Testing module for Project Predict Custumer Churn, Udacity ML DevOps course.
Author: Vanessa Urdaneta
Date: 24/10/2021.
"""

import logging
import joblib
from churn_library import import_data, perform_eda, \
	encoder_helper, perform_feature_engineering, train_models

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format="%(asctime)s: %(name)s - %(levelname)s - %(message)s",
    force=True)


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df_churn = import_data("./data/bank_data.csv")
        logging.info("Testing import data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import data: The file wasn't found")
        raise err

    try:
        assert df_churn.shape[0] > 0
        assert df_churn.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    df_churn = import_data("./data/bank_data.csv")
    perform_eda(df_churn)
    for image_name in [
            "Churn",
            "Customer_Age",
            "Marital_Status",
            "Total_Trans",
            "Heatmap"]:
        try:
            with open("images/eda/%s.jpg" % image_name, 'r'):
                logging.info("Testing perform_eda: SUCCESS, images generated")
        except FileNotFoundError as err:
            logging.error("Testing perform_eda: generated images missing")
            raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    df_churn = import_data("./data/bank_data.csv")
    dataframe_encoded = encoder_helper(df_churn,
                                       ["Gender",
                                        "Education_Level",
                                        "Marital_Status",
                                        "Income_Category",
                                        "Card_Category"])
    try:
        assert dataframe_encoded.shape[0] > 0
        assert dataframe_encoded.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have rows and columns")
        raise err
    try:
        for column in [
                "Gender",
                "Education_Level",
                "Marital_Status",
                "Income_Category",
                "Card_Category"]:
            assert column in dataframe_encoded
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't have the right encoded columns")
        raise err
    logging.info("Testing encoder_helper: SUCCESS, all categorical features encoded")
    return dataframe_encoded


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df_churn = import_data("./data/bank_data.csv")
    dataframe_encoded = encoder_helper(df_churn,
                                       ["Gender",
                                        "Education_Level",
                                        "Marital_Status",
                                        "Income_Category",
                                        "Card_Category"])
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        dataframe_encoded)
    try:
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("Testing feature_engineering: SUCCESS, train and test sets generated")
    except AssertionError as err:
        logging.error("Testing feature_engineering: Sequences length mismatch")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    df_churn = import_data("./data/bank_data.csv")
    dataframe_encoded = encoder_helper(df_churn,
                                       ["Gender",
                                        "Education_Level",
                                        "Marital_Status",
                                        "Income_Category",
                                        "Card_Category"])
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        dataframe_encoded)

    train_models(x_train, x_test, y_train, y_test)
    try:
        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')
        logging.info("Testing train_models: SUCCESS, models have been registered")
    except FileNotFoundError as err:
        logging.error("Testing train_models: no registered models")
        raise err
    for image_name in [
            "Logistic_Regression",
            "Random_Forest",
            "Feature_Importance",
            "ROC_curve_Comparison",
            "ROC_curve_Logistic",
            "ROC_curve_RandomF"]:
        try:
            with open("images/results/%s.jpg" % image_name, 'r'):
                logging.info(
                    "Testing train_models: SUCCESS - image reports generated")
        except FileNotFoundError as err:
            logging.error(
                "Testing train_models : not generated reports")
            raise err


if __name__ == "__main__":

    # 1 Import:
    test_import()
    # 2 Eda:
    test_eda()
    # 3 Encoder:
    test_encoder_helper()
    # 4 Feature eng
    test_perform_feature_engineering()
    # 5 Train models
    test_train_models()
