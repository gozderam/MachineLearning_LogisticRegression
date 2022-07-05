from posixpath import split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers.one_hot_encoding import OneHotEncodingTransformer


def split_and_reset_index(data: pd.DataFrame, y_colname: str):
    X = data.drop([y_colname], axis = 1)
    y = data.loc[:, y_colname]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


def load_and_preprocess_heart(path = "data/heart.dat"):
    columns = [
    ### real
    "age", 

    ### binary
    "sex",
    
    ### nominal
    "pt", # "chest pain type", 
    
    ### real
    "bpress", # "resting blood pressure",
    
    ### real
    "chol", # "serum cholesterol in mg/dl",
    
    ### binary
    "sugar", # "fasting blood sugar > 120 mg/dl",
    
    ### nominal
    "rer", # "resting electrocardiographic results (values 0,1,2)", 
    
    ### real
    "hearth_rate", #"maximum heart rate achieved", 
    
    ### binary
    "exercise", #"exercise induced angina",
    
    ### real
    "oldpeak", # "oldpeak = ST depression induced by exercise relative to rest", 
    
    ### ordered 
    "slope", # "the slope of the peak exercise ST segment", 
    
    ### real
    "vessels", # "number of major vessels (0-3) colored by flourosopy",
    
    ### nominal 
    "thal", 
    
    ### binary - predicted 
    "heart_disease" # Absence (1) or presence (2) of heart disease 
    ]

    data = pd.read_csv(path, header = None, sep=" ")
    data.columns = columns

    # check that no missing values provided
    assert(np.all(data.isna().sum() == 0))
    assert(np.all(data!=np.NaN))
    assert(np.all(data!="?"))

    # change encoding of the response varaible from 1, 2 to 0, 1 (1 - presence of hearth desease - success)
    data.heart_disease = data.heart_disease.replace(1, 0).replace(2, 1)

    # split into train & test and reset indexes on dataframes
    X_train, X_test, y_train, y_test = split_and_reset_index(data, "heart_disease")
   
    # one hot encoding
    ohe = OneHotEncodingTransformer(["pt", "rer", "thal"])
    X_train = ohe.fit_transform(X_train)
    X_test = ohe.transform(X_test)

    return X_train, X_test, y_train, y_test



def load_and_preprocess_bank(path = "data/bank.csv"):
    data = pd.read_csv(path, sep=";")

    # check that no missing values provided
    assert(np.all(data.isna().sum() == 0))
    assert(np.all(data!=np.NaN))
    assert(np.all(data!="?"))

    # change encoding of the response varaible from no, yes to 0, 1 (1 -  the client will subscribe a term deposit - success)
    data.y = data.y.replace("no", 0).replace("yes", 1)

    # convert month to numbers
    month_dict = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12
    }

    data.month.replace(month_dict, inplace=True)

    # split into train & test and reset indexes on dataframes
    X_train, X_test, y_train, y_test = split_and_reset_index(data, "y")

    # one hot encoding
    ohe = OneHotEncodingTransformer(["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"])
    X_train = ohe.fit_transform(X_train)
    X_test = ohe.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_and_preprocess_banknote_auth():
    df = pd.read_csv('./data/banknote_auth/data_banknote_authentication.txt')

    # check that no missing values provided
    assert(np.all(df.isna().sum() == 0))
    assert(np.all(df!=np.NaN))
    assert(np.all(df!="?"))

    return split_and_reset_index(df, 'class')


def load_and_preprocess_ckd():
    df = pd.read_csv('./data/ckd/CKD.csv')

    # check that no missing values provided
    assert(np.all(df.isna().sum() == 0))
    assert(np.all(df!=np.NaN))
    assert(np.all(df!="?"))

    return split_and_reset_index(df, 'Chronic Kidney Disease: yes')
