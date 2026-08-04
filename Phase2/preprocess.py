import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold,mutual_info_regression, SelectKBest



def load_data(filename):

    #DISPLAY FORMATTING
    pd.set_option('display.float_format', "{:,.3f}".format)

    df = pd.read_csv(filename, low_memory=False)
    df = df [["long_name","age","height_cm","overall", "potential", "player_positions", "value_eur", "release_clause_eur", "club_name","club_position","league_name", "league_level",
            "preferred_foot","weak_foot","skill_moves","international_reputation","work_rate","body_type"]]
    return df

def clean_data(df):
    df.drop_duplicates(inplace=True)
    df.drop(columns=["work_rate"], inplace = True) # drop as it is empty for all
    for i in df[df.release_clause_eur.isnull()].index:
        df.loc[i, "release_clause_eur"] = df.loc[i, "value_eur"] *1.25

    c = 'player_positions' #variable to hold column name
    df[c] = df[c].str.split(",").str[0].str.strip()
    df.drop(columns = "long_name", inplace=True)
    return df

    

def encode_features(df):
    nonNumeric = df.select_dtypes(exclude = ["number"])
    #now encode non numeric columns
    #no column really requires scaling, lets still go for age
    for c in nonNumeric:
        df[c] = LabelEncoder().fit_transform(df[c])

    return df


def scale_features(df):
    scaler = StandardScaler()
    columns_to_scale = ['age']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    joblib.dump(scaler, "models/age_scaler.joblib")
    return df


def remove_outliers(df):
    #in our dataset, outliers should not be removed as they are main points, so I am not removing for most columns, removing only for league_level
    c = "league_level"
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3-Q1

    lower_bound = Q1 - 1.5*IQR
    higher_bound = Q3 + 1.5*IQR

    df = df[(df[c] >= lower_bound) & (df[c] <=higher_bound)]

    return df


def feature_engineering(df):
    #apply log transform on target variable
    df['log_value'] = np.log1p(df['value_eur'])
    df['log_rc'] = np.log1p(df['release_clause_eur'])
    unnecessary_cols = ["player_positions","club_name","height_cm","preferred_foot", "value_eur", "release_clause_eur"]
    df = df.drop(columns = unnecessary_cols)

    return df

    

def feature_selection(df):
    #variance threshold
    X = df.drop(columns = ["log_value"])
    y = df["log_value"]

    numeric_X = X.select_dtypes(exclude=["str"])

    selector = VarianceThreshold(threshold=0.01)
    selector.fit(numeric_X)

    selected_columns = numeric_X.columns[selector.get_support()]

    #Select K-Best
    X = df[selected_columns]
    y = df["log_value"]

    #now select till age, k = 5
    kVal = 5
    selector = SelectKBest(score_func=mutual_info_regression, k=kVal)
    selector.fit(X, y)

    selected_columns = X.columns[selector.get_support()]

    df = pd.concat([df[selected_columns], y], axis = 1)
    #here I have dropped release clause as well as it can become a biased indicator
    df.drop(columns = ["league_name", "log_rc"],inplace = True)

    return df

def preprocess_pipeline(filename):
    df = load_data(filename)

    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)
    df = remove_outliers(df)
    df = feature_engineering(df)
    df = feature_selection(df)

    return df

