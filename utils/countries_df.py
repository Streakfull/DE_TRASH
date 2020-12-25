import sklearn
from utils.general import iterative_impute
from sklearn.impute import IterativeImputer
import numpy as np
import pandas as pd

def clean_column_names(countries_df):
    columns = ["Real Growth Rating(%)", "Literacy Rate(%)", "Inflation(%)", "Unemployement(%)"]
    for column in columns:
        countries_df[column] = countries_df[column].str.extract('(\d+)')
        countries_df[column] = countries_df[column].apply(pd.to_numeric)
    countries_df.rename({"Unemployement(%)": "Unemployment(%)"})
    return countries_df

def drop_extra_columns(countries_df):
    dropped_columns = ["Unnamed: 0","Population","Subregion"]
    countries_df = countries_df.drop(dropped_columns,axis=1)
    return countries_df

def impute_columns(countries_df):
    countries_df['Region'] = countries_df['Region'].astype(str)
    df_averages = countries_df.groupby("Region").mean()[0:6]
    df_averages = df_averages.replace(np.nan,0)
    regions = df_averages.index
    test_df = countries_df.copy()
    missing_columns = ["Unemployement(%)","Gini","Inflation(%)","Real Growth Rating(%)","Literacy Rate(%)", "Area"]
    regions
    # index = test_df.Region == "Europe"
    # select_rows = test_df[index]
    # select_rows
    # index2 = select_rows["Gini"].isna()
    # rows = select_rows[index2]
    # rows["Gini"] = df_averages[df_averages.index == "Europe"]["Gini"]

    # condition_1 = test_df.Region == "Europe"
    # condition_2 = test_df.Gini.isna()
    # test_df[condition_1 & condition_2] = df_averages[df_averages.index == "Europe"]["Gini"][0]
    # test_df[condition_1 & condition_2]
    for region in regions:
        index = test_df.Region == region
        select_rows = test_df[index]
        for column in missing_columns:
            condition_1 = countries_df.Region == region
            condition_2 = countries_df[column].isna()
            countries_df.loc[condition_1 & condition_2, column] = df_averages[df_averages.index == region][column][0]
    return countries_df

def drop_na_rows(countries_df):
    countries_df = countries_df.dropna()
    return countries_df

def remove_outliers(countries_df):
    lof = sklearn.neighbors.LocalOutlierFactor(n_neighbors=13)
    predictions = lof.fit_predict(countries_df.select_dtypes(exclude="object"))
    percentage_outliers = predictions[predictions==-1].shape[0] / countries_df.shape[0]
    percentage_outliers = percentage_outliers * 100
    not_outliers = predictions == 1
    countries_df = countries_df[not_outliers]
    return countries_df

def transform_countries(countries_df):
    countries_df = clean_column_names(countries_df)
    countries_df = drop_extra_columns(countries_df)
    countries_df = impute_columns(countries_df)
    countries_df = drop_na_rows(countries_df)
    countries_df = remove_outliers(countries_df)
    return countries_df