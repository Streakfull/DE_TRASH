import sklearn
from utils.general import iterative_impute
from sklearn.impute import IterativeImputer
import numpy as np


def impute_columns(life_expectancy_df):
    imputed_values = iterative_impute(life_expectancy_df, [
                                      "Population", "Measles", "Thinness 1-19 Years", "Under-five Deaths"], "Population")
    life_expectancy_df['Population'] = imputed_values[:, 0]
    imputed_values = iterative_impute(life_expectancy_df, [
                                      "Hepatitis B", "Diphtheria", "Polio", "Life Expectancy"], "Hepatitis B")
    life_expectancy_df['Hepatitis B'] = imputed_values[:, 0]
    imputed_values = iterative_impute(life_expectancy_df, [
                                      "GDP", "Percentage Expenditure", "Life Expectancy", "Income Composition Of Resources", "Schooling", "Alcohol", "BMI"], "GDP")
    life_expectancy_df['GDP'] = imputed_values[:, 0]
    imputed_values = iterative_impute(life_expectancy_df, [
                                      "Total Expenditure", "Alcohol", "Schooling", "BMI"], "Total Expenditure")
    life_expectancy_df['Total Expenditure'] = imputed_values[:, 0]
    imputed_values = iterative_impute(life_expectancy_df, ["Alcohol", "Schooling", "Income Composition Of Resources",
                                                           "Life Expectancy", "GDP", "Percentage Expenditure", "BMI", "Total Expenditure"], "Alcohol")
    life_expectancy_df['Alcohol'] = imputed_values[:, 0]
    imputed_values = iterative_impute(life_expectancy_df, ["Income Composition Of Resources", "Life Expectancy",
                                                           "BMI", "GDP", "Alcohol", "Diphtheria", "Percentage Expenditure", "Polio"], "Income Composition Of Resources")
    imputed_values = iterative_impute(life_expectancy_df, ["Schooling", "Alcohol", "Income Composition Of Resources",
                                                           "Life Expectancy", "GDP", "Percentage Expenditure", "BMI", "Total Expenditure"], "Schooling")
    life_expectancy_df['Schooling'] = imputed_values[:, 0]
    imputed_values = iterative_impute(life_expectancy_df, ["Income Composition Of Resources", "Life Expectancy",
                                                           "BMI", "GDP", "Alcohol", "Diphtheria", "Percentage Expenditure", "Polio"], "Income Composition Of Resources")

    life_expectancy_df['Income Composition Of Resources'] = imputed_values[:, 0]
    imputer = IterativeImputer(random_state=0)
    columns = ['Thinness 1-19 Years', 'BMI', 'Polio',
               'Diphtheria', 'Life Expectancy', 'Adult Mortality']
    data = life_expectancy_df[columns]
    imputer = imputer.fit(data)
    imputed_values = imputer.transform(data)
    life_expectancy_df[columns] = imputed_values
    return life_expectancy_df


def handle_outliers(life_expectancy_df):
    lof = sklearn.neighbors.LocalOutlierFactor(n_neighbors=20)
    predictions = lof.fit_predict(
        life_expectancy_df.select_dtypes(exclude="object"))
    sorted = life_expectancy_df[predictions == -1].groupby(
        'Country').count().sort_values("Year", ascending=False)
# Countries that appear more than 7 times are not considered outliers
    countries = sorted[sorted["Year"] >= 7][:].index
    safely_removed_countries = ~life_expectancy_df.Country.isin(countries)
    outliers = np.logical_and(predictions == -1, safely_removed_countries)
    outliers[outliers == True].shape
    # Remove
    life_expectancy_df = life_expectancy_df[~outliers]
    return life_expectancy_df


def drop_unused_columns(life_expectancy_df):
    return life_expectancy_df.drop(['Infant Deaths', 'Thinness 5-9 Years'], axis=1)


def smoothe_noise(life_expectancy_df):
    smoothed_values = life_expectancy_df.groupby('Country')['Alcohol'].transform(
        lambda x: x.ewm(span=40, adjust=False).mean())
    life_expectancy_df["Alcohol"] = smoothed_values
    for column in life_expectancy_df.select_dtypes(exclude="object").columns:
        if(column == "Year"):
            continue
        smoothed_values = life_expectancy_df.groupby('Country')[column].transform(
            lambda x: x.ewm(span=40, adjust=False).mean())
        life_expectancy_df[column] = smoothed_values
    return life_expectancy_df


def transform_life_expectancy_df(life_expectancy_df):
    life_expectancy_df = drop_unused_columns(life_expectancy_df)
    life_expectancy_df = impute_columns(life_expectancy_df)
    life_expectancy_df = handle_outliers(life_expectancy_df)
    life_expectancy_df = smoothe_noise(life_expectancy_df)
    return life_expectancy_df
