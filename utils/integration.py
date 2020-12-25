import sklearn
from utils.general import iterative_impute
from sklearn.impute import IterativeImputer
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

def drop_irrelevant_columns(life_expectancy_df, happiness_dfs, countries_df):
    happiness_df_analysis = happiness_dfs[["Happiness Rank", "Happiness Score", "Location", "Year", "Freedom", "Economy (GDP Per Capita)"]]
    happiness_df_analysis = happiness_df_analysis.rename(columns={"Economy (GDP Per Capita)":"GDP/Capita"})
    countries_df_analysis = countries_df[["Name","Region","Gini", "Literacy Rate(%)", "Unemployement(%)"]]
    life_expectancy_df_analysis = life_expectancy_df[["Country", "Status", "Life Expectancy","Percentage Expenditure","GDP","HIV/AIDS","Measles","BMI","Year"]].copy()
    return life_expectancy_df_analysis, happiness_df_analysis, countries_df_analysis

def add_years_to_life_expectancy(life_expectancy_df):
    number_columns = life_expectancy_df.select_dtypes(exclude="object").columns
    missing_years = [2016,2017,2018,2019]
    for year in missing_years:
        new_data = life_expectancy_df[life_expectancy_df.Year == 2015].copy()
        new_data[number_columns] = np.NaN
        new_data["Year"] = year
        life_expectancy_df = life_expectancy_df.append(new_data)
    return life_expectancy_df

def impute_life_expectancy_new_years(life_expectancy_df):
    missing_columns = life_expectancy_df.select_dtypes(exclude="object").columns
    countries = life_expectancy_df.Country.unique()
    for column in missing_columns:
        if(column=="Year"):
            continue
        for country in countries:
            index = life_expectancy_df.Country == country
            data = iterative_impute(life_expectancy_df[index], [column,"Year"], column)
            life_expectancy_df.loc[index,column] = data[:,0]
    index = life_expectancy_df["Measles"] < 0 
    life_expectancy_df.loc[index,"Measles"] = 0
    return life_expectancy_df

def select_life_expectancy_new_years(life_expectancy_df):
    life_expectancy_df = life_expectancy_df[life_expectancy_df.Year.isin([2015,2016,2017,2018,2019])]
    return life_expectancy_df

def merge(life_expectancy_df, happiness_dfs, countries_df):
    life_expectancy_250_world =  life_expectancy_df.merge(countries_df, how="left",left_on="Country", right_on="Name")
    happiness_dfs["Year"] = happiness_dfs["Year"].astype(int)
    life_expectancy_250_world_happiness = life_expectancy_250_world.merge(happiness_dfs, how="left", left_on=["Country","Year"], right_on=["Location","Year"])
    return life_expectancy_250_world_happiness

def impute_integrated_dataset(integrated):
    imputer = IterativeImputer(random_state=0)
    data = integrated.select_dtypes(exclude="object")
    imputer = imputer.fit(data)
    t_data = imputer.transform(data)
    integrated[data.columns] = t_data
    integrated = integrated.drop(columns=["Name","Location"])
    integrated["Year"] = integrated["Year"].astype(int)
    return integrated

def normalize_features(integrated):
    columns_normalized = ["Literacy Rate(%)","Percentage Expenditure","Gini","HIV/AIDS","Measles","BMI","Life Expectancy","Happiness Score","GDP/Capita","GDP"]
    for column in columns_normalized:
        integrated["{column}_normalized".format(column=column)] = minmax_scale(integrated[column])
    return integrated


def integrate(life_expectancy_df, happiness_dfs, countries_df):
    life_expectancy_df, happiness_dfs, countries_df = drop_irrelevant_columns(life_expectancy_df, happiness_dfs, countries_df)
    life_expectancy_df = add_years_to_life_expectancy(life_expectancy_df)
    life_expectancy_df = impute_life_expectancy_new_years(life_expectancy_df)
    life_expectancy_df = select_life_expectancy_new_years(life_expectancy_df)
    integrated = merge(life_expectancy_df, happiness_dfs, countries_df)
    integrated = normalize_features(integrated)
    return integrated
