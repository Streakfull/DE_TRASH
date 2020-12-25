from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def contstruct_string_data(df, column, type):
    rows, columns = df.shape
    unique_values = df[column].nunique()
    present_values = df[column].count()
    missing_values = rows - present_values
    missing_values_percentage = np.round((missing_values/rows)*100, 2)
    present_values_percentage = np.round((present_values/rows)*100, 2)
    mean = "N/A"
    std = "N/A"
    if(type != "object"):
        mean = np.round(np.mean(df[column]), 2)
        std = np.round(np.std(df[column]), 2)

    most_common = df[column].mode()[0]
    data = [("Unique Values", unique_values),
            ("Present Values", present_values),
            ("Present Values %", present_values_percentage),
            ("Missing Values", missing_values),
            ("Missing Values %", missing_values_percentage),
            ("Most Common", most_common),
            ("Mean", mean),
            ("STD", std)
            ]
    return data


def display_column_data(df, column, type):
    data = contstruct_string_data(df, column, type)
    print("========================= {column} data =========================".format(
        column=column))
    for name, value in data:
        percentage_sign = "%" if "%" in name else ""
        print("{name}: {value}{percentage_sign}".format(
            name=name, value=value, percentage_sign=percentage_sign))
    return data


def plot_column(df, column, index, type):
    graph = plt.figure()
    graph.suptitle("{column} Statistics".format(column=column))
    if(type == "object"):
        graph = sns.histplot(data=df, x=column)
    data = display_column_data(df, column, type)
    unique_values = df[column].nunique()
    if(unique_values > 25 and type == "object"):
        graph.set_xticklabels("")
    if(type != "object"):
        graph = sns.displot(df[column], kde=True)
        box_plot = plt.figure()
        box_plot = sns.boxplot(x=df[column])
    plt.show()
    return data


def transform_data(data):
    data_dict = []
    for row in data:
        row_dict = {}
        for column in row:
            column_name, value = column
            row_dict[column_name] = value
        data_dict.append(row_dict)
    return data_dict

# Plots all column data for a given df


def plot_column_data(df):
    data_types = df.dtypes
    # print(data_types)
    data = []
    for index, column in enumerate(df):
        column_data = plot_column(df, column, index, data_types[index])
        data.append(column_data)
    transformed_data = transform_data(data)
    column_data_df = pd.DataFrame.from_records(transformed_data)
    column_data_df = column_data_df.set_index(df.columns)
    print("========================= Columns Summary =========================")
    return column_data_df

    # data_df = pd.DataFrame.from_(data)
    # print(data_df,"DATA")


def explore_df(df):
    rows, columns = df.shape
    #print(rows, columns, "OKK??")
    print(df[0:5])
    info = df.info()
    return plot_column_data(df)


def examine_missing_column(df, group_by, column):
    index = df[column].isnull()
    grouped_df = df[index].groupby(group_by).sum()[:][column]
    return grouped_df


def iterative_impute(df, columns, column):
    imputer = IterativeImputer(random_state=0)
    df = df[columns]
    imputer = imputer.fit(df)
    return imputer.transform(df)
