import pandas as pd
import numpy as np

def get_category_levels(loc_codes):
    """
    Load the categories and split them into four levels of depth.

    Returns:
    level_1 (DataFrame): Categories at level 1.
    level_2 (DataFrame): Categories at level 2.
    level_3 (DataFrame): Categories at level 3.
    level_4 (DataFrame): Categories at level 4.
    """
    # Load the dataframe with the category names and codes
    df_translation_dict = pd.read_excel(loc_codes, index_col=0)

    # Remove all special aggregates that start with 'S'
    df_translation_dict = df_translation_dict[~df_translation_dict['codes'].str.startswith('S')]

    # Add a 'C' before all columns
    df_translation_dict['codes'] = 'C' + df_translation_dict['codes']

    # Define as level 0 the category that has only zeros
    level_0 = df_translation_dict[df_translation_dict['codes'] == 'C000000']

    # Define as level 1 the categories that have a code ending with 0000 and starts with at least 1 non-zero digit in the first two values
    level_1 = df_translation_dict[df_translation_dict['codes'].str.endswith('0000') & (df_translation_dict['codes'].str[0:3] != 'C00')]

    # Define as level 2 the categories that have a code ending with 000 and a non-zero value in the third digit
    level_2 = df_translation_dict[df_translation_dict['codes'].str.endswith('000') & (df_translation_dict['codes'].str[3] != '0')]

    # Define as level 3 the categories that have a code ending with 00 and a non-zero value in the fourth digit
    level_3 = df_translation_dict[df_translation_dict['codes'].str.endswith('00') & (df_translation_dict['codes'].str[4] != '0')]

    # Define as level 4 the categories that have a code ending with 0 and a non-zero value in the fifth digit
    level_4 = df_translation_dict[df_translation_dict['codes'].str.endswith('0') & (df_translation_dict['codes'].str[5] != '0')]

    return level_0, level_1, level_2, level_3, level_4, df_translation_dict

# Write a function that takes a column name and dataset with cpi input as functions and returns the number of non-null values, mean value, and standard deviation, and min value and max value
def get_descriptive_stats_for_col(column_name, cpi_data):
    """
    Get the number of non-null values, mean value, standard deviation, min value, and max value of a column.

    Parameters:
    column_name (str): The name of the column.
    cpi_data (DataFrame): The DataFrame with the CPI data.

    Returns:
    num_non_null (int): The number of non-null values in the column.
    mean (float): The mean value of the column.
    std (float): The standard deviation of the column.
    min (float): The minimum value of the column. 
    max (float): The maximum value of the column.
    """
    # If the column name is not in the CPI data, return None for all values
    if column_name not in cpi_data.columns:
        return None, None, None, None, None

    num_non_null = cpi_data[column_name].count()
    mean = cpi_data[column_name].mean()
    std = cpi_data[column_name].std()
    min = cpi_data[column_name].min()
    max = cpi_data[column_name].max()
    
    return num_non_null, mean, std, min, max

def convert_to_growthrate(column_name, cpi_data):
    """
    Convert a column to growth rate as 100 * log(x_t / x_{t-1}).

    Parameters:
    column_name (str): The name of the column.
    cpi_data (DataFrame): The DataFrame with the CPI data.

    Returns:
    growth_rate (Series): The growth rate of the column.
    """
    # If the column name is not in the CPI data, return None
    if column_name not in cpi_data.columns:
        return None

    growth_rate = 100 * np.log(cpi_data[column_name] / cpi_data[column_name].shift(1))
    return growth_rate

def calculate_summary_stats(dataframe):
    """
    Calculate summary statistics for the given dataframe.

    Parameters:
    dataframe (DataFrame): The dataframe for which to calculate summary statistics.

    Returns:
    length (int): Length of the dataframe.
    mean_n (float): Mean of the 'n' values.
    weighted_mean (float): Weighted mean of the means.
    total_std (float): Total standard deviation approximated using the 'n's and standard deviations.
    total_min (float): Total minimum value.
    total_max (float): Total maximum value.
    """
    length = len(dataframe)
    mean_n = dataframe['n'].mean()
    weighted_mean = np.average(dataframe['mean'], weights=dataframe['n'])
    total_std = dataframe['std'].mean() #np.sqrt(np.sum(dataframe['std']**2 * dataframe['n']) / dataframe['n'].sum())
    total_min = dataframe['min'].min()
    total_max = dataframe['max'].max()

    return length, mean_n, weighted_mean, total_std, total_min, total_max

def create_table_1(loc_cpi):

    # Derive the category levels
    level_0, level_1, level_2, level_3, level_4, df_translation_dict = get_category_levels(loc_codes=r"C:\Users\beerp\Data\HRNN\df_codes.xlsx")
    level_dfs = [level_0, level_1, level_2, level_3, level_4]

    # Load the data from "C:\Users\beerp\Data\NIPE\public_data_april.xlsx"
    cpi_data_original = pd.read_excel(loc_cpi, index_col=0)

    # Convert to growthrates
    cpi_data = cpi_data_original.apply(lambda x: convert_to_growthrate(x.name, cpi_data_original))

    # Extend the level dataframes with the descriptive statistics
    descriptive_stat_names = ['n', 'mean', 'std', 'min', 'max']
    summary_stats = pd.DataFrame(columns=['level', 'length', 'mean_n', 'weighted_mean', 'total_std', 'total_min', 'total_max'])

    for i, df in enumerate(level_dfs):
        df[descriptive_stat_names] = df['codes'].apply(lambda x: pd.Series(get_descriptive_stats_for_col(x, cpi_data))).values
        if i == 0:
            summary_stats.loc[i] = ['Headline only'] + list(calculate_summary_stats(df))
        else:
            summary_stats.loc[i] = [f'Level {i}'] + list(calculate_summary_stats(df))

    all_levels = pd.concat(level_dfs, ignore_index=True)
    summary_stats.loc[len(summary_stats)] = ['Full hierarchy'] + list(calculate_summary_stats(all_levels))

    return summary_stats
