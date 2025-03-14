from calendar import c
import re
import pandas as pd
from module_one import clean_dataset

def filter_stockcodes(df, stockcodes):
    """
    Filters the DataFrame for the given StockCodes.
    
    Args:
        df (pd.DataFrame): The main DataFrame.
        stockcodes (list): List of StockCodes to filter.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    stockcodes_list = stockcodes['StockCode'].str.replace(r'[^0-9]', '', regex=True)
    # Filter the DataFrame
    filtered_df = df[df["StockCode"].isin(stockcodes_list)]
    filtered_df.to_csv('data\dataset\CNN_Model_Train_Data_combined.csv')
    return filtered_df

def clean():
    # Apply the function
    df = clean_dataset('data\dataset\dataset.csv')
    stockcodes_to_filter = pd.read_csv('data\dataset\CNN_Model_Train_Data.csv')
    filtered_df = filter_stockcodes(df, stockcodes_to_filter)

    # Display the filtered DataFrame
    print(filtered_df)


if __name__ == "__main__":
    clean()