# import libraries
import pandas as pd
import sys
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function used to load data (Extracting Part)
    -
    Parameters: 
        -messages_filepath: file path to messages.csv
        -categories_filepath: file path to categories.csv
    Returns:
        -df: merged messages+categories dataframes
    """
    messages_df = pd.read_csv(messages_filepath) # Load the messages dataset
    categories_df = pd.read_csv(categories_filepath) # Load the categories dataset
    df = pd.merge(messages_df, categories_df, on='id') # Merge(JOIN) both datasets on column id
    return df


def clean_data(df):
    """
    Function to clean data (Transforming Part)
    Parameters:
        -df: merged dataframe of both messages+categories
    Returns:
        -df: cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    row = categories.iloc[0]

    category_colnames = list(map(lambda x: x[:-2], row))
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1.
    # Loop through each column in the DataFrame
    for column in categories:
    # Set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[-1]
        
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], errors='coerce')  # Convert non-numeric values to NaN
        
        # Replace values other than 0 and 1 with None
        categories[column] = categories[column].apply(lambda x: x if x in [0, 1] else None)
        

    
    # drop the original categories column from `df`
    df = df.drop(columns='categories')
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    
    # drop duplicates
    df = df.drop_duplicates()
    # drop nans because they are the values of 2
    df = df.dropna()
    # make sure 0 and 1 are ints
    df.iloc[:,4:]=df.iloc[:,4:].astype(int)
    
    
    return df


def save_data(df, database_filename):
    """
    Function to save dataframe in a SQLite database
    -
    Parameters:
    -df: cleaned dataframe
    -database_filename: path to the database filename
    -
    Returns: N/A
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('messages', engine, index=False,if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
