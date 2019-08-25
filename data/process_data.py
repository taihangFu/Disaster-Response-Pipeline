import sys

# import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Load 2 csv files about disaster messages and categories from given filepaths then merge them into a pandas dataframe
    
    Args:
        messages_filepath (str): contains disaster_messages.csv
        categories_filepath (str): contains disaster_categories.csv
        
    Returns:
        df: new dataframe by  merging data in messages and categories csv files
    '''
    
    # load  dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Remove duplicates rows who have he same id=df
    messages=messages.drop_duplicates('id')
    categories=categories.drop_duplicates('id')
    
    # Merge datasets
    df = messages.merge(categories, how='outer',\
                                   on='id')
    
    return df
    
def clean_data(df): 
    '''Clean the data by creating a dataframe that contains message information and relevant disaster categories
    
    Args:
        database_filepath (str): filepath where the database(created by data/process_data.py) is located
    
    Returns: 
        df: a dataframe contains messages relevant columns and 36 categories for the messages relevant columns   
    '''
    
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    #  Split categories into separate category columns
    row = categories.iloc[0]
    category_colnames = row.values # extract a list of new column names for categories
    categories.columns = category_colnames # rename the columns of `categories`
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str.get(-1)  # set each value to be the last character of the string
        categories[column] =  categories[column].astype('int32')  # convert column from string to numeric
    
    ## Clean Data
    categories = categories.rename(columns={"1":"related-1"}) # work around: rename first columns label since it is modified with unknown reason
    categories['related-1'] = categories['related-1'].replace(2, categories['related-1'].mode()[0]) # replace label 2 to Mode as we need all columns binary class but 'related-1' has 3 classes
    
    # Replace categories column in df with new category columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df.reset_index(drop=True), categories.reset_index(drop=True)], axis=1)
    
    return df
    
def save_data(df, database_filename):
    '''save df as a sqlitedatabase engine into given database_filename
    '''
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)  


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