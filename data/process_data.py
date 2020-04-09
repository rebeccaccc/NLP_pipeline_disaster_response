# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine

# function to load data 

def load_data(messages_filepath, categories_filepath):
    ''' load data in its raw form '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'outer', on = 'id')
    return df

# function to clean data 
def clean_data(df):
    ''' Clean data to have an indicator variable for each message category.'''
    # create a dataframe of the 36 individual category columns
    categories = pd.Series(df.categories).str.split(';', expand = True)
    #extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories
    df.drop('categories', axis = 'columns', inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    df = df.drop_duplicates(subset = 'message', keep = 'first')
    return df

# function to save data 
def save_data(df, database_filename):
    ''' Save data into a table in a sql database '''
    engine = create_engine('sqlite:///'+ str(database_filename))
    df.to_sql('response', engine, index=False)
    pass  


def main():
    '''The main function to excute the chained functions.'''
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