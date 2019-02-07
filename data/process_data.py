import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Function that load and merge 2 raw datasets: messages and categories
    Arg:
    messages_filepath: filepath of messages
    categories_filepath: filepath of  categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,left_on='id', right_on='id', how='left')
    return df


def clean_data(df):
    '''
    Function that prepare the dataset for modeling, including transform target variables, drop duplicate and remove useless varaibles
    
    Arg:
    df : the merged dataset
    '''
    
    # transform original categories column,from single column that has values like'related-1;request-0;offer-0;aid_related-0;
    # ...' to several columns with names as related, request, offer, aid_related and so on, values as 1 or 0.
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
    
    # drop original categories column
    df=df.drop(['categories'],axis=1)
    
    #merga original messages with new categories column
    df = pd.concat([df, categories], axis=1)
    
    #drop duplicate messages
    df=df.drop_duplicates()
    
    #since those messages with 2 in related all have 0 in other categories, they are actullay unrelated messages, so I change 
    # the value from 2 to 0.
    df.loc[df['related']==2,'related']=0
    
    # since all values in child_alone category is 0, it's meaningless to predict this category, so I remove this category
    df=df.drop(['child_alone'],axis=1)
    
    return df


def save_data(df, database_filename):
    '''
    Function that save the clean dataset into an sqlite database
    
    Arg:
    df : the cleaned dataset
    database_filename : name of sqlite database
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster message', engine, index=False)


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
