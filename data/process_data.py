import sys
import pandas as pd
import sqlalchemy as sql

def load_data(messages_filepath, categories_filepath):
    """Load & merge the messages and categories datasets
    
    inputs:
    messages_filepath: (str). Path where the messages csv file is stored.
    categories_filepath: (str). Path where the categories csv file is stored.
       
    returns:
    df: dataframe. A data frame containing the messages and categories merged on 'id' column.
    """
    mess_df = pd.read_csv(messages_filepath)
    cat_df = pd.read_csv(categories_filepath)
    
    return mess_df.merge(cat_df, how='left', on='id')

def clean_data(df):
    """
    Cleans the data frame containing messages and categories. ie. remove duplicate messages,
    create dummy variables for all categories.
    
    inputs:
    dataframe: (df). Dataframe containing messages and categories data
       
    returns:
    df: dataframe. A data frame that has been cleaned  
    """    
    
    categories = df.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    #slice the categories at position -2
    category_colnames = row.apply(lambda x: x[:-2])
    
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1:])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])  
        
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)  
        
    #Concatenate original df with categories df
    df = pd.concat([df, categories], axis=1)
        
    #Drop duplicate messages
    df = df.drop_duplicates(subset=['message'])
        
    #Create dummy variables for the 'genre' column
    df = pd.concat([df, pd.get_dummies(df['genre'])], axis=1)
    df = df.drop(['genre', 'social'], axis=1)     
        
    return df


def save_data(df, database_filename):
    """
    Saves the data from a data frame in an sql database
    
    inputs:
    df: (dataframe). Dataframe containing messages and categories data
    database_filename: (str). The file name for the data base
       
    returns:
    None. 
    """      
    
    engine = sql.create_engine('sqlite:///{}.db'.format(database_filename))
    
    tbname = 'dr_messages_tbl'
    
    df.to_sql(tbname, engine, index=False, if_exists='replace')

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