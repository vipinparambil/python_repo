
import pandas as pd
import numpy as np
import os

def read_data():
    raw_data_path = os.path.join(os.path.pardir, 'data','raw')
    test_file_path = os.path.join(raw_data_path, "test.csv")
    train_file_path = os.path.join(raw_data_path, "train.csv")    
    train_df = pd.read_csv(train_file_path, index_col='PassengerId')
    test_df = pd.read_csv(test_file_path, index_col='PassengerId')   
    test_df['Survived'] = -888    
    df = pd.concat((train_df, test_df), axis =0, sort=True)
    return df   

def get_title(name):
    title_group = {'mr' : 'Mr',
                  'mrs' : 'Mrs',
                 'miss' : 'Miss',
                 'master' : 'Master',
                 'don' : 'Sir',
                  'rev' : 'Sir',
                  'dr' : 'Officer',
                  'mme' : 'Mrs',
                  'ms' : 'Mrs',
                  'major' : 'Officer',
                  'lady' : 'Lady',
                  'sir' : 'Sir',
                  'mlle' : 'Miss',
                  'col' : 'Officer',
                  'capt' : 'Officer',
                  'the countess' : 'Lady',
                  'jonkheer' : 'Sir',
                  'dona' : 'Lady'
                    }
    name_su = name.split(',')[1]
    name_t = name_su.split('.')[0]
    title = name_t.strip().lower()
    return title_group[title]

def fill_missing_values(df):
    df.Embarked.fillna('C', inplace=True)    
    median_fare = df.loc[(df.Pclass == 3) & (df.Embarked == 'S'), 'Fare'].median()
    df.Fare.fillna(median_fare, inplace=True)
    age_by_title = df.groupby('Title').Age.transform('median')
    df.Age.fillna(age_by_title, inplace=True)
    return df
    
def get_deck(Cabin):
    return np.where(pd.notnull(Cabin), str(Cabin)[0].upper(),'Z')

def write_data(df):
    processed_data_path = os.path.join(os.path.pardir, "data", "processed")
    train_path = os.path.join(processed_data_path, 'train.csv')
    test_path = os.path.join(processed_data_path, "test.csv")
    ## write train csv
    df.loc[df.Survived != -888].to_csv(train_path)
    ### write csv to test without Srvived column
    columns = [ c  for c in df.columns if c != 'Survived']
    df.loc[df.Survived == -888, columns].to_csv(test_path)
    
def process_data(df):
    # using the method chaining concept
    return (df
         # create title attribute - then add this 
         .assign(Title = lambda x: x.Name.map(get_title))
         # working missing values - start with this
         .pipe(fill_missing_values)
         # create fare bin feature
         .assign(Fare_Bin = lambda x: pd.qcut(x.Fare, 4, labels=['very_low','low','high','very_high']))
         # create age state
         .assign(AgeState = lambda x : np.where(x.Age >= 18, 'Adult','Child'))
         .assign(FamilySize = lambda x : x.Parch + x.SibSp + 1)
         .assign(IsMother = lambda x : np.where(((x.Sex == 'female') & (x.Parch > 0) & (x.Age > 18) & (x.Title != 'Miss')), 1, 0))
          # create deck feature
         .assign(Cabin = lambda x: np.where(x.Cabin == 'T', np.nan, x.Cabin)) 
         .assign(Deck = lambda x : x.Cabin.map(get_deck))
         # feature encoding 
         .assign(IsMale = lambda x : np.where(x.Sex == 'male', 1,0))
         .pipe(pd.get_dummies, columns=['Deck', 'Pclass','Title', 'Fare_Bin', 'Embarked','AgeState'])
         # add code to drop unnecessary columns
         .drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'], axis=1)
         # reorder columns
         .pipe(reorder_columns)
         )
            
            
def reorder_columns(df):
    columns = [c for c in df.columns if c!= 'Survived' ]
    columns = ['Survived'] + columns
    df = df[columns]
    return df


if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df) 
