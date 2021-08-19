from env import user, password, host
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def get_db_url(username: str, hostname: str , password: str, database_name: str):
    '''
    Takes username, hostname, password and database_name and 
    returns a connection string
    '''
    connection = f'mysql+pymysql://{username}:{password}@{hostname}/{database_name}'
    
    return connection


def get_titanic_data():
    filename = "titanic.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)

    else:
        conn = get_db_url(username=user, password=password, hostname=host, database_name='titanic_db')
        
        sql = '''
        select * 
        from passengers
        '''
        df = pd.read_sql(sql, conn)

        df.to_csv(filename)

        return df


def get_iris_data():
    filename = "iris.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)

    else:
        conn = get_db_url(username=user, password=password, hostname=host, database_name='iris_db')
        
        sql = '''
        select * 
        from measurements
        join species
        on species.species_id = measurements.species_id
        '''
        df = pd.read_sql(sql, conn)

        df.to_csv(filename)

        return df


def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test