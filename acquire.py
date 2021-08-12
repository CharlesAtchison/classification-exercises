from env import user, password, host
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
