import pandas as pd
from acquire import get_iris_data


def prep_data(iris_data):
    '''Takes iris_dataframe and drop columns and then returns dummy
    DataFrames for each species type.
    '''
    args = [col_name for col_name in iris_data.columns if col_name not in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species_name']]
    iris_data.drop(columns=[*args], inplace=True)
    iris_data.rename(columns={'species_name': 'species'}, inplace=True)
    dummy_vars = [iris_data[iris_data.species == s] for s in iris_data.species.unique()]
    
    return iris_data, dummy_vars
    

