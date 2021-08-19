import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
import acquire
import prepare

import warnings
warnings.filterwarnings('ignore')

def confusion_table(df: pd.DataFrame) -> str:
    '''Takes DataFrame and prints a formatted Confusion Table/Matrix in
    markdown for Juypter notebooks. The first column must be the actual values and all
    the other columns have to be model values or predicted values.
    
    Parameters
    ----------
    
    df : pandas DataFrame
        Requires the 'actual' values to be the first column 
        and all other columns to be the predicted values.
        
    Returns
    -------
    str 
        string that is formatted with HTML and markdown
        for Juypter Notebooks so that it can be copied and pasted into a 
        markdown cell and easier to view the values.
        
    '''
    result = str()
    table_names = str()
    tables = str()
    actual = df.columns[0]
    col_names = [str(col) for col in df.columns if col != actual]
    for col in col_names:
        table_names += f'<th><center>{str(col.capitalize())}</center></th>'
    for col in col_names:
        
        # Crosstab the model row vs the actual values
        val = pd.crosstab(df[col], df[actual], rownames=['Pred'], colnames=['Actual']).reset_index()
        
        # Generate report values, precision, recall, accuracy
        report = pd.DataFrame(classification_report(df[actual], df[col], output_dict=True))
        
        # Get all the uniques in a list
        uniques = [str(col) for col in val.columns if col not in ['Pred']]
        
        # Make a line break in table for Accuracy
        accuracy_row = ['Accuracy']
        accuracy_row.extend(['-----' for n in range(len(uniques))])
        accuracy_row[-1] = report.accuracy[0] * 100
        
        # Ensure all columns names are strings
        val = val.rename(columns=lambda x: str(x))
        
        # Create a divider of len n
        divider = ['-----' for n in range(len(uniques)+1)]
        val.loc[len(val.index)] = divider
        # Input the accuracy
        val.loc[len(val.index)] = accuracy_row
        val.loc[len(val.index)] = divider
        
        for unique in uniques:
            # Iterate through all uniques and fetch their precision and 
            # Recall values to put into the table.
            precision = report[str(unique)][0] * 100
            recall = report[str(unique)][1] * 100
            df2 = [{'Pred': 'Precision', unique: precision},
                  {'Pred': 'Recall', unique: recall}]
            
            # Add the values to the bottom of the table
            val = val.append(df2, ignore_index=True)
        
        # Collapse the index under Pred to have the table smaller
        new_df = val.set_index('Pred')
        # Put the table to markdown
        tab = new_df.to_markdown()
        
        
        tables += f'<td>\n\n{tab}\n\n</td>\n\n'

    result += f'''<table>
    <tr>{table_names}</tr>
    <tr>{tables}</tr></table>'''

    return result


def replace_obj_cols(daf: pd.DataFrame, dropna=False) -> (pd.DataFrame, dict, dict):
    '''Takes a DataFrame and will return a DataFrame that has
    all objects replaced with int values and the respective keys are return
    and a revert key is also generated.
    
    Parameters
    ----------
    
    df : pandas DataFrame
        Will take all object/str based column data types and convert their values
        to integers to be input into a ML algorithm.
    
    dropna: bool
        If this is True, it will drop all rows with any column that has NaN 
        
    Returns
    -------
    DataFrame 
        The returned DataFrame has all the str/object values replaced with integers
        
    dict - replace_key
        The returned replace_key shows what values replaced what str
        
    dict - revert_key
        The returned revert_key allows it to be put into a df.replace(revert_key) 
        to put all the original values back into the DataFrame
    
    Example
    -------
    >>>dt = {'Sex':['male', 'female', 'female', 'male', 'male'],
        'Room':['math', 'math', 'gym', 'gym', 'reading'],
        'Age':[11, 29, 15, 16, 14]}

    >>>test = pd.DataFrame(data=dt)
    
    >>>test, rk, revk  = replace_obj_cols(test)
       Sex  Room  Age
    0    0     0   11
    1    1     0   29
    2    1     1   15
    3    0     1   16
    4    0     2   14,
    
    {'Sex': {'male': 0, 'female': 1},
    'Room': {'math': 0, 'gym': 1, 'reading': 2}},
    
    {'Sex': {0: 'male', 1: 'female'},
    'Room': {0: 'math', 1: 'gym', 2: 'reading'}}
    
    >>>test.replace(revk, inplace=True)
          Sex     Room  Age
    0    male     math   11
    1  female     math   29
    2  female      gym   15
    3    male      gym   16
    4    male  reading   14
        
    '''
    df = daf.copy(deep=True)
    replace_key = {}
    revert_key = {}
    col_names = df.select_dtypes('object').columns
    if dropna:
        df.dropna(inplace=True)
    for col in col_names:
        uniques = list(df[col].unique())
        temp_dict = {}
        rev_dict = {}
        for each_att in uniques:
            temp_dict[each_att] = uniques.index(each_att)
            rev_dict[uniques.index(each_att)] = each_att
        replace_key[col] = temp_dict
        revert_key[col] = rev_dict
    df.replace(replace_key, inplace=True)
    
    return df, replace_key, revert_key

