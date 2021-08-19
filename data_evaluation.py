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


def model_generator():
    pass

