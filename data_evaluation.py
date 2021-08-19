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
    col_names = [col for col in df.columns if col != actual]
    for col in col_names:
        table_names += f'<th><center>{str(col.capitalize())}</center></th>'
    for col in col_names:
        val = pd.crosstab(df[col], df[actual], rownames=['Pred'], colnames=['Actual']).reset_index()
        report = pd.DataFrame(classification_report(df[actual], df[col], output_dict=True))
        uniques = [col for col in val.columns if col not in ['Pred']]
        
        
        accuracy_row = ['Accuracy']
        accuracy_row.extend(['-----' for n in range(len(uniques))])
        accuracy_row[-1] = report.accuracy[0] * 100
        
        
        divider = ['-----' for n in range(len(uniques)+1)]
        val.loc[len(val.index)] = divider
        val.loc[len(val.index)] = accuracy_row
        val.loc[len(val.index)] = divider
        
        for unique in uniques:
            df2 = [{'Pred': 'Precision', unique: report[unique][0] * 100},
                  {'Pred': 'Recall', unique: report[unique][1] * 100}]
            val = val.append(df2, ignore_index = True)
            
        new_df = val.set_index('Pred')
        tab = new_df.to_markdown()
        
        
        tables += f'<td>\n\n{tab}\n\n</td>\n\n'

    result += f'''<table>
    <tr>{table_names}</tr>
    <tr>{tables}</tr></table>'''

    return result


def model_generator():
    pass

