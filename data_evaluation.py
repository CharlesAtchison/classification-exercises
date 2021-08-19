import pandas as pd

def confusion_table(df: pd.DataFrame, target:str) -> str:
    '''Takes DataFrame and prints a formatted Confusion Table/Matrix in
    markdown for Juypter notebooks.
    
    Parameters
    ----------
    
    df : pandas DataFrame
        Requires the 'actual' values to be a column named 'actual'
        and all other columns to be the predicted values.
    target : str
        This is the target predicted value that will be used to measure
        for recall and precision in a confusion matrix.
        
    Returns
    -------
    str 
        string that is formatted with HTML and markdown
        for Juypter Notebooks so that it can be copied and pasted into a 
        markdown cell and easier to view the values.
        
    '''
    table_names = str()
    tables = str()
    
    col_names = [col for col in df.columns if col != 'actual']
    for col in col_names:
        table_names += f'<th><center>{str(col.capitalize())}</center></th>'
    
    for col in col_names:
        val = pd.crosstab(df[col], df.actual, rownames=['Pred'], colnames=['Actual']).to_markdown()
        recall_subset = df[df.actual == target]
        recall_val = (recall_subset.actual == recall_subset[col]).mean() * 100
        precision_subset = df[df[col] == target]
        precision_val = (precision_subset.actual == precision_subset[col]).mean() * 100
        
        val += f'\n| Precision|---|{precision_val:0.2f}%|\n| Recall|---|{recall_val:0.2f}%|'
        tables += f'<td>\n\n{val}\n\n</td>\n\n'
        
    result = f'''<center><h1>{target.capitalize()}</h1></center>
    <table>
    <tr>{table_names}</tr>
    <tr>{tables}</tr></table>'''

    return result