import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import acquire, prepare, warnings
import numpy as np

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
        val = pd.crosstab(df[col], df[actual], rownames=['Prediction'], colnames=['Actual']).reset_index()
        
        # Generate report values, precision, recall, accuracy
        report = pd.DataFrame(classification_report(df[actual], df[col], output_dict=True))
        
        # Get all the uniques in a list
        uniques = [str(col) for col in val.columns if col not in ['Prediction']]
        
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
            # Iterate through all uniques and fetch their precision, 
            # recall, f1-score and support values to put into the table.
            precision = report[str(unique)][0] * 100
            recall = report[str(unique)][1] * 100
            f1_score = report[str(unique)][2] * 100
            support = report[str(unique)][3]
            df2 = [{'Prediction': 'Precision', unique: precision},
                  {'Prediction': 'Recall', unique: recall},
                  {'Prediction': 'f1-score', unique: f1_score},
                  {'Prediction': 'support', unique: support}]
            
            # Add the values to the bottom of the table
            val = val.append(df2, ignore_index=True)
        
        # Collapse the index under Prediction to have the table smaller
        new_df = val.set_index('Prediction')
        # Put the table to markdown
        tab = new_df.to_markdown()
        
        
        tables += f'<td>\n\n{tab}\n\n</td>\n\n'

    result += f'''<div><center><h3>{actual}</h3>
    <table>
    <tr>{table_names}</tr>
    <tr>{tables}</tr></table></center></div>'''

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

def explore_validation_curve(X : pd.DataFrame, y : pd.DataFrame, param_name : str, num_est : np.ndarray, model, color_args={'training': ['black', 'orange'], 'validation': ['red', 'cyan']}) -> pd.DataFrame:
    '''Function that will print out plot of the single selected hyperparameter for the validation
    curve the plotted mean for each nth value and the standard deviation for each nth value. This requires
    some model generated. 

    
    Parameters
    ----------
    X : pandas DataFrame
        Some x_values dataframe to be put into the validation_curve.
    
    y : pandas DataFrame
        Some y_values dataframe to be put into the validation_curve.

    param_name : str
        What hyperparmeter you would like to explore within the validation_curve.
    
    num_est : numpy ndarray
        The range of values to test within the validation_curve.
    
    model : Sklearn model
        Can check sklearn models, verified currently compatible with:
            DecisionTreeClassifier(),
            RandomForestClassifier()

    color_args : dict
        Not required, default values:
        {'training': ['black', 'orange'],
         'validation': ['red', 'cyan']}
        
        can personalize but must be in the format of
        # training_line    line_color      standard_dev fill color
        {'training}    :   ['black'     ,  'orange']

        # validation_line    line_color      standard_dev fill color
        {'validation}    :   ['red'     ,  'cyan']

    Returns
    -------
    pandas DataFrame 
        The returned DataFrame contains mean, std, n_val for all n values for the defined param_name
    
    Examples
    -------
    >>> val = explore_validation_curve(X_train, y_train, 'max_depth', np.arange(1, 13, 1),
                                DecisionTreeClassifier())

    
    	mean	std	        n_val
    0	0.799102	0.059686	1
    1	0.775102	0.052757	2
    2	0.817265	0.044255	3
    3	0.789184	0.049054	4
    4	0.799184	0.029710	5


    >>> val = explore_forest_validation_curve(X_train, y_train, 'n_estimators', np.arange(1, 300, 3),
                                RandomForestClassifier(), color_args={'training': ['green', 'purple'],
                                                                     'validation': ['orange', 'red']})
    >>> val.head(n=5)

            mean	std	    n_val
    0	0.710939	0.081079	1
    1	0.795061	0.035173	4
    2	0.809143	0.051167	7
    3	0.837347	0.049380	10
    4	0.815102	0.049550	13

    >>> print(X_train.shape[1])
    12

    >>> val = explore_forest_validation_curve(X_train, y_train, 'max_depth', np.arange(1, 13, 1),
                                RandomForestClassifier(n_estimators=178, criterion='gini', min_samples_leaf=3))
    >>> val.head(n=5)

            mean	std	    n_val
    0	0.787143	0.049357	1
    1	0.797184	0.052523	2
    2	0.815265	0.043630	3
    3	0.823224	0.035994	4
    4	0.835143	0.039832	5
    '''
    num_est_df = pd.DataFrame({'n_val':num_est})

    # Check that if the param_name is 'max_depth' that the range is not greater than the number of attributes in model.
    if param_name == 'max_depth' and len(num_est) > X.shape[1]:
        raise Exception(f"Sorry, your range cannot be larger than the number of attributes ({X.shape[1]}) when using 'max_depth")
        
    # Calculate validation curve and return as array
    x_score, y_score= validation_curve(model, 
                                                X = X, y = y,
                                                param_name = param_name, cv=10,
                                                param_range = num_est, scoring='accuracy', n_jobs=-1)


    # Calculate the mean and std for the training_score (x)
    x_mean = np.mean(x_score, axis=1)
    x_std = np.std(x_score, axis=1)


    # Calculate the mean and std for the validate score (y) and put to dataframes for merging
    y_mean = np.mean(y_score, axis=1)
    y_std = np.std(y_score, axis=1)
    y_std_df = pd.DataFrame({'std': y_std})
    y_mean_df = pd.DataFrame({'mean': y_mean})

    # Merge dataframes to find the best attributes
    merged_1_df = y_mean_df.merge(y_std_df, left_index=True, right_index=True, how='inner')
    merged_df = merged_1_df.merge(num_est_df, left_index=True, right_index=True, how='inner')
    best_avg = merged_df['mean'].max()
    best_n = merged_df[merged_df['mean'] == best_avg]
    best_n = best_n['n_val']

    # Plot the training score and the scross validation scores
    plt.plot(num_est, x_mean, label='Training Score', color=color_args['training'][0])
    plt.plot(num_est, y_mean, label='Cross-Validation Score', color=color_args['validation'][0])

    # Fill the standard deviation above and below the mean values
    plt.fill_between(num_est, x_mean -  x_std, x_mean + x_std, color=color_args['training'][1])
    plt.fill_between(num_est, y_mean -  y_std, y_mean + y_std, color=color_args['validation'][1])

    # Make title, x & y labels, and legend for the plt and annotate the best metric on average then show plt
    plt.title(f'Validation Curve for {param_name}')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.xlabel(f"n_val for {param_name}")
    plt.ylabel('Accuracy')
    plt.annotate(f'Best mean at N = {int(best_n)} with {best_avg:0.2f}',
            xy=(best_n, best_avg), xycoords='data',
            xytext=(0, 20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"))
    plt.show()

    # Return the dataframe that has the results for each nth value.
    return merged_df
