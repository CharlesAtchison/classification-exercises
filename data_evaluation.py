import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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

def explore_validation_curve(X : pd.DataFrame, y : pd.DataFrame, param_grid : dict, model, cv=None, color_args={'train': ['black', 'orange'], 'test': ['red', 'cyan']}):
    '''Function that will print out plot of the single or multiple input hyperparameter(s) for the validation
    curves the plotted mean for each nth value and the standard deviation for each nth value. This requires
    some model generated and will return a sklearn.model_select.GridSearchCV class.

    
    Parameters
    ----------
    X : pandas DataFrame
        Some x_values dataframe to be put into the validation_curve.
    
    y : pandas DataFrame
        Some y_values dataframe to be put into the validation_curve.

    param_grid : str
        What hyperparmeter you would like to explore within the validation_curve and an associated numpy range.
        With each additional hyperparameter, you'll have combinatoric possibilities of 
        n!/r!(n − r)!, where r is the number of hyperparameters and n is the number
        of n values for each hyperparameter.

        format :
                        {
                            __some_hyperparameter__: __some_numpy_range__,
                            __some_hyperparameter__: __some_numpy_range__
                        }
        
        Examples
        --------
        Single Hyperparameter
        param_grid = {'n_estimators' : np.arange(1, 200, 2)}
        
        Multi Hyperparameter
        param_grid = {'n_estimators' : np.arange(1, 200, 2),
                      'max_depth' : np.arange(1, 13, 1)}

    
    model : Sklearn model
        Can check sklearn models, verified currently compatible with:
            DecisionTreeClassifier,
            RandomForestClassifier,
            KNeighborsClassifier
            
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.


    color_args : dict
        Not required, default values:
        {'train': ['black', 'orange'],
         'test': ['red', 'cyan']}
        
        can personalize but must be in the format of
        # train_line    line_color      standard_dev fill color
        {'train}    :   ['black'     ,  'orange']

        # test_line    line_color      standard_dev fill color
        {'test'}    :   ['red'     ,  'cyan']

    Returns
    -------
    sklearn GridSearchCV
        The returned class is a the GridSearchCV with associated selectable attributes.
        
    
    Examples
    -------
    >>> param_grid = {'n_estimators' : np.arange(1, 200, 2),
                      'max_depth' : np.arange(1, 13, 1)}
                      
    >>> val = explore_validation_curve(X_train, y_train, param_grid, RandomForestClassifier())

    
    >>> print(type(val))
    
    <class 'sklearn.model_selection._search.GridSearchCV'>
    
    --------------------------------------------------------------------------------------------------

    >>> param_grid = {'n_estimators' : np.arange(1, 200, 2),
                      'max_depth' : np.arange(1, 13, 1)}
    
    >>> val = explore_validation_curve(X_train, y_train, param_grid, DecisionTreeClassifier(), cv=5,
                                color_args={'train': ['green', 'purple'],
                                            'test': ['orange', 'red']})
    >>> print(type(val))
    
    <class 'sklearn.model_selection._search.GridSearchCV'>

    --------------------------------------------------------------------------------------------------

    >>> param_grid = {'n_neighbors' : np.arange(1, 30, 2),
                      'max_depth' : np.arange(1, 13, 1)}
    
    >>> val = explore_validation_curve(X_train, y_train, param_grid, KNeighborsClassifier(), cv=5,
                                color_args={'train': ['green', 'purple'],
                                            'test': ['orange', 'red']})
    >>> print(type(val))
    
    <class 'sklearn.model_selection._search.GridSearchCV'>

    '''
    
    # Check that if the param_name is 'max_depth' that the range is not greater than the number of attributes in model.
    if 'max_depth' in param_grid.keys() and len(param_grid['max_depth']) > X.shape[1]:
        raise Exception(f"Sorry, your range cannot be larger than the number of attributes ({X.shape[1]}) when using 'max_depth")
        
    # Calculate validation curve and return as array
    grid = GridSearchCV(model, param_grid, cv=cv, return_train_score=True)
    grid.fit(X, y)

    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    best_vals = dict()
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        best_vals[p_k] = p_v
        masks.append(list(results['param_'+p_k].data==p_v))

    params=grid.param_grid

    ## Ploting results
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        plt.title(f'Validation Curve for {p}')
        # Check if there is only 1 hyperparameter to test
        if len(masks_names) > 1:
            # Stack the masks to find the best index
            m = np.stack(masks[:i] + masks[i+1:])
            pram_preformace_in_best
            best_parms_mask = m.all(axis=0)
            # Map the best index 
            best_index = np.where(best_parms_mask)[0]
        else:
            best_index = np.arange(len(means_test))
        x = np.array(params[p])
        # Find the test_mean and train mean for each hyperparameter
        test_mean = np.array(means_test[best_index])
        test_std = np.array(stds_test[best_index])
        train_mean = np.array(means_train[best_index])
        train_std = np.array(stds_train[best_index])
        best_mean = means_test[best_index][best_vals[p]-1]
        # Build the plot for each hyperparameter
        plt.plot(x, train_mean, label='Training score', color=color_args['train'][0])
        plt.plot(x, test_mean, label='Test score', color=color_args['test'][0])
        plt.fill_between(x, test_mean - test_std, test_mean + test_std, linestyle='--', label='test', color=color_args['test'][1])
        plt.fill_between(x, train_mean - train_std, train_mean + train_std, linestyle='-', label='train' , color=color_args['train'][1])
        plt.xlabel(p.upper())
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.annotate(f'Best {p} at N = {best_vals[p]}\nat {best_mean:0.2f}',
            xy=(best_vals[p], best_mean), xycoords='data',
            xytext=(0, 20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"))
        plt.show()
        
    print(grid.best_params_)

    # Return a GridSearchCV class with the associated attributes to examine.
    return grid
