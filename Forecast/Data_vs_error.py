import sys
sys.path.append('/media/m3rg2000/mounted/Junaid/Heidelberg/Emissions/new_emission/Notebooks')
import importlib
import functions  # First import
importlib.reload(functions)  # Reloads the module to reflect changes
from functions import *  # Now import functions.py

df_merged = pd.read_pickle('/media/m3rg2000/mounted/Junaid/Heidelberg/Emissions/new_emission/Notebooks/Saved_files/data/df_merged.pkl')
df_merged_Na_removed = df_merged.dropna(axis = 0, how = 'any', inplace = False)   
df_merged_Na_removed.drop(columns = ['timestamp', 'index_present_original'], inplace = True)

def get_score_here(base='Training', actual=0,predicted=0):
    mape = mean_absolute_percentage_error(actual,predicted)
    return ((mape*100).round(2))
np.random.seed(1002)
random.seed(0)
mape_train = []
mape_test = []
two_week_points = 2*7*24*60
ALL = df_merged_Na_removed[(df_merged_Na_removed['NOx content in the raw gas (preheater outlet)'] > 90) & (df_merged_Na_removed['NOx content in the raw gas (preheater outlet)'] < 1000)]
for interation in tqdm(range(25, 40)):
    df_prune = ALL.drop(ALL.sample(n=two_week_points*interation, random_state=42).index)
    # filtered = filter_df(df_prune)
    # X= filtered.iloc[:,4:]  # -6 for no timeseries, -1 for timeseries
    # y = filtered.loc[:,['NOx content in the raw gas (preheater outlet)']]
    X= df_prune.iloc[:,4:]  # -6 for no timeseries, -1 for timeseries
    y = df_prune.loc[:,['NOx content in the raw gas (preheater outlet)']]
    
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.20,random_state=0)

    model_nox = XGBRegressor(
    subsample = 0.9,              
    reg_lambda = 1,        
    reg_alpha = 1,         
    n_estimators = 1600,   
    min_child_weight = 6,  
    max_depth = 11,          
    learning_rate = 0.025, 
    colsample_bytree = 0.8,  
    random_state = 42,  
    tree_method = 'gpu_hist',  
    n_jobs = -1  

    ).fit(X_train,y_train)
    tr = get_score_here('Training',actual=y_train,predicted=model_nox.predict(X_train))
    ts = get_score_here('Test',actual=y_test,predicted=model_nox.predict(X_test))
    mape_train.append(tr)
    mape_test.append(ts)

df = pd.DataFrame({'train': mape_train, 'test':mape_test})
print(df)
df.to_csv('/home/m3rg2000/Junaid_temporary/saved_data/general/data_vs_error.csv')
