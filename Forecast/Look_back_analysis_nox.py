import sys
sys.path.append('/media/m3rg2000/mounted/Junaid/Heidelberg/Emissions/new_emission/Notebooks')
import importlib
import functions  # First import
importlib.reload(functions)  # Reloads the module to reflect changes
from functions import *  # Now import functions.py

df_merged = pd.read_pickle('/media/m3rg2000/mounted/Junaid/Heidelberg/Emissions/new_emission/Notebooks/Saved_files/data/df_merged.pkl')

emission_name = 'NOx content in the raw gas (preheater outlet)'
raw = df_merged[emission_name].values
noise_filtered = moving_average_n_window(raw, 5)
df_merged[emission_name] = noise_filtered

minutely = df_merged[['timestamp', 'no_missing', emission_name]].dropna(axis = 0, how = 'any', inplace = False)   
minutely.reset_index(inplace=True)

df_mae = []
df_mape = []
for look_back_trial in tqdm(range(5,125,5)):
    look_back_steps = look_back_trial
    look_back_period = look_back_steps + 1 #20 minutes

    missing_index = []     # indicies of df_merge at which outlier occur.
    for i in range(len(minutely)):
        val = minutely[emission_name][i]
        if (val <= 9  or val >= 1000):         ##### ***** change here if you want to exclude zeros !
            missing_index.append(i)      # append the required index of the hourly df.
    # print(f'{len(missing_index)} outside range out of {len(minutely)} minutes datapoits')

    count = 0    # cont of windows available for model development
    window = []   # filtered window indicies for minutely
    len_of_window = []
    non_zero_windows = []   # filtered non-zero window indicies for minuterly
    for i in range(len(missing_index)):
        if count ==0:
            window.append((0,missing_index[i]))
            count += 1
        else:
            window.append((missing_index[i-1]+1,missing_index[i]))
            count += 1
            if i == len(missing_index) - 1:
                window.append((missing_index[i]+1,len(minutely)))
    window_start = []
    window_end = []
    for i in window:
        a,b = i
        length = b-a
        if length >= look_back_period+1:
            non_zero_windows.append(i)
            len_of_window.append(length)
            window_start.append(a)
            window_end.append(b)
    # len_win_hrs = [i/4 for i in len_of_window ] 
    working_windows = pd.DataFrame({'start':window_start, 'index range': non_zero_windows,'end':window_end,  'win length':len_of_window})
    working_windows.reset_index(inplace=True)
    working_windows.rename(columns={'index':'window number'}, inplace=True)
    
    train_index = 881280
    buffer_index_1 = 881280
    buffer_index_2 = 969120
    test_index = 969120

    train_val = working_windows[working_windows['end']<=train_index] #upto Aug 2021
    buffer = working_windows[(working_windows['start']>=buffer_index_1 )& (working_windows['end']<=buffer_index_2)] #Sept-Oct 2021
    test = working_windows[working_windows['start']>=test_index]#Nov-Dec 2021
    
    m1 = train_val['window number'].min()
    M1 =train_val['window number'].max()
    m2 = buffer['window number'].min()
    M2 =buffer['window number'].max()
    m3 = test['window number'].min()
    M3 =test['window number'].max()
    
    df_names = []
    for i in range(look_back_period-1):
        df_names.append(i+1)
    df_names.append('y')
    df_names = ['window number'] + df_names
    df_backbone = pd.DataFrame(columns = df_names)

    all_rows = []
    all_index_no_missing = []
    window_number = 0
    for start, finish in non_zero_windows:
        piece_meal = minutely[start:finish][emission_name].to_list()
        piece_meal_no_missing = minutely[start:finish]['no_missing'].to_list()
        for i in range(len(piece_meal) - look_back_steps):  #2
            values = piece_meal[i:i + look_back_period]
            all_index_no_missing.append(piece_meal_no_missing[i + look_back_period-1])
            all_rows.append([window_number] + values)
        window_number +=1

    # After collecting everything, create DataFrame once
    df_backbone = pd.DataFrame(all_rows)
    df_backbone.columns = df_names
    df_backbone['no_missing'] = all_index_no_missing
    df_backbone = df_backbone.merge(df_merged[['no_missing','timestamp' ]], on = 'no_missing', how = 'left')
    
    Ts_train = df_backbone[df_backbone['window number']<=M1]
    Ts_buffer = df_backbone[(df_backbone['window number']>=m2) & (df_backbone['window number']<=M2)]
    Ts_test = df_backbone[df_backbone['window number']>=m3]
    X_train = Ts_train.iloc[:, 1:-3]
    y_train = Ts_train.iloc[:,[-3]]

    X_test= Ts_test.iloc[:, 1:-3]
    y_test= Ts_test.iloc[:,[-3]]

    y = pd.concat([y_train,y_test])
    
    np.random.seed(1002)
    random.seed(0)
    # Separate scalers for X and y
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Scale inputs (X)
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)

    # Scale outputs (y)
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten() # Convert y_df to NP arrays before reshaping
    y_test_s = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
    model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),  # deeper + more capacity
    max_iter=100,                      # allow more training epochs
    batch_size=512,                    # increase batch size to speed up + stabilize
    learning_rate_init=0.001,         # keep as-is (try 0.0005 if overfitting)
    solver='adam',                    # good for large datasets
    random_state=21,
    verbose=True,
    early_stopping=True,             
    validation_fraction=0.1,
    n_iter_no_change=10               # optional: longer patience for early stoppin
    )

    model.fit(X_train_s,y_train_s)

    test_pred_s = model.predict(X_test_s).reshape(-1,1)
    # train_pred_s = model.predict(X_train_s).reshape(-1,1)
    test_pred = scaler_y.inverse_transform(test_pred_s)
    # train_pred = scaler_y.inverse_transform(train_pred_s)

    _ , mae, mape  = get_score('Test', actual=y_test, predicted=test_pred)
    df_mae.append(mae)
    df_mape.append(mape)

pd.DataFrame({'mae': df_mae, 'mape':df_mape}).to_csv('/home/m3rg2000/Junaid_temporary/saved_data/forecast/nox/look_back_analysis_nox_2hr.csv')