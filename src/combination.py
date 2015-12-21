import pandas as pd
import numpy as np

def load_pred_data(path):
    df = pd.read_csv(path)
    y_prob = df.values
    return y_prob


path = 'data/prediction/'    
filename_xgb = 'xgboost'
filename_nn2 = 'pred_nn2_dataset'
filename_com = 'com'
filename_rf = 'calibratedRandomForest'

for idx in range(0, 10):
	y_xgb = load_pred_data(path + filename_xgb + str(idx) + '.csv')
	y_nn2 = load_pred_data(path + filename_nn2 + str(idx) + '.csv')
	y_rf = load_pred_data(path + filename_rf + str(idx) + '.csv')
	y_prob = np.multiply(y_xgb,0.40) + np.multiply(y_nn2,0.35) + np.multiply(y_rf, 0.25)
	my_df = pd.DataFrame(y_prob)
	my_df.to_csv(path + filename_com + str(idx) + '.csv', index = False, header = True)
