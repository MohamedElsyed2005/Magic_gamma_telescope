import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

data = pd.read_csv("magic04.data", header = None)
num_columns = ['fLength', 'fWidth', 'fSize', 'fConc','fConc1',
                'fAsym','fM3Long','fM3Trans','fAlpha', 'fDist']

cat_columns = ['class']

data.columns = num_columns + cat_columns

data = data.drop_duplicates()

data_num =  data.select_dtypes(include = [np.number])
data_cat = data.select_dtypes(include = object)

std_scaler = StandardScaler()
data_num_scaled = pd.DataFrame(std_scaler.fit_transform(data_num),
                               columns = num_columns)

under_sample = RandomUnderSampler(random_state = 42)
data_num_scaled_balanced, data_cat_balanced = under_sample.fit_resample(data_num_scaled, data_cat)

data_cat_balanced = (data_cat_balanced == 'g')
train_data , test_data , train_target, test_target = train_test_split(data_num_scaled_balanced,
                                                                      data_cat_balanced, test_size = 0.15, random_state = 42)
train_data, valid_data, train_target, valid_target = train_test_split(train_data, train_target,
                                                                      test_size = 0.15 , random_state = 42) 
