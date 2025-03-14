import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("magic04.data", header=None)

num_columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1',
            'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
cat_columns = ['class']
data.columns = num_columns + cat_columns

data = data.drop_duplicates()

data_num =  data.select_dtypes(include = [np.number])
data_cat = data.select_dtypes(include = object)

data_cat = (data_cat == 'g')

std_scaler = StandardScaler()
data_num_scaled = pd.DataFrame(std_scaler.fit_transform(data_num), columns=num_columns)

train_data, test_data, train_target, test_target = train_test_split(
    data_num_scaled, data_cat, test_size=0.15, random_state=42, stratify=data_cat)

train_data, valid_data, train_target, valid_target = train_test_split(
    train_data, train_target, test_size=0.15, random_state=42, stratify=train_target)

under_sample = RandomUnderSampler(random_state=42)
train_data_balanced, target_data_balanced = under_sample.fit_resample(train_data, train_target)

target_data_balanced = target_data_balanced.values.ravel()

param_grid = {'n_neighbors': range(1, 25)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring="f1_weighted", cv=5)
grid_search.fit(train_data_balanced, target_data_balanced)
best_k = grid_search.best_params_['n_neighbors']

knn = KNeighborsClassifier(n_neighbors = best_k)
knn.fit(train_data_balanced, target_data_balanced)
train_pred = knn.predict(train_data_balanced)

cm = confusion_matrix(target_data_balanced, train_pred)
accuracy = accuracy_score(target_data_balanced, train_pred)
precision = precision_score(target_data_balanced, train_pred)
recall = recall_score(target_data_balanced, train_pred)
f1 = f1_score(target_data_balanced, train_pred)
clf_report = classification_report(target_data_balanced, train_pred)

# =============================================================================
# grid_search.fit(valid_data, valid_target.values.ravel())
# best_k = grid_search.best_params_['n_neighbors']
# =============================================================================

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(train_data_balanced, target_data_balanced)
test_pred = final_knn.predict(test_data)