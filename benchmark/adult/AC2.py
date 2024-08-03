#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *


# In[2]:


train_path = '../../data/adult/adult.data'
test_path = '../../data/adult/adult.test'

column_names = ['age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'income-per-year']
na_values=['?']

train = pd.read_csv(train_path, header=None, names=column_names, 
                    skipinitialspace=True, na_values=na_values)
test = pd.read_csv(test_path, header=0, names=column_names,
                   skipinitialspace=True, na_values=na_values)

df = pd.concat([test, train], ignore_index=True)

seed = randrange(100)
y1_train, y1_test = train_test_split(df, test_size = 0.3, random_state = seed) # 


# In[3]:


##### Process na values
dropped = y1_train.dropna()
count = y1_train.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
y1_train = dropped

dropped = y1_test.dropna()
count = y1_test.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
y1_test = dropped

# Fill Missing Category Entries
y1_train["workclass"] = y1_train["workclass"].fillna("X")
y1_train["occupation"] = y1_train["occupation"].fillna("X")
y1_train["native-country"] = y1_train["native-country"].fillna("United-States")

dropped = y1_test.dropna()
count = y1_test.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
y1_test = dropped


# Create a one-hot encoding of the categorical variables.
cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
y1_df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

for feature in cat_feat:
    le = LabelEncoder()
    y1_train[feature] = le.fit_transform(y1_train[feature])
    y1_test[feature] = le.transform(y1_test[feature])


# In[4]:


pro_att_name = ['race'] # ['race', 'sex']
priv_class = ['White'] # ['White', 'Male']
reamining_cat_feat = []

y1_data_orig_train, y1_X_train, y1_y_train = load_adult_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_adult_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)


# In[5]:


y1_model =  RandomForestClassifier(n_estimators=250,max_features=5)
y1_mdl = y1_model.fit(y1_X_train, y1_y_train)


# In[6]:


plot_model_performance(y1_mdl, y1_X_test, y1_y_test)

