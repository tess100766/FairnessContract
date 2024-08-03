#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
from xgboost import XGBClassifier
dir = 'res/titanic2-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


# Load data
train = pd.read_csv('../../../data/titanic/train.csv')
test = pd.read_csv('../../../data/titanic/test.csv')
df = train


# In[3]:


## BASIC PREP
df['Sex'] = df['Sex'].replace({'female': 0.0, 'male': 1.0})

y1_df = df.copy()
## Feature Engineering
df['Family'] =  df["Parch"] + df["SibSp"]
df['Family'].loc[df['Family'] > 0] = 1
df['Family'].loc[df['Family'] == 0] = 0
df = df.drop(['SibSp','Parch'], axis=1)
df = df.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)

y1_df = y1_df.drop(['PassengerId','Name'], axis=1)


# In[4]:


# Missing value
average_age_titanic   = y1_df["Age"].mean()
std_age_titanic       = y1_df["Age"].std()
count_nan_age_titanic = y1_df["Age"].isnull().sum()
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
y1_df["Age"][np.isnan(y1_df["Age"])] = rand_1

y1_df["Embarked"] = y1_df["Embarked"].fillna("S")
y1_df["Fare"].fillna(y1_df["Fare"].median(), inplace=True)

y1_df['Fare'] = y1_df['Fare'].astype(int)
y1_df[ 'Cabin' ] = y1_df.Cabin.fillna( 'U' )
y1_df[ 'Ticket' ] = y1_df.Ticket.fillna( 'X' )
y1_df = y1_df.dropna()
# One-hot encoder
cat_feat = ['Embarked', 'Ticket', 'Cabin']
y1_df = pd.get_dummies(y1_df, columns=cat_feat, prefix_sep='=')


average_age_titanic   = df["Age"].mean()
std_age_titanic       = df["Age"].std()
count_nan_age_titanic = df["Age"].isnull().sum()
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
df["Age"][np.isnan(df["Age"])] = rand_1

df["Embarked"] = df["Embarked"].fillna("S")
df["Fare"].fillna(df["Fare"].median(), inplace=True)

df['Fare'] = df['Fare'].astype(int)
df = df.dropna()
# One-hot encoder
cat_feat = ['Embarked']
df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')


# In[5]:



seed = randrange(100)
y2_train, y2_test = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['loan']
y1_train, y1_test = train_test_split(y1_df, test_size = 0.3, random_state = seed) # 

pro_att_name = ['Sex']
priv_class = [1]
reamining_cat_feat = []

y2_data_orig_train, y2_X_train, y2_y_train = load_titanic_data(y2_train, pro_att_name, priv_class, reamining_cat_feat)
y2_data_orig_test, y2_X_test, y2_y_test = load_titanic_data(y2_test, pro_att_name, priv_class, reamining_cat_feat)

y1_data_orig_train, y1_X_train, y1_y_train = load_titanic_data(y1_train, pro_att_name, priv_class, reamining_cat_feat)
y1_data_orig_test, y1_X_test, y1_y_test = load_titanic_data(y1_test, pro_att_name, priv_class, reamining_cat_feat)

y2_xgb = RandomForestClassifier(n_estimators=100)
y2_mdl = y2_xgb.fit(y2_X_train, y2_y_train)

y1_xgb = RandomForestClassifier(n_estimators=100)
y1_mdl = y1_xgb.fit(y1_X_train, y1_y_train)



# plot_model_performance(model_1, X_test_1, y_test_1)
y1_pred, y1_fair = get_fair_metrics_and_plot('filename', y1_data_orig_test, y1_mdl)
y2_pred, y2_fair = get_fair_metrics_and_plot('filename', y2_data_orig_test, y2_mdl)



y1_fair = y1_fair.drop(['DI', 'CNT', 'TI'], axis=1)
y2_fair = y2_fair.drop(['DI', 'CNT', 'TI'], axis=1)
CVR, CVD, AVR_EOD, AVD_EOD, AVR_SPD, AVD_SPD, AVD_AOD, AV_ERD = compute_new_metrics(y2_data_orig_test, y1_pred, y2_pred)
row_y1 = y1_fair.iloc[[0]].values[0].tolist()
row_y2 = y2_fair.iloc[[0]].values[0].tolist()
diff = []



# diff.append(CVR)
# diff.append(CVD)
diff.append(AVD_SPD)
diff.append(AVD_EOD)
diff.append(AVD_AOD)
diff.append(AV_ERD)

for i in range(len(row_y2)):
    if(i < 2):
        change = row_y2[i] - row_y1[i]
    else:
        break;
    diff.append(change)

stage = 'Custom(feature)'
model_name = 'titanic2'
# diff = diff_df.iloc[0].values.tolist()
diff.insert(0, stage)
diff.insert(0, model_name)

cols = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
# metrics = pd.DataFrame(data=obj_fairness, index=['y1'], columns=cols)
diff_df = pd.DataFrame(data=[diff], columns  = cols, index = ['Diff']).round(3)

with open(diff_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(diff)

