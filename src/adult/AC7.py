#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../../../')
from utils.packages import *
from utils.ml_fairness import *
from utils.standard_data import *
import time
start_time = time.time()
dir = 'res/adult7-'
d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
diff_file = dir + 'fairness' + '.csv'
if(not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)


# In[2]:


train_path = '../../../data/adult/adult.data'
test_path = '../../../data/adult/adult.test'

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

##### Drop na values
dropped = df.dropna()
count = df.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df = dropped


# In[3]:


df_nocs = df.copy()

df['education'].replace('Preschool', 'dropout',inplace=True)
df['education'].replace('10th', 'dropout',inplace=True)
df['education'].replace('11th', 'dropout',inplace=True)
df['education'].replace('12th', 'dropout',inplace=True)
df['education'].replace('1st-4th', 'dropout',inplace=True)
df['education'].replace('5th-6th', 'dropout',inplace=True)
df['education'].replace('7th-8th', 'dropout',inplace=True)
df['education'].replace('9th', 'dropout',inplace=True)
df['education'].replace('HS-Grad', 'HighGrad',inplace=True)
df['education'].replace('HS-grad', 'HighGrad',inplace=True)
df['education'].replace('Some-college', 'CommunityCollege',inplace=True)
df['education'].replace('Assoc-acdm', 'CommunityCollege',inplace=True)
df['education'].replace('Assoc-voc', 'CommunityCollege',inplace=True)
df['education'].replace('Bachelors', 'Bachelors',inplace=True)
df['education'].replace('Masters', 'Masters',inplace=True)
df['education'].replace('Prof-school', 'Masters',inplace=True)
df['education'].replace('Doctorate', 'Doctorate',inplace=True)

df['marital-status'].replace('Never-married', 'NotMarried',inplace=True)
df['marital-status'].replace(['Married-AF-spouse'], 'Married',inplace=True)
df['marital-status'].replace(['Married-civ-spouse'], 'Married',inplace=True)
df['marital-status'].replace(['Married-spouse-absent'], 'NotMarried',inplace=True)
df['marital-status'].replace(['Separated'], 'Separated',inplace=True)
df['marital-status'].replace(['Divorced'], 'Separated',inplace=True)
df['marital-status'].replace(['Widowed'], 'Widowed',inplace=True)

df['age'] = pd.cut(df['age'], 20)
df['hours-per-week'] = pd.cut(df['hours-per-week'], 10)


# In[4]:


# Create a one-hot encoding of the categorical variables.
cat_feat = ['age', 'hours-per-week', 'sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
# df = pd.get_dummies(df, columns=cat_feat, prefix_sep='=')

## Implement label encoder instead of one-hot encoder
for feature in cat_feat:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
for feature in cat_feat:
    le = LabelEncoder()
    df_nocs[feature] = le.fit_transform(df_nocs[feature])
    


# In[5]:




pro_att_name = ['race'] # ['race', 'sex']
priv_class = ['White'] # ['White', 'Male']
reamining_cat_feat = []

# data, X, y = load_adult_data(df, pro_att_name, priv_class, reamining_cat_feat)
# data_nocs, X_nocs, y_nocs = load_adult_data(df_nocs, pro_att_name, priv_class, reamining_cat_feat)

# data, att = data.convert_to_dataframe()
# data = data.astype(int)
# data.to_csv("ac7_" + str(len(data.columns)), index=False)
# #
# data_nocs, att = data_nocs.convert_to_dataframe()
# data_nocs = data_nocs.astype(int)
# data_nocs.to_csv("ac7_nocs_" + str(len(data_nocs.columns)), index=False)


# In[6]:

seed = randrange(100)
data_train_1, data_test_1 = train_test_split(df, test_size = 0.3, random_state = seed) # stratify=df['race']
data_train_2, data_test_2 = train_test_split(df_nocs, test_size = 0.3, random_state = seed) #

data_orig_train_1, X_train_1, y_train_1 = load_adult_data(data_train_1, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_1, X_test_1, y_test_1 = load_adult_data(data_test_1, pro_att_name, priv_class, reamining_cat_feat)

data_orig_train_2, X_train_2, y_train_2 = load_adult_data(data_train_2, pro_att_name, priv_class, reamining_cat_feat)
data_orig_test_2, X_test_2, y_test_2 = load_adult_data(data_test_2, pro_att_name, priv_class, reamining_cat_feat)


sc = StandardScaler()

trained = sc.fit(X_train_1)
X_train_1 = trained.transform(X_train_1)
X_test_1 = trained.transform(X_test_1)

X_train_2 = trained.transform(X_train_2)
X_test_2 = trained.transform(X_test_2)

data_orig_train_1.features = X_train_1
data_orig_test_1.features = X_test_1

data_orig_train_2.features = X_train_2
data_orig_test_2.features = X_test_2

model_1 = RandomForestClassifier()
model_1 = model_1.fit(X_train_1, y_train_1)

model_2 = RandomForestClassifier()
model_2 = model_2.fit(X_train_2, y_train_2)
print("intial time",  start_time - time.time())
y_pred_1 = model_1.predict(X_test_1)
y_pred_2 = model_2.predict(X_test_2)

pro_att = 'race'
from predefine import disparate_impact, statistical_parity_difference, equal_opportunity_difference, average_odds_difference
import sklearn

print("Accuracy: ", sklearn.metrics.accuracy_score(y_test_1, y_pred_1))
print("DI: ", disparate_impact(data_orig_test_1, y_pred_1, pro_att))
print("SPD: ", statistical_parity_difference(data_orig_test_1, y_pred_1, pro_att))
print("EOD: ", equal_opportunity_difference(data_orig_test_1, y_pred_1, y_test_1, pro_att))
print("AOD: ", average_odds_difference(data_orig_test_1, y_pred_1, y_test_1, pro_att))

print("Accuracy: ", sklearn.metrics.accuracy_score(y_test_2, y_pred_2))
print("DI: ", disparate_impact(data_orig_test_2, y_pred_2, pro_att))
print("SPD: ", statistical_parity_difference(data_orig_test_2, y_pred_2, pro_att))
print("EOD: ", equal_opportunity_difference(data_orig_test_2, y_pred_2, y_test_2, pro_att))
print("AOD: ", average_odds_difference(data_orig_test_2, y_pred_2, y_test_2, pro_att))


privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]
from aif360.algorithms.postprocessing import EqOddsPostprocessing, CalibratedEqOddsPostprocessing, \
    RejectOptionClassification
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
data_train = [data_orig_train_1, data_orig_train_2]
data_test = [data_orig_test_1, data_orig_test_2]
trained_model = [model_1, model_2]
model = RandomForestClassifier()
import time
start_time = time.time()


# for data_orig_train, data_orig_test, tr_model in zip(data_train, data_test, trained_model):

    # ### Reweighing
    # start_time = time.time()
    # RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    # data_transf_train = RW.fit_transform(data_orig_train)
    # # Train and save the model_1
    # rf_transf = model.fit(data_transf_train.features,
    #                      data_transf_train.labels.ravel())
    #
    # data_transf_test = RW.transform(data_orig_test)
    # fair = get_fair_metrics_and_plot(data_transf_test, rf_transf)
    # print("RW", fair, start_time - time.time())
    # ### Disperate Impact
    # start_time = time.time()
    # DIR = DisparateImpactRemover()
    # data_transf_train = DIR.fit_transform(data_orig_train)
    # # Train and save the model_1
    # rf_transf = model.fit(data_transf_train.features,
    #                      data_transf_train.labels.ravel())
    #
    # fair = get_fair_metrics_and_plot(data_orig_test, rf_transf)
    # print("DI", fair, start_time - time.time())
    # # #  Prejudice Remover Regularizer
    # # debiased_model = PrejudiceRemover()
    # # # Train and save the model_1
    # # debiased_model.fit(data_orig_train_1)
    # # fair = get_fair_metrics_and_plot(data_orig_test_1, debiased_model, model_aif=True)
    # # y_pred_1 = debiased_model.predict(data_orig_test_1)
    #
    # data_orig_test_pred = data_orig_test.copy(deepcopy=True)
    # # Prediction with the original RandomForest model_1
    # scores = np.zeros_like(data_orig_test.labels)
    # scores = tr_model.predict_proba(data_orig_test.features)[:,1].reshape(-1,1)
    # data_orig_test_pred.scores = scores
    #
    # preds = np.zeros_like(data_orig_test.labels)
    # preds = tr_model.predict(data_orig_test.features).reshape(-1,1)
    # data_orig_test_pred.labels = preds
    #
    # def format_probs(probs1):
    #     probs1 = np.array(probs1)
    #     probs0 = np.array(1-probs1)
    #     return np.concatenate((probs0, probs1), axis=1)
    #
    # # Equality of Odds
    # start_time = time.time()
    # EOPP = EqOddsPostprocessing(privileged_groups = privileged_groups,
    #                              unprivileged_groups = unprivileged_groups,
    #                              seed=42)
    # EOPP = EOPP.fit(data_orig_test, data_orig_test_pred)
    # data_transf_test_pred = EOPP.predict(data_orig_test_pred)
    #
    # fair = fair_metrics(data_orig_test, data_transf_test_pred, pred_is_dataset=True)
    # print("EO", fair, start_time - time.time())
    # ### Calibrated Equality of Odds
    # start_time = time.time()
    # cost_constraint = "fnr"
    # CPP = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
    #                                      unprivileged_groups = unprivileged_groups,
    #                                      cost_constraint=cost_constraint,
    #                                      seed=42)
    #
    # CPP = CPP.fit(data_orig_test, data_orig_test_pred)
    # data_transf_test_pred = CPP.predict(data_orig_test_pred)
    #
    # fair = fair_metrics(data_orig_test, data_transf_test_pred, pred_is_dataset=True)
    # print("CEO", fair, start_time - time.time())
    # ### Reject Option Classifier
    # start_time = time.time()
    # ROC = RejectOptionClassification(privileged_groups = privileged_groups,
    #                              unprivileged_groups = unprivileged_groups)
    #
    # ROC = ROC.fit(data_orig_test, data_orig_test_pred)
    # data_transf_test_pred = ROC.predict(data_orig_test_pred)
    #
    # fair = fair_metrics(data_orig_test, data_transf_test_pred, pred_is_dataset=True)
    # print("ROC", fair, start_time - time.time())
    # print('SUCCESS: completed 1 model_1.')
start_time = time.time()
import aif360_utils
mim_aif = aif360_utils.FaXAIF(data_orig_train_1, y_train_1, pro_att, model=model_1, model_type='MIM')
y_pred_1 = mim_aif.predict(data_orig_test_1)
print("Accuracy: ", sklearn.metrics.accuracy_score(y_test_1, y_pred_1))
print("DI: ", disparate_impact(data_orig_test_1, y_pred_1, pro_att))
print("SPD: ", statistical_parity_difference(data_orig_test_1, y_pred_1, pro_att))
print("EOD: ", equal_opportunity_difference(data_orig_test_1, y_pred_1, y_test_1, pro_att))
print("AOD: ", average_odds_difference(data_orig_test_1, y_pred_1, y_test_1, pro_att))
print(start_time - time.time())

start_time = time.time()
import aif360_utils
mim_aif = aif360_utils.FaXAIF(data_orig_train_2, y_train_2, pro_att, model=model_2, model_type='MIM')
y_pred_2 = mim_aif.predict(data_orig_test_2)
print("Accuracy: ", sklearn.metrics.accuracy_score(y_test_2, y_pred_2))
print("DI: ", disparate_impact(data_orig_test_2, y_pred_2, pro_att))
print("SPD: ", statistical_parity_difference(data_orig_test_2,y_pred_2, pro_att))
print("EOD: ", equal_opportunity_difference(data_orig_test_2, y_pred_2, y_test_2, pro_att))
print("AOD: ", average_odds_difference(data_orig_test_2, y_pred_2, y_test_2, pro_att))
print(start_time - time.time())