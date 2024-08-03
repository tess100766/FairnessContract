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
dir = 'res/adult1-'

d_fields = ['Pipeline', 'Stage', 'SF_SPD', 'SF_EOD', 'SF_AOD', 'SD_ERD', 'Acc', 'F1']
diff_file = dir + 'fairness' + '.csv'
if (not os.path.isfile(diff_file)):
    with open(diff_file, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(d_fields)

train_path = '../../../data/adult/adult.data'
test_path = '../../../data/adult/adult.test'

column_names = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'income-per-year']
na_values = ['?']

train = pd.read_csv(train_path, header=None, names=column_names,
                    skipinitialspace=True, na_values=na_values)
test = pd.read_csv(test_path, header=0, names=column_names,
                   skipinitialspace=True, na_values=na_values)

df_good = pd.concat([test, train], ignore_index=True)

##### Drop na values
dropped = df_good.dropna()
count = df_good.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df_good = dropped
print("--- %s drop: ---" % (time.time() - start_time))
df_bad = df_good.copy()
# Create a one-hot encoding of the categorical variables.
cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
df_bad = pd.get_dummies(df_bad, columns=cat_feat, prefix_sep='=')

## Implement label encoder instead of one-hot encoder
for feature in cat_feat:
    le = LabelEncoder()
    df_good[feature] = le.fit_transform(df_good[feature])
seed = randrange(100)
good_train, good_test = train_test_split(df_good, test_size=0.3, random_state=seed)  #
bad_train, bad_test = train_test_split(df_bad, test_size=0.3, random_state=seed)  # stratify=df['race']
print("--- %s split ---" % (time.time() - start_time))

pro_att_name = ['race']  # ['race', 'sex']
priv_class = ['White']  # ['White', 'Male']
reamining_cat_feat = []

good_data_orig_train, good_X_train, good_y_train = load_adult_data(good_train, pro_att_name, priv_class, reamining_cat_feat)
good_data_orig_test, good_X_test, good_y_test = load_adult_data(good_test, pro_att_name, priv_class, reamining_cat_feat)

bad_data_orig_train, bad_X_train, bad_y_train = load_adult_data(bad_train, pro_att_name, priv_class, reamining_cat_feat)
bad_data_orig_test, bad_X_test, bad_y_test = load_adult_data(bad_test, pro_att_name, priv_class, reamining_cat_feat)

sc = StandardScaler()

trained = sc.fit(good_X_train)
good_X_train = trained.transform(good_X_train)
good_X_test = trained.transform(good_X_test)

good_data_orig_train.features = good_X_train
good_data_orig_test.features = good_X_test
# trained_1 = sc.fit(bad_X_train)
# bad_X_train = trained_1.transform(bad_X_train)
# bad_X_test = trained_1.transform(bad_X_test)
#
bad_data_orig_train.features = bad_X_train
bad_data_orig_test.features = bad_X_test

# In[5]:
good_model = LogisticRegression()
good_model = good_model.fit(good_X_train, good_y_train)

bad_model = LogisticRegression()
bad_model = bad_model.fit(bad_X_train, bad_y_train)
print("intial time",  start_time - time.time())


good_y_pred = good_model.predict(good_X_test)
bad_y_pred = bad_model.predict(bad_X_test)

from predefine import disparate_impact, statistical_parity_difference, equal_opportunity_difference, average_odds_difference
import sklearn
pro_att = 'race'

print("Accuracy: ", sklearn.metrics.accuracy_score(good_y_test, good_y_pred))
print("DI: ", disparate_impact(good_data_orig_test, good_y_pred, pro_att))
print("SPD: ", statistical_parity_difference(good_data_orig_test, good_y_pred, pro_att))
print("EOD: ", equal_opportunity_difference(good_data_orig_test, good_y_pred, good_y_test, pro_att))
print("AOD: ", average_odds_difference(good_data_orig_test, good_y_pred, good_y_test, pro_att))

print("Accuracy: ", sklearn.metrics.accuracy_score(bad_y_test, bad_y_pred))
print("DI: ", disparate_impact(bad_data_orig_test, bad_y_pred, pro_att))
print("SPD: ", statistical_parity_difference(bad_data_orig_test, bad_y_pred, pro_att))
print("EOD: ", equal_opportunity_difference(bad_data_orig_test, bad_y_pred, bad_y_test, pro_att))
print("AOD: ", average_odds_difference(bad_data_orig_test, bad_y_pred, bad_y_test, pro_att))


from aif360.algorithms.postprocessing import EqOddsPostprocessing, CalibratedEqOddsPostprocessing, \
    RejectOptionClassification
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]

data_train = [good_data_orig_train, bad_data_orig_train]
data_test = [good_data_orig_test, bad_data_orig_test]
trained_model = [good_model, bad_model]
model = LogisticRegression()

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
mim_aif = aif360_utils.FaXAIF(good_data_orig_train, good_y_train, pro_att, model=good_model, model_type='MIM')
good_y_pred = mim_aif.predict(good_data_orig_test)
print("Accuracy: ", sklearn.metrics.accuracy_score(good_y_test, good_y_pred))
print("DI: ", disparate_impact(good_data_orig_test, good_y_pred, pro_att))
print("SPD: ", statistical_parity_difference(good_data_orig_test, good_y_pred, pro_att))
print("EOD: ", equal_opportunity_difference(good_data_orig_test, good_y_pred, good_y_test, pro_att))
print("AOD: ", average_odds_difference(good_data_orig_test, good_y_pred, good_y_test, pro_att))
print(start_time - time.time())

start_time = time.time()
import aif360_utils
mim_aif = aif360_utils.FaXAIF(bad_data_orig_train, bad_y_train, pro_att, model=bad_model, model_type='MIM')
bad_y_pred = mim_aif.predict(bad_data_orig_test)
print("Accuracy: ", sklearn.metrics.accuracy_score(bad_y_test, bad_y_pred))
print("DI: ", disparate_impact(bad_data_orig_test, bad_y_pred, pro_att))
print("SPD: ", statistical_parity_difference(bad_data_orig_test, bad_y_pred, pro_att))
print("EOD: ", equal_opportunity_difference(bad_data_orig_test, bad_y_pred, bad_y_test, pro_att))
print("AOD: ", average_odds_difference(bad_data_orig_test, bad_y_pred, bad_y_test, pro_att))
print(start_time - time.time())