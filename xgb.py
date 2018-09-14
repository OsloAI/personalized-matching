import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
print (train.shape)



train = train[train.gender != '\\N']
# test = test[test.gender != '\\N']
train['gender'] = train['gender'].apply(lambda x : int(x))
test['gender'] = test['gender'].apply(lambda x : int(x))

train = train[train.age != '\\N']
# test = test[test.age != '\\N']
train['age'] = train['age'].apply(lambda x : int(x))
test['age'] = test['age'].apply(lambda x : int(x))

train = train[train['2_total_fee'] != '\\N']
# test = test[test['2_total_fee'] != '\\N']
test.loc[test['2_total_fee'] == '\\N','2_total_fee'] = 0.0
train['2_total_fee'] = train['2_total_fee'].apply(lambda x : float(x))
test['2_total_fee'] = test['2_total_fee'].apply(lambda x : float(x))

train = train[train['3_total_fee'] != '\\N']
# test = test[test['3_total_fee'] != '\\N']
test.loc[test['3_total_fee'] == '\\N','3_total_fee'] = 0.0
train['3_total_fee'] = train['3_total_fee'].apply(lambda x : float(x))
test['3_total_fee'] = test['3_total_fee'].apply(lambda x : float(x))


label = train.pop('current_service')
le = LabelEncoder()
label = le.fit_transform(label)


feature = [value for value in train.columns.values if
                   value not in ['user_id']]



#xgb模型
def XGB():
    clf = xgb.XGBClassifier(max_depth=12, learning_rate=0.05,
                            n_estimators=2000, silent=True,
                            objective="multi:softmax",
                            nthread=4, gamma=0,
                            max_delta_step=0, subsample=1, colsample_bytree=0.9, colsample_bylevel=0.9,
                            reg_alpha=1, reg_lambda=1, scale_pos_weight=1,
                            base_score=0.5, seed=2018, missing=None,num_class=15)
    return clf




online = False
# online = True
if online:
        print ('online')

        model = XGB()
        model.fit(train[feature], label, eval_set=[(train[feature], label)], verbose=1,)
        pred = model.predict(test[feature])
        pred = le.inverse_transform(pred)
        test['predict'] = pred

        test[['user_id', 'predict']].to_csv('./sub.csv', index=False)
else:
        print ('offline')
        train_x,test_x,train_y,test_y = train_test_split(train[feature],label,test_size=0.1,shuffle=True,random_state=2018)
        model = XGB()
        model.fit(train_x[feature], train_y, eval_set=[(test_x[feature], test_y)], verbose=1,early_stopping_rounds=100)
        pred = model.predict(test_x)
        print(f1_score(test_y,pred,average='weighted'))

        # feature_list = model.feature_importances_
        # pd.DataFrame(
        #         {
        #                 'feature':feature,
        #                 'score':feature_list,
        #         }
        # ).to_csv('./feature_importance.csv',index=False)

        # from sklearn.externals import joblib
        # joblib.dump(model, 'gbm.pkl')
