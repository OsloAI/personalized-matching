直接跑xgb的代码就能出csv提交。

比赛网址：https://www.datafountain.cn/competitions/311/details

比赛数据：https://www.datafountain.cn/competitions/311/details/data-evaluation

比赛类型：多分类问题

A榜排名

方案1：LightGBM model 0.81245 (2018-09-13 23:13)

方案2：XGBoost model 0.8254 排名 30th/1153 (2018-09-14 14:23:04)

XGBoost的关键参数

max_depth=12, learning_rate=0.05,
n_estimators=752, silent=True,
objective="multi:softmax",
nthread=4, gamma=0,
max_delta_step=0, subsample=1, colsample_bytree=0.9, colsample_bylevel=0.9,
reg_alpha=1, reg_lambda=1, scale_pos_weight=1,
base_score=0.5, seed=2018, missing=None,num_class=15

