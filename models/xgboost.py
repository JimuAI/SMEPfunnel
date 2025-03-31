import xgboost as xgb



def xgboost_classifier():
    # 初始化XGBoost分类器
    model = xgb.XGBClassifier(
        objective='binary:logistic',  # 二分类问题
        n_estimators=100,            # 树的数量
        max_depth=3,                 # 每棵树的最大深度
        learning_rate=0.1,           # 学习率
        colsample_bytree=0.8,        # 每棵树使用的特征比例
        random_state=42,
        eval_metric='logloss'        # 评估指标
    )
    return model

def xgboost_regressor():
    # 初始化XGBoost回归器
    model = xgb.XGBRegressor(
        objective='reg:squarederror',   # 回归问题
        n_estimators=100,               # 树的数量
        max_depth=3,                    # 每棵树的最大深度
        learning_rate=0.1,              # 学习率
        colsample_bytree=0.8,           # 每棵树使用的特征比例
        random_state=42,
        eval_metric='rmse'              # 评估指标
    )
    return model