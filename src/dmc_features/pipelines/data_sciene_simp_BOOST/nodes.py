from catboost import CatBoostClassifier, Pool, metrics

import logging
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, precision_score # cohen_kappa_score


def log_results(data, enc_cols, parameters):
    data = data["train_test"]
    log = logging.getLogger(__name__)
    cols = []
    for key in enc_cols.keys(): # all different configs
        cols += enc_cols[key]
    cols = [parameters["artnr_col"]] + cols
    cols_dict = parameters["coldict"]
    cont_cols = cols_dict["cont"]
    cat_cols = cols_dict['cat_pass'] + cols_dict['cat_catboost'] #  + ["GEWICHT_IMP"]
    cat_cols.remove(parameters["artnr_col"])

    # use train(0) and val_set (1) for evaluation
    train = data[data["val_set"]==0]
    test = data[data["val_set"]==1]
    results = {}
    outpu_col = parameters["target_col"]

    # add_cols = ["ARTIKELNR", "ART_CATBOOST", 20, '20_base', '20_5_True_lift', '20_5_True_lift_base']
    for add_col in cols:
        if add_col == parameters["artnr_col"]:
            categorical = cat_cols + [add_col]
            continuous = cont_cols
        else:
            continuous = cont_cols + [add_col]
            categorical = cat_cols
        X_train = train[categorical+continuous]
        y_train = train[outpu_col]
        # multi-Warenkorb
        X_test_p = test[test["basketNArts"]>1][categorical+continuous]
        y_test_p = test[test["basketNArts"]>1][outpu_col]
        X_test = test[categorical+continuous]
        y_test = test[outpu_col]
        categorical_features_indices = np.array(range(0,len(categorical)))
        # if add_col == parameters["artnr_col"]:
        #     weights = pd.Series(1, index=range(train.shape[0]))
        #     weights.loc[train[train[outpu_col]==1].index] = train.shape[0]/ train[train[outpu_col]==1].shape[0]
        #     weights = weights.to_list()
        #     train_pool = Pool(train[categorical+continuous], train[outpu_col], cat_features=categorical_features_indices, weight=weights)
        # else:
        train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
        validate_pool = Pool(X_test, y_test, cat_features=categorical_features_indices)
        
        params = {
            'iterations': 1000,
            #'learning_rate': 0.05,
            'eval_metric': metrics.MCC(),
            'random_seed': 42,
            'logging_level': 'Silent',
            'use_best_model': True,
            #'depth': 10,
            'allow_writing_files': False,
            'one_hot_max_size': 16, # including state for OHE; other features handled by catboost
            'od_type': 'Iter',  # https://catboost.ai/en/docs/features/overfitting-detector-desc#od_wait
            'od_wait': 50,
        }
        results[add_col] = {}
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=validate_pool)
        y_pred = model.predict(X_test)
        y_pred_p = model.predict(X_test_p)
        results[add_col] = {
            'MCC_test':matthews_corrcoef(y_test, y_pred),
            'F1_test':f1_score(y_test, y_pred),
            'Recall_test': recall_score(y_test, y_pred),
            'Prec_test': precision_score(y_test, y_pred),
            'MCC_test_multiarts': matthews_corrcoef(y_test_p, y_pred_p),
            'Recall_test_multiarts': recall_score(y_test_p, y_pred_p),
            'Prec_test_multiarts': precision_score(y_test_p, y_pred_p),
            "feat_imp": pd.Series(data = model.get_feature_importance(), index = X_test.columns),
        }
        log.info("MCC: {}. Column used to encode articles: {}.".format(results[add_col]['MCC_test'], add_col))
    return results