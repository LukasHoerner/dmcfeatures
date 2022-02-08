import pandas as pd
import numpy as np
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def combine_dfs(orders_train, orders_test, orders_test_y):
    orders_test["returnShipment"] = orders_test_y["returnShipment"]
    combined = pd.concat([orders_train, orders_test])
    combined["val_set"] = 1
    combined.iloc[:len(orders_train), -1] = 0
    combined.drop(columns = ["orderItemID"], inplace = True)
    combined.replace("?", np.nan, inplace = True)
    combined.drop(combined[combined["color"].isna()].index, inplace = True)
    combined.reset_index(drop=True, inplace=True)
    combined.loc[:, "color"] = combined["color"].str.replace("brwon", "brown").str.replace("blau","blue")
    combined.loc[:, "size"] = combined["size"].str.lower()
    return combined

def generate_date_features(orders):
    # correct typo
    orders.loc[orders[orders["dateOfBirth"]=="1655-04-19"].index, "dateOfBirth"]= "1955-04-19"
    # convert to daytime
    time_cols = ["orderDate", "deliveryDate", "creationDate", "dateOfBirth"]
    for time_col in time_cols:
        orders.loc[:, time_col] = pd.to_datetime(orders[time_col])
        
    # correct senseless dates
        # age: over ~100years or underaged (border year 2000)
    weird_age = orders[(all(
        pd.to_datetime(orders["orderDate"]).dt.year < 1915) or(
            pd.to_datetime(orders["dateOfBirth"]).dt.year >= 2000))].index
    orders.loc[weird_age, "dateOfBirth"]= np.nan
        # some bug with delivery_Date in 1990
    orders.loc[orders[orders["deliveryDate"].dt.year == 1990].index, "deliveryDate"]= np.nan
        # set same day deliveries to na
    same_day_inc = orders[orders["orderDate"]==orders["deliveryDate"]].index
    orders.loc[same_day_inc ,"deliveryDate"]= np.nan
        # set day received to median time needed
    delivery_median = ((orders["deliveryDate"]-orders["orderDate"]).dt.days).median()
    delivery_nas = orders[orders["deliveryDate"].isna()].index
    orders.loc[delivery_nas, "deliveryDate"] = orders.loc[
        delivery_nas, "orderDate"] + pd.DateOffset(days = round(delivery_median))
    # adjust for sundays (no deliveries)
    sundays_inc = orders[orders["deliveryDate"].dt.weekday==6].index
    orders.loc[sundays_inc, "deliveryDate"] = orders.loc[
        sundays_inc, "orderDate"] + pd.DateOffset(days = 1)

    # create dict for features
    feature_cols_time = {}

    # create day features
    feature_cols_time["weekdayOrdered"] = orders["orderDate"].dt.dayofweek
    feature_cols_time["weekdayReceived"] = orders["deliveryDate"].dt.dayofweek

    feature_cols_time["durationDelivery"] = (orders["deliveryDate"]-orders["orderDate"]).dt.days
    we_del_ind = feature_cols_time["weekdayOrdered"][
        (feature_cols_time["weekdayOrdered"]>feature_cols_time["weekdayReceived"])&
        (feature_cols_time["weekdayOrdered"] != 6)].index
    feature_cols_time["weDelivery"] = pd.Series(data=0, index = orders.index)
    feature_cols_time["weDelivery"].loc[we_del_ind] = 1 # package not late for those: ppl used to it
    user_age = round((orders["orderDate"]-orders["dateOfBirth"]).dt.days/365)
    user_age_imp = pd.Series(0, index = user_age.index)
    user_age_imp.loc[user_age[user_age.isna()].index] = 1
    user_age = user_age.fillna(user_age.mean()).astype(np.int32)
    feature_cols_time["UserAgeOrder"] = user_age
    feature_cols_time["userAgeImp"] = user_age_imp

    # binary == 1 if account was made on oldest date -> system import
    old_sys_inc = orders["creationDate"][orders["creationDate"]==min(orders["creationDate"])].index
    feature_cols_time["accOldSys"] = pd.Series(data=0, index = orders.index)
    feature_cols_time["accOldSys"].loc[old_sys_inc] = 1
    # ~140 freshly made for order
    feature_cols_time["accAgeAtOrder"] = (orders["orderDate"]-orders["creationDate"]).dt.days
    return feature_cols_time


def generate_other_features(orders):
    """based on train dataset
    """
    orders["basketID"] = orders["customerID"].astype(str) + orders["orderDate"].astype(str)

    n_arts_lookup = orders["basketID"].value_counts()
    basket_price_lookup = orders.groupby("basketID")["price"].sum()

    basket_price = pd.Series(data = basket_price_lookup.loc[orders["basketID"]].values, index = orders.index)
    basketNArts =  pd.Series(data = n_arts_lookup.loc[orders["basketID"]].values, index = orders.index)

    train = orders[orders["val_set"]==0]
    test = orders[orders["val_set"]==1]
    max_prices = train.groupby("itemID")["price"].max()
    max_prices = pd.concat([max_prices,
    test[~test["itemID"].isin(train["itemID"])].groupby("itemID")["price"].max()])
    discount = 1 - orders.apply(lambda x: x["price"]/max_prices.loc[x["itemID"]], axis = 1)
    discount.loc[discount[discount.isna()].index] = 1
    return {"discount": discount,
            "item_color_cat": orders["itemID"].astype(str)+"_"+orders["color"],
            "item_size_cat": orders["itemID"].astype(str)+"_"+orders["size"],
            "basket_price": basket_price,
            "basketNArts": basketNArts
            }


def generate_ret_p(data, parameters):
    data["basketID"] = data["customerID"].astype(str) + data["orderDate"].astype(str)
    ret_col = parameters['ret_col'] # y
    ordnr_col = parameters['ordnr_col'] # indicator
    artnr_col = parameters['artnr_col'] # X (data)
    
    hyperparams = parameters['ret_p_hyperparams']
    min_n_arts = hyperparams["min_joint_baskets"]
    top_fs = hyperparams["top_freq_arts"]
    weights = hyperparams["weights"]
    drop_double_rets = hyperparams["drop_double_returns"]
    
    pd.options.mode.chained_assignment = None
    # feature generation: cut predictions before 
    df = data[data["val_set"]==0]
    test_df = data[data["val_set"]==1]
    
    # fit
    # mean_ret_op = data[parameters['return_type']].mean()
    multi_arts = df[df[ordnr_col].duplicated(keep=False)]
    multi_arts = multi_arts[[artnr_col, ordnr_col, ret_col]]

    ret_arts = multi_arts[multi_arts[ret_col]==1][artnr_col].unique()
    print("Bestellungen in Multi-Art-Warenkörben: ", multi_arts.shape[0], " von ", df.shape[0])
    print("Number of in Multi-Arts-Baskets that got returned ", multi_arts[artnr_col].isin(ret_arts).sum())
    
    ret_arts_frame = multi_arts[multi_arts[artnr_col].isin(ret_arts)]
    lookup_ret_perc_art = ret_arts_frame.groupby(artnr_col)[ret_col].mean().to_dict()

    lookup_tot_ord = multi_arts.groupby(artnr_col).size()
    n_transactions = multi_arts.shape[0]
    lookup_rel_ord = (lookup_tot_ord/n_transactions).to_dict()
    
    # generate empty nested dicts for every combination
    lookup_lift_arts = {key: {} for key in min_n_arts}
    lookup_basket_occ = {key: {} for key in min_n_arts}
    lookup_basket_ret_p = {key: {} for key in min_n_arts}
    lookup_droped_ret_arts = {key: [] for key in min_n_arts}
    for article in ret_arts:
        art_df = multi_arts[multi_arts[artnr_col]==article][[ordnr_col, ret_col]]
        ret_orders = art_df[art_df[ret_col]>=1][ordnr_col]

        basket_df = multi_arts[(multi_arts[ordnr_col].isin(art_df[ordnr_col])) &    # all articles which were ordered with article
                      (multi_arts[artnr_col] != article)]
        if drop_double_rets:   # ignores probability if both articles got returned
            basket_df = basket_df.drop(basket_df[basket_df[ret_col]==1].index)
        else:
            basket_df.loc[:, ret_col] = 0   # set all ret_cols to 0
        ret_incides = basket_df[basket_df[ordnr_col].isin(ret_orders)].index
        basket_df.loc[ret_incides, ret_col] = 1 # set ret_col = 1 if returned
        basket_grouped = basket_df.groupby(artnr_col).size()
        
        # for every min_joint_baseket combination
        for n_arts in min_n_arts:
            arts_basket_occ = basket_grouped[basket_grouped>=n_arts]   # only add arts with sufficient occurances
            if arts_basket_occ.shape[0] > 0:
                lookup_basket_occ[n_arts][article] = arts_basket_occ   # .to_dict() # minimum of co-occurances
                lookup_basket_ret_p[n_arts][article] = basket_df[basket_df[artnr_col].isin(arts_basket_occ.keys())].groupby(
                    artnr_col)[ret_col].mean()  #.to_dict()
                lookup_lift_arts[n_arts][article] = arts_basket_occ.index.to_series().apply(lambda x: (arts_basket_occ.loc[x]/n_transactions)/
                                      (lookup_rel_ord[article]*lookup_rel_ord[x]))
            else:
                lookup_droped_ret_arts[n_arts] += [article]
    
    used_cols = [artnr_col, ordnr_col, ret_col]
    arts_used_ret_p = {}
    arts_ret_p = {}
    for n_arts in min_n_arts:
        for top_f in top_fs:
            for weight in weights:
                colname = "min_"+str(n_arts)+"_top_"+str(top_f)+ "_"+str(weight) + "_"
                data[colname+"_lift"] = np.nan
                data[colname+"_conf"] = np.nan
                # create df for test and train data
                subframe_train = df[df[artnr_col].isin(lookup_basket_ret_p[n_arts].keys()) & # article got returned (+ has at least min_joint_basket occurances)
                        (df[ordnr_col].duplicated(keep=False))][used_cols]    # includes ret_col for train_data to regularize own influence
                subframe_test = test_df[test_df[artnr_col].isin(lookup_basket_ret_p[n_arts].keys()) & # article got returned (+ has at least min_joint_basket occurances)
                        (test_df[ordnr_col].duplicated(keep=False))][[artnr_col, ordnr_col]]
                
                # apply function for lift and confidence to training data
                ret_pcts_conf_train = subframe_train.apply(lambda x: get_ret_perc_train(
                    x[1], x[2], df[used_cols], lookup_basket_ret_p[n_arts][(x[0])], lookup_basket_occ[n_arts][(x[0])],
                    lookup_basket_occ[n_arts][(x[0])], top_f, weight), axis=1, result_type='expand')
                ret_pcts_lift_train = subframe_train.apply(lambda x: get_ret_perc_train(
                    x[1], x[2], df[used_cols], lookup_basket_ret_p[n_arts][(x[0])], lookup_basket_occ[n_arts][(x[0])],
                    lookup_lift_arts[n_arts][(x[0])], top_f, weight), axis=1, result_type='expand')

                # apply function for lift and confidence to test data
                ret_pcts_conf_test = subframe_test.apply(lambda x: get_ret_perc_test(
                    x[1], test_df[used_cols], lookup_basket_ret_p[n_arts][(x[0])], lookup_basket_occ[n_arts][(x[0])],
                    lookup_basket_occ[n_arts][(x[0])], top_f, weight), axis=1, result_type='expand')
                ret_pcts_lift_test = subframe_test.apply(lambda x: get_ret_perc_test(
                    x[1], test_df[used_cols], lookup_basket_ret_p[n_arts][(x[0])], lookup_basket_occ[n_arts][(x[0])],
                    lookup_lift_arts[n_arts][(x[0])], top_f, weight), axis=1, result_type='expand')
                print(ret_pcts_lift_train.iloc[:, 0].head(), ret_pcts_conf_train.iloc[:, 0].head())
                
                arts_used_ret_p[colname+"lift"] = pd.concat([ret_pcts_lift_train.iloc[:, 1],
                                                             ret_pcts_lift_test.iloc[:, 1]])
                arts_used_ret_p[colname+"conf"] = pd.concat([ret_pcts_conf_train.iloc[:, 1],
                                                              ret_pcts_conf_test.iloc[:, 1]])
                
                arts_ret_p[colname+"lift"] = pd.concat([ret_pcts_lift_train.iloc[:, 0],
                                                             ret_pcts_lift_test.iloc[:, 0]])
                arts_ret_p[colname+"conf"] = pd.concat([ret_pcts_conf_train.iloc[:, 0],
                                                              ret_pcts_conf_test.iloc[:, 0]])
        print(colname, "subframe_train:", subframe_train.shape[0], " subframe_test:", subframe_test.shape[0])
                
    return [arts_ret_p, lookup_basket_occ, lookup_basket_ret_p, lookup_ret_perc_art, 
            lookup_tot_ord, lookup_lift_arts, lookup_droped_ret_arts, arts_used_ret_p]

def get_ret_perc_train(bestnr, ret_ind, data, ret_p_art, occ_arts, sort_arts, top_f, weights):
    # sort arts: use df with lift if lift, else no_occurances
    # function init: pick all articles from basket 
    basket_arts = data[(data.iloc[:, 1] == bestnr) &    # iloc[:,1] = Column ordernr
             (data.iloc[:, 0].isin(ret_p_art.index))].iloc[:, 0]     # iloc[:,0] = Column article
    if len(basket_arts)==0:
        return np.nan, 0
    else: 
        # sorting specified by series in input (from lookup_lift_arts or lookup_basket_occ)
        sorted_arts = sort_arts.loc[basket_arts].sort_values().index.to_series()

        if top_f:
            if type(top_f) == str: # for values like var_5 -> variable 5 or if len(arts)>5 -> take half of the most frq
                n_min = [int(s) for s in top_f if s.isdigit()][0]
                top_f = max(n_min, round(len(sorted_arts)/2))
            sorted_arts = sorted_arts.iloc[-top_f:]

        # adapt for training data (fix ret_p) to prevent overfitting
        ret_p_basket = ret_p_art.loc[sorted_arts]
        occ_basket = occ_arts.loc[sorted_arts]
        no_ret_ind = 1 - ret_ind
        ret_p_basket = (ret_p_basket*occ_basket-ret_ind   # numerator = no returns of article
                                )/(occ_basket-no_ret_ind)  # denominator = no orders of article

        if weights: # if no weigths: only use mean of pcts
            weight = range(1, sorted_arts.shape[0]+1)
            if weights == "sqrt_w":
                weight = np.sqrt(weight)
            ret_pcts = (ret_p_basket*weight).sum()/sum(weight)
        else:
            ret_pcts = ret_p_basket.mean()

        return ret_pcts, occ_basket.sum()

def get_ret_perc_test(bestnr, data, ret_p_art, occ_arts, sort_arts, top_f, weights):
    # function init: pick all articles from basket 
    basket_arts = data[(data.iloc[:, 1] == bestnr) &    # iloc[:,1] = Column ordernr
             (data.iloc[:, 0].isin(ret_p_art.index))].iloc[:, 0]     # iloc[:,0] = Column article
    if len(basket_arts)==0:
        return np.nan, 0
    else:
        # sorting specified by series in input (from lookup_lift_arts or lookup_basket_occ)
        sorted_arts = sort_arts.loc[basket_arts].sort_values().index.to_series()

        if top_f:
            if type(top_f) == str: # for values like var_5 -> variable 5 or if len(arts)>5 -> take half of the most frq
                n_min = [int(s) for s in top_f if s.isdigit()][0]
                top_f = max(n_min, round(len(sorted_arts)/2))
            sorted_arts = sorted_arts.iloc[-top_f:]

        if weights: # if no weigths: only use mean of pcts
            weight = range(1, sorted_arts.shape[0]+1)
            if weights == "sqrt_w": # middle parth for sorting importance-wise
                weight = np.sqrt(weight)
            ret_pcts = (ret_p_art.loc[sorted_arts]*weight).sum()/sum(weight)
        else:
            ret_pcts = ret_p_art.loc[sorted_arts].mean()

        return ret_pcts, occ_arts.loc[sorted_arts].sum()


def subframes_budgets(train_arts, test_arts, budget:int):
    # helper function to seperate dataset into budget amount of ~equally sized chunks. Seperated by articles
    budgets = list(range(1, budget+1))
    
    arts = train_arts.value_counts().index
    iterations = int(len(arts)/budget)
    remainder = len(arts) - iterations*budget
    # print(budgets, arts, iterations, remainder)
    train_enc = pd.Series(data=budgets*iterations +  budgets[:remainder], index = arts)
    
    rest = test_arts[~test_arts.isin(train_arts)].value_counts().index
    iterations = int(len(rest)/budget)
    remainder = len(rest) - iterations*budget
    rest_enc = pd.Series(data=budgets*iterations +  budgets[:remainder], index = rest)
    enc = pd.concat([train_enc, rest_enc])
    all_arts = pd.concat([train_arts, test_arts])
    budgets = all_arts.apply(lambda x: enc.loc[x])
    return budgets


def subframes_budgets(train_arts, test_arts, budget):
    budgets = list(range(1, budget+1))
    
    arts = train_arts.value_counts().index
    iterations = int(len(arts)/budget)
    remainder = len(arts) - iterations*budget
    # print(budgets, arts, iterations, remainder)
    train_enc = pd.Series(data=budgets*iterations +  budgets[:remainder], index = arts)
    
    rest = test_arts[~test_arts.isin(train_arts)].value_counts().index
    iterations = int(len(rest)/budget)
    remainder = len(rest) - iterations*budget
    rest_enc = pd.Series(data=budgets*iterations +  budgets[:remainder], index = rest)
    enc = pd.concat([train_enc, rest_enc])
    all_arts = pd.concat([train_arts, test_arts])
    budgets = all_arts.apply(lambda x: enc.loc[x])
    return budgets


def generate_feature_frame(orders, time_features, other_features, arts_ret_p, parameters):
    coldict = parameters["coldict"]
    cat_pass = coldict["cat_pass"]
    cat_catboost = coldict["cat_catboost"]
    cont = coldict["cont"]
    output_col = coldict["output_col"]

    features = pd.DataFrame({**other_features, **time_features})
    orders_features = pd.merge(orders, features, right_index = True, left_index = True)

    train = orders_features[orders_features["val_set"]==0]
    test = orders_features[orders_features["val_set"]==1]
    budgets = subframes_budgets(train["itemID"], test["itemID"], parameters["budgets"])

    # delete unused features and add budget (data splits by articles)
    orders_features = orders_features[cat_pass+cat_catboost+cont+[output_col]+["val_set"]]
    orders_features.loc[:, cat_pass+cat_catboost] = orders_features[cat_pass+cat_catboost].astype("category")
    orders_features["budgets"] = budgets

    # create dataframe for joint_ret_p, including imputation for single + multi basket + indicator columns
    ret_p_cols = pd.Series(arts_ret_p.keys())
    non_imp_inc = {key: arts_ret_p[key].dropna().index for key in ret_p_cols}
    multi_baskets = orders_features[orders_features["basketNArts"]>1]
    single_baskets = orders_features[orders_features["basketNArts"]==1]
    multi_mean = multi_baskets[multi_baskets["val_set"]==0][output_col].mean()
    single_mean = single_baskets[single_baskets["val_set"]==0][output_col].mean()
    # columns to indicate impution
    imp_frame = pd.DataFrame(index = orders.index, columns = ret_p_cols+"_Imp").fillna(1)
    ret_p_frame = pd.merge(pd.DataFrame(arts_ret_p, index = orders.index), imp_frame, right_index = True, left_index = True)
    print(ret_p_frame.shape[0])
    
    for col in ret_p_cols:  # set non-imputed columns to zero
        ret_p_frame.loc[non_imp_inc[col], col+"_Imp"] = 0
        ret_p_frame.loc[multi_baskets.index, col] = ret_p_frame.loc[multi_baskets.index, col].fillna(multi_mean)
        ret_p_frame.loc[single_baskets.index, col] = ret_p_frame.loc[single_baskets.index, col].fillna(single_mean)


    catboost_enc = CatBoostEncoder()
    catboost_enc.fit(train[cat_catboost], train[output_col])
    catcoded_cols = catboost_enc.transform(orders_features[cat_catboost]).add_suffix("Catboost")

    orders_features = pd.merge(orders_features, catcoded_cols, right_index = True, left_index = True)
    orders_features = pd.merge(orders_features, ret_p_frame, right_index = True, left_index = True)
    return orders_features