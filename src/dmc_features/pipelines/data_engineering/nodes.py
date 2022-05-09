import pandas as pd
import numpy as np
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.cat_boost import CatBoostEncoder


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
    """_summary_

    :param orders: _description_
    :type orders: _type_
    :return: _description_
    :rtype: _type_
    """
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
    feature_cols_time["monthOrdered"] = orders["orderDate"].dt.month
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
    
    feature_cols_time["accAgeAtOrder"] = (orders["orderDate"]-orders["creationDate"]).dt.days
    return feature_cols_time


def generate_other_features(orders):
    """generate features only based on train dataset

    :param orders: _description_
    :type orders: _type_
    :return: _description_
    :rtype: _type_
    """
    train = orders[orders["val_set"]==0]
    test = orders[orders["val_set"]==1]
    # orders= orders.loc[train.index.to_list() + test.index.to_list()]
    orders["basketID"] = orders["customerID"].astype(str) + orders["orderDate"].astype(str)

    n_arts_lookup = orders["basketID"].value_counts()
    basket_price_lookup = orders.groupby("basketID")["price"].sum()

    basket_price = pd.Series(data = basket_price_lookup.loc[orders["basketID"]].values, index = orders.index)
    basketNArts =  pd.Series(data = n_arts_lookup.loc[orders["basketID"]].values, index = orders.index)
    
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


def target_encode(X, stats, prior_mean, N_min, y = None):
    """ helper function for BetaLooE

    :param X: _description_
    :type X: _type_
    :param stats: _description_
    :type stats: _type_
    :param prior_mean: _description_
    :type prior_mean: _type_
    :param N_min: _description_
    :type N_min: _type_
    :param y: _description_, defaults to None
    :type y: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    values = pd.Series(index = X.index, dtype=float)
    for i, stats_partframe in enumerate(stats):
        if type(y) == pd.core.series.Series:
            trans_frame = X[y==i]
        else:
            trans_frame = X

        df_stats = pd.merge(trans_frame, stats_partframe, how='left').set_axis(trans_frame.index)
        n = pd.Series(data = df_stats['n'].copy(), index = trans_frame.index)
        N = pd.Series(data = df_stats['N'].copy(), index = trans_frame.index)

        nan_indexs = np.isnan(n)
        n[nan_indexs] = prior_mean
        N[nan_indexs] = 1

        N_prior = np.maximum(N_min-N, 0)
        alpha_prior = prior_mean*N_prior
        beta_prior = (1-prior_mean)*N_prior

        alpha = alpha_prior + n
        beta =  beta_prior + N-n

        num = alpha
        dem = alpha+beta
        value = num/dem
        value.fillna(prior_mean, inplace=True)
        values[trans_frame.index] = value
    return values


class BetaLooE():
    def __init__(self, N_min = 20):
        self.N_min = N_min
    def fit(self, data, targets):
        self.prior_mean = targets.mean()
        df = pd.merge(data, targets, left_index = True, right_index = True)
        stats = df.groupby(data.name)
        stats = stats.agg(['sum', 'count'])[targets.name]
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)

        stats_noret = stats.copy()

        # stats_noret.loc[stats_noret["N"]>1, "N"] = stats_noret[stats_noret["N"]>1]["N"] - 1
        stats_noret.loc[:, "N"] = stats_noret["N"]-1
            # minimaler unterschied
        stats_ret = stats_noret.copy()
        stats_ret.loc[stats_ret["n"]>0, "n"] = stats_noret["n"] - 1
            # does also work for multi-arts (no BestNr)

        self.stats = stats
        self.stats_noret = stats_noret
        self.stats_ret = stats_ret       
        return self
    
    def transform(self, data, y=None):
        prior_mean = self.prior_mean
        N_min = self.N_min
        
        if type(y) == pd.core.series.Series:
            stats_noret = self.stats_noret
            stats_ret = self.stats_ret
            values = target_encode(data, [stats_noret, stats_ret], prior_mean, N_min, y)
        else:
            stats = self.stats
            values = target_encode(data, [stats], prior_mean, N_min)
        return values


class BetaLooE2d():
    """class object betalooe encoder 
    """
    def __init__(self, N_min = 20, variants=False):
        self.variants = variants
        self.N_min = N_min
    def fit(self, data, targets, meta_group):
        N_min = self.N_min
        # get colnames (using pandas .name attribute -> needed atm)
        data_col = data.name
        target_col = targets.name
        meta_col = meta_group.name

        # generate data used for aricle-wise stage
        prior_mean = targets.mean() # mean of dataset
        self.prior_mean = prior_mean

        df = pd.merge(data, targets, left_index = True, right_index = True)

        # first stage: equal to (leave one out) beta encoder
        stats = df.groupby(data_col)
        stats = stats.agg(['sum', 'count'])[target_col]
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats = stats.reset_index(level=0).set_index(data_col)
        # generate arts_lookup for single-arts
        stats_f = stats.copy()
        stats_f.loc[:, "N"] = stats_f["N"] - 1
        # stats_f.loc[stats_f["N"]>1, "N"] = stats_f[stats_f["N"]>1]["N"] - 1
        stats_t = stats_f.copy()
        stats_t.loc[stats_t["n"]>0, "n"] = stats_f["n"] - 1

        # first stage: article-meta stage for article-basket data
        dff = pd.merge(meta_group, df, left_index = True, right_index = True)

        #
        meta_data = dff.groupby(meta_col)[data_col].apply(np.array)
        meta_data = meta_data.apply(np.unique) # inserted

        meta_lens = meta_data.apply(lambda x: len(x))
        meta_multi = meta_lens[meta_lens>1].index.to_series()

        meta_arts = np.sort(dff[dff[meta_col].isin(meta_multi)][data_col].unique())
        self.meta_arts = pd.Series(meta_arts, name=data_col)

        dff = dff[dff[meta_col].isin(meta_multi)] # change )&  (dff[data_col].isin(meta_arts))
        dff_rets = dff[dff[target_col]==1]
        
        data_meta = dff.groupby(data_col)[meta_col].apply(np.array)
        data_meta = data_meta.apply(lambda x: np.unique(x)) # unique: dont take multi-arts in same basket into account
        data_ret_meta = dff_rets.groupby(data_col)[meta_col].apply(np.array)
        data_ret_meta = data_ret_meta.apply(lambda x: np.unique(x))

        joint_datapoints_N = data_meta.apply(lambda x: np.concatenate(meta_data.loc[x].to_numpy())) 
        joint_datapoints_n = data_ret_meta.apply(lambda x: np.concatenate(meta_data.loc[x].to_numpy()))

        counted_N = joint_datapoints_N.apply(lambda x: np.unique(x, return_counts=True))
        counted_n = joint_datapoints_n.apply(lambda x: np.unique(x, return_counts=True))

        N = counted_N.apply(lambda x: pd.Series(data=x[1], index=x[0]))
        n = counted_n.apply(lambda x: pd.Series(data=x[1], index=x[0]))
        N = N.loc[meta_arts, meta_arts]
        n = n.reindex(meta_arts, axis=1).reindex(meta_arts)

        if self.variants == True:
            # problem -> get 
            no_dub = dff[dff[["basketID", "itemID"]].duplicated(keep=False)]
            data_meta_grouped = dff.groupby(["basketID", "itemID"])["returnShipment"].mean()
            data_meta_index = no_dub.drop_duplicates(keep=False).groupby(["basketID", "itemID"]).size().index
            repair_data = data_meta_grouped.loc[data_meta_index].reset_index()
            repair_data.loc[:, meta_col] = meta_data.loc[repair_data[meta_col]].to_numpy()
            for x in repair_data.itertuples():
                n.loc[x[2], x[1]] -= x[3]
            for art in meta_arts:
                n.loc[art, art] = prior_mean
                N.loc[art,art] = 1
            grouped = dff.groupby(data_col)[meta_col].value_counts()
            grouped_N = grouped[grouped>1]
            
            # variants -> stattdessen count?
            variants_N = grouped_N.groupby(data_col).sum()
            # drops = alle arts die 
            drops = grouped[grouped==1]
            grouped_ret = dff_rets.groupby(data_col)[meta_col].value_counts()
            drop_inc = drops.index.get_level_values(0).unique().to_series()
            ret_inc = grouped_ret.index.get_level_values(0).unique().to_series()
            # drop_inc: only arts that got returned (grouped_ret)
            drop_inc = drop_inc[drop_inc.isin(ret_inc)]
            for inc in drop_inc:
                # if grouped_ret.loc[inc].drop(drops.loc[inc].index, errors='ignore'):
                n.loc[inc,inc] = grouped_ret.loc[inc].drop(drops.loc[inc].index, errors='ignore').sum()        
            for item in variants_N.index:
                N.loc[item, item] = variants_N.loc[item]

            """multi_counts = data_meta.apply(lambda x: len(x))
            # auch bei returns
            multi_ret_counts = data_ret_meta.apply(lambda x: len(x))
            # abziehen von df
            for item in multi_counts.index:
                N.loc[item, item] = N.loc[item, item] - multi_counts.loc[item]
            for item in multi_ret_counts.index:
                n.loc[item, item] = n.loc[item, item] - multi_ret_counts.loc[item]"""
        #else:
        #    for item in meta_arts:
        #        N.loc[item, item] = np.nan
        #        n.loc[item, item] = np.nan

        N_train = np.nan_to_num((N - 1).to_numpy(dtype=float))# .astype(int)
        N = np.nan_to_num(N.to_numpy(dtype=float))# .astype(int)
        n_train_t = np.nan_to_num((n-1).to_numpy(dtype=float))# .astype(int)
        n = np.nan_to_num(n.to_numpy(dtype=float))# .astype(int)

        # generate lift and confidence for sorting
        sorting = {}
        n_transactions = data.shape[0]
        rel_freq_arts = (stats.loc[meta_arts, "N"]/n_transactions).to_numpy()
        lift_frame = (N/n_transactions)/ (np.atleast_2d(rel_freq_arts).T * rel_freq_arts)
        lift_frame[np.where(lift_frame==0)] = 1
        sorting["lift"] = np.argsort(lift_frame, axis=1)
        sorting["conf"] = np.argsort(N, axis=1)
        self.sorting = sorting

        frame_name = ["train_f", "train_t", "test"]

        arts_posterior = {}
        stats_frames = [stats_f, stats_t, stats]

        arts_ij_posterior = {}
        ns = [n,n_train_t, n]
        Ns = [N_train, N_train, N]
        # multi_statsframes = [multi_f_stats, multi_t_stats, multi_stats]
        for i, name in enumerate(frame_name):
            statsframe = stats_frames[i].copy()
            ni = statsframe["n"]
            Ni = statsframe["N"]
            # Stufe 1: data-column based -> create stuff based on index of arts; other stuff seperately
                # if Ni (Anzahl Artikel i) > Ni_prior: prior = 0; else N_min-AnzahlArts
            Ni_prior = np.maximum(N_min-Ni, 0) # 1D-array

            # bayes regularization for articles i: use mean_prior from dataset (no leave-one-out)
            alpha_i_prior = prior_mean*Ni_prior
            beta_i_prior = (1-prior_mean)*Ni_prior

            alpha_i = alpha_i_prior + ni
            beta_i = beta_i_prior + (Ni-ni)
            values_i = alpha_i/(alpha_i+beta_i)
            # option -> if no variants: generate addidional alpha etc. from multi-art orders

            arts_posterior[name] = values_i

                # reset index to arts (only articles that have been in orders)
            Ni_prior = Ni_prior.loc[meta_arts].to_numpy().reshape(len(meta_arts),-1)
            alpha_i_prior = alpha_i_prior.loc[meta_arts].to_numpy().reshape(len(meta_arts),-1)
            beta_i_prior = beta_i_prior.loc[meta_arts].to_numpy().reshape(len(meta_arts),-1)
            values_i = values_i.loc[meta_arts].to_numpy().reshape(len(meta_arts),-1)
            ni = ni.loc[meta_arts].to_numpy().reshape(len(meta_arts),-1)
            Ni = Ni.loc[meta_arts].to_numpy().reshape(len(meta_arts),-1)

            # second stage: data<>meta-group combinations
            nij = ns[i].copy()
            Nij = Ns[i].copy()

            # bayes regularization for article combinations ij
                # leaves out evidence from existing combination ij (e.g. Ni-N)
            nij_reg = ni-nij # number of positive observations *not* including combination nij
            Nij_reg = Ni-Nij # number of total observations *not* including combination nij

            prior_means = nij_reg/Nij_reg # prior_mean not taking combination ij into account (seperate effects) 
            inds_zero = np.where(np.isnan(prior_means))
            prior_means[inds_zero] = np.take(values_i, inds_zero[0]) # replace nas with article-wise priors
            # if N=1 the regularization can cause division by 0 -> replace with mean
            inds_inf = np.where(np.isinf(prior_means))
            prior_means[inds_inf] = np.take(values_i, inds_inf[0])

            Nij_prior = np.maximum(N_min-Nij, 0) 
            Nij_prior = np.maximum(Nij_prior - Ni_prior,0) # Nij_prior >= Ni_prior

            alpha_ij_prior = alpha_i_prior+prior_means*Nij_prior
            beta_ij_prior = beta_i_prior+(1-prior_means)*Nij_prior

            alpha_ij = alpha_ij_prior + nij
            beta_ij = beta_ij_prior + (Nij-nij)

            values_ij = alpha_ij/(alpha_ij+beta_ij)

            arts_ij_posterior[name] = values_ij
        self.arts_posterior = arts_posterior
        self.arts_ij_posterior = arts_ij_posterior
        return self # [arts_posterior, arts_ij_posterior, meta_arts, sorting, prior_mean]
        
    def transform(self, data, meta_group, targets=None, weighted=True, basket_max=5, sorting_type="lift"):
        """uses meta_arts as indexer, since arts_ij_posterior is an array

        :param data: _description_
        :type data: _type_
        :param meta_group: _description_
        :type meta_group: _type_
        :param targets: _description_, defaults to None
        :type targets: _type_, optional
        :param weighted: _description_, defaults to True
        :type weighted: bool, optional
        :param basket_max: _description_, defaults to 5
        :type basket_max: int, optional
        :param sorting_type: _description_, defaults to "lift"
        :type sorting_type: str, optional
        :return: _description_
        :rtype: _type_
        """

        arts_posterior = self.arts_posterior
        arts_ij_posterior = self.arts_ij_posterior
        prior_mean = self.prior_mean
        meta_arts = self.meta_arts
        sorting = self.sorting[sorting_type]
        
        data_col = data.name
        meta_col = meta_group.name
        
        # working with int for meta-group
        if meta_group.dtype != int:
            new_ind = pd.Series(index=meta_group.unique(), data = range(meta_group.nunique()))
            meta_group = pd.Series(new_ind.loc[meta_group].to_numpy(), index = meta_group.index, name=meta_col)
        df = pd.merge(data, meta_group, left_index = True, right_index = True)

        # drop new datapoints and set them to prior mean
            # will not use them for calculating basket_mean as well
        known_data = arts_posterior["train_f"].index.to_series()
        new_data = df[~df[data_col].isin(known_data)].index
        df.drop(new_data, inplace=True)
        new_data = pd.Series(prior_mean, index = new_data)

        # replace meta-category to new dummy for datapoints not in meta_arts
        new_meta_inc = df[~df[data_col].isin(meta_arts)].index
        new_meta_imp = max(df[meta_col])+1
        df.loc[new_meta_inc, meta_col] = range(new_meta_imp, new_meta_imp+len(new_meta_inc))

        # drop datapoints with no joint meta-categoty (including dummmy)
        arts_inc = df[meta_col].drop_duplicates(keep=False).index
        df_arts = df.loc[arts_inc]
        arts_data = pd.Series(index=arts_inc, dtype=float)
        df.drop(arts_inc, inplace=True)

        # set categories in data to integer (matching arts_ij_posterior index)
        meta_arts_inv = meta_arts.reset_index().set_index(data_col)
        df[data_col] = meta_arts_inv.loc[df[data_col]]["index"].to_list() # 
        setted = df.groupby(meta_col)[data_col].agg("unique")
        if self.variants == True:
            singles = df[[data_col, meta_col]].drop_duplicates(keep=False)
            singles = singles.apply(lambda x:
                                        setted.loc[x[meta_col]]
                                        [setted.loc[x[meta_col]]!=x[data_col]], axis=1)
            doubles = df[df[[data_col, meta_col]].duplicated(keep=False)].apply(lambda x:
                                        setted.loc[x[meta_col]], axis=1)
            data_basket = pd.concat([singles, doubles]).sort_index()
        else:
            data_basket = df.apply(lambda x:
                                setted.loc[x[meta_col]]
                                [setted.loc[x[meta_col]]!=x[data_col]], axis=1)
        df["basket"] = data_basket
        # basket_arts = basket[~basket.str.len().eq(0)].index
        basket_data = pd.Series(index=df.index, dtype=float)

        if type(targets) == pd.core.series.Series:
            target_col = targets.name
            df_arts = pd.merge(df_arts, targets, left_index = True, right_index = True)
            df = pd.merge(df, targets, left_index = True, right_index = True)

            # entries in meta-group
            # df_arts = dff.loc[arts_inc]
            # df_basket_arts = dff.loc[basket_arts]
            for i, trainset in enumerate(['train_f', 'train_t']):
                arts_get = df_arts[df_arts[target_col]==i]
                arts_lookup = arts_posterior[trainset]
                arts_data.loc[arts_get.index] = arts_lookup.loc[arts_get[data_col]].to_list()

                basket_get = df[df[target_col]==i]
                basket_lookup = arts_ij_posterior[trainset]
                ij_baskets = basket_get.apply(lambda x:
                                              basket_lookup[x[data_col],
                                                            x["basket"][np.argsort(sorting[x[data_col],x["basket"]])][-basket_max:]],
                                              axis=1)   # sorts basket according to sorting and picks the articles accordingly
                if weighted == True:
                    weights = ij_baskets.apply(lambda x: range(1, len(x)+1))
                    weigthed = (weights * ij_baskets)
                    avg = weigthed.apply(lambda x: sum(x)
                                        ) / weights.apply(lambda x: sum(x))
                else:
                    avg = ij_baskets.apply(lambda x: x.mean())
                basket_data.loc[basket_get.index] = avg.to_list()
        else:
            arts_lookup = arts_posterior['test']
            arts_data.loc[arts_inc] = arts_lookup.loc[df_arts[data_col]].to_list()

            basket_lookup = arts_ij_posterior['test']
            ij_baskets = df.apply(lambda x: 
                                  basket_lookup[x[data_col],
                                                x["basket"][np.argsort(sorting[x[data_col],x["basket"]])][-basket_max:]],
                                  axis=1)
            if weighted == True:
                weights = ij_baskets.apply(lambda x: range(1, len(x)+1))
                weigthed = (weights * ij_baskets)
                avg = weigthed.apply(lambda x: sum(x)
                                    ) / weights.apply(lambda x: sum(x))
                # print(avg.head())
            else:
                avg = ij_baskets.apply(lambda x: x.mean())
                # print(avg.head())
            basket_data.loc[avg.index] = avg.to_list()
        values = pd.concat([new_data, arts_data, basket_data])
        self.values_distibution = {
            "new": new_data.shape[0],
            "arts": arts_data.shape[0],
            "basket": basket_data.shape[0]}
        
        return values.sort_index()


def finalize_features(data, parameters: dict):
    """_summary_

    :param data: _description_
    :type data: _type_
    :param parameters: _description_
    :type parameters: dict
    :return: _description_
    :rtype: _type_
    """
    data[parameters['meta_col']] = data["customerID"].astype(str) + data["orderDate"].astype(str)
    target_col = parameters['target_col'] # y
    data_col = parameters['artnr_col'] # X (data)
    meta_col = parameters['meta_col'] # meta-group
        # convert articles to integer numbers for encoder
    # artenc = pd.Series(index = data[data_col].unique(),
    #             data=range(data[data_col].nunique()))
    # data["artenc"] = artenc.loc[data[data_col]].to_list()
    # data_col = "artenc"
    train = data[data["val_set"]==0][[data_col, meta_col, target_col]]
    test = data[data["val_set"]==1][[data_col, meta_col, target_col]]
    
    hyperparams = parameters['ret_p_hyperparams']
    min_n_arts = hyperparams["N_min"]
    sorting_type = hyperparams["sorting_type"]
    weights = hyperparams["weights"]
    basket_max = hyperparams["basket_max"]
    enc_mE_dict = {}
    enc_mE_smooth_dict = {}
    enc_1D_dict = {}
    enc_1D_dict_base = {}
    enc_2D_dict = {}
    enc_2D_dict_base = {}
    for n_min in min_n_arts:
        print(n_min)
        mEnc = MEstimateEncoder(m=n_min, cols=data_col)
        mEnc_rand = MEstimateEncoder(m=n_min, random_state=parameters["random_state"], randomized=True, cols=data_col)
        enc_train = mEnc.fit_transform(train[data_col], train[target_col]).iloc[:, 0]
        enc_test = mEnc.transform(test[data_col]).iloc[:, 0]
        enc_mE_dict[n_min] = pd.concat([enc_train, enc_test]).sort_index()
        enc_train = mEnc_rand.fit_transform(train[data_col], train[target_col]).iloc[:, 0]
        enc_test = mEnc_rand.transform(test[data_col]).iloc[:, 0]
        enc_mE_smooth_dict[n_min] = pd.concat([enc_train, enc_test]).sort_index()

        enc1d = BetaLooE(n_min)
        enc1d.fit(train[data_col], train[target_col])
        enc_train = enc1d.transform(train[data_col], train[target_col])
        enc_test = enc1d.transform(test[data_col])
        enc_1D_dict[n_min] = pd.concat([enc_train, enc_test])
        enc_1D_dict_base[n_min] = enc1d.transform(data[data_col])
        enc2d = BetaLooE2d(n_min, variants = True)
        enc2d.fit(train[data_col], train[target_col], train[meta_col])
        for maxbasket in basket_max:
            for weigthing in weights:
                for sorting in sorting_type:
                    entry_name = str(str(n_min) + "_" + str(maxbasket) + "_" + str(weigthing) + "_" + sorting)
                    enc_train = enc2d.transform(train[data_col], train[meta_col], train[target_col], weighted=weigthing, basket_max=maxbasket, sorting_type=sorting)
                    enc_test = enc2d.transform(test[data_col], test[meta_col], weighted=weigthing, basket_max=maxbasket, sorting_type=sorting)
                    enc_2D_dict[entry_name] = pd.concat([enc_train, enc_test])
                    enc_2D_dict_base[entry_name] = enc2d.transform(data[data_col], data[meta_col], weighted=weigthing, basket_max=maxbasket, sorting_type=sorting)
    enc_mE_df = pd.DataFrame(enc_mE_dict).add_suffix("_m")
    enc_mE_smooth_df = pd.DataFrame(enc_mE_smooth_dict).add_suffix("_m_smooth")
    enc_1D_df_base = pd.DataFrame(enc_1D_dict_base).add_suffix("_base")
    enc_1D_df = pd.DataFrame(enc_1D_dict)
    enc_2D_df = pd.DataFrame(enc_2D_dict)
    enc_2D_df_base = pd.DataFrame(enc_2D_dict_base).add_suffix("_base")
    enc_cols = {
    "m":list(enc_mE_df.columns),
    "m_smooth":list(enc_mE_smooth_df.columns),
    "1D": list(enc_1D_df.columns),
    "1D_base": list(enc_1D_df_base.columns),
    "2D": list(enc_2D_df.columns),
    "2D_base": list(enc_2D_df_base.columns)
    }
    return [enc_cols, pd.concat([enc_1D_df, enc_1D_df_base, enc_2D_df, enc_2D_df_base, enc_mE_df, enc_mE_smooth_df], axis=1)]


def generate_val_test_features(data, parameters):
    """_summary_

    :param data: _description_
    :type data: _type_
    :param parameters: _description_
    :type parameters: _type_
    :return: _description_
    :rtype: _type_
    """
    target_feats = {"train_val": {}, "train_test": {}}
    target_feats["train_test"]["other_feats"] = generate_other_features(data)
    [enc_cols, enc_frame] = finalize_features(data, parameters)
    target_feats["train_test"]["enc_frame"] = enc_frame
    # split to train_val with same size like original test-set
    cut_inc = data.shape[0] - 2* (data["val_set"]==1).sum()
    data.loc[cut_inc: , "val_set"] += 1
    data = data[data["val_set"]<2]
    target_feats["train_val"]["other_feats"] = generate_other_features(data)
    [enc_cols, enc_frame] = finalize_features(data, parameters)
    target_feats["train_val"]["enc_frame"] = enc_frame
    return [target_feats, enc_cols]


def subframes_budgets(train_arts, test_arts, budget:int):
    """function to create budgets (to use for HpBandSTer) - not used in experiments

    :param train_arts: _description_
    :type train_arts: _type_
    :param test_arts: _description_
    :type test_arts: _type_
    :param budget: _description_
    :type budget: int
    :return: _description_
    :rtype: _type_
    """
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


def generate_feature_frame(orders, time_features, target_features, parameters): #other_features, enc_frame,
    """generates dataframe for validation and train set

    :param orders: _description_
    :type orders: _type_
    :param time_features: _description_
    :type time_features: _type_
    :param target_features: _description_
    :type target_features: _type_
    :param parameters: _description_
    :type parameters: _type_
    :return: _description_
    :rtype: _type_
    """
    sets = ['train_test', 'train_val']
    orders_features_sets = {}
    time_features = pd.DataFrame(time_features)
    # fix length of different data
    for subset in sets:
        if subset == "train_val":
            cut_inc = orders.shape[0] - 2* (orders["val_set"]==1).sum()
            orders.loc[cut_inc: , "val_set"] += 1
            keep_inc = orders[orders["val_set"]<2].index
            orders = orders.loc[keep_inc]
            time_features = time_features.loc[keep_inc]
        target_feats = pd.DataFrame(target_features[subset]["other_feats"])


        coldict = parameters["coldict"]
        cat_pass = coldict["cat_pass"]
        cat_catboost = coldict["cat_catboost"]
        cont = coldict["cont"]
        output_col = parameters["target_col"]

        features = pd.merge(target_feats, time_features, right_index = True, left_index = True)
        orders_features = pd.merge(orders, features, right_index = True, left_index = True)

        # delete unused features and add budget (data splits by articles)
        orders_features = orders_features[cat_pass+cat_catboost+cont+[output_col]+["val_set"]]
        train = orders_features[orders_features["val_set"]==0]
        test = orders_features[orders_features["val_set"]==1]
        budgets = subframes_budgets(train["itemID"], test["itemID"], parameters["budgets"])
        orders_features["budgets"] = budgets

        # create dataframe for joint_ret_p, including imputation for single + multi basket + indicator columns
        # multi_baskets = orders_features[orders_features["basketNArts"]>1]
        # single_baskets = orders_features[orders_features["basketNArts"]==1]
        # multi_mean = multi_baskets[multi_baskets["val_set"]==0][output_col].mean()
        # single_mean = single_baskets[single_baskets["val_set"]==0][output_col].mean()
        # columns to indicate impution
        # imp_frame = pd.DataFrame(index = orders.index, columns = ret_p_cols+"_Imp").fillna(1)
        # ret_p_frame = pd.merge(pd.DataFrame(enc_frame, index = orders.index), imp_frame, right_index = True, left_index = True)    
        # for col in ret_p_cols:  # set non-imputed columns to zero
        #     ret_p_frame.loc[non_imp_inc[col], col+"_Imp"] = 0
        #     ret_p_frame.loc[multi_baskets.index, col] = ret_p_frame.loc[multi_baskets.index, col].fillna(multi_mean)
        #     ret_p_frame.loc[single_baskets.index, col] = ret_p_frame.loc[single_baskets.index, col].fillna(single_mean)

        orders_features.loc[:, cat_pass+cat_catboost] = orders_features[cat_pass+cat_catboost].astype("category")
        train.loc[:, cat_pass+cat_catboost] = train[cat_pass+cat_catboost].astype("category")
        print(cat_catboost)
        catboost_enc = CatBoostEncoder()
        catboost_enc.fit(train[cat_catboost], train[output_col])
        print(catboost_enc)
        catcoded_cols = catboost_enc.transform(orders_features[cat_catboost]).add_suffix("Catboost")
        print(catcoded_cols.head())

        orders_features = pd.merge(orders_features, catcoded_cols, right_index = True, left_index = True)
        orders_features = pd.merge(orders_features, target_features[subset]["enc_frame"], right_index = True, left_index = True)

        orders_features_sets[subset] = orders_features

    return orders_features_sets