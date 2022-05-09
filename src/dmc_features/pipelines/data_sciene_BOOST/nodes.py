from catboost import CatBoostClassifier, Pool, metrics

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score, precision_score

# import argparse
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from datetime import datetime

import logging

class CatBoostWorker(Worker):
    def __init__(self, data, cat_cols, cont_cols, retp_cols, output_col, experiment, random_state=42, **kwargs):
        super().__init__(**kwargs)
        self.random_state=random_state
        self.experiment = experiment

        self.cat_cols = cat_cols # for embeddings; day_of_week + month as embedding
        self.cont_cols = cont_cols
        self.output_col = output_col
        # later adapt catinc etc
        cat_inc = [i for i in range(len(self.cat_cols))]
        # cont_inc = [i for i in range(len(self.cat_cols), len(self.cat_cols) + len(self.cont_cols))]
        train = data[data["val_set"]==0]
        self.train_multi_inc = train[train["basketNArts"]>1].index
        self.train = train[cat_cols+cont_cols+retp_cols+[output_col]]
        test = data[data["val_set"]==1]
        self.test = test[cat_cols+cont_cols+retp_cols+[output_col]]
        self.test_multi_inc = test[test["basketNArts"]>1].index
        # data = pickle.load(open("/home/ubuntu/orders-returns/data/04_feature/preprocessed_frame.p", "rb"))
        # self.X_train = data[data['val_set']== 0][self.cat_cols + self.cont_cols].to_numpy()
        # self.y_train = data[data['val_set']== 0][output_col].to_numpy()
        # self.X_test = data[data['val_set']== 1][self.cat_cols + self.cont_cols].to_numpy()
        # self.y_test = data[data['val_set']== 1][output_col].to_numpy()
        self.categorical_features_indices = np.array(range(len(cat_cols)))

        # emb_dims = [(x, min(200, (x + 1) // 2)) for x in cat_dims] # from fastai library
        # self.emb_dims = [(x + 1, min(600, round(1.6 * x **0.56))) for x in cat_dims]
        
        # train_dataset = TabularDataset(X_train, y_train, cat_inc, cot_inc)
        # test_dataset = TabularDataset(X_test, y_test, cat_inc, cot_inc)
        # # dataloader = DataLoader(train_dataset, batchsize, shuffle=True) used the code from example
        # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(N_train))
        # validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(N_train, N_train+N_valid))
        # self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
        # self.validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1024, sampler=validation_sampler)
        # self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)


    def compute(self, config, search_run=True, iterations = None, budget=None, working_directory=None, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """
        log =self.logger
        cat_features=self.categorical_features_indices
        if self.experiment == "catboost_enc":
            cat_features = np.append(cat_features, len(cat_features))
            used_retp_col = "itemID"
            X_train = self.train[[used_retp_col] + self.cat_cols+self.cont_cols]
            X_test = self.test[[used_retp_col] + self.cat_cols+self.cont_cols]
        else:
            if self.experiment == "BetaLoo2D":
                n_min = config['n_min']
                max_basket = config['max_basket']
                weight = config['weight']
                sort = config['sort']
                used_retp_col = str(str(n_min) + "_" + str(max_basket) + "_" + str(weight) + "_" + sort)
            elif self.experiment == "mEstimate":
                used_retp_col = str(config['m']) + "_" + config['smooth']
            
            X_train = self.train[self.cat_cols+self.cont_cols+ [used_retp_col]]
            X_test = self.test[self.cat_cols+self.cont_cols+ [used_retp_col]]

        y_train = self.train[self.output_col]
        y_test = self.test[self.output_col]
        y_test_multi = y_test.loc[self.test_multi_inc]
        # if config['drop_articles']:
        #     print("drop artikelnr from cols")
        # if config['catboostcol']:
        #     if type(config['drop_articles'])==str:
        #         print("impute")
        #         # impute to used_retp_col
        #     else:
        #         print("add CATBOOST to cols")
        # else:
        #     print(used_retp_col)

        # use catcol: yes, no, impute
        # use art: yes, no
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        validate_pool = Pool(X_test, y_test, cat_features=cat_features)

        params = {
            'iterations': 1000,
            'learning_rate': config['lr'],
            'eval_metric': metrics.MCC(),
            'random_seed': self.random_state,
            'logging_level': 'Silent',
            'use_best_model': True,
            'od_type': 'Iter',  # https://catboost.ai/en/docs/features/overfitting-detector-desc#od_wait
            'od_wait': 50,
            'allow_writing_files': False,
            'one_hot_max_size': 16, # low cardinality features as OHE
            'l2_leaf_reg': config['l2_leaf_reg'],
            'depth': config['depth']
        }
        if iterations:
            params["use_best_model"] = False
            params["iterations"] = iterations
            del params["od_type"]
            del params["od_wait"]
            # maybe it is also necessary to set od_type and od_wait

        # validate_pool
        model = CatBoostClassifier(**params)    # balanced
                                                # problem: dealing with imbalanced data -> encoding
        model.fit(train_pool, eval_set=validate_pool)
        y_pred = model.predict(X_test)
        y_pred_multi = model.predict(X_test.loc[self.test_multi_inc])
        MCC_test= matthews_corrcoef(y_test, y_pred)

        metricsdict = {"MCC_test": MCC_test,
                    'iterations': model.tree_count_,
                    'cols': used_retp_col,
                    'F1_test': f1_score(y_test, y_pred),
                    'Recall_test': recall_score(y_test, y_pred),
                    'Prec_test': precision_score(y_test, y_pred),
                    'MCC_test_multiarts': matthews_corrcoef(y_test_multi, y_pred_multi),
                    'Recall_test_multiarts': recall_score(y_test_multi, y_pred_multi),
                    'Prec_test_multiarts': precision_score(y_test_multi, y_pred_multi),
                    'depth': config['depth'],
                    'l2_leaf_reg': config['l2_leaf_reg'],
                    'learning_rate': config['lr'],
                    "X_train_shape": X_train.shape[0],
                    "X_test_shape": X_test.shape[0]}
        log.info(metricsdict)
        if search_run:
            return ({
                    'loss': 1-MCC_test, # remember: HpBandSter always minimizes!
                    'info': 
                    {'iterations': model.tree_count_,
                    'F1_test': f1_score(y_test, y_pred),
                    'Recall_test': recall_score(y_test, y_pred),
                    'Prec_test': precision_score(y_test, y_pred),
                    'MCC_test_multiarts': matthews_corrcoef(y_test_multi, y_pred_multi),
                    'Recall_test_multiarts': recall_score(y_test_multi, y_pred_multi),
                    'Prec_test_multiarts': precision_score(y_test_multi, y_pred_multi),
                    'depth': config['depth'],
                    'l2_leaf_reg': config['l2_leaf_reg'],
                    'learning_rate': config['lr'],}
                    })
        else:
            return [model, metricsdict]
            # log.info({"MCC_test": MCC_test,
            #     'cols': used_retp_col,
            #     'F1_test': f1_score(y_test, y_pred),
            #     'Recall_test': recall_score(y_test, y_pred),
            #     'Prec_test': precision_score(y_test, y_pred),
            #     'MCC_test_multiarts': matthews_corrcoef(y_test_multi, y_pred_multi),
            #     'Recall_test_multiarts': recall_score(y_test_multi, y_pred_multi),
            #     'Prec_test_multiarts': precision_score(y_test_multi, y_pred_multi),
            #     'depth': config['depth'],
            #     'l2_leaf_reg': config['l2_leaf_reg'],
            #     'learning_rate': config['lr'],}
            #     )


    @staticmethod
    def get_configspace(parameters, experiment):
            """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
            cs = CS.ConfigurationSpace(seed=parameters["random_state"])
            ret_p_hyperparams = parameters["ret_p_hyperparams"]
            N_min = ret_p_hyperparams["N_min"]
            weights = ret_p_hyperparams["weights"]
            sorting_type = ret_p_hyperparams["sorting_type"]
            basket_max = ret_p_hyperparams["basket_max"]
            
            # algorithm (catboost) hyperparams
            lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=3e-1, default_value=1e-1) #, log=True
            l2_leaf_reg =  CSH.UniformIntegerHyperparameter('l2_leaf_reg', lower=2, upper=30, default_value=3)
            depth =  CSH.UniformIntegerHyperparameter('depth', lower=1, upper=16, default_value=6)
            cs.add_hyperparameters([lr, l2_leaf_reg, depth])

            # Hyperparameters used for picking the joint_ret_p
            if experiment == "BetaLoo2D":
                min_joint_baskets = CSH.CategoricalHyperparameter('n_min', N_min, default_value=N_min[1])
                max_metaitems = CSH.CategoricalHyperparameter('max_basket', basket_max, default_value=basket_max[1])
                weighting = CSH.CategoricalHyperparameter('weight',  weights, default_value=weights[0])
                sorting = CSH.CategoricalHyperparameter('sort', sorting_type, default_value=sorting_type[0])
                cs.add_hyperparameters([min_joint_baskets, max_metaitems, weighting, sorting])
            elif experiment == "mEstimate":
                m = CSH.CategoricalHyperparameter('m', N_min, default_value=N_min[1])
                smooth = CSH.CategoricalHyperparameter('smooth', ["m", "m_smooth"], default_value="m")
                cs.add_hyperparameters([m, smooth])
            # add no hyperparams if item is itemID

            # deprecated params
            # drop_articles = CSH.CategoricalHyperparameter('drop_articles', [True, False], default_value=True)
            # catboostcol = CSH.CategoricalHyperparameter('catboostcol', [True, False, "impute"], default_value=True)
            # cs.add_hyperparameters([drop_articles, catboost])

            return cs


# def evaluate_cohen_kappa(self, model, data_loader):
#         model.eval()
#         cohen_kappas=[]
#         with torch.no_grad():
#                 for y, cont_x, cat_x in data_loader:
#                         output = torch.sigmoid(model(cont_x, cat_x))
#                         pred = torch.round(output)
#                         #test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
#                         cohen_kappa = cohen_kappa_score(y, pred)
#                         cohen_kappas += [cohen_kappa]
#         #import pdb; pdb.set_trace()
#         cohen_kappa = sum(cohen_kappas)/len(cohen_kappas)
#         return(cohen_kappa)

# # only needed when script is called as main
# if __name__ == "__main__":
#     worker = PyTorchWorker(run_id='0')
#     cs = worker.get_configspace()

#     config = cs.sample_configuration().get_dictionary()
#     print(config)
#     res = worker.compute(config=config, budget=2, working_directory='.')
#     print(res)

# def create_args(description, min_budget, max_budget, n_iterations, n_workers):
#     parser = argparse.ArgumentParser(description=description)
#     parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.', default=min_budget) #9
#     parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.', default=max_budget)  #243
#     parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=n_iterations) #5
#     parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=n_workers)    #32
#     parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
#     args=parser.parse_args()
#     return args

def find_incumbent(data, parameters, enc_cols, host_nn='lo'):
    np.random.seed(parameters["random_state"])
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger(__name__)
    # use test-validation set
    data = data['train_val']

    cols_dict = parameters["coldict"]
    cat_cols = cols_dict['cat_pass'] + cols_dict['cat_catboost']
    cat_cols.remove(parameters["artnr_col"])
    cont_cols = cols_dict["cont"]
    outpu_col = parameters["target_col"]
    bohb_args = parameters["bohb_catboost"]
    resultdict = {}

    experiments = {"catboost_enc": [parameters["artnr_col"]],
                    "mEstimate": enc_cols["m"] + enc_cols["m_smooth"],
                    "BetaLoo2D": enc_cols["2D"]}

    for experiment in experiments.keys():
        host = hpns.nic_name_to_host(host_nn)
        ret_p_cols = experiments[experiment]
        run_id = experiment + datetime.now().strftime("%M")
        log.info('Starting ' + experiment)
        # args = create_args(bohb_args["description"], bohb_args["min_budget"], bohb_args["max_budget"], bohb_args["n_iterations"],)
        # Step 1: Start a nameserver (see example_1)
        NS = hpns.NameServer(run_id=run_id, host=host, port=None)
        NS.start()
        # Step 2: Start the workers
        # Now we can instantiate the specified number of workers. To emphasize the effect,
        # we introduce a sleep_interval of one second, which makes every function evaluation
        # take a bit of time. Note the additional id argument that helps separating the
        # individual workers. This is necessary because every worker uses its processes
        # ID which is the same for all threads here.
        workers=[]
        print(workers)
        for i in range(bohb_args["n_workers"]):
            print(i)
            w = CatBoostWorker(nameserver=host,logger=log, run_id=run_id, id=i, data=data,cat_cols=cat_cols,
            cont_cols=cont_cols, retp_cols=ret_p_cols, output_col=outpu_col, experiment=experiment)
            w.run(background=True)
            workers.append(w)
        print(w)
        # Step 3: Run an optimizer
        # Now we can create an optimizer object and start the run.
        # We add the min_n_workers argument to the run methods to make the optimizer wait
        # for all workers to start. This is not mandatory, and workers can be added
        # at any time, but if the timing of the run is essential, this can be used to
        # synchronize all workers right at the start.
        configspace = w.get_configspace(parameters, experiment)
        print(configspace, run_id, bohb_args)

        bohb = BOHB(configspace = w.get_configspace(parameters, experiment), run_id = run_id,
            min_budget=bohb_args["min_budget"], max_budget=bohb_args["max_budget"])
        res = bohb.run(n_iterations=bohb_args["n_iterations"], min_n_workers=bohb_args["n_workers"])
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()
        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        all_runs = res.get_all_runs()

        log.info('Best found configuration:' + str(id2config[incumbent]['config']))
        log.info('A total of {} unique configurations where sampled.'.format(len(id2config.keys())))
        log.info('A total of {} runs where executed.'.format(len(res.get_all_runs())))
        log.info('Total budget corresponds to {} full function evaluations.'.format((sum([r.budget for r in all_runs])/bohb_args["max_budget"])))
        log.info('The run took  %.1f seconds to complete.'.format((all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started'])))
        resultdict[experiment] = {"res": res, "id2config": id2config, "incumbent": incumbent, "all_runs": all_runs}

    return resultdict


def test_params(data, resultdict, parameters):
    data = data['train_test']
    cols_dict = parameters["coldict"]
    cat_cols = cols_dict['cat_pass'] + cols_dict['cat_catboost']
    cat_cols.remove(parameters["artnr_col"])
    cont_cols = cols_dict["cont"]
    outpu_col = parameters["target_col"]
    best_models = {}
    best_metrics = {}
    for experiment in resultdict.keys():
        results = resultdict[experiment]
        config = results["id2config"][results["incumbent"]]["config"]
        res = results["res"]
        iterations = res.get_runs_by_id(res.get_incumbent_id())[-1]["info"]["iterations"] # get optimal iterations
        if experiment == "BetaLoo2D":
            n_min = config['n_min']
            max_basket = config['max_basket']
            weight = config['weight']
            sort = config['sort']
            used_retp_col = [str(str(n_min) + "_" + str(max_basket) + "_" + str(weight) + "_" + sort)]
        elif experiment == "mEstimate":
            used_retp_col = [str(config['m']) + "_" + config['smooth']]
        else:
            used_retp_col = ["itemID"]
        worker = CatBoostWorker(run_id=experiment, data=data,cat_cols=cat_cols,
            cont_cols=cont_cols, retp_cols=used_retp_col, output_col=outpu_col, experiment=experiment)
        [model, metricsdict] = worker.compute(config, search_run=False, iterations=iterations)
        print(metricsdict)
        for key in parameters["metrics"]:
            if key in metricsdict.keys():
                best_metrics[experiment + "_" + key] = metricsdict[key]
        best_models[experiment] = model

    return [best_metrics, best_models]

def test_metrics(inputdict):
    return inputdict