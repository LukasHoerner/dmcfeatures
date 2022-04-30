import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader #, TensorDataset
from torchmetrics import MatthewsCorrCoef, F1Score, Precision, Recall, MetricCollection
from sklearn.preprocessing import OrdinalEncoder

import numpy as np
import pandas as pd
import pickle

from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from datetime import datetime

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import logging
logging.basicConfig(level=logging.DEBUG)

class TabularDataset(Dataset):
    def __init__(self, X, y, cat_feat=[], cont_feat=[]):
        """
        # https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/
        Characterizes a Dataset for PyTorch
        Parameters
        ----------
        X: array with features
        y: 1D array with explainable variable
        cat_feat: int or List of ints with incides
        The names of the categorical columns in the data.
        These columns will be passed through the embedding
        layers in the model. These columns must be
        label encoded beforehand. 
        output_col: string
        The name of the output variable column in the data
        provided.
        """
        self.n = X.shape[0]

        self.y = torch.from_numpy(np.array(y.astype(np.float32))).reshape(-1, 1)

        self.cat_feat = cat_feat # if any(cat_feat) else []
        self.cont_feat = cont_feat # if cont_feat else []

        self.cont_X = torch.from_numpy(
            X[:, cont_feat].astype(np.float32) if any(self.cont_feat) else np.zeros((self.n, 1)))
        self.cat_X = torch.from_numpy(
            X[:, cat_feat].astype(np.int32) if any(self.cat_feat) else np.zeros((self.n, 1)))

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]


class PyTorchWorker(Worker):
    def __init__(self, data, parameters, retp_cols, output_col, experiment, **kwargs):
        super().__init__(**kwargs)
        ret_p_hyperparams = parameters["ret_p_hyperparams"]
        self.N_min = ret_p_hyperparams["N_min"]
        self.basket_max = ret_p_hyperparams["basket_max"]

        cols_dict = parameters["coldict"]
        cat_cols = cols_dict['cat_pass'] + cols_dict['cat_catboost']
        cat_cols.remove(parameters["artnr_col"]) # remove art and add it depending on config
        cont_cols = cols_dict["cont"]
        # output_col = parameters["ret_col"]
        self.random_state=parameters["random_state"]
        self.experiment = experiment
        self.patience = parameters["patience_torch"]

        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.item_col = parameters["artnr_col"]
        self.output_col = output_col
        # later adapt catinc etc
        # cat_inc = [i for i in range(len(self.cat_cols))]
        # cont_inc = [i for i in range(len(self.cat_cols), len(self.cat_cols) + len(self.cont_cols))]
        train = data[data["val_set"]==0]

        self.train = train[cat_cols+cont_cols+retp_cols+[output_col]]
        test = data[data["val_set"]==1]
        self.test = test[cat_cols+cont_cols+retp_cols+[output_col]]

        # budget_dict = {}
        # budget_dict["train"] = {}
        # budget_dict["test"] = {}

        # for budget in data["budgets"].unique():
        #     budget_dict["train"][budget] = train[train["budgets"]<=budget].index
        #     budget_dict["test"][budget] = test[test["budgets"]<=budget].index
        # self.budget_dict = budget_dict

        self.categorical_features_indices = np.array(range(len(cat_cols)))

       # enable opportunity to choose between oversampling and weights
        # self.weights = None 
        # if parameters["imbalance"]: # train-val-test set with 0 - 1 - 2 in val_set column
        #     oversamp_inc = round((1 - (3-data["val_set"].max())*parameters["val_test_pct"])/(parameters["val_test_pct"]) * data[data["val_set"]==data["val_set"].max()].shape[0])
        #     self.oversamp_inc = oversamp_inc
        #     weights = pd.Series(1, index=range(train.shape[0]))
        #     weights.loc[train[train[output_col]==1].index] = 1/train.loc[:oversamp_inc, output_col].mean()
        #     weights = weights.loc[:oversamp_inc]
        #     self.weights = weights.to_list()
        #     print(oversamp_inc)

        # batch_size = 128
        # self.cat_cols = cat_cols # for embeddings; day_of_week + month as embedding
        # self.cot_cols = cot_cols
        # self.output_col = output_col
        # cat_inc = [i for i in range(len(self.cat_cols))]
        # cot_inc = [i for i in range(len(self.cat_cols), len(self.cat_cols) + len(self.cot_cols))]
        # # data = pickle.load(open("/home/ubuntu/orders-returns/data/04_feature/preprocessed_frame.p", "rb"))
        # X_train = data[data['val_set']== 0][self.cat_cols + self.cot_cols].to_numpy()
        # y_train = data[data['val_set']== 0][output_col].to_numpy()
        # X_test = data[data['val_set']== 1][self.cat_cols + self.cot_cols].to_numpy()
        # y_test = data[data['val_set']== 1][output_col].to_numpy()
        

        # dataloader = DataLoader(train_dataset, batchsize, shuffle=True) used the code from example

    def compute(self, config, budget, working_directory, search_run=True, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """
        


        log=self.logger
        cat_inc=self.categorical_features_indices

        if self.experiment == "baseline":
            cat_inc = np.append(cat_inc, len(cat_inc))
            used_retp_col = [self.item_col]
            X_train = self.train[used_retp_col + self.cat_cols+self.cont_cols]
            X_test = self.test[used_retp_col + self.cat_cols+self.cont_cols]
        else:
            # used_retp_col=[]
            if self.experiment == "BetaLoo2D":
                n_min = self.N_min[config['n_min']-1]
                max_basket = self.basket_max[config['max_basket']-1]
                weight = config['weight']
                sort = config['sort']
                used_retp_col = [str(str(n_min) + "_" + str(max_basket) + "_" + str(weight) + "_" + sort)]
            elif self.experiment == "mEstimate":
                if config['smooth']:
                    used_retp_col = [str(self.N_min[config['m']-1]) + "_" + config['smooth']]
                else:
                    used_retp_col = [self.N_min[config['m']-1]]
            X_train = self.train[self.cat_cols+self.cont_cols+ used_retp_col]
            X_test = self.test[self.cat_cols+self.cont_cols+ used_retp_col]
        print(used_retp_col)
        # train_budget = self.budget_dict["train"][int(budget)]
        # test_budget = self.budget_dict["test"][int(budget)]
        # X_train = X_train.loc[train_budget]
        # X_test = X_test.loc[test_budget]

        y_train = self.train[self.output_col].to_numpy()
        y_test = self.test[self.output_col].to_numpy()
        # y_train = y_train[train_budget.to_list()]
        # y_test = y_test[test_budget.to_list()]

        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        print(budget, X_train.shape, X_test.shape)

        cont_inc = np.array(range(len(cat_inc), X_train.shape[1]))
        cat_dims = [len(np.unique(X_train[:, inc])) for inc in cat_inc]
        # emb_dims = [(x, min(200, (x + 1) // 2)) for x in cat_dims] # current rule of thumb from fastai library
        emb_dims = [(x + 1, min(600, round(1.6 * x **0.56))) for x in cat_dims]
        
        # create datasets and dataloader
        batch_size = 1024
        train_dataset = TabularDataset(X_train, y_train, cat_inc, cont_inc)
        test_dataset = TabularDataset(X_test, y_test, cat_inc, cont_inc)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

        # initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # layers_size = config['first_layer_sz']

        if config['no_layers'] == 2:
            lin_layer_sizes = [200, 100]
        elif config['no_layers'] == 1:
            lin_layer_sizes = [100]

        # for i in range(config['no_layers']):
        #     lin_layer_sizes += [int(layers_size)]
        #     layers_size -= layers_size/2
        dropouts = [config['lin_layer_dropout']] * config['no_layers']

        model = FeedForwardNN(emb_dims, no_of_cont=len(cont_inc), lin_layer_sizes=lin_layer_sizes,
                            output_size=1, emb_dropout=config['emb_dropout'], # drop emd_dropout? -> or add 0
                            lin_layer_dropouts=dropouts)
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay= config['wd'], eps=1e-05) # eps to fastai default
        # print(model)
        # init metrics
        metrics = MetricCollection([MatthewsCorrCoef(num_classes=2), F1Score(), Precision(), Recall()])
        train_metrics = metrics.clone(prefix='train_')
        valid_metrics = metrics.clone(prefix='val_') 

        prev_mcc = 0
        best_epoch = 0
        patience = self.patience
        train_dict = {0: None}
        # train model
        for epoch in range(int(budget)): # 
            model.train()            
            loss = 0
            for i, (y, cont_x, cat_x) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(cont_x, cat_x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                y_pred = torch.sigmoid(output)
                train_metrics(y_pred, y.type(torch.int8))
            model.eval()
            for i, (y, cont_x, cat_x) in enumerate(test_loader):
                output = model(cont_x, cat_x)
                y_pred = torch.sigmoid(output)
                valid_metrics(y_pred, y.type(torch.int8))
            epoch_val_metrics = valid_metrics.compute()
            epoch_train_metrics = train_metrics.compute()
            print(epoch, loss.item(), epoch_train_metrics["train_MatthewsCorrCoef"], epoch_val_metrics)
            if epoch_val_metrics["val_MatthewsCorrCoef"].item() > prev_mcc:
                train_dict[epoch] = epoch_val_metrics
                best_epoch = epoch
                prev_mcc = epoch_val_metrics["val_MatthewsCorrCoef"].item()
                patience = self.patience
            else:
                patience -= 1
                if patience == 0:
                    break
            train_metrics.reset()
            valid_metrics.reset()


        metricsdict = {'mcc':prev_mcc,
                'F1_test': train_dict[best_epoch]["val_F1Score"],
                'Recall_test': train_dict[best_epoch]["val_Recall"],
                'Prec_test': train_dict[best_epoch]["val_Precision"],
                'data': X_train.shape,
                'epoch': best_epoch,
                'cols': used_retp_col,
                "lr": config["lr"],
                "wd": config["wd"],
                "emb_dropout": config["emb_dropout"],
                "lin_layer_dropout": config['lin_layer_dropout'],
                "no_layers": config["no_layers"],
                # "first_layer_sz": config["first_layer_sz"],
                }
        log.info(metricsdict)
        if search_run:
            return ({
                    'loss': 1-prev_mcc, # remember: HpBandSter always minimizes!
                    'info': {'best_metrics': prev_mcc,
                            'best_epoch': best_epoch,
                            'cols': used_retp_col,
                            "lr": config["lr"],
                            "wd": config["wd"],
                            "emb_dropout": config["emb_dropout"],
                            "lin_layer_dropout": config['lin_layer_dropout'],
                            "no_layers": config["no_layers"],
                            # "first_layer_sz": config["first_layer_sz"],
                            'number of parameters': model.number_of_parameters(),
                            }
                    })
        else:
            return [model, metricsdict]


    @staticmethod
    def get_configspace(parameters, experiment):
            """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
            ret_p_hyperparams = parameters["ret_p_hyperparams"]
            N_min = ret_p_hyperparams["N_min"]
            weights = ret_p_hyperparams["weights"]
            sorting_type = ret_p_hyperparams["sorting_type"]
            basket_max = ret_p_hyperparams["basket_max"]

            cs = CS.ConfigurationSpace()
            wd = CSH.UniformFloatHyperparameter('wd', lower=1e-2, upper=3e-1, default_value=1e-1)
            lr = CSH.UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-3, default_value=1e-4, log=True)
            lin_layer_dropout = CSH.UniformFloatHyperparameter('lin_layer_dropout', lower=0.0, upper=0.1, default_value=0.01, log=False)
            emb_dropout = CSH.UniformFloatHyperparameter('emb_dropout', lower=0.0, upper=0.1, default_value=0.01, log=False)
            # For demonstration purposes, we add different optimizers as categorical hyperparameters.

            cs.add_hyperparameters([wd, lr, lin_layer_dropout, emb_dropout])

            no_layers =  CSH.UniformIntegerHyperparameter('no_layers', lower=1, upper=2, default_value=2)
            # first_layer_sz =  CSH.UniformIntegerHyperparameter('first_layer_sz', lower=100, upper=500, default_value=300)

            # cs.add_hyperparameters([no_layers, first_layer_sz])
            cs.add_hyperparameter(no_layers)

            # Hyperparameters used for picking the joint_ret_p
            if experiment == "BetaLoo2D":
                min_joint_baskets = CSH.UniformIntegerHyperparameter('n_min', lower=1, upper=len(N_min), default_value=2)
                max_metaitems = CSH.UniformIntegerHyperparameter('max_basket', lower=1, upper=len(basket_max), default_value=2)
                # min_joint_baskets = CSH.CategoricalHyperparameter('n_min', N_min, default_value=N_min[1])
                # max_metaitems = CSH.CategoricalHyperparameter('max_basket', basket_max, default_value=basket_max[1])
                weighting = CSH.CategoricalHyperparameter('weight',  weights, default_value=weights[0])
                sorting = CSH.CategoricalHyperparameter('sort', sorting_type, default_value=sorting_type[0])
                cs.add_hyperparameters([min_joint_baskets, max_metaitems, weighting, sorting])
            elif experiment == "mEstimate":
                m = CSH.UniformIntegerHyperparameter('m', lower=1, upper=len(N_min), default_value=2)
                # m = CSH.CategoricalHyperparameter('m', N_min, default_value=N_min[1])
                smooth = CSH.CategoricalHyperparameter('smooth', ["m", "m_smooth", "", "base"], default_value="m")
                cs.add_hyperparameters([m, smooth])
            # # add no hyperparams if using article
            # if parameters["imbalance"] == True:
            #     # 1 indicating weights, 0 oversampling
            #     imbalance = CSH.UniformIntegerHyperparameter('weights', lower=0, upper=1, default_value=1)
            #     # imbalance = CSH.CategoricalHyperparameter('imb', ["oversamp", "weights"], default_value="oversamp")
            #     cs.add_hyperparameter(imbalance)
            return cs


class FeedForwardNN(nn.Module):
    def __init__(
        self,
        emb_dims,
        no_of_cont,
        lin_layer_sizes,
        output_size,
        emb_dropout,
        lin_layer_dropouts,
    ):

        """
        Parameters
        ----------
        emb_dims: List of two element tuples
        This list will contain a two element tuple for each
        categorical feature. The first element of a tuple will
        denote the number of unique values of the categorical
        feature. The second element will denote the embedding
        dimension to be used for that feature.
        no_of_cont: Integer
        The number of continuous features in the data.
        lin_layer_sizes: List of integers.
        The size of each linear layer. The length will be equal
        to the total number
        of linear layers in the network.
        output_size: Integer
        The size of the final output.
        emb_dropout: Float
        The dropout to be used after the embedding layers.
        lin_layer_dropouts: List of floats
        The dropouts to be used after each linear layer.
        """
        super().__init__()

        # Embedding layers
        self.embeds = nn.ModuleList([nn.Embedding(x, y)
                                         for x, y in emb_dims])
        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont
        # if len(lin_layer_sizes) == 0:
        #     lin_layer_sizes = [2]
        # Linear Layers
        first_lin_layer = nn.Linear(
            self.no_of_embs + self.no_of_cont, lin_layer_sizes[0]
        )

        self.lin_layers = nn.ModuleList(
            [first_lin_layer]
            + [
                nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                for i in range(len(lin_layer_sizes) - 1)
            ]
        )

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(size) for size in lin_layer_sizes]
        )

        # Dropout Layers
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList(
            [nn.Dropout(size) for size in lin_layer_dropouts]
        )

    def forward(self, cont_data, cat_data):

        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i]) for i, emb_layer in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_dropout(x)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_cont_data], 1)
            else:
                x = normalized_cont_data

        for lin_layer, dropout_layer, bn_layer in zip(
            self.lin_layers, self.droput_layers, self.bn_layers
        ):
            x = torch.relu(lin_layer(x))
            # x = torch.tanh(lin_layer(x))
            x = bn_layer(x) # batch-norm
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))


def find_incumbent(data, parameters, enc_cols, host_nn='lo'):
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger(__name__)
    # use test-validation set
    data = data['train_val']

    # new data: encode categorical variables
    ord_encode =  parameters["coldict"]["ord_encode"]
    train = data[data["val_set"]==0]
    test = data[data["val_set"]==1]
    enc = OrdinalEncoder(handle_unknown= "use_encoded_value", unknown_value=np.nan) # )
    enc_train = enc.fit_transform(train[ord_encode])
    enc_test = enc.transform(test[ord_encode])
    data.loc[:, ord_encode] = np.nan
    data.loc[train.index, ord_encode] = enc_train
    data.loc[test.index, ord_encode] = enc_test
    for column in ord_encode:
        data[column].fillna(max(data[column]), inplace=True)
    catenc= parameters["coldict"]["cat_pass"] + parameters["coldict"]["cat_catboost"]
    data.loc[:, catenc] = data[catenc].apply(lambda x: pd.to_numeric(x))
    # oversamp_inc = None
    # if parameters["imbalance"] == True:
    #     oversamp_inc = round((1 - data["val_set"].max()*parameters["val_test_pct"])/(parameters["val_test_pct"]) * data[data["val_set"]==data["val_set"].max()].shape[0])

    # cols_dict = parameters["cols_dict"]
    # cat_cols = cols_dict['pass_cat_cols'] + cols_dict['enc_cat_cols']
    # cat_cols.remove(parameters["artnr_col"])
    # cont_cols = cols_dict["cont_cols"]
    # output_col = parameters["ret_col"]
    bohb_args = parameters["bohb_pytorch"]
    resultdict = {}
    experiments = {"baseline": [parameters["artnr_col"]],
                    "mEstimate": enc_cols["m"] + enc_cols["m_smooth"] + enc_cols['1D'] + enc_cols['1D_base'],
                    "BetaLoo2D": enc_cols["2D"]}

    for experiment in experiments.keys():
        host = hpns.nic_name_to_host(host_nn)
        ret_p_cols = experiments[experiment]
        run_id = experiment + datetime.now().strftime("%M")
        # log.info('Starting ' + experiment)
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
        for i in range(bohb_args["n_workers"]):
            w = PyTorchWorker(nameserver=host,run_id=run_id, logger=log, id=i, data=data,parameters=parameters,
            retp_cols=ret_p_cols, output_col=parameters["target_col"], experiment=experiment)
            w.run(background=True)
            workers.append(w)
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