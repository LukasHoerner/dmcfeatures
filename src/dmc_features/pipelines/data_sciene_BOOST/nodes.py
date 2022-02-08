from catboost import CatBoostClassifier, Pool, metrics

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import cohen_kappa_score


import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)

class CatBoostWorker(Worker):
    def __init__(self, data, cat_cols, cot_cols, output_col, **kwargs):
        super().__init__(**kwargs)

        self.cat_cols = cat_cols # for embeddings; day_of_week + month as embedding
        self.cot_cols = cot_cols
        self.output_col = output_col
        cat_inc = [i for i in range(len(self.cat_cols))]
        cot_inc = [i for i in range(len(self.cat_cols), len(self.cat_cols) + len(self.cot_cols))]
        # data = pickle.load(open("/home/ubuntu/orders-returns/data/04_feature/preprocessed_frame.p", "rb"))
        X_train = data[data['val_set']== 0][self.cat_cols + self.cot_cols].to_numpy()
        y_train = data[data['val_set']== 0][output_col].to_numpy()
        X_test = data[data['val_set']== 1][self.cat_cols + self.cot_cols].to_numpy()
        y_test = data[data['val_set']== 1][output_col].to_numpy()
        cat_dims = [len(np.unique(X_train[:, inc])) for inc in cat_inc]

        # emb_dims = [(x, min(200, (x + 1) // 2)) for x in cat_dims] # from fastai library
        self.emb_dims = [(x + 1, min(600, round(1.6 * x **0.56))) for x in cat_dims]
        
        train_dataset = TabularDataset(X_train, y_train, cat_inc, cot_inc)
        test_dataset = TabularDataset(X_test, y_test, cat_inc, cot_inc)
        # dataloader = DataLoader(train_dataset, batchsize, shuffle=True) used the code from example

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(N_train))
        validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(N_train, N_train+N_valid))


        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
        self.validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1024, sampler=validation_sampler)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)


    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """
        n_arts = config['min_joint_baskets']
        top_f = config['top_freq_arts']
        weight = config['weights']
        sort = config['sorting']
        used_retp_col = "min_"+str(n_arts)+"_top_"+str(top_f)+ "_"+str(weight) + "_" + sort

        if config['drop_articles']:
            print("drop artikelnr from cols")
        if config['catboostcol']:
            if type(config['drop_articles'])==str:
                print("impute")
                # impute to used_retp_col
            else:
                print("add CATBOOST to cols")
        else:
            print(used_retp_col)

                
        categorical_features_indices = [1]
        # use catcol: yes, no, impute
        # use art: yes, no

        params = {
            'iterations': 1000,
            'learning_rate': config['lr'],
            'eval_metric': metrics.F1(),
            'random_seed': 42,
            'logging_level': 'Silent',
            'use_best_model': True,
            'od_type': 'Iter',  # https://catboost.ai/en/docs/features/overfitting-detector-desc#od_wait
            'od_wait': 50,
            'allow_writing_files': False,
            'l2_leaf_reg': config['l2_leaf_reg'],
            'depth': config['depth']
        }

        train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
        validate_pool
        model = CatBoostClassifier(**params)

        train_kappa = self.evaluate_cohen_kappa(model, self.train_loader)
        validation_kappa = self.evaluate_cohen_kappa(model, self.validation_loader)
        test_kappa = self.evaluate_cohen_kappa(model, self.test_loader)

        return ({
                'loss': 1-validation_kappa, # remember: HpBandSter always minimizes!
                'info': {   'test kappa': test_kappa,
                            'train kappa': train_kappa,
                            'validation kappa': validation_kappa,
                            'number of parameters': model.number_of_parameters(),
                                }

        })

    def evaluate_cohen_kappa(self, model, data_loader):
            model.eval()
            cohen_kappas=[]
            with torch.no_grad():
                    for y, cont_x, cat_x in data_loader:
                            output = torch.sigmoid(model(cont_x, cat_x))
                            pred = torch.round(output)
                            #test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                            cohen_kappa = cohen_kappa_score(y, pred)
                            cohen_kappas += [cohen_kappa]
            #import pdb; pdb.set_trace()
            cohen_kappa = sum(cohen_kappas)/len(cohen_kappas)
            return(cohen_kappa)


    @staticmethod
    def get_configspace():
            """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
            cs = CS.ConfigurationSpace()

            lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=3e-1, default_value='1e-2', log=True)
            l2_leaf_reg =  CSH.UniformIntegerHyperparameter('l2_leaf_reg', lower=2, upper=30, default_value=3)
            depth =  CSH.UniformIntegerHyperparameter('depth', lower=1, upper=16, default_value=6)

            cs.add_hyperparameters([lr, l2_leaf_reg, depth])

            # Hyperparameters used for picking the joint_ret_p
            min_joint_baskets = CSH.CategoricalHyperparameter('min_joint_baskets', ["0", "5", "10", "15"], default_value="10")
            top_freq_arts = CSH.CategoricalHyperparameter('top_freq_arts', ["", "3", "5", "10", "var_5"], default_value="3")
            weights = CSH.CategoricalHyperparameter('weights', ["", "lin_w", "sqrt_w"], default_value="lin_w")
            sorting = CSH.CategoricalHyperparameter('sorting', ["_lift", "_conf"], default_value="_lift")

            cs.add_hyperparameters([min_joint_baskets, top_freq_arts, weights, sorting])

            # feature selection properties
            drop_articles = CSH.CategoricalHyperparameter('drop_articles', [True, False], default_value=True)
            catboostcol = CSH.CategoricalHyperparameter('catboostcol', [True, False, "impute"], default_value=True)

            cs.add_hyperparameters([drop_articles, catboost])

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

            x = F.relu(lin_layer(x))
            x = bn_layer(x) # batch-norm
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x
    
    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))


# needed??
# if __name__ == "__main__":
#     worker = PyTorchWorker(run_id='0')
#     cs = worker.get_configspace()

#     config = cs.sample_configuration().get_dictionary()
#     print(config)
#     res = worker.compute(config=config, budget=2, working_directory='.')
#     print(res)

def create_args(description, min_budget, max_budget, n_iterations, n_workers, run_id):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.', default=min_budget) #9
    parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.', default=max_budget)  #243
    parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=n_iterations) #5
    parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=n_workers)    #32
    parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
    args=parser.parse_args()
    return args

def find_incumbent(args, host_nn='lo'):
    host = hpns.nic_name_to_host(host_nn)

    # Step 1: Start a nameserver (see example_1)
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=None)
    NS.start()

    # Step 2: Start the workers
    # Now we can instantiate the specified number of workers. To emphasize the effect,
    # we introduce a sleep_interval of one second, which makes every function evaluation
    # take a bit of time. Note the additional id argument that helps separating the
    # individual workers. This is necessary because every worker uses its processes
    # ID which is the same for all threads here.
    workers=[]
    for i in range(args.n_workers):
        w = PyTorchWorker(nameserver=host,run_id=args.run_id, id=i)
        w.run(background=True)
        workers.append(w)

    # Step 3: Run an optimizer
    # Now we can create an optimizer object and start the run.
    # We add the min_n_workers argument to the run methods to make the optimizer wait
    # for all workers to start. This is not mandatory, and workers can be added
    # at any time, but if the timing of the run is essential, this can be used to
    # synchronize all workers right at the start.
    bohb = BOHB(  configspace = w.get_configspace(),
        run_id = args.run_id,
        min_budget=args.min_budget, max_budget=args.max_budget
                )
    res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    all_runs = res.get_all_runs()

    return [res, id2config, incumbent, all_runs]