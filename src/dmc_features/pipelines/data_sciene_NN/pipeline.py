from kedro.pipeline import Pipeline, node

from .nodes import find_incumbent, test_params


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
               func= find_incumbent,
               inputs= ["orders_features", "parameters", "enc_cols"],
               outputs= "resultdict_pytorch",
               name="search_pytorch",
            ),
            node(
               func= test_params,
               inputs= ["orders_features", "resultdict_pytorch", "parameters"],
               outputs= ["metrics_pytorch", "models_pytorch"],
               name="test_pytorch",
            )
        ]
    )