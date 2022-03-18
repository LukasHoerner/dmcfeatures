from kedro.pipeline import Pipeline, node

from .nodes import find_incumbent, test_params


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
               func= find_incumbent,
               inputs= ["orders_features", "parameters", "enc_cols"],
               outputs= "resultdict",
               name="find_incumbent",
            ),
            node(
               func= test_params,
               inputs= ["orders_features" , "resultdict",  "parameters"],
               outputs= "models_catboost",
               name="test_params",
            )
        ]
    )