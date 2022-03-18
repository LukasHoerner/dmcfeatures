from kedro.pipeline import Pipeline, node

from .nodes import find_incumbent


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
               func= find_incumbent,
               inputs= ["orders_features", "parameters", "enc_cols"],
               outputs= "results_pytorch",
               name="search_pytorch",
            )
        ]
    )