from kedro.pipeline import Pipeline, node

from .nodes import log_results



def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
               func= log_results,
               inputs= ["orders_features", "enc_cols", "parameters"],
               outputs= "results_cols",
               name="log_results",
            )
        ]
    )