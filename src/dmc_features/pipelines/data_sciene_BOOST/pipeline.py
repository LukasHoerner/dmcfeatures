from kedro.pipeline import Pipeline, node

# from .nodes import firstfunc


def create_pipeline(**kwargs):
    return Pipeline(
        [
            #node(
            #    func= repair_cols,
            #    inputs= ["orders", "orders_positions", "returns_positions", "returns"],
            #    outputs= ["orders_colcheck", "orders_positions_colcheck",
            #     "returns_positions_colcheck", "returns_colcheck"],
            #    name="repair_cols",
            #)
        ]
    )