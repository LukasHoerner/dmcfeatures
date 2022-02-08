from kedro.pipeline import Pipeline, node

from .nodes import combine_dfs, generate_date_features, generate_other_features, generate_ret_p, generate_feature_frame


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func= combine_dfs,
                inputs= ["orders_train", "orders_test", "orders_test_y"],
                outputs= "orders_full",
                name="combine_dfs",
            ),
            node(
                func=generate_date_features,
                inputs= "orders_full",
                outputs="time_features",
                name="time_feat",
            ),
            node(
                func=generate_other_features,
                inputs= "orders_full",
                outputs="rest_features",
                name="rest_feat",
            ),
            node(
                func=generate_ret_p,
                inputs= ["orders_full", "parameters"],
                outputs=["arts_ret_p", "lookup_basket_occ", "lookup_basket_ret_p","lookup_ret_perc_art",
                "lookup_tot_ord", "lookup_lift_arts", "lookup_droped_ret_arts", "arts_used_ret_p"],
                name="ret_p_feat",
            ),
            node(
                func=generate_feature_frame,
                inputs= ["orders_full", "time_features", "rest_features", "arts_ret_p", "parameters"],
                outputs="orders_features",
                name="feature_frame",
            ),
        ]
    )