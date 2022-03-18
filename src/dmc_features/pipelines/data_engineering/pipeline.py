from kedro.pipeline import Pipeline, node

from .nodes import combine_dfs, generate_date_features, generate_val_test_features, generate_feature_frame


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
                func=generate_val_test_features,
                inputs= ["orders_full", "parameters"],
                outputs=["target_features", "enc_cols"],
                name="target_feat",
            ),
            # node(
            #     func=generate_other_features,
            #     inputs= "orders_full",
            #     outputs="rest_features",
            #     name="rest_feat",
            # ),
            # node(
            #     func=finalize_features,
            #     inputs= ["orders_full", "parameters"],
            #     outputs=["enc_cols", "enc_frame"],
            #     name="features_final",
            # ),
            node(
                func=generate_feature_frame,
                inputs= ["orders_full", "time_features", "target_features", "parameters"],
                outputs="orders_features",
                name="feature_frame",
            ),
        ]
    )