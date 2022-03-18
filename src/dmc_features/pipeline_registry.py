"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from dmc_features.pipelines import data_engineering as de
from dmc_features.pipelines import data_sciene_simp_BOOST as ds_simp_B
from dmc_features.pipelines import data_sciene_BOOST as ds_B
from dmc_features.pipelines import data_sciene_NN as ds_NN


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_engineering_pipeline = de.create_pipeline()
    data_science_simp_pipeline = ds_simp_B.create_pipeline()
    data_science_BOOST_pipeline = ds_B.create_pipeline()
    data_science_NN_pipeline = ds_NN.create_pipeline()

    pipeline_all = data_engineering_pipeline + data_science_simp_pipeline + data_science_BOOST_pipeline
    return {"__default__": Pipeline([pipeline_all]),
            "de": data_engineering_pipeline,
            "ds_simp_B": data_science_simp_pipeline,
            "dsBoost": data_science_BOOST_pipeline,
            "ds_NN": data_science_NN_pipeline}
