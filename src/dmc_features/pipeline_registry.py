"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from dmc_features.pipelines import data_engineering as de
# from dmc_features.pipelines import data_sciene_BOOST as dsB


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_engineering_pipeline = de.create_pipeline()

    pipeline_all = data_engineering_pipeline # + data_sciene_BOOST_pipeline
    return {"__default__": Pipeline([pipeline_all]),
            "de": data_engineering_pipeline}
