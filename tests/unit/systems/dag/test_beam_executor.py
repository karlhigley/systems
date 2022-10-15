#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from distutils.spawn import find_executable
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

import apache_beam as beam
from apache_beam.dataframe.transforms import DataframeTransform

from merlin.dag import ColumnSelector, DictArray, Graph
from merlin.dag.base_operator import BaseOperator
from merlin.schema import ColumnSchema, Schema, Tags
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.executors import BeamExecutor

loader_tf_utils = pytest.importorskip("nvtabular.loader.tf_utils")

# everything tensorflow related must be imported after this.
loader_tf_utils.configure_tensorflow()
tf = pytest.importorskip("tensorflow")

from merlin.systems.dag.ops.tensorflow import PredictTensorflow  # noqa
from merlin.systems.dag.ops.workflow import TransformWorkflow  # noqa

from nvtabular import Workflow  # noqa
from nvtabular import ops as wf_ops  # noqa


@pytest.mark.parametrize("engine", ["parquet"])
def test_run_dag_on_dataframe_with_beam(tmpdir, paths, dataset, engine):
    # Create a Workflow
    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [Tags.USER]
        )
    selector = ColumnSelector(["x", "y", "id"])

    workflow_ops = selector >> wf_ops.Rename(postfix="_nvt")
    workflow = Workflow(workflow_ops["x_nvt"])
    workflow.fit(dataset)

    breakpoint()

    # Create Tensorflow Model
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(name="x_nvt", dtype=tf.float64, shape=(1,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, name="output"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[],
    )

    op_chain = selector >> TransformWorkflow(workflow, cats=["x_nvt"]) >> PredictTensorflow(model)
    merlin_ensemble = Ensemble(op_chain, schema)

    def systems_transform(records):
        df = pd.DataFrame.from_records(records)
        result = BeamExecutor(transform_method="transform_batch").transform(df, [merlin_ensemble.graph.output_node])
        return result

    class Record(NamedTuple):
        x: float
        y: float
        z: float
        id: int
        timestamp: int
        name_cat: str
        name_string: str
        label: int


    from pathlib import Path
    directory = Path(paths[0]).parent
    records = None
    with beam.Pipeline() as p:
        records = p | "Read" >> beam.io.ReadFromParquet(f"{directory}/*").with_output_types(Record) \
                    | beam.BatchElements(min_batch_size=1000) \
                    | beam.ParDo(systems_transform)
                    # | DataframeTransform(systems_transform) \
                    # | beam.io.parquetio.WriteToParquet(f"{tmpdir}/output")
        _ = records | "Write" >> beam.io.textio.WriteToText(f"{tmpdir}/output")

    result = p.run()
    result.wait_until_finish()

    # TODO: Assert that the output dir

    # ins_exec = InstrumentedExecutor(transform_method="transform_batch")
    # for _ in range(0, 10):
    #     response = ins_exec.transform(dataset.to_ddf().compute(), [merlin_ensemble.graph.output_node])

