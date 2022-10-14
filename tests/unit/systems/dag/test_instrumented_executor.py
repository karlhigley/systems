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

import numpy as np
import pandas as pd
import pytest

from merlin.dag import ColumnSelector, DictArray, Graph
from merlin.dag.base_operator import BaseOperator
from merlin.schema import ColumnSchema, Schema, Tags
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.executors import InstrumentedExecutor

loader_tf_utils = pytest.importorskip("nvtabular.loader.tf_utils")

# everything tensorflow related must be imported after this.
loader_tf_utils.configure_tensorflow()
tf = pytest.importorskip("tensorflow")

from merlin.systems.dag.ops.tensorflow import PredictTensorflow  # noqa
from merlin.systems.dag.ops.workflow import TransformWorkflow  # noqa

from nvtabular import Workflow  # noqa
from nvtabular import ops as wf_ops  # noqa


TRITON_SERVER_PATH = find_executable("tritonserver")

def test_instrumented_executor_with_dataframe_like():
    df = DictArray(
        {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}, dtypes={"a": np.int64, "b": np.int64}
    )
    schema = Schema([ColumnSchema("a", dtype=np.int64), ColumnSchema("b", dtype=np.int64)])
    operator = ["a"] >> BaseOperator()
    graph = Graph(operator)
    graph.construct_schema(schema)

    executor = InstrumentedExecutor()
    result = executor.transform(df, [graph.output_node])

    assert all(result["a"] == df["a"])
    assert "b" not in result.columns



@pytest.mark.parametrize("engine", ["parquet"])
def test_run_complex_dag_on_dataframe_with_ray_executor(tmpdir, dataset, engine):
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

    ins_exec = InstrumentedExecutor(transform_method="transform_batch")
    for _ in range(0, 10):
        response = ins_exec.transform(dataset.to_ddf().compute(), [merlin_ensemble.graph.output_node])



