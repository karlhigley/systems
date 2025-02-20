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

import os
import pathlib
from copy import deepcopy

import numpy as np
import pytest

from merlin.dag import ColumnSelector, Graph
from merlin.schema import ColumnSchema, Schema

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format  # noqa

model_config = pytest.importorskip("tritonclient.grpc.model_config_pb2")
tf_op = pytest.importorskip("merlin.systems.dag.ops.tensorflow")

tf = pytest.importorskip("tensorflow")


def test_tf_op_exports_own_config(tmpdir):
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(name="input", dtype=tf.int32, shape=(784,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, name="output"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    # Triton
    triton_op = tf_op.PredictTensorflow(model)
    triton_op.export(tmpdir, None, None)

    # Export creates directory
    export_path = pathlib.Path(tmpdir) / triton_op.export_name
    assert export_path.exists()

    # Export creates the config file
    config_path = export_path / "config.pbtxt"
    assert config_path.exists()

    # Read the config file back in from proto
    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)

        # The config file contents are correct
        assert parsed.name == triton_op.export_name
        assert parsed.backend == "tensorflow"


def test_tf_op_compute_schema():
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(name="input", dtype=tf.int32, shape=(784,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, name="output"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    # Triton
    triton_op = tf_op.PredictTensorflow(model)

    out_schema = triton_op.compute_output_schema(Schema(["input"]), ColumnSelector(["input"]), None)
    assert out_schema.column_names == ["output"]


def test_tf_schema_validation():
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(name="input", dtype=tf.int32, shape=(784,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, name="output"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    # Triton
    tf_node = [] >> tf_op.PredictTensorflow(model)
    tf_graph = Graph(tf_node)

    with pytest.raises(ValueError) as exception_info:
        deepcopy(tf_graph).construct_schema(Schema([]))
    assert "Missing column 'input'" in str(exception_info.value)

    with pytest.raises(ValueError) as exception_info:
        deepcopy(tf_graph).construct_schema(Schema(["not_input"]))
    assert "Missing column 'input'" in str(exception_info.value)

    with pytest.raises(ValueError) as exception_info:
        deepcopy(tf_graph).construct_schema(Schema(["input", "not_input"]))
    assert "Mismatched dtypes for column 'input'" in str(exception_info.value)


def test_tf_op_infers_schema_for_input_tuples():
    inputs = (tf.keras.Input(shape=(128,)), tf.keras.Input(shape=(128,)))
    towers = (tf.keras.layers.Dense(64)(inputs[0]), tf.keras.layers.Dense(64)(inputs[1]))
    outputs = tf.keras.layers.dot(towers, [1, 1])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    # Triton
    triton_op = tf_op.PredictTensorflow(model)
    assert triton_op.input_schema == Schema(
        [
            ColumnSchema(
                name="input_1",
                tags=set(),
                properties={"value_count": {"min": 128, "max": 128}},
                dtype=np.dtype("float32"),
                is_list=True,
                is_ragged=False,
            ),
            ColumnSchema(
                "input_2",
                properties={"value_count": {"min": 128, "max": 128}},
                dtype=np.float32,
                is_list=True,
                is_ragged=False,
            ),
        ]
    )
    assert triton_op.output_schema == Schema(
        [ColumnSchema("dot", dtype=np.float32, is_list=False, is_ragged=False)]
    )
