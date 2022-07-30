import json
import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2

from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.operator import InferenceDataFrame
from merlin.systems.dag.ops.softmax_sampling import SoftmaxSampling
from nvtabular import ColumnSelector


def test_softmax_output_dtype_keeps_input_dtype():
    # We expect the method to not change the output dtype

    s = SoftmaxSampling("rel_col", _input_cols=["input_col"])

    input_col_schema = ColumnSchema(
        name="input_col", dtype=pd.StringDtype, is_list=False, is_ragged=False
    )
    input_schema = Schema([ColumnSchema(name="rel_col"), input_col_schema])

    actual = s.compute_output_schema(input_schema, ColumnSelector(["input_col"]))

    assert actual == Schema([input_col_schema])


@pytest.mark.parametrize("dtype", [np.float32, pd.StringDtype])
@pytest.mark.parametrize(
    "input_cols", [["input_col1"], ["input_col2"], ["input_col1", "input_col2"]]
)
def test_softmax_output_dtype__with_multiple_inputs_keeps_input_dtype(input_cols, dtype):
    # We expect the method to not change the output dtype

    s = SoftmaxSampling("rel_col", _input_cols=input_cols)

    input_col_schema = [
        ColumnSchema(name=input_col, dtype=dtype, is_list=False, is_ragged=False)
        for input_col in input_cols
    ]

    input_schema = Schema([ColumnSchema(name="rel_col")] + input_col_schema)

    actual = s.compute_output_schema(input_schema, ColumnSelector(input_cols))
    assert actual == Schema(input_col_schema)


@pytest.mark.parametrize(
    "input_cols", [["input_col1"], ["input_col2"], ["input_col1", "input_col2"]]
)
def test_softmax_ordering(input_cols):
    s = SoftmaxSampling("rel_col", _input_cols=input_cols)
    tensors = {col: np.random.uniform(0, 1, 3) for col in input_cols}
    tensors.update({"rel_col": np.array([1, 0.1, 0.01])})
    df = InferenceDataFrame(tensors)

    transformed = s.transform(df)
    assert sorted(transformed.tensors.keys()) == sorted(input_cols)


def test_softmax_export(tmpdir):
    input_col = "input"
    s = SoftmaxSampling("rel_col", _input_cols=input_col)

    input_schema = Schema([ColumnSchema(name=input_col), ColumnSchema(name="sel_col")])
    output_schema = s.compute_output_schema(input_schema, col_selector=ColumnSelector(["sel_col"]))

    returned_config = s.export(tmpdir, input_schema, output_schema)

    expected_config_text = """
name: "softmaxsampling"
platform: "op_runner"
input {
  name: "input"
  data_type: TYPE_FP64
  dims: -1
  dims: -1
}
input {
  name: "sel_col"
  data_type: TYPE_FP64
  dims: -1
  dims: -1
}
output {
  name: "sel_col"
  data_type: TYPE_FP64
  dims: -1
  dims: -1
}
parameters {
  key: "operator_names"
  value {
    string_value: "[\\"softmaxsampling\\"]"
  }
}
parameters {
  key: "softmaxsampling"
  value {
    string_value: "{\\"module_name\\": \\"merlin.systems.dag.ops.softmax_sampling\\", \\"class_name\\": \\"SoftmaxSampling\\", \\"input_dict\\": \\"{\\\\\\"input\\\\\\": {\\\\\\"dtype\\\\\\": \\\\\\"float64\\\\\\", \\\\\\"is_list\\\\\\": false, \\\\\\"is_ragged\\\\\\": false}, \\\\\\"sel_col\\\\\\": {\\\\\\"dtype\\\\\\": \\\\\\"float64\\\\\\", \\\\\\"is_list\\\\\\": false, \\\\\\"is_ragged\\\\\\": false}}\\", \\"output_dict\\": \\"{\\\\\\"sel_col\\\\\\": {\\\\\\"dtype\\\\\\": \\\\\\"float64\\\\\\", \\\\\\"is_list\\\\\\": false, \\\\\\"is_ragged\\\\\\": false}}\\", \\"params\\": \\"{\\\\\\"input_cols\\\\\\": \\\\\\"input\\\\\\", \\\\\\"relevance_col\\\\\\": \\\\\\"rel_col\\\\\\", \\\\\\"temperature\\\\\\": 20.0, \\\\\\"topk\\\\\\": 10}\\"}"
  }
}
backend: "python"
"""  # noqa
    expected_config = text_format.Parse(expected_config_text, model_config_pb2.ModelConfig())

    with open(
        os.path.join(tmpdir, "softmaxsampling", "config.pbtxt"), "r", encoding="utf-8"
    ) as fin:
        actual_config = text_format.Parse(fin.read(), model_config_pb2.ModelConfig())
    assert expected_config == actual_config
    assert returned_config == actual_config


@pytest.mark.parametrize(
    "input_cols", [["input_col1"], ["input_col2"], ["input_col1", "input_col2"]]
)
def test_softmax_compute_input_schema(input_cols):
    s = SoftmaxSampling("relevance_col")

    parents_schema = Schema(
        [ColumnSchema(name=input_col) for input_col in input_cols]
        + [ColumnSchema(name="relevance_col")]
    )
    root_schema = Schema([ColumnSchema(name="root_col")])
    deps_schema = Schema([ColumnSchema(name="deps_col")])
    s.compute_input_schema(
        root_schema, parents_schema, deps_schema, ColumnSelector(names=["relevance_col"])
    )

    assert s._input_col_names == input_cols + ["relevance_col"]
    assert s.relevance_col.selector.names == ["relevance_col"]

    # TODO: these two seem wrong.
    assert s._relevance_col_name == "deps_col"
    assert s.dependencies.selector.names == ["relevance_col"]


@pytest.mark.parametrize("input_cols", ["input_col1", ["input_col1"], ["input_col1", "input_col2"]])
def test_softmax_from_config(input_cols):
    parameters = {
        "relevance_col": "rel_col",
        "input_col": input_cols,
        "temperature": 10.0,
        "topk": 2,
    }
    config = {"params": json.dumps(parameters)}
    SoftmaxSampling.__init__ = MagicMock(return_value=None)
    s = SoftmaxSampling.from_config(config)

    if isinstance(input_cols, str):
        input_cols = [input_cols]
    s.__init__.assert_called_once_with("rel_col", temperature=10.0, topk=2, _input_cols=input_cols)
