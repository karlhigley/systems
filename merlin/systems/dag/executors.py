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
import logging

from merlin.core.dispatch import concat_columns, is_list_dtype, list_val_dtype
from merlin.dag.executors import LocalExecutor

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


resource = Resource.create(attributes={
    SERVICE_NAME: "merlin-systems"
})
provider = TracerProvider(resource=resource)
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

processor = BatchSpanProcessor(jaeger_exporter)
provider.add_span_processor(processor)

LOG = logging.getLogger("merlin-systems")
TRACER = trace.get_tracer("instrumented_executor")



# def OpenTelemetry(func):
#     with TRACER.start_as_current_span(f"{func.__class__.__name__}):
        


class InstrumentedExecutor:
    """
    An executor for running Merlin operator DAGs locally with tracing
    """

    def __init__(self, transform_method=""):
        self.transform_method = transform_method

    def transform(
        self,
        transformable,
        nodes,
        output_dtypes=None,
        additional_columns=None,
        capture_dtypes=False,
    ):
        """
        Transforms a single dataframe (possibly a partition of a Dask Dataframe)
        by applying the operators from a collection of Nodes
        """
        with TRACER.start_as_current_span(f"{self.__class__.__name__}.transform()"):
            return self._transform_impl(
                transformable,
                nodes,
                output_dtypes=output_dtypes,
                additional_columns=additional_columns,
                capture_dtypes=capture_dtypes,
            )



    def _transform_impl(
        self,
        transformable,
        nodes,
        output_dtypes=None,
        additional_columns=None,
        capture_dtypes=False,
    ):
        """
        Transforms a single dataframe (possibly a partition of a Dask Dataframe)
        by applying the operators from a collection of Nodes
        """
        output_data = None

        for node in nodes:
            input_data = self._build_input_data(node, transformable, capture_dtypes=capture_dtypes)

            if node.op:
                transformed_data = self._transform_data(
                    node, input_data, capture_dtypes=capture_dtypes
                )
            else:
                transformed_data = input_data

            output_data = self._combine_node_outputs(node, transformed_data, output_data)

        if additional_columns:
            output_data = concat_columns(
                [output_data, transformable[_get_unique(additional_columns)]]
            )

        return output_data

    def _build_input_data(self, node, transformable, capture_dtypes=False):
        """
        Recurse through the graph executing parent and dependency operators
        to form the input dataframe for each output node
        Parameters
        ----------
        node : Node
            Output node of the graph to execute
        transformable : Transformable
            Dataframe to run the graph ending with node on
        capture_dtypes : bool, optional
            Overrides the schema dtypes with the actual dtypes when True, by default False
        Returns
        -------
        Transformable
            The input DataFrame or DictArray formed from
            the outputs of upstream parent/dependency nodes
        """
        node_input_cols = _get_unique(node.input_schema.column_names)
        addl_input_cols = set(node.dependency_columns.names)

        if node.parents_with_dependencies:
            # If there are parents, collect their outputs
            # to build the current node's input
            input_data = None
            seen_columns = None

            for parent in node.parents_with_dependencies:
                parent_output_cols = _get_unique(parent.output_schema.column_names)
                parent_data = self._transform_impl(transformable, [parent], capture_dtypes=capture_dtypes)
                if input_data is None or not len(input_data):
                    input_data = parent_data[parent_output_cols]
                    seen_columns = set(parent_output_cols)
                else:
                    new_columns = set(parent_output_cols) - seen_columns
                    input_data = concat_columns([input_data, parent_data[list(new_columns)]])
                    seen_columns.update(new_columns)

            # Check for additional input columns that aren't generated by parents
            # and fetch them from the root DataFrame or DictArray
            unseen_columns = set(node.input_schema.column_names) - seen_columns
            addl_input_cols = addl_input_cols.union(unseen_columns)

            # TODO: Find a better way to remove dupes
            addl_input_cols = addl_input_cols - set(input_data.columns)

            if addl_input_cols:
                input_data = concat_columns([input_data, transformable[list(addl_input_cols)]])
        else:
            # If there are no parents, this is an input node,
            # so pull columns directly from root data
            if addl_input_cols:
                addl_input_cols = list(addl_input_cols)
            else:
                addl_input_cols = []
            input_data = transformable[node_input_cols + addl_input_cols]

        return input_data

    def _transform_data(self, node, input_data, capture_dtypes=False):
        """
        Run the transform represented by the final node in the graph
        and check output dtypes against the output schema
        Parameters
        ----------
        node : Node
            Output node of the graph to execute
        input_data : Transformable
            Dataframe to run the graph ending with node on
        capture_dtypes : bool, optional
            Overrides the schema dtypes with the actual dtypes when True, by default False
        Returns
        -------
        Transformable
            The output DataFrame or DictArray formed by executing the final node's transform
        Raises
        ------
        TypeError
            If the transformed output columns don't have the same dtypes
            as the output schema columns
        RuntimeError
            If no DataFrame or DictArray is returned from the operator
        """
        try:
            # use input_columns to ensure correct grouping (subgroups)
            selection = node.input_columns.resolve(node.input_schema)

            span_name = f"{node.op.__class__.__name__}.transform([{selection.names}])"
            with TRACER.start_as_current_span(span_name):
                output_data = getattr(node.op, self.transform_method, node.op.transform)(
                    selection, input_data
                )
                # output_data = node.op.transform(selection, input_data)

            # Update or validate output_data dtypes
            for col_name, output_col_schema in node.output_schema.column_schemas.items():
                col_series = output_data[col_name]
                col_dtype = col_series.dtype  # tf.int32
                is_list = is_list_dtype(col_series)

                if is_list:
                    col_dtype = list_val_dtype(col_series)
                if hasattr(col_dtype, "as_numpy_dtype"):
                    col_dtype = col_dtype.as_numpy_dtype()
                elif hasattr(col_series, "numpy"):
                    col_dtype = col_series[0].cpu().numpy().dtype

                # col_dtype = convert_to_merlin_dtype(col_dtype)
                output_data_schema = output_col_schema.with_dtype(
                    col_dtype, is_list=is_list, is_ragged=is_list
                )

                if capture_dtypes:
                    node.output_schema.column_schemas[col_name] = output_data_schema
                elif len(output_data):
                    if output_col_schema.dtype != output_data_schema.dtype:
                        raise TypeError(
                            f"Dtype discrepancy detected for column {col_name}: "
                            f"operator {node.op.label} reported dtype "
                            f"`{output_col_schema.dtype}` but returned dtype "
                            f"`{output_data_schema.dtype}`."
                        )
        except Exception:
            LOG.exception("Failed to transform operator %s", node.op)
            raise
        if output_data is None:
            raise RuntimeError(f"Operator {node.op} didn't return a value during transform")

        return output_data

    def _combine_node_outputs(self, node, transformed_data, output):
        node_output_cols = _get_unique(node.output_schema.column_names)

        # dask needs output to be in the same order defined as meta, reorder partitions here
        # this also selects columns (handling the case of removing columns from the output using
        # "-" overload)
        if output is None:
            output = transformed_data[node_output_cols]
        else:
            output = concat_columns([output, transformed_data[node_output_cols]])

        return output

class BeamExecutor(LocalExecutor):
    """
    An executor for running Merlin operator DAGs on Beam
    """

    def __init__(self, transform_method=""):
        self.transform_method = transform_method

    def _transform_data(self, node, input_data, capture_dtypes=False):
        """
        Run the transform represented by the final node in the graph
        and check output dtypes against the output schema
        Parameters
        ----------
        node : Node
            Output node of the graph to execute
        input_data : Transformable
            Dataframe to run the graph ending with node on
        capture_dtypes : bool, optional
            Overrides the schema dtypes with the actual dtypes when True, by default False
        Returns
        -------
        Transformable
            The output DataFrame or DictArray formed by executing the final node's transform
        Raises
        ------
        TypeError
            If the transformed output columns don't have the same dtypes
            as the output schema columns
        RuntimeError
            If no DataFrame or DictArray is returned from the operator
        """
        try:
            # use input_columns to ensure correct grouping (subgroups)
            selection = node.input_columns.resolve(node.input_schema)
            output_data = getattr(node.op, self.transform_method, node.op.transform)(
                selection, input_data
            )

            # # Update or validate output_data dtypes
            # for col_name, output_col_schema in node.output_schema.column_schemas.items():
            #     col_series = output_data[col_name]
            #     col_dtype = col_series.dtype 
            #     is_list = is_list_dtype(col_series)

            #     if is_list:
            #         col_dtype = list_val_dtype(col_series)
            #     if hasattr(col_dtype, "as_numpy_dtype"):
            #         col_dtype = col_dtype.as_numpy_dtype()
            #     elif hasattr(col_series, "numpy"):
            #         col_dtype = col_series[0].cpu().numpy().dtype

            #     # col_dtype = convert_to_merlin_dtype(col_dtype)
            #     output_data_schema = output_col_schema.with_dtype(
            #         col_dtype, is_list=is_list, is_ragged=is_list
            #     )

            #     if capture_dtypes:
            #         node.output_schema.column_schemas[col_name] = output_data_schema

        except Exception:
            LOG.exception("Failed to transform operator %s", node.op)
            raise
        if output_data is None:
            raise RuntimeError(f"Operator {node.op} didn't return a value during transform")

        return output_data

    def _combine_node_outputs(self, node, transformed_data, output):
        node_output_cols = _get_unique(node.output_schema.column_names)

        # dask needs output to be in the same order defined as meta, reorder partitions here
        # this also selects columns (handling the case of removing columns from the output using
        # "-" overload)
        if output is None:
            output = transformed_data[node_output_cols]
        else:
            output = concat_columns([output, transformed_data[node_output_cols]])

        return output


def _get_unique(cols):
    # Need to preserve order in unique-column list
    return list({x: x for x in cols}.keys())

