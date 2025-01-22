# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import importlib
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Mapping, Protocol, cast

from haystack import logging
from haystack.core.errors import ComponentError
from haystack.core.component.sockets import Sockets
from haystack.core.component.types import OutputSocket, InputSocket
from haystack.core.component.component import ComponentMeta

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PydanticComponent(Protocol):
    """Same as haystack.core.component.Component, but with run method (possibly) returning a Pydantic model"""

    def run(self, *args: Any, **kwargs: Any) -> BaseModel | dict[str, Any]: ...


class HaystackPydanticComponentMeta(ComponentMeta):
    @staticmethod
    def _parse_and_set_output_sockets(instance: Any):
        has_async_run = hasattr(instance, "run_async")

        # If `component.set_output_types()` was called in the component constructor,
        # `__haystack_output__` is already populated, no need to do anything.
        if not hasattr(instance, "__haystack_output__"):
            # If that's not the case, we need to populate `__haystack_output__`
            #
            # If either of the run methods were decorated, they'll have a field assigned that
            # stores the output specification. If both run methods were decorated, we ensure that
            # outputs are the same. We deepcopy the content of the cache to transfer ownership from
            # the class method to the actual instance, so that different instances of the same class
            # won't share this data.

            run_func: Callable | None = getattr(instance, "run", None)
            if run_func is None:
                raise ComponentError(
                    f"Component {instance.__class__.__name__} must have a 'run' method. See the docs for more information."
                )

            # we priortize the pydantic output type specified in the run method signature
            # if it's not available, we fallback to the output_types decorator
            output_type = inspect.signature(run_func).return_annotation
            output_sockets: Mapping[str, InputSocket | OutputSocket]
            if issubclass(output_type, BaseModel):
                # set with the return type of the run method
                output_sockets = {
                    field_name: OutputSocket(
                        name=field_name, type=field_info.annotation
                    )
                    for field_name, field_info in output_type.model_fields.items()
                }
            elif output_type == inspect.Signature.empty:
                # the ones set with `output_types` decorator
                run_output_types = getattr(instance.run, "_output_types_cache", {})
                async_run_output_types = (
                    getattr(instance.run_async, "_output_types_cache", {})
                    if has_async_run
                    else {}
                )

                if has_async_run and run_output_types != async_run_output_types:
                    raise ComponentError(
                        "Output type specifications of 'run' and 'run_async' methods must be the same"
                    )
                output_types_cache = run_output_types

                output_sockets = cast(
                    dict[str, OutputSocket], deepcopy(output_types_cache)
                )
            else:
                raise ComponentError(
                    f"Output type must extend pydantic BaseModel if specified as return type signature, but got {output_type}"
                )

            instance.__haystack_output__ = Sockets(
                instance, cast(dict[str, InputSocket | OutputSocket], output_sockets), OutputSocket
            )


# patch the ComponentMeta class in the haystack.core.component.component module
component_module = importlib.import_module("haystack.core.component.component")
component_module.ComponentMeta = HaystackPydanticComponentMeta  # type: ignore
