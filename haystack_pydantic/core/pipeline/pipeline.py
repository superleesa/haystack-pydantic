import inspect
from contextlib import ExitStack
from typing import Any, Callable, TYPE_CHECKING, Protocol

from haystack import Pipeline
from pydantic import BaseModel

from haystack_pydantic.core.component.component import LenientComponent


def create_run_method(original_run_method: Callable[..., BaseModel]) -> Callable:
    def run_and_return_typeddict(*args, **kwargs) -> dict[str, Any]:
        pydantic_output = original_run_method(*args, **kwargs)
        return {
            field_name: getattr(pydantic_output, field_name)
            for field_name in pydantic_output.model_fields.keys()
        }

    return run_and_return_typeddict


class AddComponentSig(Protocol):
    def __call__(self, name: str, instance: LenientComponent) -> None:
        ...


class PydanticWrappedPipeline(Pipeline):
    if TYPE_CHECKING:
        add_component: AddComponentSig  # type: ignore
    
    
    def _convert_to_model(
        self, component_name: str, data: dict[str, Any]
    ) -> BaseModel | dict:
        component = self.graph.nodes[component_name]["instance"]
        model = inspect.signature(component.run).return_annotation
        if not issubclass(model, BaseModel):
            return data
        return model(**data)

    def patch_back_to_modeled_run_methods(
        self, component_name_to_original_run_method: dict[str, Callable]
    ) -> None:
        for (
            component_name,
            original_run_method,
        ) in component_name_to_original_run_method.items():
            self.graph.nodes[component_name]["instance"].run = original_run_method

    def run(
        self, data: dict[str, Any], include_outputs_from: set[str] | None = None
    ) -> dict[str, BaseModel | dict]:
        # patch run methods of all components to return dict instead of Pydantic models (because Pipeline internally uses dict)
        component_name_to_modeled_run_method: dict[str, Callable] = {}
        for node_name, node_data in self.graph.nodes(data=True):
            component_name, component = node_name, node_data["instance"]

            original_run_method = component.run
            original_run_method_return_type = inspect.signature(
                original_run_method
            ).return_annotation
            if issubclass(original_run_method_return_type, BaseModel):
                component.run = create_run_method(original_run_method)
                component_name_to_modeled_run_method[component_name] = (
                    original_run_method
                )

        # the default behaviour of `include_outputs_from`=None would only return leaf outputs in the pipeline graph,
        # which can cause some components to have partial outputs
        # (e.g. even though there is attr1 and attr2, attr2 can be missing if it is used by another component),
        # leading to model parsing errors
        # so to avoid this, we include all components in the output, if `include_outputs_from` is not provided
        if include_outputs_from is None:
            include_outputs_from = set(self.graph.nodes)  # all components

        with ExitStack() as stack:
            # patch back to the original run methods, to avoid side effects
            # this is done in a stack, so that the original run methods are patched back even if an exception occurs
            stack.callback(
                self.patch_back_to_modeled_run_methods,
                component_name_to_modeled_run_method,
            )
            output = super().run(data, include_outputs_from=include_outputs_from)

        # dict => pydantic model
        return {
            component_name: self._convert_to_model(component_name, component_output)
            for component_name, component_output in output.items()
        }
