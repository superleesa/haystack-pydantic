import importlib
from typing import cast

from haystack.core.component.component import _Component

# NOTE: we can't import directly because `component` exists as another variable
# (which represents Component object, rather than the component module) in the __init__.py file
component_module = importlib.import_module("haystack.core.component.component")

from haystack_pydantic.core.component import (  # noqa: E402 F401
    component as __,
)  # ComponentMeta will be patched, as side effect

component: _Component = cast(_Component, component_module.component)
