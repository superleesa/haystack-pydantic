# haystack-pydantic

A thin wrapper around haystack Component and Pipeline for typing (outputs) with Pydantic models

```shell
pip install haystack-pydantic
```

```python
from haystack_pydantic import Pipeline
from haystack_pydantic import component
from pydantic import BaseModel

class TestOutput(BaseModel):
    output1: str
    output2: str

@component
class TestComponent:
    # define pydantic model as output type (instead of the `output_types` decorator)
    def run(self, input1: str, input2: str) -> TestOutput:  
        return TestOutput(output1=input1, output2=input2)  # return the pydantic model

# use the component standalone
test_component = TestComponent()
test_component.run("input1", "input2")  # TestOutput(output1="input1", output2="input2")

# use the component within a pipeline
pipeline = Pipeline()
pipeline.add_component("test_component", test_component)
pipeline.run(data={"input1": "input1", "input2": "input2"})  # {"test_component": TestOutput(output1="input1", output2="input2")}
```