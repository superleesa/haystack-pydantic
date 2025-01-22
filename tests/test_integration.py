from pydantic import BaseModel

from haystack_pydantic import Pipeline, component


class SampleOutput1(BaseModel):
    attr1: str
    attr2: int


class SampleOutput2(BaseModel):
    attr1: str


@component
class SampleComponent1:
    def run(self) -> SampleOutput1:
        return SampleOutput1(attr1="sample1", attr2=1)


@component
class SampleComponent2:
    def run(
        self,
        input1: str,
    ) -> SampleOutput2:
        return SampleOutput2(
            attr1=input1,
        )


def test_pipeline_run() -> None:
    pipeline = Pipeline()
    pipeline.add_component(name="sample_comp1", instance=SampleComponent1())
    pipeline.add_component(name="sample_comp2", instance=SampleComponent2())
    pipeline.connect("sample_comp1.attr1", "sample_comp2.input1")

    output = pipeline.run(data={"input1": "sample_input"})
    assert output == {
        "sample_comp1": SampleOutput1(attr1="sample1", attr2=1),
        "sample_comp2": SampleOutput2(attr1="sample1"),
    }


def test_component_run() -> None:
    component = SampleComponent1()
    output = component.run()
    assert output == SampleOutput1(attr1="sample1", attr2=1)
