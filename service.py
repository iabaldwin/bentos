import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

simple_runner = bentoml.picklable_model.get("simple_model:latest").to_runner()

svc = bentoml.Service("simple_model", runners=[simple_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = simple_runner.run(input_series)
    return result
