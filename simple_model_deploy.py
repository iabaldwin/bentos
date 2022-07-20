from typing import List
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

def simple_model(inputs: NumpyNdarray) -> int:
    return np.sum(inputs)

# `save_model` saves a given python object or function
saved_model = bentoml.picklable_model.save_model(
    'simple_model',
    simple_model,
    signatures={"__call__": {"batchable": True}}
)
print(f"Model saved: {saved_model}")
loaded_model = bentoml.picklable_model.load_model("simple_model:latest")
