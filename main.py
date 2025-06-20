import logging
from contextlib import asynccontextmanager
from copy import copy

import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

TITLE = "Supernova"
DESCRIPTION = "Neural Network for predicting data types from raw strings eg csv, json etc"
VERSION = "1.0.0"

ml_models = {}

CLASS_NAMES = {
    0: 'int',
    1: 'float',
    2: 'boolean',
    3: 'time',
    4: 'date',
    5: 'datetime',
    6: 'uuid',
    7: 'string'
}

VOCAB_SIZE: int = 128
MAX_LENGTH: int = 100


def preprocess_string(input_str: str) -> np.ndarray:
    input_str = input_str[:MAX_LENGTH - 1].strip().upper()
    encoded = [ord(c) % VOCAB_SIZE for c in input_str]
    if len(encoded) < MAX_LENGTH:
        encoded.extend([0] * (MAX_LENGTH - len(encoded)))
    return np.array(encoded)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
    model = tf.keras.models.load_model('model.keras')
    ml_models['supernova'] = model
    yield
    ml_models.clear()


app = FastAPI(
    title=TITLE,
    description=DESCRIPTION,
    version=VERSION,
    lifespan=lifespan
)


class PredictRequest(BaseModel):
    input_str: str


@app.post('/predict/')
async def predict(predict_request: PredictRequest):
    input_str = copy(predict_request.input_str)
    x = preprocess_string(predict_request.input_str)
    x = np.expand_dims(x, axis=0)
    model = ml_models['supernova']
    predictions = model.predict(x)[0]
    max_proba_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[max_proba_index]
    probability = float(predictions[max_proba_index])
    return {
        'input-str': input_str,
        'predicted-class': predicted_class,
        'probability': probability
    }


@app.get('/classes')
async def classes():
    return list(CLASS_NAMES.values())


@app.get('/version')
async def version():
    return {'Product Name': TITLE, 'Version': VERSION}
