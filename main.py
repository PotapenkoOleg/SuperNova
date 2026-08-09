import logging
from contextlib import asynccontextmanager
from copy import copy

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

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


class VotePredictRequest(BaseModel):
    input_strs: list[str] = Field(..., min_length=1)
    soft_vote: bool = False


class BulkPredictRequest(BaseModel):
    input_strs: list[str] = Field(..., min_length=1)

@app.get('/')
async def root():
    return Response(status_code=200) # makes load balancer happy

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

@app.post('/bulk_predict/')
async def bulk_predict(bulk_predict_request: BulkPredictRequest):
    input_strs = bulk_predict_request.input_strs
    x = np.stack([preprocess_string(input_str) for input_str in input_strs])
    model = ml_models['supernova']
    predictions = model.predict(x, verbose=0)
    max_proba_indices = np.argmax(predictions, axis=1)
    return [
        {
            'input-str': input_str,
            'predicted-class': CLASS_NAMES[int(max_proba_index)],
            'probability': float(prediction[max_proba_index])
        }
        for input_str, prediction, max_proba_index
        in zip(input_strs, predictions, max_proba_indices)
    ]

@app.post('/vote_predict/')
async def vote_predict(vote_predict_request: VotePredictRequest):
    input_strs = vote_predict_request.input_strs
    soft_vote = vote_predict_request.soft_vote
    x = np.stack([preprocess_string(input_str) for input_str in input_strs])
    model = ml_models['supernova']
    predictions = model.predict(x, verbose=0)
    prob_sums = predictions.sum(axis=0)
    votes = np.bincount(np.argmax(predictions, axis=1), minlength=len(CLASS_NAMES))
    # Whichever criterion is primary decides the winner; the other one breaks ties
    primary, secondary = (prob_sums, votes) if soft_vote else (votes, prob_sums)
    tied = [i for i in CLASS_NAMES if primary[i] == primary.max()]
    winner = max(tied, key=lambda i: (secondary[i], -i))
    return {
        'predicted-class': CLASS_NAMES[winner],
        'soft-vote': soft_vote,
        'sample-count': len(input_strs),
        'votes': {CLASS_NAMES[i]: int(votes[i]) for i in CLASS_NAMES},
        'probability-sums': {CLASS_NAMES[i]: float(prob_sums[i]) for i in CLASS_NAMES},
        'tie-break-used': len(tied) > 1
    }

@app.get('/classes')
async def classes():
    return list(CLASS_NAMES.values())


@app.get('/version')
async def version():
    return {'Product Name': TITLE, 'Version': VERSION}
