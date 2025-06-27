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

VOTING_TYPE_HARD = 'hard'
VOTING_TYPE_SOFT = 'soft'


class VotingClassifier:
    def __init__(self, voting_type: str):
        if voting_type not in [VOTING_TYPE_HARD, VOTING_TYPE_SOFT]:
            raise ValueError(f"Voting type must be either '{VOTING_TYPE_HARD}' or '{VOTING_TYPE_SOFT}'")
        self.voting_type = voting_type

    @staticmethod
    def predict(input_str: str):
        x = preprocess_string(input_str)
        x = np.expand_dims(x, axis=0)
        model = ml_models['supernova']
        predictions = model.predict(x)[0]
        max_proba_index = int(np.argmax(predictions))
        predicted_class = max_proba_index
        probability = float(predictions[max_proba_index])
        return predicted_class, probability

    def bulk_predict(self, input: list[list[str]]):
        results = []
        for column in input:
            predicted_classes = []
            probabilities = []
            for row in column:
                predicted_class, probability = self.predict(row)
                predicted_classes.append(predicted_class)
                probabilities.append(probability)
            np_classes = np.array(predicted_classes)
            np_probabilities = np.array(probabilities)
            if self.voting_type == VOTING_TYPE_HARD:
                unique_elements, counts = np.unique(np_classes, return_counts=True)
                most_common_index = np.argmax(counts)
                most_common_element = unique_elements[most_common_index]
                results.append(most_common_element)
            if self.voting_type == VOTING_TYPE_SOFT:
                unique_elements = np.unique(np_classes)
                probability_sums = []
                for element in unique_elements:
                    mask = np_classes == element
                    probability_sum = np.sum(np_probabilities[mask])
                    probability_sums.append(probability_sum)
                max_prob_index = np.argmax(probability_sums)
                results.append(unique_elements[max_prob_index])
        return results


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
    predicted_class, probability = VotingClassifier.predict(input_str)
    return {
        'input-str': input_str,
        'predicted-class': CLASS_NAMES[predicted_class],
        'probability': probability
    }


@app.post('/bulk-predict/')
async def bulk_predict(predict_request: list[PredictRequest]):
    input_strings = [[req.input_str for req in predict_request]]
    classifier = VotingClassifier(voting_type=VOTING_TYPE_SOFT)
    predicted_classes = classifier.bulk_predict(input_strings)
    return [{'predicted-class': CLASS_NAMES[pred_class]} for pred_class in predicted_classes]


@app.get('/classes')
async def classes():
    return list(CLASS_NAMES.values())


@app.get('/version')
async def version():
    return {'Product Name': TITLE, 'Version': VERSION}
