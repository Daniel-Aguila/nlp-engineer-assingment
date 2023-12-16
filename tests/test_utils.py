import json
from urllib import request
import numpy as np
from main import tokenize
from nlp_engineer_assignment.api import getPredictionFromText
import pytest
from nlp_engineer_assignment.utils import count_letters, score
from fastapi.testclient import TestClient
from nlp_engineer_assignment import api


client = TestClient(api.app)

def test_count_letters():
    assert np.array_equal(count_letters("hello"), np.array([0, 0, 0, 1, 0]))
    assert np.array_equal(count_letters("world"), np.array([0, 0, 0, 0, 0]))
    assert np.array_equal(
        count_letters("hello hello"),
        np.array([0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1])
    )


def test_score():
    assert score(np.array([[0, 1, 1, 0, 1]]),
                 np.array([[0, 1, 1, 0, 1]])) == 1.0
    assert score(np.array([[0, 1, 1, 0, 1]]),
                 np.array([[1, 1, 0, 0, 1]])) == 0.6
    assert score(np.array([[0, 1, 1, 0, 1]]),
                 np.array([[0, 0, 0, 0, 0]])) == 0.4
    assert score(np.array([[0, 1, 1, 0, 1]]),
                 np.array([[1, 0, 0, 1, 0]])) == 0.0

def test_tokenize_for_incorrect_character():
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    sequence = "hello!"
    with pytest.raises(ValueError):
        tokenize(sequence,vocab)

def test_tokenize_for_character_to_number_conversion():
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    sequence = "this is a test"
    assert tokenize(sequence,vocab) == [19,7,8,18,26,8,18,26,0,26,19,4,18,19]

def test_getPredictionFromText_correct_response_given_request():
    request = {
        "text": "yaraku is a japanese"
    }
    prediction = {
        "prediction": "00010000012202020011"
    }
    response = client.post("/predictText", json=request)
    assert response.json() == prediction

def test_getPredictionFromText_status_code():
    request = {
        "text": "yaraku is a japanese"
    }
    response = client.post("/predictText", json=request)
    assert response.status_code == 200

