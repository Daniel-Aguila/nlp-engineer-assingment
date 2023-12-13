import numpy as np
import os
import uvicorn

from nlp_engineer_assignment import count_letters, print_line, read_inputs, \
    score, train_classifier, model_predict

def tokenize(sequence: str, vocab: np.array):
    token_list = [vocab.index(char) for char in sequence]
    return token_list

def preprocess(sequences: str, vocabs: np.array):
    preprocessed_sequence = []
    for sequence in sequences:
        tokenize_sequence = tokenize(sequence,vocabs)
        preprocessed_sequence.append(tokenize_sequence)
    return preprocessed_sequence

def train_model():
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    ###
    # Setup
    ###

    # Constructs the vocabulary as described in the assignment
    vocabs = [chr(ord('a') + i) for i in range(0, 26)] + [' ']

    ###
    # Train
    ###

    train_inputs = read_inputs(
        os.path.join(cur_dir, "data", "train.txt")
    )

    #Get train labels
    train_labels = []
    for train_input in train_inputs:
        train_labels.append(count_letters(train_input))

    #preprocess the train inputs by tokenizing with the vocab
    preprocessed_train_inputs = preprocess(train_inputs,vocabs)
 
    model = train_classifier(preprocessed_train_inputs,train_labels)

    ###
    # Test
    ###
    test_inputs = read_inputs(
        os.path.join(cur_dir, "data", "test.txt")
    )

    #preprocess the test inputs by tokenizing with the vocab
    preprocessed_test_inputs = preprocess(test_inputs,vocabs)
 
    #get predictions with the preprocessed test inputs
    golds = np.stack([count_letters(text) for text in test_inputs])
    predictions = model_predict(model,preprocessed_test_inputs)

    # Print the first five inputs, golds, and predictions for analysis
    for i in range(5):
        print(f"Input {i+1}: {test_inputs[i]}")
        print(
            f"Gold {i+1}: {count_letters(test_inputs[i]).tolist()}"
        )
        print(f"Pred {i+1}: {predictions[i].tolist()}")
        print_line()

    print(f"Test Accuracy: {100.0 * score(golds, predictions):.2f}%")
    print_line()


if __name__ == "__main__":
    train_model()
    uvicorn.run(
        "nlp_engineer_assignment.api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1
    )