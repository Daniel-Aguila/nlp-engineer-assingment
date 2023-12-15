import numpy as np
import os
import uvicorn
import torch

from nlp_engineer_assignment import count_letters, print_line, read_inputs, \
    score, train_classifier, model_predict

def tokenize(sequence: str, vocab: list):
    token_list = []
    for char in sequence:
        if char not in vocab:
            raise ValueError("Please make sure all the characters in the input string are a-z or space.")
        else:
            token_list.append(vocab.index(char))
    return token_list

def preprocess(sequences: list[str]):
    vocabs = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    
    preprocessed_sequences = []
    for sequence in sequences:
        sequence_lower = sequence.lower()
        tokenize_sequence = tokenize(sequence_lower,vocabs)
        preprocessed_sequences.append(tokenize_sequence)
    
    #it is faster turning np.arrays to tensors, than a list to a tensor
    preprocessed_sequences = np.array(preprocessed_sequences)
    return preprocessed_sequences

def train_model():
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    ###
    # Setup
    ###

    # Constructs the vocabulary as described in the assignment

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
    train_labels = np.array(train_labels)

    #preprocess the train inputs by tokenizing with the vocab
    preprocessed_train_inputs = preprocess(train_inputs)
 
    model = train_classifier(preprocessed_train_inputs,train_labels)

    ###
    # Test
    ###
    test_inputs = read_inputs(
        os.path.join(cur_dir, "data", "test.txt")
    )

    #preprocess the test inputs by tokenizing with the vocab
    preprocessed_test_inputs = preprocess(test_inputs)
 
    #get predictions with the preprocessed test inputs
    golds = np.stack([count_letters(text) for text in test_inputs])
    
    predictions = model_predict(model,preprocessed_test_inputs)
    torch.save(model, 'model.pth')


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