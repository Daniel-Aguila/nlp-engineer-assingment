import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    TODO: You should implement the Transformer model from scratch here. You can
    use elementary PyTorch layers such as: nn.Linear, nn.Embedding, nn.ReLU, nn.Softmax, etc.
    DO NOT use pre-implemented layers for the architecture, positional encoding or self-attention,
    such as nn.TransformerEncoderLayer, nn.TransformerEncoder, nn.MultiheadAttention, etc.
    """
    def __init__(self,input_size,output_dimension=512):
        super(Transformer,self).__init__()
        #This facilitates the residual connection. All of the model sub-layers and embedding layers produce outputs of dimension dmodel = 512
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=output_dimension)

    def PositionalEncoding(self,embeddings):
        #Since we are adding Positional Encoding to the embeddings, I figure I can just iterate through the embeddings themselves, without
        #having to create a separate tensor
        max_sequence_length = embeddings.shape[1]
        dimensions = embeddings.shape[2]
        positional_embeddings = embeddings.clone()
        for position in range(max_sequence_length):
            for i in range(dimensions):
                if i % 2 == 0:
                    positional_embeddings[:,position,i] += torch.sin(position / (10000**(2*i/dimensions))) #for even dimensions
                else:
                    positional_embeddings[:,position,i] += torch.cos(position / (10000**(2*i/dimensions))) #for odd dimensions
        return positional_embeddings


def train_classifier(train_inputs):
    # TODO: Implement the training loop for the Transformer model.
    raise NotImplementedError(
        "You should implement `train_classifier` in transformer.py"
    )
