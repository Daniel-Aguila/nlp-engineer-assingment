import torch
import torch.nn as nn

from nlp_engineer_assignment.utils import score


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
        positional_embeddings = embeddings.clone() #to keep names meaningful
        #TODO Study Pytorch to implement a more optimized way to apply the equations without a double for loop.
        #This is really slow with long sequences and high dimensions
        for position in range(max_sequence_length):
            for i in range(dimensions):
                if i % 2 == 0:
                    positional_embeddings[:,position,i] += torch.sin(position / (10000**(2*i/dimensions))) #for even dimensions
                else:
                    positional_embeddings[:,position,i] += torch.cos(position / (10000**(2*i/dimensions))) #for odd dimensions
        return positional_embeddings

    def ScaledDotProductAttention(self,query,key,value,dkey=64):

        key_transposed = torch.transpose(key,1,2) #Transposed the Key_dimension
        scores = torch.matmul(query,key_transposed) #Perform the the MatMul layer from the paper to create the scores
        scaled_scores = scores/(dkey**0.5) #Scale down the scores by a sqrt of the dimension of key
        attention_weights = torch.nn.Softmax(scaled_scores) #Apply Softmax layer
        attention_output = torch.matmul(attention_weights,value) #Final MatMul layer between the Value and Final product between Query and Key
        
        return attention_output

    def MultiHeadAttention(self):
        #TODO Implement parallelism of the ScaledDotProductAttention layers
        return

def train_classifier(train_inputs):
    # TODO: Implement the training loop for the Transformer model.
    raise NotImplementedError(
        "You should implement `train_classifier` in transformer.py"
    )
