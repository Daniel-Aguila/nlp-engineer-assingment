from turtle import forward
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
    def __init__(self,input_size,output_dimension=512,inner_layer=2048):
        super(Transformer,self).__init__()
        #This facilitates the residual connection. All of the model sub-layers and embedding layers produce outputs of dimension dmodel = 512
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=output_dimension)
        self.v_linear = nn.Linear(in_features=output_dimension,out_features=output_dimension)
        self.k_linear = nn.Linear(in_features=output_dimension,out_features=output_dimension)
        self.q_linear = nn.Linear(in_features=output_dimension,out_features=output_dimension)
        self.final_self_attention_linear = nn.Linear(in_features=output_dimension,out_features=output_dimension)

        self.linear1 = nn.Linear(in_features=input_size,out_features=inner_layer)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=inner_layer,out_features=output_dimension)

        self.gamma = nn.Parameter(torch.ones(output_dimension))
        self.beta = nn.Parameter(torch.zeros(output_dimension))

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

    def ScaledDotProductAttention(self,query,key,value,dkey=512):

        key_transposed = torch.transpose(key,1,2) #Transposed the Key_dimension
        scores = torch.matmul(query,key_transposed) #Perform the the MatMul layer from the paper to create the scores
        scaled_scores = scores/(dkey**0.5) #Scale down the scores by a sqrt of the dimension of key
        attention_weights = torch.nn.Softmax(dim=-1)(scaled_scores) #Apply Softmax to the sequence dimension
        attention_output = torch.matmul(attention_weights,value) #Final MatMul layer between the Value and Final product between Query and Key
        
        return attention_output

    #TODO Implement Multi-Head if there is available time and if model does not perform as well

    def LayerNormalization(self,tensor,beta,gamma):
        epsilon = 1e-9
        mean = tensor.mean()
        variance = tensor.var()

        norm = (tensor-mean) / (torch.sqrt(variance) + epsilon)
        output = (gamma * norm) + beta

        return output

    def forward(self,x):
        #positional embedding
        x = self.embedding(x)
        x1 = self.PositionalEncoding(x)

        #self attetion, single head
        qx = self.q_linear(x1)
        kx = self.k_linear(x1)
        vx = self.v_linear(x1)

        x2 = self.ScaledDotProductAttention(qx,kx,vx)
        x2 = self.final_self_attention_linear(x2)
        #Add + Norm
        x3 = self.LayerNormalization(x1 + x2)

        #feed forward block
        x4 = self.linear1(x3)
        x4 = self.relu(x4)
        x4 = self.linear2(x4)
        #Add + Norm
        x5 = self.LayerNormalization(x3 + x4)

        return x5

def train_classifier(train_inputs):
    # TODO: Implement the training loop for the Transformer model.
    raise NotImplementedError(
        "You should implement `train_classifier` in transformer.py"
    )
