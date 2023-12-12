from cmath import cos, sin
import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    TODO: You should implement the Transformer model from scratch here. You can
    use elementary PyTorch layers such as: nn.Linear, nn.Embedding, nn.ReLU, nn.Softmax, etc.
    DO NOT use pre-implemented layers for the architecture, positional encoding or self-attention,
    such as nn.TransformerEncoderLayer, nn.TransformerEncoder, nn.MultiheadAttention, etc.
    """
    def __init__(self,output_dimension=512,inner_layer=2048):
        super(Transformer,self).__init__()
        #This facilitates the residual connection. All of the model sub-layers and embedding layers produce outputs of dimension dmodel = 512
        self.embedding = nn.Embedding(num_embeddings=output_dimension, embedding_dim=output_dimension)
        self.v_linear = nn.Linear(in_features=output_dimension,out_features=output_dimension)
        self.k_linear = nn.Linear(in_features=output_dimension,out_features=output_dimension)
        self.q_linear = nn.Linear(in_features=output_dimension,out_features=output_dimension)
        self.final_self_attention_linear = nn.Linear(in_features=output_dimension,out_features=output_dimension)

        self.linear1 = nn.Linear(in_features=output_dimension,out_features=inner_layer)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=inner_layer,out_features=output_dimension)

        self.gamma = nn.Parameter(torch.ones(output_dimension))
        self.beta = nn.Parameter(torch.zeros(output_dimension))

    def PositionalEncoding(self,embeddings):
        #Since we are adding Positional Encoding to the embeddings, I figure I can just iterate through the embeddings themselves, without
        #having to create a separate tensor
        max_sequence_length = embeddings.shape[0]
        dimensions = embeddings.shape[1]
        positional_embeddings = embeddings.clone() #to keep names meaningful
        #TODO Study Pytorch to implement a more optimized way to apply the equations without a double for loop.
        #This is really slow with long sequences and high dimensions
        for position in range(max_sequence_length):
            for i in range(dimensions):
                if i % 2 == 0:
                    #real extracts the real part of the complex number
                    positional_embeddings[position,i] += sin(position / (10000**(2*i/dimensions))).real #for even dimensions
                else:
                    positional_embeddings[position,i] += cos(position / (10000**(2*i/dimensions))).real #for odd dimensions
        return positional_embeddings

    def ScaledDotProductAttention(self,query,key,value,dkey=512):
        key_transposed = torch.transpose(key,0,1) #Transposed the Key_dimension
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
        x3 = self.LayerNormalization(x1 + x2,beta=self.beta,gamma=self.gamma)

        #feed forward block
        x4 = self.linear1(x3)
        x4 = self.relu(x4)
        x4 = self.linear2(x4)
        #Add + Norm
        x5 = self.LayerNormalization(x3 + x4,beta=self.beta,gamma=self.gamma)

        return x5

def train_classifier(train_inputs,train_labels, epochs=10):
    # TODO: Implement the training loop for the Transformer model.
    tensor_inputs = torch.tensor(train_inputs)
    tensor_labels = torch.tensor(train_labels)
    dmodel = 512
    warmup_steps = 4000
    step_num = 0

    cross_entropy_loss = nn.CrossEntropyLoss()
    model = Transformer()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0,betas=(0.9,0.98))

    for epoch in range(epochs):
        for index,input in enumerate(tensor_inputs):
            step_num += 1
            lrate_ = (dmodel**-0.5) * min(step_num**-.5,step_num*warmup_steps**-0.5) 
            for param_group in optimizer.param_groups: #update learning rate
                param_group['lr'] = lrate_
            optimizer.zero_grad()

            outputs = model(input) #get output for the currect batch inputs
            loss = cross_entropy_loss(outputs,tensor_labels[index]) #calculate loss
            print(loss)
            loss.backward()
            optimizer.step()
    raise NotImplementedError(
        "You should implement `train_classifier` in transformer.py"
    )
