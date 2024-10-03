from torch import nn
import torch.nn.functional as F

class JokeEvaluationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(JokeEvaluationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        

        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = F.relu(self.embedding(text, offsets))
        
        x = F.relu(self.fc(embedded))
        
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)

