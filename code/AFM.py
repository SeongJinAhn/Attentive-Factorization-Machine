import torch
import torch.nn as nn
import torch.nn.functional as F

class afm(nn.Module):
    def __init__(self,config,num_user):
        super(afm,self).__init__()

        self.ToTensor = torch.LongTensor
        self.GPU = config.GPU_available
        if self.GPU:
            self.ToTensor = torch.cuda.LongTensor
    
        self.person_embed = nn.Embedding(num_user,config.embedding_dim)
        self.prediction = nn.Linear(3,1)
        self.attention= Attention(config.embedding_dim)

    def forward(self,group,config):
        gr_len = len(group)
        gr_embed = []
        a = torch.FloatTensor()
        if config.GPU_available:
            a = torch.cuda.FloatTensor()

        for x in group:
            gr_embed.append(self.person_embed(self.ToTensor(x)))

        for gr in gr_embed:
            b = torch.FloatTensor()
            if config.GPU_available:
                b = torch.cuda.FloatTensor()
    
            for i, v_i in enumerate(gr):
                for v_j in gr[i+1:]:
                    v_ij = torch.mul(v_i,v_j)
                    a_ij = self.attention(v_ij)

                    b = torch.cat((b,a_ij))
            a = torch.cat((a,b.reshape(1,3)),dim=0)
        a = self.prediction(a)
        a = nn.Sigmoid()(a) * 2 -1
#        a = nn.Softmax(dim=0)(a.flatten())
        return a
        

    
class Attention(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(Attention, self).__init__()
        self.L1 = nn.Linear(embedding_dim,4)
        self.L2 = nn.Linear(4,1)


    def forward(self, x):
        out = self.L1(x)
        out = nn.Sigmoid()(out)
        out = self.L2(out)
        return out