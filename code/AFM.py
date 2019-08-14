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
            self.device = torch.device('cuda')
    
        self.person_embed = nn.Embedding(num_user+1,config.embedding_dim)
        self.prediction = nn.Linear(config.embedding_dim,1)
        self.attention= Attention(config.embedding_dim)

        self.p = nn.Parameter(torch.rand(1,config.embedding_dim))
        self.first = nn.Embedding(num_user,1)
        self.gen_bias = nn.Parameter(torch.rand(1))

    def forward(self,group,config):
        first = self.FirstOrder(group,config)
        second = self.AttentiveSecond(group,config)
        asd = nn.Sigmoid()(first+second)*2-1
        return asd

    def FirstOrder(self,group,config):
        return torch.sum(self.first(group),dim=1).reshape(-1)

    def AttentiveSecond(self,group,config):
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
    
            v_ij=torch.Tensor()
            a_ij=torch.Tensor()

            if config.GPU_available :
                v_ij=v_ij.to(self.device)
                a_ij=a_ij.to(self.device)   

            for i, v_i in enumerate(gr):
                for v_j in gr[i+1:]:
                    element_wise = torch.mul(v_i,v_j)
                    v_ij = torch.cat((v_ij,element_wise.reshape(-1,1)),dim=-1)
                    a_ij = torch.cat((a_ij,self.attention(element_wise)))

            v_ij = v_ij.reshape(config.embedding_dim,-1)
            a_hat_ij = nn.Softmax(dim=0)(a_ij) # relative impact

            product = torch.mm(self.p,torch.mul(v_ij,a_hat_ij)).sum()
            a = torch.cat((a,product.reshape(1)))
#        a = nn.Softmax(dim=0)(a.flatten())
        return a
        

    
class Attention(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(Attention, self).__init__()
        self.L1 = nn.Linear(embedding_dim,30)
        self.L2 = nn.Linear(30,1)


    def forward(self, x):
        out = self.L1(x)
        out = nn.Sigmoid()(out)
        out = self.L2(out)
        out = nn.Sigmoid()(out)
        return out
