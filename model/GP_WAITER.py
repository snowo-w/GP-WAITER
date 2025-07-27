from torch import nn
import torch
class PatchEmbedding(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.conv0=nn.Conv2d(1,1,(61,101),(1,1),(30,50))
        # self.conv0 = nn.Conv2d(1, 1, (90, 101), (1, 4))
        self.conv1 = nn.Conv2d(1, 1, (90, 101), (1, 4))
        self.conv2 = nn.Conv2d(1, 1, (60, 101), (1, 2))
        self.conv3 = nn.Conv2d(1, 1, (1, 101), (1, 1))
        self.conv4 = nn.Conv2d(1, 1, (1, 101), (1, 1))
        self.sigmoid = nn.Sigmoid()
        self.batchnorm2d = nn.BatchNorm2d(1)
        self.w=w
        self.learned_w =nn.parameter(torch.rand(10))
        
    def forward(self,x):
        x = x.unsqueeze(1)
        l0=torch.mul(x,self.w).permute(0,2,3,1)
        l0=torch.matmul(l0,self.w).permute(0,3,1,2)
        l1 = self.batchnorm2d(self.conv1(l0))
        l2 = self.sigmoid(self.conv2(l1))
        print("l2.shape:", l2.shape)
        l3 = self.sigmoid(self.conv3(l2))
        print("l3.shape:",l3.shape)
        l4=self.sigmoid(self.conv4(l3))
        l4=self.batchnorm2d(l4)
       
        return l4.squeeze()

class AttentionBlock(nn.Module):
    def __init__(self,embed_size1,embed_size2,num_heads):
        super().__init__()
        self.layer_norm_1=nn.LayerNorm(embed_size1)
        self.attn=nn.MultiheadAttention(embed_size1,num_heads,batch_first=True)
        self.layer_norm_2=nn.LayerNorm(embed_size1)
        self.linear_transformer=nn.Sequential(
            nn.Linear(embed_size1, embed_size2),
            nn.Sigmoid()
        )
        self.linear=nn.Sequential(
            nn.Linear(embed_size1,embed_size2),
            nn.Sigmoid(),
            # nn.Dropout(drop_p),
            nn.Linear(embed_size2,embed_size2)
        )
    def forward(self,x):    
        x=self.layer_norm_1(x)
        x=x+self.attn(x,x,x)[0]
        weight=self.attn(x,x,x)[1]
        print("weight.shape:",weight.shape)
        x=self.linear_transformer(x)+self.linear(self.layer_norm_2(x))
        return x


class TModel(nn.Module):
    def __init__(self,embed_size,w,param,num_layers):
        super().__init__()
        self.patchembed=PatchEmbedding(w)
        self.layers=nn.Sequential(*(AttentionBlock(embed_size1=param[i]["embed_size1"],embed_size2=param[i]["embed_size2"],num_heads=param[i]["num_heads"]) for i in range(num_layers)))
        self.out=nn.Sequential(nn.LayerNorm(embed_size),nn.Linear(embed_size,1),nn.Sigmoid())
        self.out_r=nn.Sequential(nn.Conv1d(1,1,20,2),nn.Sigmoid(),nn.Conv1d(1,1,7,1),nn.Sigmoid())
        self.ln=nn.Linear(3,1)
        self.s=nn.Sigmoid()
        self.batchnorm1d = nn.BatchNorm1d(1)
    def forward(self,x):
        output=self.out(self.layers(self.patchembed(x))).squeeze()


        return self.out_r(self.batchnorm1d(output.unsqueeze(1))).squeeze()
