import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import time
from model import GP_WAITER as gp
import shutil
import os
import torch
from torch import nn
import torch.optim as optim
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class EarlyStopping:
   
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): 验证损失没有提升的容忍 epoch 数，之后停止
            verbose (bool): 若为 True，每次保存最佳模型时打印信息
            delta (float): 最小变化，小于该值视为无提升
            path (str): 模型保存路径
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0           # 记录连续未提升的 epoch 数
        self.best_score = None      # 最佳验证指标（如损失）
        self.early_stop = False     # 是否触发早停
        self.val_loss_min = np.Inf   # 当前最小验证损失

    def __call__(self, val_loss, model):
        score = -val_loss   # 损失越小越好，取负后越大表示越好

        if self.best_score is None:
            # 第一次调用，直接保存模型
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # 没有提升（score 没有大于 best_score + delta）
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 有提升，更新最佳分数，重置计数器，保存模型
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存当前最佳模型（验证损失最小）"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def transform(line_gen):
    line_gen = line_gen.strip('\n')
    splited_line_gen = line_gen.split(',')
    g = list(map(lambda x: float(x), splited_line_gen[1:]))
    g = np.array(g).reshape(180, 4589)
    g = g.astype(np.float32)
    return g


def train(phe_s,num_epochs,batch_size,lr):
    filepath_gen = "/data/rest/rest1/" + phe_s + "_gen.txt"
    filepath_phe = "/data/rest/rest1/" + phe_s + "_phe.csv"
   
    phe=pd.read_csv(filepath_phe)
  
    phe=phe[phe_s]/100
  
    g_array=[]

    site=pd.read_csv("/data/rest/random_train_data/train_score/"+phe_s+"_sitescore.csv")
    env_SiteScore=site.iloc[:, 2].to_numpy()
  
    env_SiteScore = env_SiteScore.reshape(180, 4589)
 
    with open(filepath_gen, "r") as f:
        line = f.readline()
        while line:      
            gen=transform(line)
            g_array.append(gen)
            line = f.readline()
        
    print(len(g_array))
    print(np.array(g_array).shape)


    xtrain, xtest, ytrain, ytest = train_test_split(g_array, phe, test_size=0.2, random_state=100)
    xtrain=np.array(xtrain)
    ytrain=np.array(ytrain)
    xtest=np.array(xtest)
    ytest=np.array(ytest)
    print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)

    params_path = os.path.join('parameters', "T_CNN_SiteScore", phe_s+'_80w_8layers_'+str(batch_size)+'b_0.001lr')
    print('params_path:', params_path)
    if (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    else:
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:0')
    # DEVICE = torch.device('cpu')
    print("CUDA:", USE_CUDA, DEVICE)
    param=[{"embed_size1":312,"embed_size2":260,"num_heads":12},{"embed_size1":260,"embed_size2":100,"num_heads":10},{"embed_size1":100,"embed_size2":20,"num_heads":5}]
    env_SiteScore=torch.Tensor(env_SiteScore).to(DEVICE)
    model=gp.TModel(embed_size=20,w=env_SiteScore,param=param,num_layers=3)
  
    if torch.cuda.is_available():
        model.cuda()  

    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    Optimizer = optim.Adam(model.parameters(), lr=lr)

    sw = SummaryWriter(log_dir=params_path, flush_secs=5)
    xtrain=torch.Tensor(xtrain)
    ytrain = torch.Tensor(ytrain)
    xtest = torch.Tensor(xtest)
    ytest = torch.Tensor(ytest)
    train_dataset=Data.TensorDataset(xtrain,ytrain)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=1, drop_last=True,
                                   shuffle=True)
    test_dataset = Data.TensorDataset(xtest,ytest)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=1, drop_last=False,
                                   shuffle=True)
    early_stopping = EarlyStopping(patience=5, verbose=True, path=params_path+'/best_model.params')
    for epoch in range(num_epochs):
        #time_epoch_start = time.time()
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch) 

        t_labels = []
        t_outputs = []
        train_loss=0.0
        for batch_index, (train_data, train_label) in enumerate(train_loader):
            if torch.cuda.is_available():
                train_data = train_data.cuda()
                train_label = train_label.squeeze().cuda()
            # input_data = train_data.view(train_data.size(0), -1)
                output = model(train_data)
                loss = criterion(output, train_label)
                train_loss += loss.item()
                print('output:',output.shape)
                print('train_label:',train_label.shape)

                a = output.to('cpu')
                t_outputs.append(a.detach().numpy())
                b = train_label.cpu()
                t_labels.append(b.detach().numpy())
                
                Optimizer.zero_grad()
                loss.backward()
                Optimizer.step()

                print('Epoch: {}, batch_index:{},Loss: {:.5f}'.format(epoch, batch_index, loss))


        t_outputs = np.concatenate(t_outputs, axis=0)
   
        t_labels = np.concatenate(t_labels, axis=0)
     
        m = np.corrcoef(t_outputs, t_labels)
        print("correlation coefficient of train:", m)
        train_loss /= len(train_loader.dataset)
        sw.add_scalar("training loss per epoch",  train_loss, epoch)
        sw.add_scalar("correlation coefficient of training_dataset per epoch", m[0, 1], epoch)
        torch.save(model.state_dict(), params_filename)
        test_prediction=[]
        test_all_labels=[]
        with torch.no_grad():
            model.eval()
            val_loss=0.0
            for batch_index, (test_data, test_label) in enumerate(test_loader):
                if torch.cuda.is_available():
                    test_data = test_data.cuda()

                    test_output = model(test_data)
                    test_loss = criterion(test_output, test_label)
                    val_loss += test_loss.item()
            
                    a = test_output.to('cpu').numpy()
                    test_prediction.append(a)
                    b = test_label.cpu().numpy()
                    test_all_labels.append(b)
                print("batch_index=======",batch_index)
                print("a.shape,b.shape:",a.shape,b.shape)
            val_loss /= len(test_loader.dataset)
            test_outputs = np.concatenate(test_prediction, axis=0)

            test_labels = np.concatenate(test_all_labels, axis=0)

            print("test_outputs.shape,test_labels.shape:",test_outputs.shape,test_labels.shape)
            m = np.corrcoef(test_outputs, test_labels)
            print("correlation coefficient of test:", m)
            sw.add_scalar("correlation coefficient of testing dataset per epoch", m[0, 1], epoch)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break



if __name__ == '__main__':
    train(phe_s="O.H17",num_epochs=200,batch_size=32,lr=0.001)
