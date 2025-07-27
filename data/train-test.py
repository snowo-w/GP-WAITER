import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import time
from model.GP_WAITER import TModel
import shutil
import os
import torch
from torch import nn
import torch.optim as optim
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

phenotype='O.H17'
num_epochs = 200
batch_size = 16
lr = 0.001
num_samples=1257
embed_size=90
num_layers=2
num_heads=10

def transform(line_gen):
    line_gen = line_gen.strip('\n')
    splited_line_gen = line_gen.split(',')
    g = list(map(lambda x: float(x), splited_line_gen[1:]))
    g = np.array(g).reshape(180, 4589)
    g = g.astype(np.float32)
    return g


def train(phe_s,num_epochs,batch_size,lr):
    filepath_gen = "/home/o/projects/geno-pheno-ViTransformer/data/rest/rest1/" + phe_s + "_gen.txt"
    filepath_phe = "/home/o/projects/geno-pheno-ViTransformer/data/rest/rest1/" + phe_s + "_phe.csv"
   
    phe=pd.read_csv(filepath_phe)
    # phe=phe.fillna(phe.mean(axis=1))
    # numerator=phe.sub(phe.min(axis=1),axis=0)
    # denominator=
    # phe=(phe-phe.min())/(phe.max()-phe.min())
    phe=phe[phe_s]/100
    # print("phe",np.array(phe))
    g_array=[]

    site=pd.read_csv("data/sites_score/sitescore_"+phe_s+".csv")
    env_SiteScore=site.values#(10,826020)
  
    env_SiteScore = env_SiteScore.reshape(180, 4589)
    #print("SiteScore.shape:",SiteScore.shape)
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
    model=TModel(embed_size=20,w=env_SiteScore,param=param,num_layers=3)
    # model.load_state_dict(torch.load("parameters/"+phe_s+"/"+phe_s+"_80w_8layers_64b_0.001lr/epoch_99.params"))
    if torch.cuda.is_available():
        model.cuda()  # 注:将模型放到GPU上,因此后续传入的数据必须也在GPU上

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
    # global_step = 0
    for epoch in range(num_epochs):
        testing_loss = 0.0
        time_epoch_start = time.time()
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
        train_loss = 0.0

        t_labels = []
        t_outputs = []
        for batch_index, (train_data, train_label) in enumerate(train_loader):
            if torch.cuda.is_available():
                train_data = train_data.cuda()
                train_label = train_label.squeeze().cuda()
            # input_data = train_data.view(train_data.size(0), -1)
                output = model(train_data)

                print('output:',output.shape)
                print('train_label:',train_label.shape)

                a = output.to('cpu')
                t_outputs.append(a.detach().numpy())
                b = train_label.cpu()
                t_labels.append(b.detach().numpy())

                loss = criterion(output, train_label)
                # corr=torch.corrcoef(output,train_label)
                train_loss = train_loss + loss
                Optimizer.zero_grad()
                loss.backward()
                Optimizer.step()

                print('Epoch: {}, batch_index:{},Loss: {:.5f}'.format(epoch, batch_index, loss))


        t_outputs = np.concatenate(t_outputs, axis=0)
        # print('train-prediction-outputs=======================', np.std(t_outputs))
        t_labels = np.concatenate(t_labels, axis=0)
     
        m = np.corrcoef(t_outputs, t_labels)
        print("correlation coefficient of train:", m)
        sw.add_scalar("training loss per epoch", train_loss / (batch_index + 1), epoch)
        sw.add_scalar("correlation coefficient of training_dataset per epoch", m[0, 1], epoch)
        torch.save(model.state_dict(), params_filename)
        test_prediction=[]
        test_all_labels=[]
        with torch.no_grad():
            for batch_index, (test_data, test_label) in enumerate(test_loader):
                if torch.cuda.is_available():
                    test_data = test_data.cuda()

                    test_output = model(test_data)
                    a = test_output.to('cpu').numpy()
                    test_prediction.append(a)
                    b = test_label.cpu().numpy()
                    test_all_labels.append(b)
                print("batch_index=======",batch_index)
                print("a.shape,b.shape:",a.shape,b.shape)
            test_outputs = np.concatenate(test_prediction, axis=0)
            # to_std=np.std(test_outputs)
            # print('test_prediction_outputs=======================', to_std)
            test_labels = np.concatenate(test_all_labels, axis=0)

            print("test_outputs.shape,test_labels.shape:",test_outputs.shape,test_labels.shape)
            m = np.corrcoef(test_outputs, test_labels)
            print("correlation coefficient of test:", m)
            sw.add_scalar("correlation coefficient of testing dataset per epoch", m[0, 1], epoch)



if __name__ == '__main__':#TIF.B17_80w_5layers_0.3d_16b_0.001lr_continue :10heads
    train(phe_s="O.H17",num_epochs=200,batch_size=32,lr=0.001)
