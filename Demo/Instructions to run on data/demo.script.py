# 1. Standard library imports
import os
import shutil
import time
import json

# 2. Third-party imports
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import torch.optim as optim
import GP_WAITER_demo as gp

    
def log_training_info(logfile, epoch, epoch_loss, corr_train, corr_test):
    log_entry = {
        "epoch": epoch,
        "epoch_loss":float(epoch_loss),
        "corr_train":float(corr_train),
        "corr_test": float(corr_test)
    }
    with open(logfile, 'a') as f:
        json.dump(log_entry, f)
        f.write('\n')  # 换行分隔每条记录


def log_best_results(summary_logfile, phe_s, best_test_corr, best_epoch):

    summary_entry = {
        "phe_s": phe_s,
        "best_test_corr": float(best_test_corr),  
        "best_epoch": best_epoch  
    }
    with open(summary_logfile, 'a') as f:
        json.dump(summary_entry, f)
        f.write('\n')

def transform(line_gen):
    line_gen = line_gen.strip('\n').split(',')
    # 解析数据并转换为 NumPy 数组
    g = np.array([float(x) for x in line_gen[1:]], dtype=np.float32).reshape(112,597)

    return g

class EarlyStopping:
   
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.params'):
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
        
def train(phe_s,root_path, divide, num_epochs, batch_size, lr, summary_logfile):
    
    # 检查GPU是否可以使用
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda:0')
    print(f"CUDA Available: {USE_CUDA}, Device: {DEVICE}")
    
    filepath_gen = f"{root_path}/demo.genotype.{phe_s}.txt"
    filepath_phe = f"{root_path}/demo.phenotype.{phe_s}.csv"
    filepath_site = f"{root_path}/demo.weighted.{phe_s}.csv"

    # 读取表型文件
    phe = pd.read_csv(filepath_phe, index_col=0).to_numpy()
    phe = phe / divide
    print(pd.DataFrame(phe[:5]))
    print(f"表型文件 {phe_s} 的形状:{phe.shape}")

    # 读取权重文件
    site = pd.read_csv(filepath_site)
    site = site["c2"].to_numpy()
    SiteScore = np.log10(1/site).reshape(112,597)#180, 4589  112,597
    SiteScore = torch.tensor(SiteScore, dtype=torch.float32).to(DEVICE)
    print(f"权重文件的形状:{SiteScore.shape}")

    # 读取基因型文件
    g_array=[]
    # 逐行读取文件并处理
    with open(filepath_gen, "r") as f:
        #next(f)  # 跳过表头行
        for line in f:
            gen = transform(line)
            g_array.append(gen)

    g_array = np.array(g_array)
    print(f'基因型文件的形状:{g_array.shape}')

    xtrain, xtest, ytrain, ytest = train_test_split(g_array, phe, test_size=0.2, random_state=100)
    # 随机打乱后 进行数据的划分
    
    print("The shape of xtrain, ytrain, xtest, ytest datasets:")
    print(f"xtrain shape: {xtrain.shape}")
    print(f"ytrain shape: {ytrain.shape}")
    print(f"xtest shape: {xtest.shape}")
    print(f"ytest shape: {ytest.shape}")

    # 构建参数目录
    params_path = os.path.join('parameters', phe_s)
    print('params_path:', params_path)
    # 检查目录是否存在，并根据情况删除旧目录或者创建新目录
    if os.path.exists(params_path):
        shutil.rmtree(params_path)  # 删除目录
        print(f'Deleted old params directory: {params_path}')

    os.makedirs(params_path, exist_ok=True)  # 创建新目录
    print(f'Created new params directory: {params_path}')

    #some parameters
    param=[
        {"embed_size1":198,"embed_size2":150,"num_heads":9},
        {"embed_size1":150,"embed_size2":100,"num_heads":10},
        {"embed_size1":100,"embed_size2":20,"num_heads":5}
        ]
        
    # 初始化模型对象，传递设置好的参数，还没有开始训练
    model=gp.TModel(embed_size=20,w=SiteScore,param=param,num_layers=3).to(DEVICE)

    criterion = nn.MSELoss()
    Optimizer = optim.Adam(model.parameters(), lr=lr)

    
    # 将数据转换为Tensor的统一处理
    xtrain, ytrain, xtest, ytest = map(torch.Tensor, [xtrain, ytrain, xtest, ytest])
    #组合成训练数据集并加载
    train_dataset=Data.TensorDataset(xtrain,ytrain)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    #组合成测试数据集并加载    
    test_dataset = Data.TensorDataset(xtest,ytest)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, drop_last=False, shuffle=False)
    
    # 初始化单独日志文件
    logfile = os.path.join(params_path, f"{phe_s}_training_log.json")
    os.makedirs(params_path, exist_ok=True)
    
    with open(logfile, 'w') as f:
        f.write('')  # 清空旧日志内容

    #### 训练集部分
    best_test_corr = None  # 初始化最佳验证相关系数
    best_epoch = None      # 初始化最佳 epoch
    best_outputs = []  # 存储最佳模型下的所有预测值
    best_labels = []   # 存储最佳模型下的所有真实值
    
    early_stopping = EarlyStopping(patience=5, verbose=True, path=params_path+'/best_model.params')
    for epoch in range(num_epochs):
        model.train()
        train_labels = []         # 每个通道的真实标签
        train_outputs = []        # 每个通道的预测输出
        epoch_loss = 0.0
        for batch_index, (train_data, train_label) in enumerate(train_loader):

            if torch.cuda.is_available():
                train_data = train_data.cuda()
                train_label = train_label.cuda()

            output = model(train_data)  # output: (batch_size, 3)
            print(f'output:{output.shape}')
            train_label = train_label.squeeze()
            print(f'train_lable:{train_label.shape}')

            loss = criterion(output, train_label)
            epoch_loss = epoch_loss + loss

            # 统一反向传播和优化
            Optimizer.zero_grad()
            loss.backward()  #此步开始进行反向传播
            Optimizer.step()  #此步进行参数的更新
            
            # 存储预测值和真实标签
            train_outputs.append(output.detach().cpu().numpy())
            train_labels.append(train_label.detach().cpu().numpy())

            print(f"Epoch: {epoch}, batch_index: {batch_index}, Loss: {loss:.5f}")
        
        train_outputs = np.concatenate(train_outputs, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        print(f'the shape of train_outputs:{train_outputs.shape}')
        print(f'the shape of train_labels:{train_labels.shape}')
        corr_train = np.corrcoef(train_outputs, train_labels)[0, 1]
        print(f"Training correlation in Epoch:{epoch} is {corr_train}")


        ##### 测试集部分
        test_outputs = []
        test_labels = []
        val_loss=0
        with torch.no_grad():
            model.eval()
            for batch_index, (test_data, test_label) in enumerate(test_loader):
                if torch.cuda.is_available():
                    test_data = test_data.cuda()

                test_output = model(test_data)  # test_output: (batch_size, 3)
                
                test_loss = criterion(test_output, test_label.cuda())
                val_loss += test_loss.item()
                test_label = test_label.squeeze()
                # 累积测试数据
                test_outputs.append(test_output.cpu().numpy())
                test_labels.append(test_label.cpu().numpy())
        
            test_outputs = np.concatenate(test_outputs, axis=0)  # 合并所有的预测值
            test_labels = np.concatenate(test_labels, axis=0)  # 合并所有的真实标签  
            print(f'the shape of test_outputs:{test_outputs.shape}')
            print(f'the shape of test_labels:{test_labels.shape}')
            corr_test = np.corrcoef(test_outputs, test_labels)[0, 1]
            print(f"Testing correlation in Epoch:{epoch} is {corr_test}")
        
        
        # 初始化 best_model_path 只需要做一次
        best_model_path = os.path.join(params_path, f'best_{phe_s}.params')

        # 只有当测试集的相关系数更高时才保存模型
        if best_test_corr is None or corr_test > best_test_corr:
            best_test_corr = corr_test
            best_epoch = epoch

            # 如果存在旧的最佳模型，删除旧文件
            if os.path.exists(best_model_path):
                os.remove(best_model_path)

            # 保存当前最佳模型
            #torch.save(model.state_dict(), best_model_path)

            # 记录最佳模型下测试集所有预测值和真实值
            best_outputs = test_outputs
            best_labels = test_labels
            print(f"New best test correlation: {best_test_corr} at epoch {best_epoch}")

        # 记录训练日志
        log_training_info(logfile, epoch, epoch_loss, corr_train, corr_test)
        val_loss /= len(test_loader.dataset)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    # 在汇总日志中记录最佳结果
    log_best_results(summary_logfile, phe_s,  best_epoch, best_test_corr)
    print(f"Training complete for {phe_s}. Best test correlation: {best_test_corr} at epoch {best_epoch}.")

    # 输出最佳模型下的所有预测值和真实值（保存为 CSV 文件）
    best_output_file = os.path.join(params_path, f'best_{phe_s}_predictions.csv')

    # 确保路径存在
    os.makedirs(os.path.dirname(best_output_file), exist_ok=True)

    # 获取测试集预测值和真实值的数量
    num_predictions = len(best_outputs)
    num_true_labels = len(best_labels)

    # 创建一个包含数量信息的 DataFrame
    header_data = pd.DataFrame([['Number of predictions', num_predictions], ['Number of true labels', num_true_labels]], columns=['Metric', 'Count'])

    # 使用 pandas 保存数量信息（覆盖文件）
    header_data.to_csv(best_output_file, index=False)

    # 创建测试集的形状信息
    test_shape = pd.DataFrame([['ytrain shape', str(ytrain.shape)], ['ytest shape', str(ytest.shape)]], columns=["Metric", "Shape"])

    # 追加测试集的形状信息
    test_shape.to_csv(best_output_file, mode='a', header=True, index=False)

    # 使用 pandas 保存预测值和真实值
    best_data = np.column_stack((best_outputs, best_labels))
    best_df = pd.DataFrame(best_data, columns=['Predictions', 'True_Labels'])

    # 追加预测值和真实值
    best_df.to_csv(best_output_file, mode='a', header=True, index=False)

    print(f"Best model predictions saved to: {best_output_file}")




if __name__ == '__main__':
    
    phe_list = ["O"]
    num_epochs = 200
    batch_size = 32
    lr = 0.001
    summary_logfile = 'best_results_summary.json'  # 汇总日志文件
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    # 各文件路径
    root_path = "./demo_data" ### Please modify the file path
    divide = 100
    # 记录任务开始时间
    with open(summary_logfile, 'a') as f:
        f.write(f"--- Model running started at {current_time} ---\n")

    for phe_s in phe_list:
        print(f"Training for phe_s: {phe_s}" )
        train(phe_s, root_path, divide, num_epochs, batch_size, lr, summary_logfile)
        print(f"Finished training for phe_s: {phe_s}\n")
