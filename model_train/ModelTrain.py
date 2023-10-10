'''
作者：yxl
日期：2022年11月18日
'''
import torch
import numpy as np

class ModelTrainer(object):
    @staticmethod
    def train(data_loader,model,loss_f,optimizer,epoch_idx,device,log_interval,max_epoch):
        model.train()

        category=data_loader.dataset.category
        conf_mat=np.zeros((category,category))#(5,5),用来算混淆矩阵
        loss_sigma=[]#用来记录每个batch的损失值，再用np.mean求均值
        loss_mean=0
        acc_avg=0
        for i,data in enumerate(data_loader):
            inputs,labels=data
            inputs,labels=inputs.to(device),labels.to(device)

            outputs=model(inputs)
            optimizer.zero_grad()
            loss=loss_f(outputs.cpu(),labels.cpu())
            loss.backward()
            optimizer.step()

            #统计loss
            loss_sigma.append(loss.item())
            loss_mean=np.mean(loss_sigma)

            _,predicted=torch.max(outputs.data,1)

            for j in range(len(labels)):
                cate_i=int(labels[j])
                pre_i=int(predicted[j])
                conf_mat[cate_i,pre_i]+=1
            acc_avg=conf_mat.trace()/conf_mat.sum()

            # 每10个iteration 打印一次训练信息
            if i % log_interval == log_interval - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".
                      format(epoch_idx + 1, max_epoch, i + 1, len(data_loader), loss_mean, acc_avg))

        return loss_mean,acc_avg,conf_mat

    @staticmethod
    def valid(data_loader,model,loss_f,device):
        model.eval()

        category = data_loader.dataset.category
        conf_mat = np.zeros((category, category))  # (5,5),用来算混淆矩阵
        loss_sigma = []  # 用来记录每个batch的损失值，再用np.mean求均值
        loss_mean= 0
        acc_avg = 0

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_f(outputs.cpu(), labels.cpu())

            loss_sigma.append(loss.item())
            loss_mean= np.mean(loss_sigma)

            _, predicted = torch.max(outputs.data, 1)

            for j in range(len(labels)):
                cate_i = int(labels[j])
                pre_i = int(predicted[j])
                conf_mat[cate_i, pre_i] += 1
            acc_avg = conf_mat.trace() / conf_mat.sum()
        return loss_mean,acc_avg,conf_mat

    @staticmethod
    def text(data_loader,model,device):
        model.eval()

        kll={"0":['枇杷',100],'1':['香蕉',20],'2':['蓝莓',10],'3':['橘子',30],'4':['梨',40]}
        count=0
        for i,data in enumerate(data_loader):
            inputs,labels=data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs=model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            #print(len(labels))
            for j in range(len(labels)):
                pre = int(predicted[j])
                print(pre)
                for i in kll.keys():
                    if pre==int(i):
                        count+=kll[i][1]
                        print(f"今天吃了{kll[i][0]},有{kll[i][1]}卡路里",end=',')
        print(f"一共吃了{count}卡路里")