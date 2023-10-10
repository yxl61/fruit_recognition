'''
作者：yxl
日期：2022年11月24日
'''
from collections import OrderedDict
from data_deal_with.dealwith import FruitDataset
import torchvision.transforms as transforms
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from model_train.ModelTrain import ModelTrainer

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,'..'))
#将自己写的模块的目录添加到变量中
device="cpu"
if __name__=="__main__":
    #训练集和验证集的路径
    train_dir=r"E:\视觉\作业\作业6\datasets\fruit_data\train"
    valid_dir=r"E:\视觉\作业\作业6\datasets\fruit_data\valid"
    text_dir=r"E:\视觉\作业\作业6\datasets\fruit_data\text"

    #densenet121gg的预训练权重路径
    path_state_dict=r"E:\视觉\上课内容\第12讲 图片分类完整项目构建\pretrained_model\resnet18-5c106cde.pth"

    train_bs=64
    valid_bs=64
    workers=0
    log_interval=10

    lr=0.01   #学习率
    momentum=0.9
    weight_decay= 1e-4
    max_epoch=2
    #factor = 0.1  # 学习率更新下降的比例,学习率调整倍数，默认为0.1，即下降10倍
    #milestones = [30, 45]

    #处理数据
    norm_mean=[0.485,0.456,0.406]
    norm_std=[0.229,0.224,0.225]
    normTransform=transforms.Normalize(norm_mean,norm_std)
    #训练集数据的处理
    transforms_train=transforms.Compose([
        transforms.Resize((224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normTransform,
    ])
    #验证集数据的处理
    transforms_valid=transforms.Compose([
        transforms.Resize((224)),
        transforms.ToTensor(),
        normTransform,
    ])
    transforms_text=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normTransform,
    ])
    train_data=FruitDataset(root_dir=train_dir,transform=transforms_train)
    valid_data=FruitDataset(root_dir=valid_dir,transform=transforms_valid)
    text_data=FruitDataset(root_dir=text_dir,transform=transforms_text)

    train_loader=DataLoader(dataset=train_data,batch_size=train_bs,shuffle=True,num_workers=workers)
    valid_loader=DataLoader(dataset=valid_data,batch_size=train_bs,num_workers=workers)
    text_loader=DataLoader(dataset=text_data,batch_size=2)

    #定义模型
    model=resnet18()
    if os.path.exists(path_state_dict):
        pretrained_state_dict = torch.load(path_state_dict, map_location="cpu")
        # map_location用于修改模型能在gpu上运行还是cpu上运行， torch.load加载模型

        model.load_state_dict(pretrained_state_dict)
    else:
        print("path:{} is not exists".format(path_state_dict))

    #修改densenet121最后一层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, train_data.category)

    model.to(device)

    #定义损失函数，优化器
    loss_f=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=factor, milestones=milestones)
    # torch.optim.lr_scheduler.MultiStepLR表示按需求有外部设置调整学习率,初始lr=0.01,在第30轮时变成lr=0.001,在第45轮时变成lr=0.0001

    loss_rec={"train":[],"valid":[]}
    acc_rec={"train":[],"valid":[]}
    for epoch in range(max_epoch):
        #train
        loss_train, acc_train, mat_train = ModelTrainer.train(
            train_loader, model, loss_f, optimizer, epoch, device, log_interval, max_epoch)

        # valid
        loss_valid, acc_valid, mat_valid = ModelTrainer.valid(
            valid_loader, model, loss_f, device)

        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}". \
              format(epoch + 1, max_epoch, acc_train, acc_valid, loss_train, loss_valid,
                     optimizer.param_groups[0]["lr"]))
        #scheduler.step()

    ModelTrainer.text(text_loader,model,device)

    path=os.path.join(BASE_DIR,'..','fruit_resnet_state_dict.pth')
    torch.save(model.state_dict(),path)