'''
作者：yxl
日期：2022年11月24日
'''
import torch
import os
from PIL import Image
from torch import nn
import torchvision.transforms as transforms
from torchvision.models import resnet18

kll={"0":['枇杷',16],'1':['香蕉',20],'2':['蓝莓',35],'3':['橘子',21],'4':['梨',44]}
count=0

path_state_dict=r"E:\视觉\作业\作业6\fruit_resnet_state_dict.pth"
model=resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)

if os.path.exists(path_state_dict):
    train_state_dict=torch.load(path_state_dict)
    model.load_state_dict(train_state_dict)

else:
    print(f"this path{path_state_dict} is not exists")


model.eval()

test_path=r"E:\视觉\作业\作业6\fruit_test"
trans_mean=[0.485,0.456,0.406]
trans_std=[0.229,0.224,0.225]
img_transforms=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=trans_mean,std=trans_std)
])
test_img_path=[os.path.join(test_path,n) for n in os.listdir(test_path)]
imgs=[]
for img_path in test_img_path:
    img=Image.open(img_path).convert('RGB')
    img=img_transforms(img)
    #imgs.append(img1)

    #a=np.array(imgs,np.float32)
    #print(a.shape)
    #inputs=torch.from_numpy(a)
    input_img=img.unsqueeze(0)
    output=model(input_img)
    _, predicted = torch.max(output.data, 1)
    pre=int(predicted)
    for j in kll.keys():
        if pre==int(j):
            count += kll[j][1]
            print(f"今天买了{kll[j][0]},{kll[j][1]}元", end=',')

print(f"一共买了{count}元")