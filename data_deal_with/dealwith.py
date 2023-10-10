'''
作者：yxl
日期：2022年11月18日
目的：定义一个类，把每个图片数据和标签对应上，其中{Apricot:0,Banana:1,Blueberry:2,Orange:3,Pear:4}，返回的是经过处理的图片数据和标签
'''
import os
from PIL import Image
from torch.utils.data import Dataset

class FruitDataset(Dataset):
    category=5

    def __init__(self,root_dir,transform=None):
        self.root_dir=root_dir
        self.transform=transform
        self.img_info=[]
        self._get_img_info()

    def __getitem__(self,index):
        path_img,label=self.img_info[index]
        img=Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img=self.transform(img)

        return img,label

    def __len__(self):
        if len(self.img_info)==0:
            raise Exception("\ndata_dir:{} is a empty dir!".format(self.root_dir))
        return len(self.img_info)

    def _get_img_info(self):
        Apricot_data_dir=os.path.join(self.root_dir,"Apricot")
        Banana_data_dir=os.path.join(self.root_dir,"Banana")
        Blueberry_data_dir=os.path.join(self.root_dir,"Blueberry")
        Orange_data_dir=os.path.join(self.root_dir,"Orange")
        Pear_data_dir=os.path.join(self.root_dir,"Pear")

        Apricot_imgs=[os.path.join(Apricot_data_dir,n) for n in os.listdir(Apricot_data_dir)]
        Banana_imgs=[os.path.join(Banana_data_dir,n) for n in os.listdir(Banana_data_dir)]
        Blueberry_imgs=[os.path.join(Blueberry_data_dir,n) for n in os.listdir(Blueberry_data_dir)]
        Orange_imgs=[os.path.join(Orange_data_dir,n) for n in os.listdir(Orange_data_dir)]
        Pear_imgs=[os.path.join(Pear_data_dir,n) for n in os.listdir(Pear_data_dir)]

        Apricot_info=[(p,0) for p in Apricot_imgs]
        Banana_info=[(p,1) for p in Banana_imgs]
        Blueberry_info=[(p,2) for p in Blueberry_imgs]
        Orange_info=[(p,3) for p in Orange_imgs]
        Pear_info=[(p,4) for p in Pear_imgs]

        for i in [Apricot_info,Banana_info,Blueberry_info,Orange_info,Pear_info]:
            self.img_info.extend(i)

if __name__ == "__main__":
    root_dir=r"E:\视觉\作业\作业6\datasets\fruit_data\text"
    train_dataset=FruitDataset(root_dir)

    print(len(train_dataset))
    print(next(iter(train_dataset)))
    print(train_dataset.img_info)