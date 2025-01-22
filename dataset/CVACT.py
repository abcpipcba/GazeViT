import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import scipy.io as sio

def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

class CVACT(torch.utils.data.Dataset):
    def __init__(self, mode = '', root1 = 'D:\\20240513\\', root2 = 'D:\\20240513\\test_8884\\', root3 = 'E:\\CVACT\\ANU_data_small\\train\\',args=None):
        super(CVACT, self).__init__()

        self.args = args
        self.root1 = root1   # Train set 1
        self.root2 = root2   # test set
        self.root3 = root3   # Train set 2
        self.mode = mode
        self.sat_size = [256, 256]  # [512, 512]
        self.grd_size = [112, 224]  # [224, 1232]
        if args.sat_res != 0:
            self.sat_size = [args.sat_res, args.sat_res]

        
        print(self.sat_size, self.grd_size)

        self.transform_street = input_transform(size=self.grd_size)
        self.transform_aerial = input_transform(size=self.sat_size)
        self.to_tensor = transforms.ToTensor()
        
        
        ## Train set 1
        path_street = os.path.join(self.root1, 'street')                        
        street_list1 = os.listdir(path_street)
        
        path_street_heatmap = os.path.join(self.root1, 'street_heatmap')         
        street_list_heatmap1 = os.listdir(path_street_heatmap)
        
        path_aerial = os.path.join(self.root1, 'aerial')                 
        aerial_list1 = os.listdir(path_aerial)
        
        path_aerial_heatmap = os.path.join(self.root1, 'aerial_heatmap') 
        aerial_list_heatmap1 = os.listdir(path_aerial_heatmap)
        
        # self.train_list = []
        # for i in range(len(street_list)):
        #     aerial_list[i] = 'aerial\\' + aerial_list[i]
        #     street_list[i] = 'street\\' + street_list[i]
        #     aerial_list_heatmap[i] = 'aerial\\' + aerial_list_heatmap[i]
        #     street_list_heatmap[i] = 'street\\' + street_list_heatmap[i]
        #     self.train_list.append([aerial_list[i], street_list[i], aerial_list_heatmap[i], street_list_heatmap[i]])
            
        # self.id_idx_list = self.train_list
        
        ## Train set 2
        name_list = pd.read_csv(open('E:\\CVACT\\ANU_data_small\\train_name_list.csv', encoding = 'utf-8'))
        name_list = np.array(name_list)
        name_list = name_list.tolist()

        aerial_list2 = []
        street_list2 = []
        aerial_list_heatmap2 = []
        street_list_heatmap2 = []
        for i in range(len(name_list)):
            
            aerial_list2.append((name_list[i][0] + '_satView_polish.jpg'))
            street_list2.append(name_list[i][0] + '_grdView.jpg')
            aerial_list_heatmap2.append(str(i)+ '.png')
            street_list_heatmap2.append(str(i)+ '.png')   
        
        self.train_list = []
        for i in range(len(street_list1)):
            # Train set 1
            aerial_list1[i] = 'aerial\\' + aerial_list1[i]
            street_list1[i] = 'street\\' + street_list1[i]
            aerial_list_heatmap1[i] = 'aerial_heatmap\\' + aerial_list_heatmap1[i]
            street_list_heatmap1[i] = 'street_heatmap\\' + street_list_heatmap1[i]
            
            self.train_list.append([aerial_list1[i], street_list1[i], aerial_list_heatmap1[i], street_list_heatmap1[i]])
            
        for i in range(len(name_list)):
            # Train set 2
            aerial_list2[i] = 'satview_polish\\' + aerial_list2[i]
            street_list2[i] = 'streetview\\' + street_list2[i]
            
            self.train_list.append([aerial_list2[i], street_list2[i], aerial_list_heatmap2[i], street_list_heatmap2[i]])
            
        self.id_idx_list = self.train_list
        
        ## test set
        anuData = sio.loadmat(os.path.join(self.root2, 'ACT_data.mat'))

        self.id_all_list = []
        self.id_idx_all_list = []
        idx = 0
        missing = 0
        for i in range(0, len(anuData['panoIds'])):
            grd_id = 'streetview/' + anuData['panoIds'][i] + '_grdView.jpg'
            sat_id = 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.jpg'

            self.id_all_list.append([sat_id, grd_id])
            self.id_idx_all_list.append(idx)
            idx += 1

        print('CVACT: load',' data_size =', len(self.id_all_list))

        self.val_inds = anuData['valSet']['valInd'][0][0] - 1
        self.valNum = len(self.val_inds)
        
        print('CVACT val:', self.valNum)

        self.id_test_list = []
        self.id_test_idx_list = []
        for k in range(self.valNum):
            sat_id = self.id_all_list[self.val_inds[k][0]][0]
            grd_id = self.id_all_list[self.val_inds[k][0]][1]
            if not os.path.exists(os.path.join(self.root2, grd_id)) or not os.path.exists(
                    os.path.join(self.root2, sat_id)):
                    print('val', k, grd_id, sat_id)
                    missing += 1
            else:
                self.id_test_list.append(self.id_all_list[self.val_inds[k][0]])
                self.id_test_idx_list.append(k)
        
        print('missing:', missing)  # may miss some images

    def __getitem__(self, index, debug=False):
        if self.mode== 'train':
            idx = index % len(self.id_idx_list)
            
            # street-view image
            img_street = Image.open(self.root1 + self.train_list[idx][1]).convert('RGB')
            img_street = img_street.crop((0,img_street.size[1]//4,img_street.size[0],img_street.size[1]//4*3))
            
            # heatmap
            img_street_heatmap = Image.open(self.root1 + self.train_list[idx][2]).convert('RGB')
            img_street_heatmap = img_street_heatmap.crop((0,img_street_heatmap.size[1]//4,img_street_heatmap.size[0],img_street_heatmap.size[1]//4*3))
            
            # aerial-view image
            img_aerial = Image.open(self.root1 + self.train_list[idx][0]).convert('RGB')
            
            # heatmap
            img_aerial_heatmap = Image.open(self.root1 + self.train_list[idx][3]).convert('RGB')
            
            img_street = self.transform_street(img_street)
            img_aerial = self.transform_aerial(img_aerial)
            
            img_street_heatmap = self.transform_street(img_street_heatmap)
            img_aerial_heatmap = self.transform_aerial(img_aerial_heatmap)
            if self.args.crop:
                atten_sat = Image.open(os.path.join(self.args.resume.replace(self.args.resume.split('/')[-1],''),'attention','train',str(idx)+'.png')).convert('RGB')
                return img_street, img_aerial, torch.tensor(idx), torch.tensor(idx), 0, self.to_tensor(atten_sat)
            return img_street, img_aerial, torch.tensor(idx), torch.tensor(idx), 0, 0, img_street_heatmap, img_aerial_heatmap

        elif 'scan_val' in self.mode:
            img_aerial = Image.open(self.root2 + self.id_test_list[index][0]).convert('RGB')
            img_aerial = self.transform_aerial(img_aerial)
            img_street = Image.open(self.root2 + self.id_test_list[index][1]).convert('RGB')
            img_street = img_street.crop((0, img_street.size[1] // 4, img_street.size[0], img_street.size[1] // 4 * 3))
            img_street = self.transform_street(img_street)
            return img_street, img_aerial, torch.tensor(index), torch.tensor(index), 0, 0, 0, 0

        elif 'test_aerial' in self.mode:
            img_aerial = Image.open(self.root2 + self.id_test_list[index][0]).convert('RGB')
            img_aerial = self.transform_aerial(img_aerial)
            if self.args.crop:
                atten_sat = Image.open(os.path.join(self.args.resume.replace(self.args.resume.split('/')[-1],''),'attention','val',str(index)+'.png')).convert('RGB')
                return img_aerial, torch.tensor(index), self.to_tensor(atten_sat)
            return img_aerial, torch.tensor(index), 0

        elif 'test_street' in self.mode:
            img_street = Image.open(self.root2 + self.id_test_list[index][1]).convert('RGB')
            img_street = img_street.crop((0, img_street.size[1] // 4, img_street.size[0], img_street.size[1] // 4 * 3))
            img_street = self.transform_street(img_street)
            return img_street, torch.tensor(index), torch.tensor(index)
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        if self.mode == 'train':
            return len(self.id_idx_list)
        elif 'scan_val' in self.mode:
            return len(self.id_test_list)
        elif 'test_aerial' in self.mode:
            return len(self.id_test_list)
        elif 'test_street' in self.mode:
            return len(self.id_test_list)
        else:
            print('not implemented!')
            raise Exception
