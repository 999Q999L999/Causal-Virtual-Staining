import os
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random
import itertools
from PIL import Image

def get_cfe_data(source="TCGA"):
    if source == "TCGA":
        return CFE_Data_TCGA
    elif source == "regh2i":
        return CFE_Data_regh2i
    elif source == "mist_her2":
        return CFE_Data_MIST_her2
    elif source == "mist_her2_ihc":
        return CFE_Data_MIST_her2_ihc
    elif source == "BRCA":
        return CFE_Data_BRCA
    elif source == "color":
        return Color_T_Data
    elif source == "regh2i_intervene":
        return CFE_Data_regh2i_intervene
    else:
        raise ValueError("Invalid source: {}".format(source))


default_tcga_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


default_regh2i_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.CenterCrop(256),
    transforms.CenterCrop(1024),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

default_mist_her2_transforms = default_regh2i_transforms
mist_her2_ihc_transforms = default_regh2i_transforms
transforms_func_dict = {}


class CFE_Data_regh2i_intervene(Dataset):
    def __init__(self,
                 mode = "train",
                 transform_func = None,
                 seed = 0,
                 root = "/data18/lwp/CI_virtual_staining/CFE/Data/RegH2I_intervene_AB",
                 pos_ratio = 0.5,
                 num_samples = 1e5
    ):
        # 初始化参数
        self.transform_func = transforms_func_dict.get(transform_func, default_regh2i_transforms)
        self.root = root
        self.pos_ratio = pos_ratio
        self.num_img_pair = num_samples
        # 设定随机种子
        random.seed(seed)
        # 获取所有可能的img组合
        self.img_dir = os.path.join(self.root, mode+"A")
        img_list = os.listdir(self.img_dir)
        img_pair_list = list(itertools.combinations(img_list, 2))
        random.shuffle(img_pair_list)
        # 加载存储每个img对应slide的json文件
        img_slide_dict = json.load(open("/data18/lwp/CI_virtual_staining/CFE/ColorFE/img_slide_dict.json", "r"))
        # 生成样本对，形式为（img0_path, img1_path, label）
        num_pos = int(self.num_img_pair * self.pos_ratio)
        num_neg = int(self.num_img_pair * (1 - self.pos_ratio))
        pos_count, neg_count = 0, 0
        pos_samples, neg_samples = [], []
        for i in range(len(img_pair_list)):
            img_file0, img_file1 = img_pair_list[i]
            img0_slide = img_slide_dict[img_file0].split("/")[0]
            img1_slide = img_slide_dict[img_file1].split("/")[0]
            label = int(img0_slide == img1_slide)
            if label == 0:
                if neg_count < num_neg:
                    neg_count += 1
                    neg_samples.append([img_file0, img_file1, label])
            elif label == 1:
                if pos_count < num_pos:
                    pos_count += 1
                    pos_samples.append([img_file0, img_file1, label])
            else:
                raise ValueError("Invalid label: {}".format(label))
        
        self.samples = pos_samples + neg_samples
        # 保存样本结果
        sample_dic = {"pos": pos_samples, "neg": neg_samples}
        json.dump(sample_dic, open("regh2i_intervene_{}_seed_{}.json".format(mode, seed), "w"))

    
    def __getitem__(self, index):
        img_pair = self.samples[index]
        img0 = self.transform_func(Image.open(os.path.join(self.img_dir, img_pair[0])))
        img1 = self.transform_func(Image.open(os.path.join(self.img_dir, img_pair[1])))
        label = img_pair[2]
        return img0, img1, label
        # return os.path.join(self.img_dir, img_pair[0]), os.path.join(self.img_dir, img_pair[1]), label

    def __len__(self):
        return len(self.samples)


class Color_T_Data(Dataset):
    def __init__(self,
                 mode = "train",
                 transform_func = None,
                 root = "/data18/lwp/CI_virtual_staining/SimCLR-custom/img_pair",
    ):  
        self.transform_func = transforms_func_dict.get(transform_func, default_tcga_transforms)
        img_pair_dir = os.path.join(root, mode)
        img_pair_file_list = os.listdir(img_pair_dir)
        img_pair_label_res_list = []
        for pair_file in img_pair_file_list:
            img_dir = os.path.join(img_pair_dir, pair_file)
            img0_path = os.path.join(img_pair_dir, img_dir, "original.jpg")
            img1_path = os.path.join(img_pair_dir, img_dir, "transform.jpg")
            label = pair_file.split("_")[-1]
            img_pair_label_res_list.append((img0_path, img1_path, label))
        self.samples = img_pair_label_res_list
    
    def __getitem__(self, index):
        img_pair = self.samples[index]
        img0 = self.transform_func(Image.open(img_pair[0]))
        img1 = self.transform_func(Image.open(img_pair[1]))
        label = int(img_pair[2])
        return img0, img1, label
        # return img_pair[0], img_pair[1], label

    def __len__(self):
        return len(self.samples)

class CFE_Data_BRCA(Dataset):
    def __init__(self,
                 mode = "train",
                 transform_func = None,
                 seed = 0,
                 root = "/data7/lwp/DataBiasAnalysis/Data/patch_img",
                 pos_ratio = 0.5,
                 num_samples = 1e5
    ):
        # 初始化参数
        self.transform_func = transforms_func_dict.get(transform_func, default_tcga_transforms)
        self.root = root
        self.pos_ratio = pos_ratio
        self.num_img_pair = num_samples
        # 设定随机种子
        random.seed(seed)
        # 读入对应的cancer type
        data_split = json.load(open("ColorFE/tcga_brca_data_split.json", "r"))
        print("Dataset used mode: {}".format(mode))
        # 获取slide 列表
        slide_list = data_split[mode]
        slide_list = [os.path.join(self.root, "Breast_invasive_carcinoma", str(0), slide) for slide in slide_list]
        self.slide_list = slide_list
        # print(self.slide_list[0:10])
        # 生成样本对，形式为（img0_path, img1_path, label）
        num_pos = int(self.num_img_pair * self.pos_ratio)
        num_neg = int(self.num_img_pair * (1 - self.pos_ratio))
        pos_samples = self._generate_pos_sample(num_pos)
        print("finished generate pos samples: {}".format(len(pos_samples)))
        neg_samples = self._generate_neg_sample(num_neg)
        print("finished generate neg samples: {}".format(len(neg_samples)))
        self.samples = pos_samples + neg_samples
        # 保存样本结果
        sample_dic = {"pos": pos_samples, "neg": neg_samples}
        json.dump(sample_dic, open("ColorFE/config/seed_{}_{}_{}.json".format(seed, mode, self.num_img_pair), "w"))

    
    def __getitem__(self, index):
        img_pair = self.samples[index]
        img0 = self.transform_func(Image.open(img_pair[0]))
        img1 = self.transform_func(Image.open(img_pair[1]))
        label = img_pair[2]
        return img0, img1, label
        # return img_pair[0], img_pair[1], label

    def __len__(self):
        return len(self.samples)


    def _generate_pos_sample(self, num):
        pos_sample_list = []
        count = 0
        index = 0
        batch = 10
        num_slide = len(self.slide_list)
        while count < num:
            # 循环取slide
            slide = self.slide_list[index % num_slide]
            # 获取对应的patch列表
            patch_dir = os.path.join(self.root, slide)
            patch_list = os.listdir(patch_dir)
            # 获取所有的正样本对
            img_pair_list = list(itertools.combinations(patch_list, 2))
            random.shuffle(img_pair_list)
            for i in range(batch):
                sample = img_pair_list[i]
                pos_sample_list.append((os.path.join(patch_dir, sample[0]), os.path.join(patch_dir, sample[1]), 1))
            count += batch
            index += 1
        return pos_sample_list
    
    def _generate_neg_sample(self, num):
        neg_sample_list = []
        count = 0
        index = 0
        batch = 10
        while count < num:
            # 随机取一对slide
            slide_pair = random.sample(self.slide_list, 2)
            # 随机获取对应的patch列表
            patch_0_list = random.sample(os.listdir(slide_pair[0]), batch)
            patch_1_list = random.sample(os.listdir(slide_pair[1]), batch)
            # 获取所有的负样本对
            img_pair_list = [(os.path.join(slide_pair[0], img0), os.path.join(slide_pair[1], img1), 0) for img0, img1 in zip(patch_0_list, patch_1_list)]
            neg_sample_list.extend(img_pair_list)
            count += batch
            index += 1
        return neg_sample_list

class CFE_Data_TCGA(Dataset):
    def __init__(self,
                 mode = "train",
                 transform_func = None,
                 seed = 0,
                 root = "/data7/lwp/DataBiasAnalysis/Data/patch_img",
                 pos_ratio = 0.3,
                 num_samples = 1e5
    ):
        # 初始化参数
        self.transform_func = transforms_func_dict.get(transform_func, default_tcga_transforms)
        self.root = root
        self.pos_ratio = pos_ratio
        self.num_img_pair = num_samples
        # 设定随机种子
        random.seed(seed)
        # 读入对应的cancer type
        data_split = json.load(open("ColorFE/data_split_info.json", "r"))
        print("Dataset used mode: {}".format(mode))
        self.cancer_list = data_split[mode]
        # 获取slide 列表
        slide_list = []
        for cancer in self.cancer_list:
            slides = os.listdir(os.path.join(self.root, cancer, str(0)))
            slides = [cancer+"/0/"+slide for slide in slides]
            slide_list.extend(slides)
        self.slide_list = slide_list
        # 生成样本对，形式为（img0_path, img1_path, label）
        num_pos = int(self.num_img_pair * self.pos_ratio)
        num_neg = int(self.num_img_pair * (1 - self.pos_ratio))
        pos_samples = self._generate_pos_sample(num_pos)
        print("finished generate pos samples: {}".format(len(pos_samples)))
        neg_samples = self._generate_neg_sample(num_neg)
        print("finished generate neg samples: {}".format(len(neg_samples)))
        self.samples = pos_samples + neg_samples
        # 保存样本结果
        sample_dic = {"pos": pos_samples, "neg": neg_samples}
        json.dump(sample_dic, open("ColorFE/config/seed_{}_{}_{}.json".format(seed, mode, self.num_img_pair), "w"))

    
    def __getitem__(self, index):
        img_pair = self.samples[index]
        img0 = self.transform_func(Image.open(img_pair[0]))
        img1 = self.transform_func(Image.open(img_pair[1]))
        label = img_pair[2]
        return img0, img1, label
        # return img_pair[0], img_pair[1], label

    def __len__(self):
        return len(self.samples)


    def _generate_pos_sample(self, num):
        pos_sample_list = []
        count = 0
        index = 0
        batch = 10
        num_slide = len(self.slide_list)
        while count < num:
            # 循环取slide
            slide = self.slide_list[index % num_slide]
            # 获取对应的patch列表
            patch_dir = os.path.join(self.root, slide)
            patch_list = os.listdir(patch_dir)
            # 获取所有的正样本对
            img_pair_list = list(itertools.combinations(patch_list, 2))
            random.shuffle(img_pair_list)
            for i in range(batch):
                sample = img_pair_list[i]
                pos_sample_list.append((os.path.join(patch_dir, sample[0]), os.path.join(patch_dir, sample[1]), 1))
            count += batch
            index += 1
        return pos_sample_list
    
    def _generate_neg_sample(self, num):
        neg_sample_list = []
        count = 0
        index = 0
        batch = 10
        num_cancer = len(self.cancer_list)
        while count < num:
            # 随机取一对slide
            cancer = self.cancer_list[index % num_cancer]
            slide_pair = random.sample(os.listdir(os.path.join(self.root, cancer, str(0))), 2)
            # 随机获取对应的patch列表
            patch_0_dir = os.path.join(self.root, cancer, str(0), slide_pair[0])
            patch_0_list = random.sample(os.listdir(patch_0_dir), batch)
            patch_1_dir = os.path.join(self.root, cancer, str(0), slide_pair[1])
            patch_1_list = random.sample(os.listdir(patch_1_dir), batch)
            # 获取所有的负样本对
            img_pair_list = [(os.path.join(patch_0_dir, img0), os.path.join(patch_1_dir, img1), 0) for img0, img1 in zip(patch_0_list, patch_1_list)]
            neg_sample_list.extend(img_pair_list)
            count += batch
            index += 1
        return neg_sample_list


    # def _generate_neg_sample(self, num):
    #     neg_sample_list = []
    #     count = 0
    #     batch = 5
    #     while count < num:
    #         # 随机取一对slide
    #         slide_pair = random.sample(self.s, 2)
    #         # 随机获取对应的patch列表
    #         patch_0_dir = os.path.join(self.root, slide_pair[0])
    #         patch_0_list = random.sample(os.listdir(patch_0_dir), batch)
    #         patch_1_dir = os.path.join(self.root, slide_pair[1])
    #         patch_1_list = random.sample(os.listdir(patch_1_dir), batch)
    #         # 获取所有的负样本对
    #         img_pair_list = [(os.path.join(patch_0_dir, img0), os.path.join(patch_1_dir, img1), 0) for img0, img1 in zip(patch_0_list, patch_1_list)]
    #         neg_sample_list.extend(img_pair_list)
    #         count += batch
    #     return neg_sample_list



class CFE_Data_regh2i(Dataset):
    def __init__(self,
                 transform_func = None,
                 seed = 0,
                 root = "/data4/qiongp/TDKstain/datasets/RegH2I/HE/",
                 fold_list = None,
                 pos_ratio = 0.5,
                 num_samples = 1e5
    ):
        # 初始化参数
        self.transform_func = transforms_func_dict.get(transform_func, default_regh2i_transforms)
        self.root = root
        self.pos_ratio = pos_ratio
        self.num_img_pair = num_samples
        self.fold_index_list = [i for i in range(5)] if fold_list is None else fold_list
        # 设定随机种子
        random.seed(seed)
        # 获取所有可能的img组合
        img_list = []
        for index in self.fold_index_list:
            fold_img_dir = os.path.join(self.root, "fold{}".format(index))
            fold_img_file_list = os.listdir(fold_img_dir)
            img_list.extend(fold_img_file_list)
        img_pair_list = list(itertools.combinations(img_list, 2))
        random.shuffle(img_pair_list)
        # 加载存储每个img对应slide的json文件
        img_slide_dict = json.load(open("/data18/lwp/CI_virtual_staining/CFE/regh2_img_slide_dict.json", "r"))
        # 生成样本对，形式为（img0_path, img1_path, label）
        num_pos = int(self.num_img_pair * self.pos_ratio)
        num_neg = int(self.num_img_pair * (1 - self.pos_ratio))
        pos_count, neg_count = 0, 0
        pos_samples, neg_samples = [], []
        for i in range(len(img_pair_list)):
            img_file0, img_file1 = img_pair_list[i]
            img0_slide = img_slide_dict[img_file0].split("/")[0]
            img1_slide = img_slide_dict[img_file1].split("/")[0]
            label = int(img0_slide == img1_slide)
            if label == 0:
                if neg_count < num_neg:
                    neg_count += 1
                    neg_samples.append([img_file0, img_file1, label])
            elif label == 1:
                if pos_count < num_pos:
                    pos_count += 1
                    pos_samples.append([img_file0, img_file1, label])
            else:
                raise ValueError("Invalid label: {}".format(label))
        
        self.samples = pos_samples + neg_samples
        # 保存样本结果
        sample_dic = {"pos": pos_samples, "neg": neg_samples}
        json.dump(sample_dic, open("regh2i_index_{}_seed_{}.json".format(self.fold_index_list, seed), "w"))

    
    def __getitem__(self, index):
        img_pair = self.samples[index]
        img0 = self.transform_func(Image.open(os.path.join(self.root, img_pair[0][0:5], img_pair[0])))
        img1 = self.transform_func(Image.open(os.path.join(self.root, img_pair[1][0:5], img_pair[1])))
        label = img_pair[2]
        return img0, img1, label
        # return os.path.join(self.root, img_pair[0][0:5], img_pair[0]), os.path.join(self.root, img_pair[1][0:5], img_pair[1]), label

    def __len__(self):
        return len(self.samples)
    

class CFE_Data_MIST_her2(Dataset):
    def __init__(self,
                 transform_func = None,
                 seed = 0,
                 mode = "train",
                 root = "/data18/lwp/CI_virtual_staining/CFE/Data/MIST/HER2",
                 pos_ratio = 0.5,
                 num_samples = 1e5
    ):
        # 初始化参数
        self.transform_func = transforms_func_dict.get(transform_func, default_mist_her2_transforms)
        self.root = root
        self.pos_ratio = pos_ratio
        self.num_img_pair = num_samples
        # 设定随机种子
        random.seed(seed)
        # 获取所有可能的img组合
        self.img_dir = os.path.join(self.root, mode+"A")
        img_list = os.listdir(self.img_dir)
        img_pair_list = list(itertools.combinations(img_list, 2))
        random.shuffle(img_pair_list)
        # 生成样本对，形式为（img0_path, img1_path, label）
        num_pos = int(self.num_img_pair * self.pos_ratio)
        num_neg = int(self.num_img_pair * (1 - self.pos_ratio))
        pos_count, neg_count = 0, 0
        pos_samples, neg_samples = [], []
        for i in range(len(img_pair_list)):
            img_file0, img_file1 = img_pair_list[i]
            img0_slide = img_file0.split("_")[0]
            img1_slide = img_file1.split("_")[0]
            label = int(img0_slide == img1_slide)
            if label == 0:
                if neg_count < num_neg:
                    neg_count += 1
                    neg_samples.append([img_file0, img_file1, label])
            elif label == 1:
                if pos_count < num_pos:
                    pos_count += 1
                    pos_samples.append([img_file0, img_file1, label])
            else:
                raise ValueError("Invalid label: {}".format(label))
        
        self.samples = pos_samples + neg_samples
        # 保存样本结果
        sample_dic = {"pos": pos_samples, "neg": neg_samples}
        json.dump(sample_dic, open("mist_her2_{}_seed_{}.json".format(mode, seed), "w"))

    
    def __getitem__(self, index):
        img_pair = self.samples[index]
        img0 = self.transform_func(Image.open(os.path.join(self.img_dir, img_pair[0])))
        img1 = self.transform_func(Image.open(os.path.join(self.img_dir, img_pair[1])))
        label = img_pair[2]
        return img0, img1, label
        # return os.path.join(self.img_dir, img_pair[0]), os.path.join(self.img_dir, img_pair[1]), label
    
    def __len__(self):
        return len(self.samples)
    

class CFE_Data_MIST_her2_ihc(Dataset):
    def __init__(self,
                 transform_func = None,
                 seed = 0,
                 mode = "train",
                 root = "/data3/lwp/work/pytorch-CycleGAN-and-pix2pix-master/datasets/HER2/",
                 pos_ratio = 0.5,
                 num_samples = 1e5
    ):
        # 初始化参数
        self.transform_func = transforms_func_dict.get(transform_func, mist_her2_ihc_transforms)
        self.root = root
        self.pos_ratio = pos_ratio
        self.num_img_pair = num_samples
        # 设定随机种子
        random.seed(seed)
        # 获取所有可能的img组合
        self.img_dir = os.path.join(self.root, mode+"B")
        img_list = os.listdir(self.img_dir)
        img_pair_list = list(itertools.combinations(img_list, 2))
        random.shuffle(img_pair_list)
        # 生成样本对，形式为（img0_path, img1_path, label）
        num_pos = int(self.num_img_pair * self.pos_ratio)
        num_neg = int(self.num_img_pair * (1 - self.pos_ratio))
        pos_count, neg_count = 0, 0
        pos_samples, neg_samples = [], []
        for i in range(len(img_pair_list)):
            img_file0, img_file1 = img_pair_list[i]
            img0_slide = img_file0.split("_")[0]
            img1_slide = img_file1.split("_")[0]
            label = int(img0_slide == img1_slide)
            if label == 0:
                if neg_count < num_neg:
                    neg_count += 1
                    neg_samples.append([img_file0, img_file1, label])
            elif label == 1:
                if pos_count < num_pos:
                    pos_count += 1
                    pos_samples.append([img_file0, img_file1, label])
            else:
                raise ValueError("Invalid label: {}".format(label))
        
        self.samples = pos_samples + neg_samples
        # 保存样本结果
        # sample_dic = {"pos": pos_samples, "neg": neg_samples}
        # json.dump(sample_dic, open("mist_her2_{}_seed_{}.json".format(mode, seed), "w"))

    
    def __getitem__(self, index):
        img_pair = self.samples[index]
        img0 = self.transform_func(Image.open(os.path.join(self.img_dir, img_pair[0])))
        img1 = self.transform_func(Image.open(os.path.join(self.img_dir, img_pair[1])))
        label = img_pair[2]
        return img0, img1, label
        # return os.path.join(self.img_dir, img_pair[0]), os.path.join(self.img_dir, img_pair[1]), label
    
    def __len__(self):
        return len(self.samples)