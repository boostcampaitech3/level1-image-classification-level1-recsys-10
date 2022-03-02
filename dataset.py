import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List
from matplotlib import use

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform
from sklearn.model_selection import StratifiedKFold

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def mixup(inputs, labels, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = inputs.size()[0]
    index = torch.randperm(batch_size).to(device)

    inputs = lam * inputs + (1 - lam) * inputs[index, :]
    target_a, target_b = labels, labels[index]

    return inputs, lam,  target_a, target_b

def cutmix(inputs, labels, device):
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    lam = np.random.beta(1.0, 1.0)
    rand_index = torch.randperm(inputs.size()[0]).to(device)
    shuffled_labels = labels[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:,:,bbx1:bbx2, bby1:bby2] = inputs[rand_index,:,bbx1:bbx2, bby1:bby2]

    target_a = labels
    target_b = labels[rand_index]


    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2])) # 람다 조정

    return inputs, lam, target_a, target_b

    
def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] # 이미지의 width
    H = size[3] # 이미지의 height
    cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기
    cut_w = np.int(W * cut_rat)  # 패치의 너비
    cut_h = np.int(H * cut_rat)  # 패치의 높이

    # uniform
    # 기존 이미지의 크기에서 랜덤하게 값을 가져옵니다.(중간 좌표 추출)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출합니다.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class AutoAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            auto_augment_transform(config_str = 'original', hparams = {'translate_const': 100, 'img_mean': (124, 116, 104)}),
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class RandomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            rand_augment_transform(config_str='rand-m9-mstd0.5', hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}),
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)       

class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class TestAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            RandomHorizontalFlip(),
            RandomRotation(3),
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)

class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        if self.__class__.__name__ != 'AgeDataset':
            self.setup()
        #self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    # def calc_statistics(self):
    #     has_statistics = self.mean is not None and self.std is not None
    #     if not has_statistics:
    #         print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
    #         sums = []
    #         squared = []
    #         for image_path in self.image_paths[:3000]:
    #             image = np.array(Image.open(image_path)).astype(np.int32)
    #             sums.append(image.mean(axis=(0, 1)))
    #             squared.append((image ** 2).mean(axis=(0, 1)))

    #         self.mean = np.mean(sums, axis=0) / 255
    #         self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    def __init__(self, data_dir, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), val_ratio=0.2, kfold=5, k=0):
        self.indices = defaultdict(list)
        self.k = k
        self.kfold = kfold
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(kfold, k, profiles, val_ratio):
        ids = []
        gender_and_age = []

        for profile in profiles:
            id, gender, race, age = profile.split("_")
            gender_label = GenderLabels.from_str(gender)
            age_label = AgeLabels.from_number(age)
            ids.append(id)
            gender_and_age.append(str(gender_label) + str(age_label))
        ids = np.array(ids)
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
        kfold_array = []
        for train_index, valid_index in skf.split(ids, gender_and_age):
            kfold_array.append({'train' : train_index, 'valid' : valid_index})
        return kfold_array[k]

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(self.kfold, self.k, profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class MultiLabelDataset(MaskSplitByProfileDataset):
    def __init__(self, data_dir, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), val_ratio=0.2, kfold=5, k=0):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio, kfold, k)

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        multi_label = {'mask': mask_label, 'age': age_label, 'gender': gender_label, 'label': multi_class_label}

        image_transform = self.transform(image)

        return image_transform, multi_label


class AgeDataset(MultiLabelDataset) :
    _file_names = {
        "mask1_fake_A": MaskLabels.MASK,
        "mask2_fake_A": MaskLabels.MASK,
        "mask3_fake_A": MaskLabels.MASK,
        "mask4_fake_A": MaskLabels.MASK,
        "mask5_fake_A": MaskLabels.MASK,
        "incorrect_mask_fake_A": MaskLabels.INCORRECT,
        "normal_fake_A": MaskLabels.NORMAL,
        "mask1_real_A": MaskLabels.MASK,
        "mask2_real_A": MaskLabels.MASK,
        "mask3_real_A": MaskLabels.MASK,
        "mask4_real_A": MaskLabels.MASK,
        "mask5_real_A": MaskLabels.MASK,
        "incorrect_mask_fake_A": MaskLabels.INCORRECT,
        "normal_fake_A": MaskLabels.NORMAL,
        "mask1_fake_B": MaskLabels.MASK,
        "mask2_fake_B": MaskLabels.MASK,
        "mask3_fake_B": MaskLabels.MASK,
        "mask4_fake_B": MaskLabels.MASK,
        "mask5_fake_B": MaskLabels.MASK,
        "incorrect_mask_fake_B": MaskLabels.INCORRECT,
        "normal_fake_B": MaskLabels.NORMAL
    }

    def __init__(self, data_dir, age_parameter, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), val_ratio=0.2, kfold=5, k=0):
        super().__init__(data_dir, mean, std, val_ratio, kfold, k)
        self.indices = defaultdict(list)
        self.age_parameter = age_parameter.split('_')
        self.data_dir = data_dir
        self.setup()
        

    def setup(self):
        use_real, older, younger = int(self.age_parameter[0]), int(self.age_parameter[1]), int(self.age_parameter[2])
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(self.kfold, self.k, profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile,"images")

                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    
                    id, gender, race, age, _= profile.split("_")
                    
                    if _file_name not in self._file_names:
                        continue

                    if "fake_A" in _file_name and older==0:
                        #print(1)
                        continue
                    elif "fake_A" in _file_name and ((int(age)<60-older and int(age)>=30)or(int(age)<30-older)) and older>0:
                        #print(2)
                        continue
                    elif "fake_B" in _file_name and younger==0:
                        #print(3)
                        continue
                    elif "fake_B" in _file_name and ((int(age)<60 and int(age)>=30+younger)or(int(age)<30)) and younger>0:
                        #print(4)
                        continue

                    if use_real==0:
                        if "real_A" in _file_name and ((int(age)<60-older and int(age)>=30)or(int(age)<30-older)) and older>0:
                        #    print(5)
                            continue
                        if "real_A" in _file_name and ((int(age)<60 and int(age)>=30+younger)or(int(age)<30)) and younger>0:
                        #    print(6)
                            continue

                    img_path = os.path.join(self.data_dir, profile,"images", file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    img_path = os.path.join(self.data_dir, profile,"images", file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    if "fake_A" in _file_name:
                        age=str(int(age)+older)
                    if "fake_B" in _file_name:
                        age=str(int(age)-younger)
                                                    
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)
                    self.image_paths.append(img_path)
                    #img_path
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    @staticmethod
    def _split_profile(kfold, k, profiles, val_ratio):
        ids = []
        gender_and_age = []

        for profile in profiles:
            id, gender, race, age,_ = profile.split("_")
            gender_label = GenderLabels.from_str(gender)
            age_label = AgeLabels.from_number(age)
            ids.append(id)
            gender_and_age.append(str(gender_label) + str(age_label))
        ids = np.array(ids)
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
        kfold_array = []
        for train_index, valid_index in skf.split(ids, gender_and_age):
            kfold_array.append({'train' : train_index, 'valid' : valid_index})
        return kfold_array[k]

            

class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

