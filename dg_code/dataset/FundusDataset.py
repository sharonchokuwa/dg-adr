import os.path as osp
from torch.utils.data.dataset import Dataset
from PIL import Image
import os

AUG_ROOT = r'/synthetic_datasets'
AUG_SPLITS = r'/synthetic_datasets/splits_1000_samples'

# Dataset for fundus images including APTOS, DEEPDR, FGADR, IDRID, MESSIDOR, RLDR, and DDR
class FundusDataset(Dataset):

    def __init__(self, root, source_domains, target_domains, mode, trans_basic=None, use_aug=None):

        root = osp.abspath(osp.expanduser(root))
        self.use_aug = use_aug
        self.mode = mode
        self.dataset_dir = osp.join(root, "images")
        self.split_dir = osp.join(root, "splits")

        self.data = []
        self.label = []
        self.domain = []
        self.domain_name = []
        
        self.trans_basic = trans_basic

        if mode == "train":
            self._read_data(source_domains, "train")
        elif mode == "val":
            self._read_data(source_domains, "crossval")
        elif mode == "test":
            self._read_data(target_domains, "test")
        
    def _read_data(self, input_domains, split):
        items = []
        for domain, dname in enumerate(input_domains):
            if split == "test":
                file_train = osp.join(self.split_dir, dname + "_train.txt")
                impath_label_list = self._read_split(file_train)
                file_val = osp.join(self.split_dir, dname + "_crossval.txt")
                impath_label_list += self._read_split(file_val)
            elif split == "train" and self.use_aug:
                file_train = osp.join(self.split_dir, dname + "_train.txt")
                impath_label_list = self._read_split(file_train)
                if dname != 'ddr': # ddr not containing synthetic augs
                    file_train_aug = osp.join(AUG_SPLITS, dname + "_train.txt")
                    impath_label_list += self._read_split(file_train_aug)
            else:
                file = osp.join(self.split_dir, dname + "_" + split + ".txt")
                impath_label_list = self._read_split(file)

            for impath, label in impath_label_list:
                self.data.append(impath)

                self.label.append(label)
                self.domain.append(domain)
                self.domain_name.append(dname)

    def _read_split(self, split_file):
        items = []
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                if os.path.exists(osp.join(self.dataset_dir, impath)):
                    impath = osp.join(self.dataset_dir, impath)
                elif self.use_aug:
                    impath = osp.join(AUG_ROOT, impath)        
                label = int(label)
                items.append((impath, label))
                
        return items

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        data = Image.open(self.data[index]).convert("RGB")

        label = self.label[index]
        domain = self.domain[index]
        domain_name = self.domain_name[index]
        
        if self.trans_basic is not None:
            data = self.trans_basic(data)
        
        return data, label, domain, index, domain_name
    

