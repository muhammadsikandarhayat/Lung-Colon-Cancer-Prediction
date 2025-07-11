import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class CancerDataset(Dataset):
    """Custom Dataset for loading cancer images"""
    
    def __init__(self, data_dir, cancer_type, transform=None):

        self.data_dir = Path(data_dir)
        self.cancer_type = cancer_type
        self.transform = transform
        self.classes = self._get_classes()
        self.image_paths, self.labels = self._load_dataset()
        
    def _get_classes(self):
        if self.cancer_type == 'lung':
            return {
                'normal': 0,  # lung_n
                'cancer': 1   # combining lung_aca and lung_scc as cancer
            }
        else:  # colon
            return {
                'normal': 0,  # colon_n
                'cancer': 1   # colon_aca
            }
    
    def _load_dataset(self):
        image_paths = []
        labels = []
        
        # Load normal tissue images
        normal_dir = self.data_dir / f"{self.cancer_type}_image_sets" / f"{self.cancer_type}_n"
        for img_path in normal_dir.glob("*.jpeg"):
            image_paths.append(str(img_path))
            labels.append(self.classes['normal'])
        
        # Load cancer tissue images
        if self.cancer_type == 'lung':
            # Load both adenocarcinoma and squamous cell carcinoma
            cancer_types = ['aca', 'scc']
            for c_type in cancer_types:
                cancer_dir = self.data_dir / f"{self.cancer_type}_image_sets" / f"{self.cancer_type}_{c_type}"
                for img_path in cancer_dir.glob("*.jpeg"):
                    image_paths.append(str(img_path))
                    labels.append(self.classes['cancer'])
        else:
            # Load colon adenocarcinoma
            cancer_dir = self.data_dir / f"{self.cancer_type}_image_sets" / f"{self.cancer_type}_aca"
            for img_path in cancer_dir.glob("*.jpeg"):
                image_paths.append(str(img_path))
                labels.append(self.classes['cancer'])
        
        return image_paths, labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read and preprocess image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        
        return image, label

def get_data_transforms():
    """Returns dictionary of transforms for train and validation"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add some color augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    }
    return data_transforms