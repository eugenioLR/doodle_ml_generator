import numpy as np
import pandas as pd
from pathlib import Path
from torchvision.io import read_image
from torch.utils.data import Dataset

def create_classification_csv(file_path: str|Path, name: str):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    data_df = pd.DataFrame(columns=["file_name", "file_path", "class_name", "class_index"])

    for idx_i, class_dir in enumerate(file_path.glob("*")):
        print(class_dir)
        for idx_j, data_file in enumerate(class_dir.glob("*")):
            data_df = pd.concat([pd.DataFrame([[data_file.stem, data_file, class_dir.stem, idx_i]], columns=data_df.columns), data_df], ignore_index=True)
    
    data_df.to_csv(name, index=False)


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = Path(self.img_labels.iloc[idx]["file_path"])
        image = read_image(str(img_path))
        label = self.img_labels.iloc[idx]["class_index"]
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def as_matrix(self):
        input_matrix = np.empty((len(self),) + self[0][0].shape[1:])
        target_matrix = np.empty(len(self))

        for idx, data_point in enumerate(self):
            input_matrix[idx] = data_point[0][0,:,:]
            target_matrix[idx] = data_point[1]
        
        return input_matrix, target_matrix
    
    