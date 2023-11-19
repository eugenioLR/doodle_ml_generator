import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, IterableDataset
from joblib import Parallel, delayed
import math
import aiofiles
from PIL import Image
import io
import asyncio
import nest_asyncio
nest_asyncio.apply()

def create_classification_csv(file_path: str|Path, name: str):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    data_df = pd.DataFrame(columns=["file_name", "file_path", "class_name", "class_index"])

    for idx_i, class_dir in enumerate(file_path.glob("*")):
        print(class_dir)
        for idx_j, data_file in enumerate(class_dir.glob("*")):
            data_df = pd.concat([pd.DataFrame([[data_file.stem, data_file, class_dir.stem, idx_i]], columns=data_df.columns), data_df], ignore_index=True)
    
    data_df.to_csv(name, index=False)

async def read_img_as_tensor(img_path):
    async with aiofiles.open(img_path, 'rb') as img_file:
        img_data = await img_file.read()
        return Image.open(io.BytesIO(img_data))

async def read_image_list(file_list):
    tasks = [read_img_as_tensor(i) for i in file_list]
    return await asyncio.gather(*tasks)

class CustomImageDataset(IterableDataset):
    def __init__(self, annotations_file, batch_size=1, device="cpu", normalize=False, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # return math.ceil(len(self.img_labels)/self.batch_size
        return len(self.img_labels)//self.batch_size
    
    @staticmethod
    def normalize(data_matrix):
        min_values = data_matrix.min(axis=2, keepdims=True).min(axis=1, keepdims=True)
        max_values = data_matrix.max(axis=2, keepdims=True).max(axis=1, keepdims=True)

        rg = max_values - min_values
        rg = np.fmax(rg, 1e-4)

        data_matrix = (data_matrix-min_values) / rg

        return data_matrix

    def __getitem__(self, idx):
        # images = [read_img_as_tensor(i) for i in self.img_labels.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]["file_path"]]
        loop = asyncio.get_event_loop()
        images = loop.run_until_complete(read_image_list(self.img_labels.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]["file_path"]))
        # loop.close()
        # images = asyncio.run(read_image_list(self.img_labels.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]["file_path"]))

        images = np.asarray(images)
        if self.normalize:
            images = CustomImageDataset.normalize(images)
        if self.transform:
            images = self.transform(images)
        images = torch.from_numpy(images).float().to(self.device)

        labels = self.img_labels.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]["class_index"].to_numpy()
        if self.target_transform:
            labels = self.target_transform(labels)

        return images, labels
    
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
        return None
    
    def as_matrix(self):
        input_matrix = np.empty((len(self)*self.batch_size,) + self[0][0].shape[1:])
        target_matrix = np.empty(len(self)*self.batch_size)

        for idx, data_point in enumerate(self):
            pass
            # print(idx, data_point[1])
            input_matrix[idx * self.batch_size: (idx + 1) * self.batch_size] = data_point[0]
            target_matrix[idx * self.batch_size: (idx + 1) * self.batch_size] = data_point[1]
        
        return input_matrix, target_matrix

def square_dims(size, ratio_w_h=1):
    """
    Gives the dimensions of
    """

    divs = np.array(sympy.divisors(size))
    dist_to_root = np.abs(divs-np.sqrt(size)*ratio_w_h)
    i = np.argmin(dist_to_root)
    x_size = int(divs[i])
    y_size = size//x_size
    return (x_size, y_size) if x_size < y_size else (y_size, x_size)


def square_dims_vector(vector, ratio_w_h=1):
    """
    Transforms a vector into a matrix of with dimensions roughly proportional to the ratio provided.
    """

    if sympy.isprime(vector.size):
        warnings.warn("The size of the vector is a prime number and cannot be converted to a matrix. Consider adding padding to the vector.")
    
    return np.reshape(vector.copy(), square_dims(vector.size, ratio_w_h))

if __name__ == "__main__":
    # dataset = CustomImageDataset("./data/train_data.csv", batch_size=1200, normalize=True)
    # print(len(dataset))
    # print(dataset[0])
    # dataset_matrix = dataset.as_matrix()
    # print(dataset_matrix)
    # print(dataset_matrix[0].shape)

    dataset = CustomImageDataset("./data/train_data.csv", batch_size=512)
    print(len(dataset))
    print(dataset[0])
    dataset_matrix = dataset.as_matrix()
    print(dataset_matrix)
    print(dataset_matrix[0].shape) 
