import itertools

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


# Change to 6 bit string
class Symmetric8BitDataset(Dataset):
    def __init__(self):
        super().__init__()
        # Generate all 256 possible 8-bit binary vectors
        data = list(itertools.product([0, 1], repeat=8))

        # Convert to tensor with proper dtype
        input_tensor = torch.tensor(data, dtype=torch.float32)

        # Calculate labels (1 for symmetric, -1 for asymmetric)
        labels = []
        for example in data:
            # Check symmetry by comparing pairs from both ends
            symmetric = all(example[i] == example[7 - i] for i in range(4))
            labels.append(1 if symmetric else -1)

        # Convert labels to tensor and reshape
        label_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

        # Create dataset
        self.dataset = TensorDataset(input_tensor, label_tensor)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def get_dataloaders(batch_size, shuffle=True, num_elements=None):
    """
    Load MNIST dataset and create data loaders.

    Parameters:
    - batch_size (int): Batch size for data loaders.
    - shuffle (bool, optional): Whether to shuffle the training data. Defaults to True.
    - num_elements (int, optional): Number of elements to retrieve from the dataset. Defaults to None.

    Returns:
    - train_loader (DataLoader): Training data loader.
    - test_loader (DataLoader): Testing data loader.
    """
    # Load MNIST dataset
    dataset = Symmetric8BitDataset()

    # Create a data loader from my custom dataset
    data_loader = DataLoader(dataset, batch_size=batch_size)

    return data_loader


# class Simmetric_8bit_Dataset(Dataset):
#
#     def __init__(self):
#         # TODO: Create a Custom dataset
#
#         # Generate all 64 possible 6-bit binary vectors
#         data = list(itertools.product([0, 1], repeat=8))
#
#         labels = []
#         # for example in data:
#         for example in data:
#             label = 1
#             for i in range(8):
#                 if example[i] != example[-i]:
#                     label = -1
#                     break
#             labels.append(label)
#
#         torch.tensor(labels, dtype=torch.float32).view(-1, 1)
#
#         self.dataset = TensorDataset(torch.tensor(data), label)
#
#     def __gettitem__(self, index):
#         return self.dataset[index]
#
#     def __len__(self):
#         return len(self.dataset)
