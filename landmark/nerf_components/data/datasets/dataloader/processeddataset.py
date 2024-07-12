import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    """
    A dataset class for handling preprocessed data.

    This class supports loading preprocessed data from different sources, including a Ceph storage system
    or local files, based on the specified filetype. It is designed to be flexible in handling various
    data formats and storage solutions.

    Attributes:
        type (str): The type of data source, either "ceph" for Ceph storage or "file" for local files.
        client (Client): An instance of the Petrel client for accessing Ceph storage.
        num (int): The number of samples or data points in the dataset.

    Methods:
        __init__(): Initializes the dataset with the specified configuration.
    """

    def __init__(
        self,
        file_folder,
        filetype="ceph",
        conf_path="~/petreloss.conf",
    ):
        """
        Initializes the PreprocessedDataset with configuration parameters.

        Parameters:
            file_folder (str): The folder path where the data is stored.
            filetype (str): The type of data source, either "ceph" for Ceph storage or "file" for local files.
            conf_path (str): The path to the configuration file for the Petrel client, if using Ceph storage.

        Raises:
            ValueError: If an unsupported filetype is specified.
        """
        if filetype == "ceph":
            self.type = "ceph"
            from petrel_client.client import Client

            self.client = Client(conf_path)
            self.file_folder = file_folder
            data_url = self.file_folder + "num.npy"
            data_bytes = self.client.get(data_url)
            self.num = int(np.frombuffer(data_bytes, np.longlong))
        else:
            self.type = "file"
            self.file_folder = file_folder
            npzfile = np.load(self.file_folder + "num.npz")
            self.num = int(npzfile["num"])

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.type == "ceph":
            data_url = self.file_folder + str(idx).zfill(10) + ".npy"
            data_bytes = self.client.get(data_url)
            data_np = np.frombuffer(data_bytes, np.float32).reshape(8192, 9)
            data_tensor = torch.from_numpy(data_np)
            rays = data_tensor[:, :6].data
            rgbs = data_tensor[:, 6:9].data
            idxs = data_tensor[:, 9:].data
        else:
            npzfile = np.load(self.file_folder + str(idx).zfill(10) + ".npz")
            rays = torch.from_numpy(npzfile["rays"])
            rgbs = torch.from_numpy(npzfile["rgbs"])
            idxs = torch.from_numpy(npzfile["idxs"])
        return rays, rgbs, idxs
