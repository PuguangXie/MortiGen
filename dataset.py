import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import warnings
from utils import normalization, renormalization, rounding

warnings.filterwarnings("ignore")


class EICUDataset(Dataset):
    def __init__(self, mode='train', missing_ratio=0.0, seed=0):
        np.random.seed(seed)

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []

        # Load data based on the specified mode
        file_path = './data/ami_nh3.xlsx'
        if mode == 'train':
            sheet_name = 'train'
            target_column = 'hospitaldischargestatus'
        elif mode == 'test':
            sheet_name = 'test'
            target_column = 'hospitaldischargestatus'
        elif mode == 'val':
            sheet_name = 'valnh'
            target_column = 'hospital_expire_flag'
        else:
            raise ValueError("Mode should be 'train', 'test', or 'val'.")

        observed_data = pd.read_excel(file_path, sheet_name=sheet_name)
        self.status = observed_data[target_column]
        self.patient_id = observed_data["hadm_id"]
        columns_to_drop = [
            target_column, 'hadm_id', 'cardiac_arrest', 'cardiogenic_shock',
            'pulmonary_edema', 'st', 'LOS'
        ]
        observed_data = observed_data.drop(columns=columns_to_drop, axis=1)

        # One-hot encode categorical data
        observed_data = pd.get_dummies(observed_data, columns=['race_Asian0_black1_hispanic2_white3_other4'], prefix='race')
        observed_data = observed_data.astype("float32")

        # Initialize masks and replace NaNs
        self.observed_masks = ~np.isnan(observed_data)
        self.observed_values = np.nan_to_num(observed_data)
        self.gt_masks = self.observed_masks.astype("float32")

        if mode == 'train':
            self._normalize_train_data()
        elif mode in ['test', 'val']:
            self._normalize_test_data()

    def _normalize_train_data(self):
        self.observed_values[:, :31], self.norm_parameters = normalization(self.observed_values[:, :31], self.observed_masks[:, :31])

    def _normalize_test_data(self):
        mean = np.array([...])  # Replace with actual mean values
        std = np.array([...])  # Replace with actual std values
        self.observed_values[:, :31] = (self.observed_values[:, :31] - mean) / std * self.observed_masks[:, :31]
        self.norm_parameters = 0

    def __getitem__(self, index):
        return {
            "patient_id": self.patient_id[index],
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "status": self.status[index],
            "norm_parameters": self.norm_parameters,
        }

    def __len__(self):
        return len(self.observed_values)


def get_dataloader(seed=1234, batch_size=16, missing_ratio=0.1):
    """Create and return train and validation DataLoaders."""
    train_dataset = EICUDataset(mode='train', missing_ratio=missing_ratio, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = EICUDataset(mode='val', missing_ratio=missing_ratio, seed=seed)
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False)

    return train_loader, valid_loader


class MIMICDataset(Dataset):
    def __init__(self, indices=None, missing_ratio=0.0, seed=0):
        np.random.seed(seed)

        observed_data = pd.read_excel('./data/ami_nh3.xlsx', sheet_name='val3h')
        self.status = observed_data["hospital_expire_flag"]
        self.patient_id = observed_data["hadm_id"]

        columns_to_drop = [
            'hospital_expire_flag', 'hadm_id', 'cardiac_arrest', 'cardiogenic_shock',
            'pulmonary_edema', 'troponin_i', 'st', 'LOS'
        ]
        observed_data = observed_data.drop(columns=columns_to_drop, axis=1)

        observed_data = pd.get_dummies(observed_data, columns=['race_Asian0_black1_hispanic2_white3_other4'], prefix='race')
        observed_data = observed_data.astype("float32")

        self.observed_masks = ~np.isnan(observed_data)
        self.observed_values = np.nan_to_num(observed_data)
        self.gt_masks = self.observed_masks.astype("float32")

        mean = np.array([...])  # Replace with actual mean values
        std = np.array([...])  # Replace with actual std values
        self.observed_values[:, :31] = (self.observed_values[:, :31] - mean) / std * self.observed_masks[:, :31]
        self.norm_parameters = 0

        self.use_index_list = np.arange(len(self.observed_values)) if indices is None else indices

    def __getitem__(self, index):
        index = self.use_index_list[index]
        return {
            "patient_id": self.patient_id[index],
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "status": self.status[index],
            "norm_parameters": self.norm_parameters,
        }

    def __len__(self):
        return len(self.use_index_list)


def get_ext_dataloader(seed=0, batch_size=2048, missing_ratio=0):
    """Create and return an external test DataLoader."""
    dataset = MIMICDataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))
    ext_dataset = MIMICDataset(indices=indlist, missing_ratio=missing_ratio, seed=seed)
    ext_loader = DataLoader(ext_dataset, batch_size=batch_size, shuffle=False)
    return ext_loader