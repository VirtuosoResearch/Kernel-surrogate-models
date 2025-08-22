import torch

class Task:

    def __init__(self, config):
        self.config = config
        self.train_data = None
        self.valid_data = None
        self.train_num = None
        self.valid_num = None

    def get_model_hparams(self):
        raise NotImplementedError

    # Output: train_data, valid_data
    def get_data(self):
        raise NotImplementedError

    def get_train_dataset(self):
        assert self.train_data is not None
        train_dataset = torch.utils.data.TensorDataset(self.train_data)
        return train_dataset

    def get_valid_dataset(self):
        assert self.valid_data is not None
        valid_dataset = torch.utils.data.TensorDataset(self.valid_data)
        return valid_dataset

    def get_train_dataloader(self, batch_size):
        assert self.train_data is not None
        train_dataset = torch.utils.data.TensorDataset(self.train_data)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        return train_dataloader

    def get_valid_dataloader(self, batch_size):
        assert self.valid_data is not None
        valid_dataset = torch.utils.data.TensorDataset(self.valid_data)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        return valid_dataloader

    def get_loss(self):
        raise NotImplementedError

    def get_eval_metrics(self):
        raise NotImplementedError