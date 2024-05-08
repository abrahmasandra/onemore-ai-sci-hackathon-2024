"""
perform k-fold validation on GNN models
"""
import torch
import itertools
from copy import deepcopy
from typing import Callable
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from mpnn import MPNN, train_loop, test_mpnn


"""
This is a template you can use for cross-validation. The core idea is to provide a list 
of possible values for each hyperparameter in the model and optimizer. The other fields
are then considered as fixed. The function `generate_config_combs` will generate all possible
combinations of the hyperparameters. You can then use this to perform cross-validation.
"""
CV_CONFIG_TEMPLATE = {
    "optim_config": {
        "cls": "Adam", # "Adam", "SGD", "RMSprop
        "lr": [0.005, 0.01, 0.05],
        "batch_size": 64
    },
    "model_config": {
        "hidden_channels": 64,
        "num_layers": [3, 4, 5],
    }
}

CV_LR_ONLY = {
    "optim_config": {
        "cls": "Adam", # "Adam", "SGD", "RMSprop
        "lr": [0.005, 0.01, 0.05],
        "batch_size": 64
    },
    "model_config": {
        "hidden_channels": 64,
        "num_layers": 3,
    }
}


def get_cv_dataset_loader(filename="data.json", one_hot_enc_atomic_num: bool = False):
    """
    Closure to load the dataset for cross-validation without rereading the file
    """
    cv_dataset = None
    def loader():
        nonlocal cv_dataset
        if cv_dataset is not None:
            return cv_dataset
        from read_data import read_pyg_data_from_file
        cv_dataset = read_pyg_data_from_file(filename, one_hot_enc_atomic_num)
        return cv_dataset
    return loader


def generate_config_combs(config):
    """
    Generate all possible combinations of configurations from a template
    """
    values, optim_keys, model_keys = [], [], []
    # handle optim_config and model_config
    for key in config['optim_config']:
        value = config['optim_config'][key]
        if isinstance(value, list):
            values.append(value)
            optim_keys.append(key)

    for key in config['model_config']:
        value = config['model_config'][key]
        if isinstance(value, list):
            values.append(value)
            model_keys.append(key)

    all_keys = optim_keys + model_keys
    for comb in itertools.product(*values):
        solid_copy = deepcopy(config)
        for key, value in zip(all_keys, comb):
            if key in optim_keys:
                solid_copy['optim_config'][key] = value
            else:
                solid_copy['model_config'][key] = value
        
        yield solid_copy


def _verify_solid_values(config: dict):
    """
    verify that all values in the config are solid, not lists
    """
    for key in config['optim_config']:
        value = config['optim_config'][key]
        if isinstance(value, list):
            return False

    for key in config['model_config']:
        value = config['model_config'][key]
        if isinstance(value, list):
            return False
    return True


def cv_for_one_config(cfg: dict, loader: Callable, num_folds: int = 5, num_epochs: int = 100, verbose: bool = False):
    assert _verify_solid_values(cfg), "All values in the config must be solid, not lists"
    dataset = loader()

    kf = KFold(n_splits=num_folds)
    # pre-compute the number of features
    num_feat = dataset[0].x.size(1)

    test_loss = []

    # TODO: control this with another verbose?
    print(f"Starting {num_folds}-fold with config \n {cfg}")
    for train_idx, test_idx in kf.split(dataset):
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset = [dataset[i] for i in test_idx]

        train_loader = DataLoader(train_dataset, batch_size=cfg["optim_config"]["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg["optim_config"]["batch_size"], shuffle=False)

        model = MPNN(num_feat, cfg["model_config"]["hidden_channels"], 4, cfg["model_config"]["num_layers"])

        optimizer_cls = getattr(torch.optim, cfg["optim_config"]["cls"])
        optimizer = optimizer_cls(model.parameters(), lr=cfg["optim_config"]["lr"])

        # perform training
        train_loop(train_loader, model, optimizer, num_epochs, verbose)

        # testing and obtain loss
        fold_loss = test_mpnn(test_loader, model)
        if verbose:
            print(f"Fold loss: {fold_loss}")
        test_loss.append(fold_loss)
    
    avg_loss = sum(test_loss) / num_folds
    return test_loss, avg_loss
    

def cross_validation(config: dict, loader: Callable, num_folds: int = 5, num_epochs: int = 100, verbose: bool = False):
    """
    Perform cross-validation on a given configuration
    """
    for cfg in generate_config_combs(config):
        test_loss, avg_loss = cv_for_one_config(cfg, loader, num_folds, num_epochs, verbose)
        print(f"Configuration: {cfg}, Loss: {avg_loss}, Test Loss: {test_loss}")


if __name__ == '__main__':
    # Example usage:
    loader = get_cv_dataset_loader(one_hot_enc_atomic_num=True)
    cross_validation(CV_LR_ONLY, loader, num_epochs=20, num_folds=3)