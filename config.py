config = {
    'bs': 32,
    'maxlen': 5000,
    'n_channels': 500,
    'epochs': 50,
    'best_val_loss': 1e7,
    'lr_reduce_parameter': 0.5,
    'change_lr_threshold': 2,
    'early_stopping_criteria': 4,
    'tolerance': 5e-4,
    'path_to_data_train': '../train_values.csv',
    'path_to_labels_train': '../train_labels.csv',
}


config_test = {"val": False}