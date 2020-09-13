config = {
    'bs': 32,
    'maxlen': 5000,
    'n_channels': 100,
    'epochs': 27,
    'best_val_loss': 1e7,
    'lr_reduce_parametr': 0.5,
    'change_lr_treshhold': 2,
    'early_stopping_criteria': 5,
    'tolerance': 5e-4,
    'path_to_data_train': '../../train_values.csv',
    'path_to_labels_train': '../../train_labels.csv',
}
