import pandas as pd
from neural_net.data_prep import DataPreparation
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from neural_net.nn import MyCNN
from config import config_test, config


preparator_train = DataPreparation()
kernel_sizes = [i for i in range(5, 26, 10)] + [i for i in range(40, 120, 30)]
model = MyCNN(config['n_channels'], kernel_sizes)
model.cuda()
answers = []
bs = config['bs']
if config_test['val']:
    data = pd.read_csv(config['path_to_data_train'])
    train_labels = pd.read_csv(config['path_to_labels_train'])
    y = np.where(train_labels.drop('sequence_id', axis=1).values == 1.0)[1]
    X_train, X_test, y_train, y_test = train_test_split(data, y, random_state=2020, test_size=0.1)
    X_test_df = X_test.copy()
    X_test, X_test_one_hot = preparator_train.transform(5000, X_test)
    model.load_state_dict(torch.load('model_dict.h5'))
else:
    test_data = pd.read_csv('../test_values.csv')
    X_test, X_test_one_hot = preparator_train.transform(5000, test_data)
    model.load_state_dict(torch.load('trained_on_val.h5'))
with torch.no_grad():
    model.eval()
    number_of_steps_test = int(len(X_test) / bs) + 1
    for i in range(number_of_steps_test):
        x = X_test[i * bs: (i + 1) * bs].cuda()
        x_one_hot = X_test_one_hot[i * bs: min((i + 1) * bs, len(X_test))].cuda()

        y_pred = model(x, x_one_hot).softmax(-1)
        pp = y_pred.cpu().numpy()
        answers.extend(pp)

answers = np.array(answers).astype('float16')
print(answers.shape)
#

if config_test['val']:
    sample_sub = train_labels.iloc[X_test_df.index]
    sample_sub[sample_sub.columns[1:]] = answers
    sample_sub.to_csv('sample_sub_val.csv', index=False)
else:
    sample_sub = pd.read_csv('../submission_format_3TFRxH6.csv')
    sample_sub[sample_sub.columns[1:]] = answers
    sample_sub.to_csv('sample_sub_test.csv', index=False)