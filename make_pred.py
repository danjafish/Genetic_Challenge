import pandas as pd
from neural_net.data_prep import DataPreparation
import torch
import numpy as np
from neural_net.nn import MyCNN

test_data = pd.read_csv('../test_values.csv')
preparator_train = DataPreparation()
X_test, X_test_one_hot = preparator_train.transform(5000, test_data)

model = MyCNN(100)
model.cuda()
model.load_state_dict(torch.load('model_dict.h5'))

bs = 32
answers = []
with torch.no_grad():
    model.eval()
    number_of_steps_test = int(len(X_test) / bs) + 1
    for i in range(number_of_steps_test):
        x = X_test[i * bs: (i + 1) * bs].cuda()
        x_one_hot = X_test_one_hot[i * bs: (i + 1) * bs].cuda()

        y_pred = model(x, x_one_hot).softmax(-1)
        pp = y_pred.cpu().numpy()
        answers.extend(pp)

answers = np.array(answers).astype('float16')
sample_sub = pd.read_csv('../submission_format_3TFRxH6.csv')
sample_sub[sample_sub.columns[1:]] = answers
sample_sub.to_csv('sample_sub.csv', index=False)
