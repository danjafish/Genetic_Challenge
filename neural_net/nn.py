import torch
import torch.nn.functional as F


class MyCNN(torch.nn.Module):

    def __init__(self, n_channels, kernel_sizes):
        super().__init__()
        ed = 20
        self.emb = torch.nn.Embedding(num_embeddings=6, embedding_dim=ed)
        #kernel_sizes = [i for i in range(5, 100, 10)]# [50, 5, 25, 80, 10]#
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv1d(in_channels=ed, out_channels=n_channels, kernel_size=ks)
                                                for ks in kernel_sizes])
        self.bn_layers = torch.nn.ModuleList([torch.nn.BatchNorm1d(n_channels) for _ in kernel_sizes])
        self.do_layers = torch.nn.ModuleList([torch.nn.Dropout(0.2) for _ in kernel_sizes])

        self.bn0 = torch.nn.BatchNorm1d(ed)
        self.bn1 = torch.nn.BatchNorm1d(9)
        self.bn2 = torch.nn.BatchNorm1d(128)

        self.do = torch.nn.Dropout(0.1)
        self.hidden_do = torch.nn.Dropout(0.5)

        self.dense_for_one_hot = torch.nn.Linear(40, 9)
        self.one_hot_to_weights = torch.nn.Linear(40, ed)
        self.dense_hidden = torch.nn.Linear(n_channels * len(self.conv_layers) + 9, 128)
        self.dense = torch.nn.Linear(128, 1314)


    def forward(self, x, y):
        w = torch.sigmoid(self.one_hot_to_weights(y))
        x = self.emb(x)
        x = w.unsqueeze(1) * x
        x = x.transpose(-1, -2)
        x = self.bn0(x)

        output_layers = []
        for ind, layer in enumerate(self.conv_layers):
            output_layers.append(F.relu(layer(x)))
            output_layers[-1] = self.bn_layers[ind](output_layers[-1])
            output_layers[-1] = F.max_pool2d(output_layers[-1], kernel_size=(1, output_layers[-1].size()[-1]))
            #output_layers[-1] = self.do_layers[ind](output_layers[-1])
            #output_layers.append(x1)
            # if output_layers == 0:
            #     z = x1
            #     output_layers+=1
            # else:
        x = torch.cat(output_layers, dim=2)

        x = x.reshape((x.shape[0], -1))

        y = self.dense_for_one_hot(y)
        y = self.bn1(y)
        y = y.reshape((y.shape[0], -1))

        x = torch.cat([x, y], dim=1)
        # print(x.shape)
        x = self.hidden_do(x)
        x = self.dense_hidden(x)
        x = self.bn2(x)
        x = self.do(x)
        x = self.dense(x)

        return x




class Cnn_seq(torch.nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        ed = 30
        self.emb = torch.nn.Embedding(num_embeddings=6, embedding_dim=ed)
        self.conv1 = torch.nn.Conv1d(in_channels=ed, out_channels=n_channels, kernel_size=50)
        self.conv2 = torch.nn.Conv1d(in_channels=ed, out_channels=n_channels, kernel_size=5)
        self.conv3 = torch.nn.Conv1d(in_channels=ed, out_channels=n_channels, kernel_size=25)
        self.conv4 = torch.nn.Conv1d(in_channels=ed, out_channels=n_channels, kernel_size=80)
        self.conv5 = torch.nn.Conv1d(in_channels=ed, out_channels=n_channels, kernel_size=10)
        # self.conv6 = torch.nn.Conv1d(in_channels=20, out_channels=n_channels, kernel_size=40)
        self.lstm1 = torch.nn.LSTM(input_size=ed, hidden_size=128, batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)

        self.bn1 = torch.nn.BatchNorm1d(n_channels)
        self.bn0 = torch.nn.BatchNorm1d(ed)
        self.bn2 = torch.nn.BatchNorm1d(n_channels)
        self.bn3 = torch.nn.BatchNorm1d(n_channels)
        self.bn4 = torch.nn.BatchNorm1d(n_channels)
        self.bn5 = torch.nn.BatchNorm1d(n_channels)
        self.bn6 = torch.nn.BatchNorm1d(n_channels)
        self.do1 = torch.nn.Dropout(0.2)
        self.do2 = torch.nn.Dropout(0.2)
        self.do3 = torch.nn.Dropout(0.2)
        self.do4 = torch.nn.Dropout(0.2)
        self.do5 = torch.nn.Dropout(0.2)
        self.do = torch.nn.Dropout(0.1)
        self.dense_for_one_hot = torch.nn.Linear(40, 9)
        self.one_hot_to_weights = torch.nn.Linear(40, ed)
        self.dense_for_lstm_and_cnn = torch.nn.Linear(n_channels * 5 + 2432, 64)
        self.dense = torch.nn.Linear(64, 1314)

    def forward(self, x, y):
        w = torch.sigmoid(self.one_hot_to_weights(y))
        x = self.emb(x)
        x = w.unsqueeze(1) * x
        x = x.transpose(-1, -2)
        x = self.bn0(x)

        z = self.lstm1(x.transpose(-1, -2))
        z = self.lstm2(z[0])
        z = F.max_pool1d(z[0].transpose(-1, -2), kernel_size=(500), stride=250)
        z = z.reshape(z.shape[0], -1)

        x1 = F.relu(self.conv1(x))
        x1 = self.bn1(x1)
        x1 = F.max_pool2d(x1, kernel_size=(1, x1.size()[-1]))
        x1 = self.do1(x1)
        # print('first conv ', x1.shape)

        x2 = F.relu(self.conv2(x))
        x2 = self.bn2(x2)
        x2 = F.max_pool2d(x2, kernel_size=(1, x2.size()[-1]))
        x2 = self.do2(x2)

        x3 = F.relu(self.conv3(x))
        x3 = self.bn3(x3)
        x3 = F.max_pool2d(x3, kernel_size=(1, x3.size()[-1]))
        x3 = self.do3(x3)

        x4 = F.relu(self.conv4(x))
        x4 = self.bn4(x4)
        x4 = F.max_pool2d(x4, kernel_size=(1, x4.size()[-1]))
        x4 = self.do4(x4)

        x5 = F.relu(self.conv5(x))
        x5 = self.bn5(x5)
        x5 = F.max_pool2d(x5, kernel_size=(1, x5.size()[-1]))
        x5 = self.do5(x5)
        #         x6 = F.relu(self.conv6(x))
        #         x6 = self.bn6(x6)
        #         x6 = F.max_pool2d(x6, kernel_size=(1, x6.size()[-1]))

        x = torch.cat([x1, x2, x3, x5, x4], dim=2)
        # print('after cat ', x.shape)
        x = x.reshape((x.shape[0], -1))
        x = torch.cat([x, z], 1)
        x = self.dense_for_lstm_and_cnn(x)

        y = self.dense_for_one_hot(y)
        y = y.reshape((y.shape[0], -1))

        x = torch.cat([x, y], dim=1)
        # print('after reshape ', x.shape)
        x = self.do(x)
        x = self.dense(x)
        # x = F.softmax(x, dim=-1)

        return x