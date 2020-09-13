from torch.nn.functional import cross_entropy
from utils.support_func import top10acc
from sklearn.metrics import accuracy_score
import numpy as np
import torch

class TrainLoop:
    def __init__(self, bs, epochs, lr_reduce_parametr, change_lr_treshhold,
                 early_stopping_criteria, model, optimizer, X_train,
                 y_train, X_test, y_test, X_train_one_hot, X_test_one_hot, tolerance=5e-4, lr=0.001):
        self.bs = bs
        self.epochs = epochs
        self.lr_reduce_parametr = lr_reduce_parametr
        self.change_lr_treshhold = change_lr_treshhold
        self.early_stopping_criteria = early_stopping_criteria
        self.model = model
        self.optimizer = optimizer
        self.number_of_steps_train = len(X_train) // bs + 1
        self.number_of_steps_test = len(X_test) // bs + 1
        self.X_train = X_train
        self.X_test = X_test
        self.X_train_one_hot = X_train_one_hot
        self.X_test_one_hot = X_test_one_hot
        self.y_train = y_train
        self.y_test = y_test
        self.lr = lr
        self.mse_min = 1e-10
        self.change_lr = 0
        self.tolerance = tolerance

    def lr_sheduler(self, mse, optimizer, top10auc, model):
        print('=====================')
        if mse-self.mse_min > self.tolerance:
            self.change_lr = 0
            print("Loss improved from {} to {}".format(self.mse_min, mse))
            print('=====================')
            self.mse_min = mse
            torch.save(model.state_dict(), 'model_dict.h5'.format(np.round(mse, 3), np.round(top10auc, 3)))
            torch.save(optimizer.state_dict(), 'opt_dict.dict')
        else:
            print("Loss not improved ", mse)
            self.change_lr += 1
            print('loss not getting better for {} epochs'.format(self.change_lr))
            if self.change_lr % self.change_lr_treshhold == 0:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * self.lr_reduce_parametr
                print('lr reduced by {}'.format(self.lr_reduce_parametr))
            print('=====================')
            if self.change_lr == self.early_stopping_criteria:
                return True
            return False

    def train_one_epoch(self, print_step, X_train, X_train_one_hot, y_train):
        self.model.train()
        torch.cuda.empty_cache()
        train_loss = 0
        number_of_steps_train = int(len(X_train)/self.bs)+1
        for i in range(number_of_steps_train):
            x = X_train[i * self.bs: (i + 1) * self.bs].cuda()
            x_one_hot = X_train_one_hot[i * self.bs: (i + 1) * self.bs].cuda()
            y = y_train[i * self.bs: (i + 1) * self.bs].cuda()

            y_pred = self.model(x, x_one_hot)
            # print(y_pred.shape)
            loss = cross_entropy(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() / number_of_steps_train
            if i % print_step == 0:
                print(f"{i} {train_loss}")
        return train_loss

    def val_one_epoch(self):
        val_loss = 0
        acc10 = 0
        acc = 0
        with torch.no_grad():
            self.model.eval()
            for i in range(self.number_of_steps_test):
                x = self.X_test[i * self.bs: (i + 1) * self.bs].cuda()
                x_one_hot = self.X_test_one_hot[i * self.bs: (i + 1) * self.bs].cuda()
                y = self.y_test[i * self.bs: (i + 1) * self.bs].cuda()

                y_pred = self.model(x, x_one_hot)
                val_loss += torch.nn.functional.cross_entropy(y_pred, y).item() / self.number_of_steps_test

                y = y.cpu().numpy()

                p = y_pred.argmax(dim=-1).cpu().numpy()
                pp = y_pred.cpu().numpy()
                prediction_top_10 = np.argsort(pp, axis=-1)[:, -10:]
                # print(prediction_top_10[:5], y[:5])
                # print(p.shape, pp.shape, prediction_top_10.shape)
                # print(top10acc(y.cpu(), prediction_top_10))
                acc10 += top10acc(y, prediction_top_10) / self.number_of_steps_test
                acc += accuracy_score(p, y) / self.number_of_steps_test
        if not self.lr_sheduler(acc10, self.optimizer, val_loss, self.model):
            return val_loss, acc, acc10, False
        else: return val_loss, acc, acc10, True

    def train_loop(self):
        for e in range(self.epochs):
            train_loss = self.train_one_epoch(200, self.X_train, self.X_train_one_hot, self.y_train)
            val_loss, acc, acc10, stop_train = self.val_one_epoch()
            print(f"{e} {train_loss} {val_loss} {acc} {acc10}")
            if stop_train:
                break

    def train_on_val(self, epochs):
        for g in self.optimizer.param_groups:
            lr = g['lr']
            break
        if lr >= 0.0001:
            for g in self.optimizer.param_groups:
                g['lr'] = 0.0001

        X_all = torch.cat([self.X_train, self.X_test], dim=0)
        X_all_one_hot = torch.cat([self.X_train_one_hot, self.X_test_one_hot], dim=0)
        y_all = torch.cat([self.y_train, self.y_test], dim=0)

        for e in range(epochs):
            train_loss = self.train_one_epoch(200, X_all, X_all_one_hot, y_all)
            torch.save(self.model.state_dict(), 'trained_on_val.h5')
            print('Train on val ', e, train_loss)
