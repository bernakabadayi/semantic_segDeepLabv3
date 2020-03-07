from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def calculate_validation_loss_acc(self, val_loader, device, model):
        """
        Calculate validation loss and accuracy for all batches.
        """
        val_losses = []
        val_scores = []

        for images,targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss= self.loss_func(outputs, targets)
            val_losses.append(loss.detach().numpy())

            _, predicted = torch.max(outputs, 1)
            targets_mask = targets >= 0

            scores = np.mean((predicted == targets)[targets_mask].data.cpu().numpy())
            val_scores.append(scores)
        
        val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
        return val_acc, val_loss

    #Calculate train loss for every iteration, validation loss for every epoc
    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        for epoch in range(num_epochs):

            train_loss = 0
            for iteration, (images, targets) in enumerate(train_loader,1):    # 5 tane gonderdim
                images = images.to(device)
                targets = targets.to(device)
             
                optim.zero_grad()
                outputs = model(images)

                loss = self.loss_func(outputs, targets)
                loss.backward()
                optim.step()

                train_loss += loss.item()

                print("[Iteration %d/%d] TRAIN loss: %f" % (iteration, iter_per_epoch, loss))
        

            targets_mask = targets >= 0
            _, predicted = torch.max(outputs, 1)
            train_acc = np.mean((predicted == targets)[targets_mask].data.cpu().numpy())
            
            self.train_acc_history.append(train_acc)
            self.train_loss_history.append(train_loss/iteration)

            # Calculate validation loss & accuracy for every epoc
            val_acc, val_loss = self.calculate_validation_loss_acc(val_loader, device, model)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)

            print("\n[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f" % (epoch + 1,
                                                           num_epochs,
                                                           train_acc,
                                                           self.train_loss_history[-1]))

            print("[Epoch %d/%d] VAL acc/loss: %.3f/%.3f\n" % (epoch + 1, 
                                                            num_epochs, 
                                                            val_acc,
                                                            val_loss))

        print('FINISH.')
