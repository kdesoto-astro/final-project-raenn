import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader

import torchvision.transforms as transforms

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
#from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time

def get_predictions(model, iterator, device):
    """
    Given a trained model, returns the test images, test labels,
    and predicttion probabilities across all the test labels.
    """

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


def calculate_accuracy(y_pred, y):
    """
    Calculate the prediction accuracy.
    """
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


class MLP(nn.Module):
    """
    The Multi-Layer Perceptron. Sets number of layers
    ande node per layer.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        first_layer = 32
        middle_layer = 32
        last_layer = 32
        self.input_fc = nn.Linear(input_dim, first_layer)
        self.hidden_fc = nn.Linear(first_layer, middle_layer)
        self.hidden_fc2 = nn.Linear(middle_layer, middle_layer)
        self.hidden_fc3 = nn.Linear(middle_layer, middle_layer)
        self.hidden_fc4 = nn.Linear(middle_layer, last_layer)
        #self.hidden_fc5 = nn.Linear(middle_layer, last_layer)
        self.output_fc = nn.Linear(last_layer, output_dim)

    def forward(self, x):

        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        h_1 = F.relu(self.input_fc(x))
        h_2 = F.relu(self.hidden_fc(h_1))
        h_3 = F.relu(self.hidden_fc2(h_2))
        h_4 = F.relu(self.hidden_fc3(h_3))
        h_5 = F.relu(self.hidden_fc4(h_4))
        #h_6 = F.relu(self.hidden_fc4(h_5))

        y_pred = self.output_fc(h_5)

        return y_pred, h_5
    

def create_dataset(features, labels):
    """
    Creates PyTorch dataset object from numpy arrays.
    """
    tensor_x = torch.Tensor(features) # transform to torch tensor
    tensor_y = torch.Tensor(labels).type(torch.LongTensor)

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    return my_dataset


def train(model, iterator, optimizer, criterion, device):
    """
    Does one epoch of training for a given torch model.
    """

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        
        y_pred, _ = model(x)
        
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    """
    Evaluates the model for the validation set.
    """

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    """
    Sets the time it takes for each epoch to train.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run_mlp(train_data, test_data, input_dim, output_dim):
    """
    Run the MLP initialization and training. Closely follows
    the demo https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb
    """
    SEED = 1234
    EPOCHS = 400
    VALID_RATIO = 0.9
    BATCH_SIZE = 32

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    
    # Create validation dataset
    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples
    
    train_data, valid_data = data.random_split(train_data,
                                           [n_train_examples, n_valid_examples])
    
    valid_data = copy.deepcopy(valid_data)
    
    
    # Create DataLoaders


    train_iterator = data.DataLoader(train_data,
                                     shuffle=True,
                                     batch_size=BATCH_SIZE)

    valid_iterator = data.DataLoader(valid_data,
                                     batch_size=BATCH_SIZE)

    test_iterator = data.DataLoader(test_data,
                                    batch_size=BATCH_SIZE)
                                   
    #Create model
    model = MLP(input_dim, output_dim)
    lr=1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    
    best_valid_loss = float('inf')
    
    # for plotting
    train_acc_arr = []
    train_loss_arr = []
    val_acc_arr = []
    val_loss_arr = []
    for epoch in np.arange(0, EPOCHS):
        #if epoch % 25 == 0:
        #    lr /= 2.
        #    optimizer = optim.Adam(model.parameters(), lr=lr)
        #    print("learning rate updated", lr)

        start_time = time.monotonic()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'superfit-model.pt')

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        
        #plotting
        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_acc)
        val_loss_arr.append(valid_loss)
        val_acc_arr.append(valid_acc)
        
    model.load_state_dict(torch.load('superfit-model.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    
    images, labels, probs = get_predictions(model, test_iterator, device)
    pred_labels = torch.argmax(probs, 1)
    max_probs = np.amax(probs.numpy(), axis=1)
    
    #plotting of accuracy and loss for one epoch
    plt.plot(np.arange(0, EPOCHS), train_acc_arr, label="Training")
    plt.plot(np.arange(0, EPOCHS), val_acc_arr, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("figs/accuracy.png")
    plt.close()
    
    plt.plot(np.arange(0, EPOCHS), train_loss_arr, label="Training")
    plt.plot(np.arange(0, EPOCHS), val_loss_arr, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig("figs/loss.png")
    plt.close()
    
    return labels.numpy(), pred_labels.numpy(), max_probs
    
                
                                   
    
    
    
    
    



