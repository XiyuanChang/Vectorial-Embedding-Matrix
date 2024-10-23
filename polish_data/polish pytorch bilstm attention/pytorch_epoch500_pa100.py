import numpy as np

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt   
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

import pickle
from collections import Counter
from pytorchtools_save_epoch import EarlyStopping

torch.manual_seed(1)


rela_embedding=pd.read_csv('./data/new_Embedding_Model_POLISH_D128-rela.csv').drop('Unnamed: 0',axis=1)
tail_embedding=pd.read_csv('./data/tail_embedding_POLISH_D128.csv').drop('Unnamed: 0',axis=1)
x_test = pd.read_csv("./data/5year-Ts.csv")
polish_train_val=pd.read_csv("./data/5year-Tr.csv")
y_train_val=polish_train_val['LABEL']
y_test=x_test['LABEL']
x_test=x_test.drop(['LABEL','Company'],axis=1)
polish_train_val=polish_train_val.drop(['LABEL','Company'],axis=1)
x_trainval_mapping_df=pd.read_csv('./data/x_trainval_map_tailv.csv').drop('Unnamed: 0',axis=1)
x_test_mapping_df=pd.read_csv('./data/x_test_map_tailv.csv').drop('Unnamed: 0',axis=1)

filename='./data/test_notnull.pickle'
with open(filename, 'rb') as f:
    # deserialize the list and load it from the file
    loaded_test_notnull_lst = pickle.load(f)

test_notnull_len=[]
for i in range(len(loaded_test_notnull_lst)):
    test_notnull_len.append(len(loaded_test_notnull_lst[i]))

test_integrate=[]
for i in range(len(loaded_test_notnull_lst)):
  test_integrate.append(torch.tensor(loaded_test_notnull_lst[i]))



test_notnull_count = Counter(test_notnull_len)
test_batch_avg=round(np.average(list(dict(test_notnull_count).keys())))



# define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, data, labels):
        # sort the data and labels based on the length of each item in data
        self.data, self.labels = zip(*sorted(zip(data, labels), key=lambda x: len(x[0]), reverse=True))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
        y = self.labels[idx]
        return x, y

def collate_fn(data):
    input_data = [d[0] for d in data]
    y = [d[1] for d in data]
    input_data = pad_sequence(input_data, batch_first=True)
    input_data = torch.flip(input_data, dims=[1])
    return input_data, torch.tensor(y)

test_batch_avg=54


test_dataset = MyDataset(test_integrate, y_test.values.tolist())

# create a dataloader with batch size 2 and collate function
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_avg, collate_fn=collate_fn)

# iterate over batches of data and labels
for batch_data, batch_labels in test_dataloader:
    # print the batch data and labels
    print(batch_data.size())
    print(batch_labels.size())




with open('./data/train_data.pickle', 'rb') as f:
    # deserialize the list and load it from the file
    train_data = pickle.load(f)
with open('./data/val_data.pickle', 'rb') as f:
    # deserialize the list and load it from the file
    val_data = pickle.load(f)

with open('./data/train_y.pickle', 'rb') as f:
    # deserialize the list and load it from the file
    train_y = pickle.load(f)
with open('./data/val_y.pickle', 'rb') as f:
    # deserialize the list and load it from the file
    val_y = pickle.load(f)




train_batch_size_avg=46
val_batch_size_avg=55
train_dataset = MyDataset(train_data, train_y)
val_dataset=MyDataset(val_data, val_y)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size_avg, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size_avg, collate_fn=collate_fn)

for batch_data, batch_labels in val_dataloader:
    print(batch_data.size())
    print(batch_labels.size())




class bilstm_attention(nn.Module):
    def __init__(self, input_size, hidden_units,num_classes):
        super().__init__()
        self.input_size = input_size  # this is the number of features
        self.num_classes=num_classes
        self.hidden_units = hidden_units
        
        self.num_layers = 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=True,
            dropout=0.1
        )

        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(self.hidden_units * 2, 1))
        self.tanh2 = nn.Tanh()
        self.fc = nn.Linear(self.hidden_units * 2, self.num_classes)
        nn.init.uniform_(self.w, -0.1, 0.1)

       

    def forward(self, x):
        H, _ = self.lstm(x, None) # [batch_size, seq_len, hidden_size * 2]
        M = self.tanh1(H)  # [batch_size, seq_len, hidden_size * 2]
        # tensor operation
        alpha = F.softmax(torch.matmul(M, self.w), dim=1)# [batch_size, seq_len, 1]
        # When tensor elements are multiplied, tensor broadcasting occurs so that the dimensions of the tensor satisfy the condition
        out = H * alpha  # [batch_size, seq_len, hidden_size * 2]
        out = torch.sum(out, 1) # [batch_size,hidden_size * 2]  
        out = self.tanh2(out)
        out = self.fc(out)# [batch_size,num_classes]
        return out

batch_data,batch_label=next(iter(train_dataloader))


def train_model(device,train_loader, val_loader, model, patience, n_epochs):
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    num_batches_train = len(train_loader)
    total_loss = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for X, y in train_loader:
          
          X=X.to(device) 
          y=y.to(device,dtype=torch.float)
          #y=y.float() 
          output = model(X) 
          output=output.squeeze()
          
          # clear the gradients of all optimized variables
          optimizer.zero_grad()
          
          # calculate the loss
          loss = criterion(output, y)
          # backward pass: compute gradient of the loss with respect to model parameters
          loss.backward()
          # perform a single optimization step (parameter update)
          optimizer.step()
          # record training loss
          train_losses.append(loss.item())

        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for data, target in val_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            data=data.to(device) 
            target=target.to(device,dtype=torch.float)
            output = model(data)
            output=output.squeeze()
            #target=target.float()
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model,n_epochs+1, optimizer,criterion)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    checkpoint = torch.load('./model/save_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    with open('./model/avg_train_losses.pickle', 'wb') as f:
      pickle.dump(avg_train_losses, f)
    with open('./model/avg_val_losses.pickle', 'wb') as f:
      pickle.dump(avg_valid_losses, f)
    return model, avg_train_losses, avg_valid_losses

""" 
#train process
##### this piece of code is trainning process
device='cuda' if torch.cuda.is_available() else 'cpu'
model = bilstm_attention(input_size=batch_data.size()[2], hidden_units=512,num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
patience=100
bilstm_atten_savemodel_patience_300,train_loss,valid_loss = train_model(device,train_dataloader, val_dataloader, model, patience, 500)
"""

#reload model and predict
device='cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.BCEWithLogitsLoss()
model = bilstm_attention(input_size=batch_data.size()[2], hidden_units=512,num_classes=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

checkpoint_path='./model/save_checkpoint.pt'
checkpoint=torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

def predict(data_loader, model_1):
    output = torch.tensor([])
    model_1.eval()
    with torch.no_grad():
        for X, target in data_loader:
            y_star = model_1(X).squeeze()
            output = torch.cat((output, y_star), 0)

    return output


output_test = predict(test_dataloader, model).numpy()
output_val = predict(val_dataloader, model).numpy()


print(len(output_val))
print(len(output_test))

# extract original test labels and val labels from dataloaders since we've changed the order of the original labels when creating data_loader
test_labels = []
for batch_data, batch_labels in test_dataloader:
    test_labels.extend(batch_labels.tolist())

val_labels = []
for batch_data, batch_labels in val_dataloader:
    val_labels.extend(batch_labels.tolist())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

precision, recall, thresholds = precision_recall_curve(val_labels, output_val)

df_recall_precision = pd.DataFrame({'Precision': precision[:-1], 'Recall': recall[:-1], 'Threshold': thresholds})

np.seterr(divide='ignore', invalid='ignore')
f1_score = (2 * precision * recall) / (precision + recall)

# Find the optimal threshold
# findex = np.argmax(f1_score)
findex = list(f1_score).index(max(f1_score))
thresholdOpt = round(thresholds[findex], ndigits=4)
fscoreOpt = round(f1_score[findex], ndigits=4)
recallOpt = round(recall[findex], ndigits=4)
precisionOpt = round(precision[findex], ndigits=4)
print('Best Threshold: {} , F-Score: {}'.format(thresholdOpt, fscoreOpt))
print('Recall: {}, Precision: {}'.format(recallOpt, precisionOpt))

from sklearn.metrics import precision_score, classification_report, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score

y_pre_test = output_test.flatten().tolist()
defaulter_decision_test = (y_pre_test >= thresholdOpt)

classification_report_dct=classification_report(test_labels, defaulter_decision_test, output_dict=True)

print(classification_report(test_labels, defaulter_decision_test))
tn, fp, fn, tp = confusion_matrix(test_labels, defaulter_decision_test).ravel()
print(tn, fp, fn, tp)  # recall=TP/TP+FN
f1_test = f1_score(test_labels, defaulter_decision_test,
                   average='macro')  # use 'micro' or 'weighted' for multi-class problems depending on the problem
print(f'test dataset F1 Score: {f1_test}')

reportdf=pd.DataFrame(classification_report_dct).transpose()
pd.DataFrame({'Best Threshold':thresholdOpt, "Best val_f1_score": fscoreOpt, "Best val_recall ": recallOpt, "Best val_recall":precisionOpt, 'avg_test_f1_score':f1_test},index=[0]).to_csv('./model/fscore.csv')
reportdf.to_csv('./model/test_classification_report.csv')

with open('./model/avg_train_losses.pickle', 'rb') as f:
    # serialize the list and write it to the file
    avg_valid_losses = pickle.load(f)

with open('./model/avg_val_losses.pickle', 'rb') as f:
    # serialize the list and write it to the file
    avg_train_losses = pickle.load(f)

loss_df = pd.DataFrame({'train_loss': avg_train_losses, 'val_loss': avg_valid_losses,'index':range(0,len(avg_train_losses))})


# Plotting the data
plt.figure(figsize=(10, 5))
sns.lineplot(x='index', y='value', hue='variable', data=pd.melt(loss_df, ['index']), markers=True)

# Adding labels and title
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Polish PyTorch BiLSTM-Attention Loss", y=-0.2)  # y=-0.2 places the title at the middle bottom

# Display plot
plt.legend(title='Loss Type', loc='upper right', labels=['Training Loss', 'Validation Loss'])
plt.savefig("./model/pythroch_loss_plot.jpg", bbox_inches='tight')
plt.show()

