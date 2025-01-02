import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import unidecode
import re


def process_data(data_path):
    # Read data form csv
    df = pd.read_csv(data_path,
                    names=["sentiment","content"],
                    encoding="ISO-8859-1")

    # Liệt kê và đổi các class thành số
    classes = df["sentiment"].unique()
    classes = {class_name : idx for idx, class_name in enumerate(classes)}
    print( classes )

    df["sentiment"] = df["sentiment"].apply( lambda x : classes[x])

    # Tiền xử lý dữ liệu
    english_stop_words = stopwords.words("english") # Các stop words trong tiếng anh
    stemmer = PorterStemmer() # Chuyển đổi các từ về cùng thì hiện tại đơn

    def text_preprocess(text):
        text = text.lower()
        text = unidecode.unidecode(text)
        text = text.strip()
        text = re.sub(r'[^\w\s]','', text)
        text = " ".join([word for word in text.split(" ") if word not in english_stop_words])
        text = " ".join([stemmer.stem(word) for word in text.split(" ") if word not in english_stop_words])
        return text 

    df["content"] = df["content"].apply(lambda x : text_preprocess(x))
    
    return df

def build_dictionary(df):
    # Xây dựng bộ từ điển với các từ có trong dataframe
    vocab = []
    for sentence in df["content"]:
        tokens = sentence.split()
        for token in tokens:
            if token not in vocab:
                vocab.append( token )

    # Thêm 2 token UNK và PAD biểu thị lần lượt cho:
    # UNK: Các từ không có trong từ điển - stop word
    # PAD: Các câu không đủ độ dài của 1 vector từ
    vocab.append("UNK")
    vocab.append("PAD")

    dic = {word : idx 
           for idx, word in enumerate(vocab)}
    
    return dic

def transform(text, word_to_idx, max_seq_len):
    tokens = []
    for word in text.split():
        try:
            word_ids = word_to_idx[word]
        except:
            word_ids = word_to_idx["UNK"]
    
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    elif len(tokens) < max_seq_len:
        denta_len = max_seq_len - len(tokens)
        tokens.extend( [word_to_idx["PAD"]]*denta_len )
    
    return tokens

def split_data(df):
    val_size = 0.2
    test_size = 0.125
    is_shuffle = True

    labels = df["sentiment"].tolist()
    texts = df["content"].tolist()


    X_train, X_val,y_train, y_val = train_test_split(texts, labels,
                                                      test_size=val_size,
                                                      random_state=SEED,
                                                      shuffle=is_shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                         test_size=test_size,
                                                         random_state=SEED,
                                                         shuffle=is_shuffle)

    return X_train, y_train, X_val, y_val, X_test, y_test
    
class FinancialNews(Dataset):
    def __init__(self, X, y, word_to_idx, max_seq_len, transform=None):
        self.X = X
        self.y = y
        self.word_to_idx = word_to_idx
        self.max_seq_len = max_seq_len
        self.transform = transform
    
    def __len__(self):
        return len( self.y )

    def __getitem__(self, index):
        text = self.X[ index ]
        label = self.y[ index ]

        text = self.transform(text, self.word_to_idx, self.max_seq_len)
        text = torch.tensor( text )

        return text, label

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,
                 n_layers, n_classes, dropout_prob):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim,hidden_size, 
                          n_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.embedding( x )
        x, hn = self.rnn( x )
        x = x[:, -1, :]
        x = self.norm( x )
        x = self.dropout( x )
        x = self.fc1( x )
        x = self.relu( x )
        x = self.fc2( x )
        return x

def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for inputs,labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            losses.append( loss.item() )
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    loss = sum(losses) / len(losses)
    acc = correct / total

    return loss, acc


def fit(model, train_loader, val_loader, 
        criterion, optimizer, device, epochs):
    
    train_losses = []
    val_losses = []

    for epoch in range( epochs ):
        bath_train_losses = []
        model.train()

        for idx, (inputs,labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bath_train_losses.append( loss.item() )
        
        train_loss = sum(bath_train_losses)/len(bath_train_losses)
        train_losses.append( train_loss )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append( val_loss )

        print(f"Epoch {epoch+1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}")
    
    return train_losses, val_losses

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("Number of GPUs:", torch.cuda.device_count())
    else:
        print("CUDA is not available. Using CPU.")

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    SEED = 1
    set_seed(SEED)

    dataset_path = "./financial_news_cls/dataset/all-data.csv"
    df = process_data(data_path=dataset_path)
    dictionary = build_dictionary(df)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
    
    max_seq_len = 32
    train_batch_size = 128
    test_batch_size = 8

    train_dataset = FinancialNews(X_train, y_train,
                                  word_to_idx=dictionary,
                                  max_seq_len=max_seq_len,
                                  transform=transform)
    val_dataset = FinancialNews(X_val, y_val,
                                  word_to_idx=dictionary,
                                  max_seq_len=max_seq_len,
                                  transform=transform)
    test_dataset = FinancialNews(X_test, y_test,
                                  word_to_idx=dictionary,
                                  max_seq_len=max_seq_len,
                                  transform=transform)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                  batch_size=test_batch_size,
                                  shuffle=False)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=test_batch_size,
                                  shuffle=False)
    

    n_classes = 3
    embedding_dim = 64
    hidden_size = 64
    n_layers = 2
    dropout_prob = 0.2

    model = SentimentClassifier(
        vocab_size= len(dictionary),
        embedding_dim= embedding_dim,
        hidden_size= hidden_size,
        n_layers= n_layers,
        n_classes= n_classes,
        dropout_prob= dropout_prob
    ).to( device )

    lr = 1e-4
    epochs = 50

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = fit(
        model= model,
        train_loader= train_dataloader,
        val_loader= val_dataloader,
        criterion= criterion,
        optimizer= optimizer,
        device= device,
        epochs= epochs
    )
    

    val_loss, val_acc = evaluate(model, val_dataloader, criterion, device)
    test_loss, test_acc = evaluate(model, test_dataloader, criterion, device)

    print("Evaluation on val/test dataset:")
    print("val accurancy: ", val_acc)
    print("Test accurancy: ", test_acc)


















