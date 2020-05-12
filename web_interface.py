from flask import Flask
from flask import json
from flask import request
from flask_restful import Resource, Api
from flask import Response
from flask import Flask, url_for, jsonify

import torch

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from torch import nn, optim

import torch.nn.functional as F

app = Flask(__name__)


incomes = [
  { 'description': 'salary', 'amount': 5000 }
]
"""Restoring your model is easy too:"""
class Net(nn.Module):

  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 16)
    self.fc2 = nn.Linear(16, 8)
    self.fc3 = nn.Linear(8, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))



@app.route('/incomes')
def get_incomes():
  return jsonify(incomes)


@app.route('/incomes', methods=['POST'])
def add_income():
  incomes.append(request.get_json())
  return str(request.get_json())

@app.route('/features',methods=['POST'])
def eval_feaures():
    features = request.get_json()
    result = model.model_predict_exit(due=features["due"],volume=features["volume"],diff=features["diff"])
    if result:
        return '',204
    else:
        return '',205
class ModelProcessor():
    def __init__(self):
        MODEL_PATH = 'model3.pth'
        sns.set(style='whitegrid', palette='muted', font_scale=1.2)

        HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

        sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

        rcParams['figure.figsize'] = 12, 8

        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)


        df = pd.read_csv('data_test.csv')
        df.head()

        """We have a large set of features/columns here. You might also notice some *NaN*s. Let's have a look at the overall dataset size:"""

        df.shape

        """Looks like we have plenty of data. But we got to do something about those missing values.

        ## Data Preprocessing

        We'll simplify the problem by removing most of the data (mo money mo problems - Michael Scott). We'll use only 4 columns for predicting whether or not is going to rain tomorrow:
        """

        cols = ['due', 'volume', 'diff', 'inrange']

        df = df[cols]

        """Neural Networks don't work with much else than numbers. We'll convert *yes* and *no* to 1 and 0, respectively:"""

        #df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace = True)
        #df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace = True)

        """Let's drop the rows with missing values. There are better ways to do this, but we'll keep it simple:"""

        df = df.dropna(how='any')
        df.head()

        """Finally, we have a dataset we can work with. 

        One important question we should answer is - *How balanced is our dataset?* Or *How many times did it rain or not rain tomorrow?*:
        """

        sns.countplot(df.inrange);

        df.inrange.value_counts() / df.shape[0]

        """Things are not looking good. About 78% of the data points have a non-rainy day for tomorrow. This means that a model that predicts there will be no rain tomorrow will be correct about 78% of the time.

        You can read and apply the [Practical Guide to Handling Imbalanced Datasets](https://www.curiousily.com/posts/practical-guide-to-handling-imbalanced-datasets/) if you want to mitigate this issue. Here, we'll just hope for the best.

        The final step is to split the data into train and test sets:
        """

        X = df[['due', 'volume', 'diff']]
        y = df[['inrange']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

        """And convert all of it to Tensors (so we can use it with PyTorch):"""

        X_train = torch.from_numpy(X_train.to_numpy()).float()
        y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())

        X_test = torch.from_numpy(X_test.to_numpy()).float()
        y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())

        print("dataset created")
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        self.net = Net(X_train.shape[1])
        print("created NN")

        criterion = nn.BCELoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        self.net = self.net.to(self.device)

        criterion = criterion.to(self.device)

        self.net = torch.load(MODEL_PATH)

        classes = ['exit', 'stay!']

        y_pred = self.net(X_test)

        y_pred = y_pred.ge(.5).view(-1).cpu()
        y_test = y_test.cpu()

        print(classification_report(y_test, y_pred, target_names=classes))

    def model_predict_exit(self,due,volume,diff):
        t = torch.as_tensor([due,volume,diff]) \
            .float() \
            .to(self.device)
        output = self.net(t)
        return output.ge(0.5).item()

if __name__ == '__main__':
    model = ModelProcessor()
    app.run(debug=True)