from flask_cors import CORS
from flask import Flask, render_template, request, jsonify

import sys
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import bert
import os
import math
import datetime

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

from tensorflow.keras.models import load_model
model = load_model('bert.h5', custom_objects={"BertModelLayer": bert.BertModelLayer})

bert_model_name="uncased_L-12_H-768_A-12"

bert_ckpt_dir = os.path.join("model/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

class DataProcessor:
  DATA_COLUMN = "Headline"
  LABEL_COLUMN = "Stance"

  def __init__(self, test, tokenizer: FullTokenizer, classes, max_seq_len=512):
    self.tokenizer = tokenizer
    self.classes = classes
    self.max_seq_len = max_seq_len
    
    self.test_x, self.test_y = self._prepare(test)

    #print("max seq_len", self.max_seq_len)
    ##print(self.test_x)
    self.test_x = self._pad(self.test_x)

  def _prepare(self, df):
    x, y = [], []
    
    for _, row in tqdm(df.iterrows()):
      text, label = row[DataProcessor.DATA_COLUMN], row[DataProcessor.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      x.append(token_ids)
      y.append(self.classes.index(label))
    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
        input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
        z=self.max_seq_len - len(input_ids)
        y=[0]
        y=np.array(y)
        for i in range(z):
            input_ids=np.append(input_ids,y)
        x.append(np.array(input_ids))
    return np.array(x)

tokenizer = FullTokenizer(
  vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt")
)

classes = ['unrelated', 'discuss', 'agree', 'disagree']

app = Flask(__name__, template_folder='./template')
CORS(app)

def getstance(inp):
	print("-----")
	print(inp[0:5])
	test_df= pd.DataFrame(inp[0:50], columns = ['Headline', 'Stance'])
	print(test_df.shape)
	data = DataProcessor(test_df, tokenizer, classes, max_seq_len=512)
	prediction = model.predict(data.test_x).argmax(axis=-1)
	print(prediction)
	x = [0,0,0,0]
	for i in prediction :
		x[i] = x[i]+1
	return x

@app.route('/')
def index():
	return render_template('index1.html')
 
@app.route('/square/', methods=['POST'])
def square():
	req_data = request.get_json()
	org_text = req_data
	print('calling the function !')
	print((org_text[0:5]))
	#print(org_text)
	x = getstance(org_text)
	#x = [0,0,0,0]
	print(x)
	data = {'total': x}
	data = jsonify(data)
	print(data)
	return data
 
if __name__ == '__main__':
	app.run(debug=True)