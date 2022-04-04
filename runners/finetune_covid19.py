import gc
import warnings

import pandas as pd
import torch
import torch.optim as optim
from model.finetune_bert import *
from sklearn.model_selection import train_test_split
from torchtext.data import BucketIterator, Field, Iterator, TabularDataset
from transformers import BertTokenizer

warnings.filterwarnings("ignore")

MODEL_NAME = 'COVID19_SBIRS'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

MAX_SEQ_LEN = 256
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

train_data = pd.read_csv('data/EQ2-Data/covid_train.csv')
valid_data = pd.read_csv('data/EQ2-Data/news_test.csv')

train_data['text'] = train_data['text'].apply(text_cleaning)
valid_data['text'] = valid_data['text'].apply(text_cleaning)

val_split, test_split = train_test_split(valid_data,test_size = 0.5)

train_data.to_csv("data/EQ2-Data/processed_splits/covid_train.csv", index=False)
val_split.to_csv("data/EQ2-Data/processed_splits/news_valid.csv", index=False)
test_split.to_csv("data/EQ2-Data/processed_splits/news_test.csv", index=False)

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('text', text_field), ('label', label_field)]

trained, valid, test = TabularDataset.splits(path="data/EQ2-Data/processed_splits/", train='covid_train.csv', validation='news_valid.csv',
                                           test='news_test.csv', format='CSV', fields=fields, skip_header=True)

train_iter = BucketIterator(trained, batch_size=16, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)
test_iter = Iterator(test, sort_key = None, batch_size=16, device=device, train=False, shuffle=False, sort=False)

torch.cuda.empty_cache()
gc.collect()

model = BERT().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train(model=model, optimizer=optimizer, train_loader=train_iter, valid_loader=valid_iter, num_epochs=1, eval_every=len(train_iter)//2, model_name=MODEL_NAME)

metrics_folder = "runners/model_metrics/"
train_loss_list, valid_loss_list, global_steps_list = load_metrics(metrics_folder + MODEL_NAME + '_metrics.pt', device)
plot_loss(global_steps_list, train_loss_list, valid_loss_list, MODEL_NAME)

torch.cuda.empty_cache()
gc.collect()

weights_folder = "runners/model_weights/"
best_model = BERT().to(device)
loaded_model = load_checkpoint(load_path=weights_folder + MODEL_NAME + '.pt', model = best_model, device=device)
evaluate(loaded_model, test_iter, plot_name=MODEL_NAME)
