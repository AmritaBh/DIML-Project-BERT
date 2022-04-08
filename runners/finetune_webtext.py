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

MODEL_NAME = 'WEBTEXT_SBIRS'
TRAINING_DATA = 'webtext'
TESTING_DATA = 'sbirs'

TRAIN_DATASET = 'data/EQ2-Data/' + TRAINING_DATA + '_train.csv'
TEST_DATASET = 'data/EQ2-Data/' + TESTING_DATA + '_test.csv'  

NUM_EPOCHS = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

MAX_SEQ_LEN = 256
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

train_data = pd.read_csv(TRAIN_DATASET)
test_data = pd.read_csv(TEST_DATASET)

train_data['text'] = train_data['text'].apply(text_cleaning)
test_data['text'] = test_data['text'].apply(text_cleaning)

train_split, val_split = train_test_split(train_data,test_size = 0.2)

train_split.to_csv("data/EQ2-Data/processed_splits/" + TRAINING_DATA + "_train.csv", index=False)
val_split.to_csv("data/EQ2-Data/processed_splits/" + TRAINING_DATA + "_valid.csv", index=False)
test_data.to_csv("data/EQ2-Data/processed_splits/" + TESTING_DATA + "_test.csv", index=False)

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('text', text_field), ('label', label_field)]

trained, valid, test = TabularDataset.splits(path="data/EQ2-Data/processed_splits/", train=TRAINING_DATA + '_train.csv', validation=TRAINING_DATA + '_valid.csv',
                                           test=TESTING_DATA + '_test.csv', format='CSV', fields=fields, skip_header=True)

train_iter = BucketIterator(trained, batch_size=16, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)
test_iter = Iterator(test, sort_key = None, batch_size=16, device=device, train=False, shuffle=False, sort=False)

torch.cuda.empty_cache()
gc.collect()

model = BERT().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train(model=model, optimizer=optimizer, train_loader=train_iter, valid_loader=valid_iter, num_epochs=NUM_EPOCHS, eval_every=len(train_iter)//2, model_name=MODEL_NAME)

metrics_folder = "runners/model_metrics/"
train_loss_list, valid_loss_list, global_steps_list = load_metrics(metrics_folder + MODEL_NAME + '_metrics.pt', device)
plot_loss(global_steps_list, train_loss_list, valid_loss_list, MODEL_NAME)

torch.cuda.empty_cache()
gc.collect()

weights_folder = "runners/model_weights/"
best_model = BERT().to(device)
loaded_model = load_checkpoint(load_path=weights_folder + MODEL_NAME + '.pt', model = best_model, device=device)
evaluate(loaded_model, test_iter, plot_name=MODEL_NAME)
