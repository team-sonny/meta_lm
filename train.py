from model import MetaLM
from transformers import AutoTokenizer
from utils.Customdataset import CustomDataset
import wandb
import argparse
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='hyper&input')
parser.add_argument('-i','--input_dir', type=str, default='clean_all_correct', help='you can use csv file. without file extention')
parser.add_argument('--prefix_projection', type=str, default=None, help='you can use csv filepath.')
parser.add_argument('--pre_seq_len', type=str, default=None, help='you can use csv filepath.')
parser.add_argument('--hidden_size', type=str, default=None, help='you can use csv filepath.')
parser.add_argument('--prefix_hidden_size', type=str, default=None, help='you can use csv filepath.')
parser.add_argument('--num_hidden_layers', type=str, default=None, help='you can use csv filepath.')
parser.add_argument('--num_attention_heads', type=str, default="~/datadisk", help='std output & model save dir')
parser.add_argument('--prompt', type=str, default=None, help='you can use csv filepath.')
parser.add_argument('-v','--validation_data', type=str, default=None, help='Optional')
parser.add_argument('--val_dir', type=str, default=None, help='Optional')
parser.add_argument('-b','--batch_size', type=int, default=16, help='default16')
parser.add_argument('-s','--valbatch_size_perdevice', type=int, default=8, help='default8')
parser.add_argument('-n','--modelname', type=str, default='PowerfulMyModel', help='Enter model name')
parser.add_argument('-p','--projectname', type=str, default='kogpt2', help='Enter model name')

args = parser.parse_args()



wandb.init()

config = wandb.config

config.update(args)

model = MetaLM(config)
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

training_data = CustomDataset(config.input_dir)
test_data = CustomDataset(config.val_dir)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

def loss_func():
    ...

def train_step(model, inputs, labels, optimizer):
    pred = model(inputs)
    loss = loss_func(pred, labels)
    loss.backward()
    optimizer.step()
    return loss, pred

def train(model):
    ...