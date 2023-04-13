from model import MetaLM
from transformers import AutoTokenizer
from utils.Customdataset import CustomDataset, collate_fn
import wandb
import argparse
from torch.utils.data import DataLoader
from torch import nn
from config import Config
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch
import torch_xla.core.xla_model as xm
import os



os.environ['XRT_TPU_CONFIG'] = "localservice;0;localhost:51011"

dev = xm.xla_device()


# model()
criterion = nn.CrossEntropyLoss()

def loss_func(pred, real):
    loss = criterion(pred,real)
    return loss

def train_step(model, inputs, labels, optimizer:torch.optim.Optimizer):
    optimizer.zero_grad()
    outputs = model(inputs)
    # loss = loss_func(pred, labels)
    loss = outputs.loss
    loss.backward()
    xm.optimizer_step(optimizer=optimizer)
    return outputs

def eval_step(model, inputs):
    outputs = model(inputs)
    # loss = loss_func(pred,labels)
    return outputs

def evaluate(model:nn.Module, dataloader: DataLoader, wandb:wandb=wandb):
    model.eval()
    t = tqdm(dataloader)
    _pred = torch.tensor([])
    _real = torch.tensor([])
    
    for data in t:
        outputs = eval_step(model,data)
        pred = torch.argmax(outputs.logits[:,-1],dim=-1)
        _pred = torch.concat([_pred,pred])
        _real = torch.concat([_real,data['labels']])
        wandb.log({"val_loss":outputs.loss.detach().cpu()})
    score = metric(_pred.detach().cpu(),_real.detach().cpu())
    model.train()
    return score
        

def metric(pred,real):
    f1 = {
        'macro_f1':f1_score(real, pred, average='macro'),
        'micro_f1':f1_score(real, pred, average='micro'),
        'weighted_f1':f1_score(real, pred, average='weighted'), 
    }
    return f1

def train(model:nn.Module, optimizer:torch.optim.Optimizer, dataloader: DataLoader, val_dataloader: DataLoader, wandb: wandb=wandb, whole_step=10000, eval_per_steps=100):
    
    model.train()
    t = tqdm(dataloader)
    step = 0
    while True:
        for idx, data in enumerate(t):
            data['labels'] = data['labels'].float()
            data = {i:j.to(device=dev) for i, j in data.items()}
            step += 1
            outputs = train_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=data,
                    labels=data['labels'],
                )
            pred = torch.argmax(outputs.logits[:,-1],dim=-1)
            # f1 = metric(pred.cpu(),data['labels'].cpu()) 데이터가 적어서 스코어 의미가 적다.
            if wandb:
                wandb.log({"loss": outputs.loss})
            if step%eval_per_steps==0:
                score = evaluate(model=model, dataloader=val_dataloader, wandb=wandb)
                wandb.log({"val_"+i:j for i,j in score})
        if step==whole_step:
            return None
        
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='hyper&input')
    parser.add_argument('-i','--input_dir', type=str, default='~/datadisk/KEMDy20_train_data.csv', help='you can use csv file. without file extention')

    parser.add_argument('--prefix_projection', type=str, default=False, help='you can use csv filepath.')
    parser.add_argument('--pre_seq_len', type=int, default=5, help='you can use csv filepath.')
    parser.add_argument('--hidden_size', type=int, default=768, help='you can use csv filepath.')
    parser.add_argument('--prefix_hidden_size', type=int, default=768, help='you can use csv filepath.')
    parser.add_argument('--num_hidden_layers', type=int, default=12, help='you can use csv filepath.')
    parser.add_argument('--num_attention_heads', type=int, default=12, help='std output & model save dir')
    parser.add_argument('--prompt', type=str, default=True, help='GPT-P-tunning.')
    parser.add_argument('--dropout', type=int, default=0.3, help='dropout.')
    parser.add_argument('--is_wav', type=bool, default=True, help='Boolean. if is True...')
    parser.add_argument('--num_labels', type=int, default=7, help='label nums')

    parser.add_argument('-v','--val_dir', type=str, default="~/datadisk/KEMDy20_val_data.csv", help='Optional')
    parser.add_argument('-t','--test_data', type=str, default="~/datadisk/KEMDy20_test_data.csv", help='Optional')
    parser.add_argument('-b','--batch_size', type=int, default=8, help='default16')
    parser.add_argument('-s','--val_batch_size', type=int, default=8, help='default8')
    parser.add_argument('-n','--modelname', type=str, default='PowerfulMyModel', help='Enter model name')
    parser.add_argument('-p','--project', type=str, default='meta-p-tunning', help='Enter project name')

    args = parser.parse_args()



    wandb.init(
        project=args.project,
        entity="smart-sprout",
        name=args.modelname
        
    )

    config = wandb.config

    config.update(args)
    config.wav_encoder = Config.from_json("wav_encoder_config.json")
    config.text_encoder = Config.from_json("text_encoder_config.json")

    model = MetaLM(config).to(device=dev)
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),lr=0.00001)
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

    training_data = CustomDataset(config.input_dir,config.is_wav)
    test_data = CustomDataset(config.val_dir,config.is_wav)

    train_dataloader = DataLoader(training_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=config.val_batch_size, shuffle=True, collate_fn=collate_fn)

    train(model=model,optimizer=optimizer,dataloader=train_dataloader,val_dataloader=test_dataloader, wandb=wandb)