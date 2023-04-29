from model import MetaLM
from transformers import AutoTokenizer
from utils.Customdataset import CustomDataset, make_collate_fn
import wandb
import argparse
from torch.utils.data import DataLoader, Dataset
from torch import nn
from config import Config
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

import os



os.environ['XRT_TPU_CONFIG'] = "localservice;0;localhost:51011"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


prompt = ["다음 내용의 감정을 맞추시오. 예제: [기쁨, 놀람, 분노, 중립, 혐오, 공포, 슬픔]\n - 소리: "," - 텍스트: "," - 감정 값: "]

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
    xm.mark_step()

    loss.backward()
    xm.mark_step()

    xm.optimizer_step(optimizer=optimizer)
    return outputs

def eval_step(model, inputs):
    outputs = model(inputs)
    xm.mark_step()

    # loss = loss_func(pred,labels)
    return outputs

def evaluate(model:nn.Module, dataloader: DataLoader, tokenizer: AutoTokenizer, gpt_tokenizer: AutoTokenizer, wandb:wandb=None):
    device = next(model.parameters()).device
    
    if config.is_wav:
        prompt_tokens_1 = gpt_tokenizer(prompt[0], return_tensors='pt').to(device)  # "다음 내용의 감정을 맞추시오. 예제: [기쁨, 놀람, 분노, 중립, 혐오, 공포, 슬픔]\n - 소리: "
    else:
        prompt_tokens_1 = gpt_tokenizer(prompt[0][:-7], return_tensors='pt').to(device)  # "다음 내용의 감정을 맞추시오. 예제: [기쁨, 놀람, 분노, 중립, 혐오, 공포, 슬픔]\n - 소리: "
    prompt_tokens_2 = gpt_tokenizer(prompt[1], return_tensors='pt').to(device)  # " - 텍스트: "
    prompt_tokens_3 = gpt_tokenizer(prompt[2], return_tensors='pt').to(device)  # " - 감정 값: "
    
    model.eval()
    
    t = tqdm(dataloader)
    _pred = torch.tensor([]).to(device)
    _real = torch.tensor([]).to(device)

    with torch.no_grad():
        for data in t:
            data["text_tokens"] = tokenizer(data["text_tokens"],return_tensors="pt",padding=True)
            data['labels'] = data['labels'].float()
            data = {i:j.to(device=device) for i, j in data.items()}
            data['p_tokens'] = [prompt_tokens_1, prompt_tokens_2, prompt_tokens_3]
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

def train(index,args):
    device = xm.xla_device()
    config = args['config']

    gpt_tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2') # tokenizer for prompt
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large') if config.is_text_encoder else gpt_tokenizer

    if config.is_wav:
        prompt_tokens_1 = gpt_tokenizer(prompt[0], return_tensors='pt').to(device)  # "다음 내용의 감정을 맞추시오. 예제: [기쁨, 놀람, 분노, 중립, 혐오, 공포, 슬픔]\n - 소리: "
    else:
        prompt_tokens_1 = gpt_tokenizer(prompt[0][:-7], return_tensors='pt').to(device)  # "다음 내용의 감정을 맞추시오. 예제: [기쁨, 놀람, 분노, 중립, 혐오, 공포, 슬픔]\n - 소리: "
        
    prompt_tokens_2 = gpt_tokenizer(prompt[1], return_tensors='pt').to(device)  # " - 텍스트: "
    prompt_tokens_3 = gpt_tokenizer(prompt[2], return_tensors='pt').to(device)  # " - 감정 값: "

    wandb.init(
        project=config.project,
        entity="smart-sprout",
        name=config.modelname,
        group=f'tpu-server{config.modelname}',
        config=config

    )
    model = model_.to(device)
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()),lr=config.lr)
    
    step = 0
    while True:
        sampler = torch.utils.data.distributed.DistributedSampler(
                    args['train_datasets'],
                    num_replicas = xm.xrt_world_size(),
                    rank = xm.get_ordinal(),
                    shuffle = True
                    )
        dataloader_ = torch.utils.data.DataLoader(
                    dataset = args['train_datasets'],
                    sampler = sampler,
                    batch_size = config.batch_size,
                    drop_last = True,
                    collate_fn=make_collate_fn(args['train_datasets'])
                    )
        dataloader = pl.ParallelLoader(
                    dataloader_, [device]
                    ).per_device_loader(device)

        t = tqdm(dataloader)
        for idx, data in enumerate(t):
            model.train()
            step += 1
            data["text_tokens"] = tokenizer(data["text_tokens"],return_tensors="pt",padding=True)
            data['labels'] = data['labels'].float()
            data = {i:j.to(device=device) for i, j in data.items()}
            data['p_tokens'] = [prompt_tokens_1, prompt_tokens_2, prompt_tokens_3]
            outputs = train_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=data,
                    labels=data['labels']
                )
            pred = torch.argmax(outputs.logits[:,-1],dim=-1)
            # f1 = metric(pred.cpu(),data['labels'].cpu()) 데이터가 적어서 스코어 의미가 적다.
            if wandb:
                wandb.log({"loss": outputs.loss})
            if step%config.eval_per_steps==0:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                            args['val_datasets'],
                            num_replicas = xm.xrt_world_size(),
                            rank = xm.get_ordinal(),
                            shuffle = True
                            )
                val_dataloader_ = torch.utils.data.DataLoader(
                            dataset = args['val_datasets'],
                            sampler = val_sampler,
                            batch_size = config.val_batch_size,
                            drop_last = True,
                            collate_fn=make_collate_fn(args['val_datasets'])
                            )
                val_dataloader =  pl.ParallelLoader(
                            val_dataloader_, [device]
                            ).per_device_loader(device)
                score = evaluate(model=model, dataloader=val_dataloader, tokenizer=tokenizer, gpt_tokenizer=gpt_tokenizer, wandb=wandb)
                wandb.log({"val_"+i:j for i,j in score.items()})
                
                with open("score.txt") as f:
                    best_score = float(f.readline().strip())
                if best_score < score[config.focus_metric]:
                    model.save(config.save_path, optimizer=optimizer)
                    with open("score.txt","w") as f:
                        f.write(str(score[config.focus_metric]))
                
            if step==config.whole_steps:
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
    parser.add_argument('--p_tunning', type=bool, default=True, help='GPT-P-tunning.')
    parser.add_argument('--dropout', type=int, default=0.3, help='dropout.')
    parser.add_argument('--is_wav', type=bool, default=True, help='Boolean. if is True...')
    parser.add_argument('--is_text', type=bool, default=True, help='Boolean. if is True...')
    parser.add_argument('--num_labels', type=int, default=7, help='label nums')
    parser.add_argument('--is_prompt', type=bool, default=False, help='is prompt')
    parser.add_argument('--is_text_encoder', type=bool, default=True, help='if True, roberta encoder is used.')
    parser.add_argument('--sep_trainable', type=bool, default=False, help='Whether training seperate token')
    parser.add_argument('--lr', type=float, default=0.000001, help='learning rate.')
    parser.add_argument('--whole_steps', type=int, default=10000, help='whole training step.')
    parser.add_argument('--eval_per_steps', type=int, default=50, help='evaluate is excuted per this args.')


    parser.add_argument('-v','--val_dir', type=str, default="~/datadisk/KEMDy20_val_data.csv", help='Optional')
    parser.add_argument('-t','--test_data', type=str, default="~/datadisk/KEMDy20_test_data.csv", help='Optional')
    parser.add_argument('-b','--batch_size', type=int, default=1, help='default16')
    parser.add_argument('-s','--val_batch_size', type=int, default=1, help='default8')
    parser.add_argument('-n','--modelname', type=str, default='PowerfulMyModel', help='Enter model name')
    parser.add_argument('-p','--project', type=str, default='meta-p-tunning', help='Enter project name')
    parser.add_argument('--save_path', type=str, default='./default_model.pt', help='Enter model save path')
    parser.add_argument('--focus_metric', type=str, default='weighted_f1', help='Enter eval metric to focus on')
    parser.add_argument('--load_path', type=str, default='', help='Enter model load path')

    config = parser.parse_args()


    # config = wandb.config

    # config.update(args)
    config.wav_encoder_config = Config.from_json("wav_encoder_config.json")
    config.text_encoder_config = Config.from_json("text_encoder_config.json")

    model_ = MetaLM(config)
    if config.load_path: model_.load(config.load_path)
    with open("score.txt","w") as f:
        f.write("0.5")

    train_datasets = CustomDataset(config.input_dir,config.is_wav)
    val_datasets = CustomDataset(config.val_dir,config.is_wav)

    FLAGS = {}
    FLAGS.update(train_datasets=train_datasets,val_datasets=val_datasets, config=config)

    xmp.spawn(train, args =(FLAGS, ), nprocs=8, start_method='fork')
    wandb.finish()