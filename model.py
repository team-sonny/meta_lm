import whisper
from torch import nn
from CustomModel import GPT2ForTokenClassification
import torch
from encoder import TextEncoder, WavEncoder, PrefixEncoder
from config import Config
import torch
import torch_xla.core.xla_model as xm

# 프롬프트 구조
# P-tunning + prompt, few shot in-context learning 으로 하면 좋을 것 같으니 비교 실험 필요.
# 질의 -
prompt = ["다음 내용의 감정을 맞추시오. 예제: [기쁨, 놀람, 분노, 중립, 혐오, 공포, 슬픔]\n - 소리: "," - 텍스트: "," - 감정 값: "]

class MetaLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.SpecialLayer = nn.Embedding(2, self.embed_dim)
        self.CLSToken = torch.LongTensor([0])
        self.SEPToken = torch.LongTensor([1])
        self.text_encoder = TextEncoder(config.text_encoder_config) if config.is_text_encoder else None
        self.wav_encoder = WavEncoder(config.wav_encoder_config)
        # GPI is Semi-Causal LM
        self.GPI = GPT2ForTokenClassification.from_pretrained('skt/kogpt2-base-v2',output_hidden_states=True,num_labels=config.num_labels)
        self.p_tunning = config.p_tunning
        self.dropout = nn.Dropout2d(config.dropout)
        for param in self.GPI.parameters():
            param.requires_grad = False
        for param in self.GPI.classifier.parameters():
            param.requires_grad = True
        self.is_prompt = config.is_prompt


        if self.p_tunning:
            self.prefix_encoder=PrefixEncoder(config)
            self.pre_seq_len = config.pre_seq_len
            self.n_layer = config.num_hidden_layers
            self.n_head = config.num_attention_heads
            self.n_embd = config.hidden_size // config.num_attention_heads
            self.prefix_tokens=torch.arange(self.pre_seq_len).long()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(next(self.parameters()).device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_cls_embedding(self, batch_size):
        CLSToken = self.CLSToken.unsqueeze(0).expand(batch_size, -1).to(next(self.parameters()).device)
        CLSEmebedding = self.SpecialLayer(CLSToken)
        return CLSEmebedding
    
    def get_sep_embedding(self, batch_size):
        SEPToken = self.SEPToken.unsqueeze(0).expand(batch_size, -1).to(next(self.parameters()).device)
        SEPEmebedding = self.SpecialLayer(SEPToken) if self.config.sep_trainable else self.GPI.transformer.wte(SEPToken)
        return SEPEmebedding

    def forward(self, inputs, order=False):
        """_summary_

        Args:
            order (bool, optional): first Audio, then Text if order is False. Defaults to False.

        Returns:
            _type_: _description_
        """
        # TODO:
        # 1. [CLS] 마지막에 붙이기
        # 2. prompt 붙이기, 다른 인코더의 아웃풋과 concat하는 코드 필요(model 내부).
        # 3. text encoder 제거
        # 4. 음성데이터 길이 줄이는 법 추가

        batch_size = inputs['text_tokens']['input_ids'].shape[0]
        past_key_values = self.get_prompt(batch_size)
        CLSEmbedding = self.get_cls_embedding(batch_size)
        SEPEmbedding = self.get_sep_embedding(batch_size)

        labels=inputs['labels']
        text_tokens, wav_tokens = inputs['text_tokens'], inputs['wav_tokens']
        text_tokens = self.text_encoder(text_tokens) if self.text_encoder is not None else self.GPI.transformer.wte(text_tokens.input_ids)
        wav_tokens = self.wav_encoder(wav_tokens)

        if self.is_prompt:
            p_token_1 = self.GPI.transformer.wte(inputs['p_tokens'][0]['input_ids'].expand(batch_size, -1))
            p_token_2 = self.GPI.transformer.wte(inputs['p_tokens'][1]['input_ids'].expand(batch_size, -1))
            p_token_3 = self.GPI.transformer.wte(inputs['p_tokens'][2]['input_ids'].expand(batch_size, -1))

            if self.config.is_text and self.config.is_wav:
                inputs = torch.concat([p_token_1, wav_tokens, p_token_2, text_tokens, p_token_3], dim=1)
            elif self.config.is_text:
                inputs = torch.concat([p_token_1, p_token_2, text_tokens, p_token_3], dim=1)
            else:
                inputs = torch.concat([p_token_1, wav_tokens, p_token_3], dim=1)
                
        else:
            # fitting seq_len for GPT with prefix
            if self.config.is_text and self.config.is_wav:
                inputs = torch.concat([text_tokens, SEPEmbedding, wav_tokens, CLSEmbedding],dim=1)
            elif self.config.is_text:
                inputs = torch.concat([text_tokens, CLSEmbedding],dim=1)
            else:
                inputs = torch.concat([wav_tokens, CLSEmbedding],dim=1)
                
            
            

        xm.mark_step()

        # inputs = inputs.view(batch_size,-1,self.n_head,self.n_embd).permute([0,2,1,3])
        outputs = self.GPI(inputs_embeds=inputs,past_key_values=past_key_values,labels=labels)
        # pred = self.classifier(outputs.hidden_states[:,-1])
        return outputs

    def save(self, PATH: str, optimizer: torch.optim.Optimizer = None) -> None:
        model_for_save = {
            "SpecialLayer": self.SpecialLayer.state_dict(),
            "prefix_encoder": self.prefix_encoder.state_dict(),
            "classifier": self.GPI.classifier.state_dict(),
        }
        model_for_save.update(**self.config.__dict__)
        model_for_save['text_encoder_config'] = model_for_save['text_encoder_config'].__dict__
        model_for_save['wav_encoder_config'] = model_for_save['wav_encoder_config'].__dict__
        if optimizer: model_for_save.update(optimizer=optimizer.state_dict())
        if self.config.is_text and self.config.is_text_encoder: model_for_save.update(text_connector=self.text_encoder.connector.state_dict())
        if self.config.is_wav: 
            model_for_save.update(wav_connector=self.wav_encoder.connector.state_dict())
            model_for_save.update(wav_layer=self.wav_encoder.layer.state_dict())
        xm.save(model_for_save,PATH)

    def load(self, PATH: str) -> torch.nn.Module:
        model = torch.load(PATH)
        self.SpecialLayer.load_state_dict(model['SpecialLayer'])
        self.prefix_encoder.load_state_dict(model['prefix_encoder'])
        self.GPI.classifier.load_state_dict(model['classifier'])
        if self.config.is_text and self.config.is_text_encoder: self.text_encoder.connector.load_state_dict(model['text_connector'])
        if self.config.is_wav: 
            self.wav_encoder.connector.load_state_dict(model['wav_connector'])
            self.wav_encoder.layer.load_state_dict(model['wav_layer'])
        return self
