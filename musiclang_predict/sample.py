"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from tokenizers import Tokenizer


class ModelLLM:


    def __init__(self, model, encode, decode, ctx, device, tokenizer):

        self.model = model
        self.encode = encode
        self.decode = decode
        self.ctx = ctx
        self.device = device
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.encode = lambda x: self.tokenizer.encode(x).ids
            self.decode = lambda x: self.tokenizer.decode(x).replace(' ', '')


    @classmethod
    def load_model(cls, name=None, path=None, update=False, **config):
        """
        Load an existing LLM model
        """
        if name is None and path is None:
            raise ValueError('One of "name" or "path" paremeter should not be empty')
        if name is not None and path is not None:
            raise ValueError('Only one of "name" or "path" should not be None')

        if name is not None:
            from .load_dataset import download_model
            out_dir = download_model(name=name, update=update, **config)
        else:
            out_dir = path

        device = config.get('device', 'cuda')
        device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
        dtype = config.get('dtype', 'bfloat16')  # 'float32' or 'bfloat16' or 'float16'
        init_from = config.get('init_from', 'resume')  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        compile = config.get('compile', False)  # use PyTorch 2.0 to compile the model to be faster

        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # model
        if init_from == 'resume':
            # init from a model saved in a specific directory
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=device)
            gptconf = GPTConfig(**checkpoint['model_args'])
            model = GPT(gptconf)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
        elif init_from.startswith('gpt2'):
            # init from a given GPT-2 model
            model = GPT.from_pretrained(init_from, dict(dropout=0.0))

        model.eval()
        model.to(device)
        if compile:
            model = torch.compile(model) # requires PyTorch 2.0 (optional)

        # look for the meta pickle in case it is available in the dataset folder
        load_meta = False
        if init_from == 'resume':
            meta_path = os.path.join(out_dir, 'meta.pkl')
            load_meta = os.path.exists(meta_path)
        if load_meta:
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            # TODO want to make this more general to arbitrary encoder/decoder schemes

            if 'stoi' in meta:
                stoi, itos = meta['stoi'], meta['itos']
                encode = lambda s: [stoi[c] for c in s.replace(' ', '').replace('\n', '').replace('\t', '')]
                decode = lambda l: ''.join([itos[i] for i in l])
                model = ModelLLM(model, encode, decode, ctx, device, tokenizer=None)
            else:
                # Load tokenizer
                print('Loading custom tokenizer ...')
                tokenizer = Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json"))
                encode = lambda s: tokenizer.encode(s).ids
                decode = lambda l: tokenizer.decode(l)
                model = ModelLLM(model, encode, decode, ctx, device, tokenizer=tokenizer)
        else:
            raise Exception('No encoding found (no meta.pkl in the directory)')

        return model



    def sample(self, start="", num_samples=1, **config):
        """
        Sample a new text from the model

        Parameters
        ----------
        start: str
                Starting prompt on which to generate text
        num_samples: int
            Number of different samples to generate
        config: dict with options (seed, max_new_tokens, temperature, top_k)
        """
        seed = config.get('seed', 1337)
        torch.manual_seed(seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(seed)
        max_new_tokens = config.get('max_new_tokens', 300)  # number of tokens generated in each sample
        temperature = config.get('temperature',
                                 0.8)  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        top_k = config.get('top_k', 200)  # retain only the top_k most likely tokens, clamp others to have 0 probability

        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read()
        start_ids = self.encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])
        # run generation
        results = []
        with torch.no_grad():
            with self.ctx:
                for k in range(num_samples):
                    y = self.model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    results.append(self.decode(y[0].tolist()))

        return results