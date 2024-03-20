"""
Sample from the trained model with PyTorch
"""
import os
import pickle
from contextlib import nullcontext
import torch
from .model import ModelArgs, Transformer
from .tokenizer import Tokenizer

import time
import os
from huggingface_hub import hf_hub_download
from musiclang_predict import MusicLangTokenizer

STOP_CHAR = None

DATA_CACHE_DIR = "data"
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"old/tok{vocab_size}.model")

class ModelLoader:
    def __init__(self, path):
        self.path = path
        self.tokenizer_path = hf_hub_download(repo_id=self.path, filename="tokenizer.model")
        self.model_path = hf_hub_download(repo_id=self.path, filename="ckpt.pt")
        self.pretokenizer = MusicLangTokenizer(self.path)
        self.CHORD_CHANGE_CHAR = self.pretokenizer.tokens_to_bytes('CHORD_CHANGE')
        self.MELODY_END_CHAR = self.pretokenizer.tokens_to_bytes('MELODY_END')

    def load_model(self, device='cuda', compile=True):
        checkpoint = self.model_path
        tokenizer = self.tokenizer_path
        checkpoint_dict = torch.load(checkpoint, map_location=device)
        gptconf = ModelArgs(**checkpoint_dict['model_args'])
        model = Transformer(gptconf)
        state_dict = checkpoint_dict['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        model.to(device)
        if compile:
            print("Compiling the model...")
            model = torch.compile(model)  # requires PyTorch 2.0 (optional)

        # load the tokenizer
        vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
        vocab_size = gptconf.vocab_size
        if tokenizer:
            # a specific tokenizer is provided, use it
            tokenizer_model = tokenizer
        else:
            # let's try to find the tokenizer model automatically. bit gross here...
            query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
            tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
        enc = Tokenizer(tokenizer_model=tokenizer_model)

        return model, enc, self.pretokenizer

def predict_one_shot(path, device='cuda', **kwargs):
    model, enc, pretokenizer = ModelLoader(path).load_model(device=device)
    result = predict_python(model, enc, device=device, **kwargs)[0]
    score = pretokenizer.untokenize_from_bytes(result)
    return score


def predict_python(model, enc, max_new_tokens=256, temperature=0.95, top_k=1000, start="_", num_samples=1 , seed=None,
                   device='cuda', stop_char=None):

    # Configure -----------------------------------------------------------------------------
    dtype = 'bfloat16' if device =='cuda' and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    # -----------------------------------------------------------------------------
    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()

    start_ids = enc.encode(start, bos=True, eos=False)
    if stop_char is not None:
        stop_id = enc.encode(stop_char, bos=False, eos=False)

    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    start_time = time.time()
    # run generation
    results = []
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                print(f'generating sample {k}')
                for i in range(max_new_tokens):
                    y = model.generate(x, 1, temperature=temperature, top_k=top_k)
                    x = y
                result = enc.decode(y[0].tolist())
                results.append(result)

    return results