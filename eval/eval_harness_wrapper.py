import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from tqdm import tqdm

class GhostQuantLM(LM):
    def __init__(self, model, tokenizer, M=768, batch_size=1, device="mps"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.M = M
        self._batch_size = batch_size
        self._device = torch.device(device)
        self.model.to(self._device)
        self.model.eval()

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.config.max_seq_len

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        with torch.no_grad():
            return self.model(inps, M=self.M)

    def loglikelihood(self, requests):
        res = []
        for req in tqdm(requests):
            context, continuation = req.args
            ctx_enc = self.tok_encode(context)
            cont_enc = self.tok_encode(continuation)
            
            inp = torch.tensor([ctx_enc + cont_enc], device=self._device)
            logits = self._model_call(inp)
            
            logprobs = F.log_softmax(logits, dim=-1)
            
            cont_start = len(ctx_enc)
            cont_logits = logprobs[0, cont_start-1:-1, :]
            cont_ids = torch.tensor(cont_enc, device=self._device).unsqueeze(-1)
            
            ll = torch.gather(cont_logits, 1, cont_ids).sum().item()
            is_greedy = torch.argmax(cont_logits, dim=-1).equal(cont_ids.squeeze(-1))
            
            res.append((ll, is_greedy))
        return res

    def loglikelihood_rolling(self, requests):
        res = []
        for (string,) in tqdm(requests):
            enc = self.tok_encode(string)
            inp = torch.tensor([enc], device=self._device)
            logits = self._model_call(inp)
            
            logprobs = F.log_softmax(logits, dim=-1)
            
            cont_logits = logprobs[0, :-1, :]
            cont_ids = torch.tensor(enc[1:], device=self._device).unsqueeze(-1)
            
            ll = torch.gather(cont_logits, 1, cont_ids).sum().item()
            res.append(ll)
        return res

    def generate_until(self, requests):
        raise NotImplementedError("Generation not required for likelihood benchmarks")
