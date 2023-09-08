"""Wrapper around HuggingFace Pipeline APIs."""
import torch
import numpy as np
from typing import Any, List, Mapping, Optional, Union

from pydantic import BaseModel, Extra
from pathlib import Path
from more_itertools import chunked

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema import Generation, LLMResult

DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text2text-generation", "text-generation")

context_length_limit = {
    'EleutherAI/gpt-neo-2.7B': 2048,
    'togethercomputer/GPT-JT-6B-v1': 2048,
    'EleutherAI/gpt-neox-20b': 2048,
    'llama-7B': 2048,
    'llama-13B': 2048,
    'bigcode/starcoder': 8192,
    'bigcode/starcoderbase': 8192,
}

def get_model_cache_dir():
    if Path('/srv/nvme0/ucinlp/shivag5/').exists():
        return Path('/srv/nvme0/ucinlp/shivag5/')
    elif Path('/srv/disk01/ucinlp/shivag5/').exists():
        return Path('/srv/disk01/ucinlp/shivag5/')
    elif Path('/persist/Documents/research/gisting/.cache/').exists():
        return Path('/persist/Documents/research/gisting/.cache/')
    else:
        raise ValueError('No model cache directory found')


def llama_path(model_size: str = '7B'):
    if (get_model_cache_dir() / f'llama_hf/{model_size}').exists():
        return get_model_cache_dir() / f'llama_hf/{model_size}'
    else:
        raise ValueError(f'Invalid llama path: {model_size}')

class HuggingFace(LLM, BaseModel):
    model: Any
    tokenizer: Any
    model_name: str = DEFAULT_MODEL_NAME
    model_kwargs: Optional[dict] = None
    generation_kwargs: Optional[dict] = None
    task: str = DEFAULT_TASK
    batch_size: int = 4
    device: str = 'cpu'

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        task: str,
        device: int = -1,
        model_kwargs: Optional[dict] = None,
        generation_kwargs: Optional[dict] = None,
        batch_size: int = 4,
        cache: Optional[bool] = None,
        verbose: bool = False,
    ) -> LLM:
        """Construct the pipeline object from model_name and task."""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )

            _model_kwargs = model_kwargs or {}
            if 'llama' not in model_name:
                if 'starcoder' not in model_name:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, **_model_kwargs, use_auth_token=True)
                else:
                    starcoder_path = get_model_cache_dir() / model_name
                    tokenizer = AutoTokenizer.from_pretrained(starcoder_path, **_model_kwargs, use_auth_token=True)
            else:
                from transformers import LlamaTokenizer
                if '7B' in model_name: model_size = '7B'
                elif '13B' in model_name: model_size = '13B'
                elif '30B' in model_name: model_size = '30B'
                elif '65B' in model_name: model_size = '65B'
                else: raise ValueError(f'Invalid llama model: {model_name}')
                llama_path = get_model_cache_dir() / f'llama_hf/{model_size}'
                tokenizer = LlamaTokenizer.from_pretrained(llama_path, **_model_kwargs)

            if task == "text-generation":
                # if model_name == 'togethercomputer/GPT-JT-6B-v1':
                if 'llama' not in model_name:
                    if 'starcoder' not in model_name:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name, torch_dtype=torch.float16, **_model_kwargs, use_auth_token=True)
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            starcoder_path, torch_dtype=torch.float16, **_model_kwargs, use_auth_token=True)
                else:
                    from transformers import LlamaForCausalLM
                    model = LlamaForCausalLM.from_pretrained(llama_path, torch_dtype=torch.float16, **_model_kwargs)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = "left"
                model.config.pad_token_id = model.config.eos_token_id
            elif task == "text2text-generation":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **_model_kwargs)
            else:
                raise ValueError(
                    f"Got invalid task {task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
            device = f'cuda:{device}' if torch.cuda.is_available() and device >= 0 else 'cpu'
            model = model.eval().to(device)
            torch.cuda.empty_cache()

            return cls(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                model_kwargs=_model_kwargs,
                generation_kwargs=generation_kwargs or {},
                task=task,
                batch_size=batch_size,
                device=device,
                cache=cache,
                verbose=verbose,
            )
        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please it install it with `pip install transformers`."
            )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_name": self.model_name},
            **{"model_kwargs": self.model_kwargs},
            **{"generation_kwargs": self.generation_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface"

    # def ppl_generate(input_texts, model, tokenizer, choices_list, device=None):
    #     loss_list = []
    #     # to support batch inference, here we assume the number of choices is equal for each instance
    #     for choices in choices_list:
    #         filled_texts = []
    #         for text, choice in zip(input_texts, choices):
    #             filled_texts.append(text+choice)
    #         loss_list.append(_evaluate_loss(filled_texts, model, tokenizer, device))
    #     lm_loss_list = np.array(loss_list)
    #     preds = lm_loss_list.argmin(axis=0).tolist()
    #     return preds


    # def _evaluate_loss(input_texts, model, tokenizer, device):
    #     with torch.no_grad():
    #         inputs = tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
    #         inputs = {k: v.to(device) for k, v in inputs.items()}
    #         outputs = model(**inputs)
    #         shift_logits = outputs.logits[..., :-1, :].contiguous()
    #         # note here we assume padding is performed on the right, left padding token will affect position_id in gpt2
    #         shift_labels = inputs["input_ids"][..., 1:].contiguous()
    #         loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
    #         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
    #             shift_labels.size())
    #         ce_loss = loss.sum(-1).cpu().detach().numpy()  # -log(p(y))
    #         lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(-1).cpu().numpy()
    #     return ce_loss / lens

    def _genprobs(self, texts: list[str]) -> list[list[float]]:
        genprobs_l = []
        with torch.no_grad():
            for idxes in chunked(np.argsort([len(p) for p in texts]), self.batch_size):
                _texts = [texts[i] for i in idxes]
                inputs = self.tokenizer(_texts, return_tensors="pt", padding=True, truncation=False).to(self.device)
                outputs = self.model(**inputs)
                shifted_input_ids = inputs.input_ids[:, 1:]
                lens = (shifted_input_ids != self.tokenizer.pad_token_id).sum(-1)
                probs = torch.log_softmax(outputs.logits, dim=-1).detach()
                gen_probs = torch.gather(probs[:, :-1, :], 2, shifted_input_ids[:, :, None]).squeeze(-1)
                genprobs_l.append(gen_probs.cpu().numpy())
        maxlen = max([genprobs.shape[-1] for genprobs in genprobs_l])
        genprobs = np.concatenate(
            [np.pad(genprobs, ((0, 0), (0, maxlen - genprobs.shape[-1])))
             for genprobs in genprobs_l],
            axis=0
        )
        return genprobs

    def _ppls(self, texts: list[str]) -> list[float]:
        ppls = np.empty(len(texts))
        with torch.no_grad():
            for idxes in chunked(np.argsort([len(p) for p in texts]), self.batch_size):
                _texts = [texts[i] for i in idxes]
                inputs = self.tokenizer(_texts, return_tensors="pt", padding=True, truncation=False).to(self.device)
                outputs = self.model(**inputs)
                shifted_input_ids = inputs.input_ids[:, 1:]
                lens = (shifted_input_ids != self.tokenizer.pad_token_id).sum(-1)
                # probs = torch.log_softmax(outputs.logits, dim=-1).detach()
                # gen_probs = torch.gather(probs[:, :-1, :], 2, shifted_input_ids[:, :, None]).squeeze(-1)
                # ppls[idxes] = [gen_probs[i, -l:].mean().item() for i, l in enumerate(lens)]
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
                gen_probs = loss_fn(outputs.logits[:, :-1, :].transpose(1, 2), shifted_input_ids)
                ppls[idxes] = (-gen_probs.sum(-1) / lens).detach().cpu().numpy()
        return ppls

    def _classify(self, prompts: list[str], choices: list[str]) -> list[str]:
        self.tokenizer.padding_side = 'left'
        # make sure all choices are tokenized to a single token
        choice_token_ids = [self.tokenizer.tokenize(c) for c in choices]
        assert all(len(c) == 1 for c in choice_token_ids)
        choice_token_ids = [c[0] for c in choice_token_ids]

        choices = [self.tokenizer.tokenize(c)[0] for c in choices]
        logprobs = np.empty((len(prompts), len(choices)))
        with torch.no_grad():
            for idxes in chunked(np.argsort([len(p) for p in prompts]), self.batch_size):
                _prompts = [prompts[i] for i in idxes]
                inputs = self.tokenizer(_prompts, return_tensors="pt", padding=True, truncation=False).to(self.device)
                outputs = self.model(**inputs)
                shifted_input_ids = inputs.input_ids[:, 1:]
                probs = torch.log_softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                choice_probs = probs[:, :, choice_token_ids]
                lens = (shifted_input_ids != self.tokenizer.pad_token_id).sum(-1)
                for i, idx in enumerate(idxes):
                    logprobs[idx] = choice_probs[i, lens[i] - 1]
                # logprobs[idxes] = probs[:, -2, choice_token_ids]
        choice_idxs = logprobs.argmax(axis=-1)
        return [choices[choice_idxs[i]] for i in range(len(prompts))]

    def _classify_v2(
        self, prompts: list[str], choices: list[str], return_losses: bool = False
    ) -> list[str]:
        self.tokenizer.padding_side = 'left'
        losses = np.empty(shape=(len(prompts), len(choices)))
        for j, choice in enumerate(choices):
            texts  = [prompt + choice for prompt in prompts]
            losses[:, j] = self._ppls(texts)
        choice_idxs = losses.argmax(axis=-1)
        answers = [choices[choice_idxs[i]] for i in range(len(prompts))]
        if not return_losses: return answers
        else: return answers, losses

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        import numpy as np
        result = LLMResult(generations=[None for _ in prompts],
                           llm_output={'token_usage': dict(
                               completion_tokens=0, prompt_tokens=0, total_tokens=0)})
        for idxes in chunked(np.argsort([len(p) for p in prompts]), self.batch_size):
            _prompts = [prompts[i] for i in idxes]
            inputs = self.tokenizer(_prompts, return_tensors="pt", padding=True, truncation=False).to(self.device)
            gen_kwargs = self.generation_kwargs
            if stop is not None:
                gen_kwargs = gen_kwargs | {"eos_token_id": self.tokenizer.encode(stop[0])[0]}
            # if inputs.attention_mask.shape[1] + gen_kwargs['max_new_tokens'] > context_length_limit[self.model_name]:
            #     breakpoint()
            assert inputs.attention_mask.shape[1] + gen_kwargs['max_new_tokens'] <= context_length_limit[self.model_name]
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.generation_kwargs)
            if self.task == "text-generation":
                outputs = outputs[:, inputs.attention_mask.shape[1]:]
            generations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # generations = [g.strip(self.tokenizer.pad_token).strip() for g in generations]
            if stop is not None:
                generations = [enforce_stop_tokens(g, stop) for g in generations]
            for i, g in zip(idxes, generations):
                result.generations[i] = [Generation(g)]
            # result.generations.extend([[Generation(g)] for g in generations])
        return result

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
        self.tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer]

        # tokenize the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = self.generation_kwargs
        if stop is not None:
            gen_kwargs = gen_kwargs | {"eos_token_id": self.tokenizer.encode(stop[0])[0]}
        assert inputs.shape[1] + gen_kwargs['max_new_tokens'] <= 2048
        with torch.no_grad():
            outputs = self.model.generate(inputs, **gen_kwargs)
        if self.task == "text-generation":
            outputs = outputs[:, inputs.shape[1]:]
        generation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            generation = enforce_stop_tokens(generation, stop)
        return generation

    def get_num_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

# def test():
if __name__ == '__main__':
    import pandas as pd
    from functools import partial
    from pathlib import Path

    from datasets import load_dataset
    # from langchain.llms.huggingface import HuggingFace
    from langchain import LLMChain
    from langchain import FewShotPromptTemplate2
    from selector import StructuralCoverageSelector, StructuralCoverageSelectorArgs
    from selector import CosineCoverageSelector, CosineCoverageSelectorArgs
    from constants import Dataset as D
    from data_utils import get_dataset, get_templates

    dataset = D.SST5
    input_feature, train_split, test_split = {
        D.SMCALFLOW_CS: ('source', 'paraphrase', 'test'),
        D.GEOQUERY: ('source', 'template_1_train', 'template_1_test'),
        D.OVERNIGHT: ('paraphrase', 'socialnetwork_template_0_train', 'socialnetwork_template_0_test'),
        D.MTOP: (None, 'train', 'validation'),
        D.BREAK: ('question_text', 'train', 'validation'),
        D.SST5: (None, 'train', 'validation'),
    }[dataset]
    ds = get_dataset(dataset, data_root=Path('../data'))
    candidates = ds[train_split].select(list(range(min(500, len(ds[train_split])))))
    # example_template = SemparseExampleTemplate(input_variables=['question_text', 'decomposition'])
    templates = get_templates(dataset, input_feature=input_feature, prompt_version='v1')
    example_template = templates['example_template']
    n_shots = 4
    # bm25_selector = StructuralCoverageSelector.from_examples(
    #     StructuralCoverageSelectorArgs(substruct='ngram', subst_size=4, depparser='spacy',n_shots=n_shots),
    #     candidates, example_template, query_examples=ds[test_split])
    cosine_selector = CosineCoverageSelector.from_examples(
        CosineCoverageSelectorArgs(emb_lm='sentence-transformers/all-mpnet-base-v2', coverage=False, n_shots=n_shots),
        candidates, example_template, query_examples=ds[test_split])
    fewshot_prompt_fn = partial(FewShotPromptTemplate2,
        input_variables=templates['example_template'].input_variables,
        example_separator='\n\n', **templates)
    prompt_template = fewshot_prompt_fn(example_selector=cosine_selector)
    prompts = [prompt_template.format(**ex) for ex in ds[test_split]]

    model_name = 'EleutherAI/gpt-neo-2.7B'
    device = 0
    generation_kwargs = dict(do_sample=False, max_new_tokens=256)
    llm = HuggingFace.from_model_name(
        model_name, device=device, task='text-generation', batch_size=4,
        generation_kwargs=generation_kwargs)

    if False:
        results = llm.generate(prompts[:10], stop=['\n'])

    if False:
        agent = LLMChain(prompt=bm25_prompt, llm=llm, verbose=True)
        completions = agent.apply([
            dict(**ex, stop=['\n']) for ex in ds[test_split].select(list(range(10)))])

    if False:
        answers, losses = llm._classify_v2(prompts[:2], choices=example_template.get_choices(), return_losses=True)

    if False:
        choices = example_template.get_choices()
        choice_token_ids = [llm.tokenizer.tokenize(c) for c in choices]
        assert all(len(c) == 1 for c in choice_token_ids)
        choice_token_ids = [c[0] for c in choice_token_ids]

        choices = [llm.tokenizer.tokenize(c)[0] for c in choices]
        logprobs = np.empty((len(prompts), len(choices)))
        with torch.no_grad():
            for idxes in chunked(np.argsort([len(p) for p in prompts]), llm.batch_size):
                _prompts = [prompts[i] for i in idxes]
                inputs = llm.tokenizer(_prompts, return_tensors="pt", padding=True, truncation=False).to(llm.device)
                outputs = llm.model(**inputs)
                probs = torch.log_softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                logprobs[idxes] = probs[:, -2, choice_token_ids]
        choice_idxs = logprobs.argmax(axis=-1)
        results = llm._classify(prompts[:2], choices=example_template.get_choices())