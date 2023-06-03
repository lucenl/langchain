"""Wrapper around HuggingFace Pipeline APIs."""
import torch
from typing import Any, List, Mapping, Optional, Union

from pydantic import BaseModel, Extra
from pathlib import Path

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema import Generation, LLMResult

DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text2text-generation", "text-generation")

def get_ssd():
    if Path('/srv/nvme0/ucinlp/shivag5/').exists():
        return Path('/srv/nvme0/ucinlp/shivag5/')
    elif Path('/srv/disk01/ucinlp/shivag5/').exists():
        return Path('/srv/disk01/ucinlp/shivag5/')
    else:
        raise ValueError('No SSD found')


def llama_path(model_size: str = '7B'):
    if (get_ssd() / f'llama_hf/{model_size}').exists():
        return get_ssd() / f'llama_hf/{model_size}'
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
                if model_name != 'bigcode/starcoder':
                    tokenizer = AutoTokenizer.from_pretrained(model_name, **_model_kwargs, use_auth_token=True)
                else:
                    starcoder_path = get_ssd() / 'bigcode/starcoder'
                    tokenizer = AutoTokenizer.from_pretrained(starcoder_path, **_model_kwargs, use_auth_token=True)
            else:
                from transformers import LlamaTokenizer
                if '7B' in model_name: model_size = '7B'
                elif '13B' in model_name: model_size = '13B'
                elif '30B' in model_name: model_size = '30B'
                elif '65B' in model_name: model_size = '65B'
                else: raise ValueError(f'Invalid llama model: {model_name}')
                llama_path = get_ssd() / f'llama_hf/{model_size}'
                tokenizer = LlamaTokenizer.from_pretrained(llama_path, **_model_kwargs)

            if task == "text-generation":
                # if model_name == 'togethercomputer/GPT-JT-6B-v1':
                if 'llama' not in model_name:
                    if model_name != 'bigcode/starcoder':
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

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        import numpy as np
        from more_itertools import chunked
        result = LLMResult(generations=[None for _ in prompts],
                           llm_output={'token_usage': dict(
                               completion_tokens=0, prompt_tokens=0, total_tokens=0)})
        for idxes in chunked(np.argsort([len(p) for p in prompts]), self.batch_size):
            _prompts = [prompts[i] for i in idxes]
            inputs = self.tokenizer(_prompts, return_tensors="pt", padding=True, truncation=False).to(self.device)
            gen_kwargs = self.generation_kwargs
            if stop is not None:
                gen_kwargs = gen_kwargs | {"eos_token_id": self.tokenizer.encode(stop[0])[0]}
            if inputs.attention_mask.shape[1] + gen_kwargs['max_new_tokens'] > 2048:
                breakpoint()
            assert inputs.attention_mask.shape[1] + gen_kwargs['max_new_tokens'] <= 2048
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

def test():
    import pandas as pd
    from functools import partial
    from pathlib import Path

    from datasets import load_dataset
    from langchain import FewShotPromptTemplate2
    from selector import StructuralCoverageSelector, StructuralCoverageSelectorArgs
    from constants import Dataset as D
    from data_utils import get_dataset, get_templates

    dataset = D.GEOQUERY
    input_feature, train_split, test_split = {
        D.SMCALFLOW_CS: ('source', 'paraphrase', 'test'),
        D.GEOQUERY: ('source', 'template_1_train', 'template_1_test'),
        D.OVERNIGHT: ('paraphrase', 'socialnetwork_template_0_train', 'socialnetwork_template_0_test'),
        D.MTOP: (None, 'train', 'validation'),
        D.BREAK: ('question_text', 'train', 'validation'),
    }[dataset]
    ds = get_dataset(dataset, data_root=Path('../data'))
    candidates = ds[train_split].select(list(range(min(500, len(ds[train_split])))))
    # example_template = SemparseExampleTemplate(input_variables=['question_text', 'decomposition'])
    templates = get_templates(dataset, input_feature=input_feature)
    example_template = templates['example_template']
    n_shots = 16
    bm25_selector = StructuralCoverageSelector.from_examples(
        StructuralCoverageSelectorArgs(substruct='ngram', subst_size=4, depparser='spacy',n_shots=n_shots),
        candidates, example_template)
    fewshot_prompt_fn = partial(FewShotPromptTemplate2,
        input_variables=templates['example_template'].input_variables,
        example_separator='\n\n', **templates)
    bm25_prompt = fewshot_prompt_fn(example_selector=bm25_selector)

    prompts = [bm25_prompt.format(**ex) for ex in ds[test_split]]

    model_name = 'EleutherAI/gpt-neo-2.7B'
    device = 0
    generation_kwargs = dict(do_sample=False, max_new_tokens=256)
    from langchain.llms.huggingface import HuggingFace
    from langchain import LLMChain
    llm = HuggingFace.from_model_name(
        model_name, device=device, task='text-generation', batch_size=4,
        generation_kwargs=generation_kwargs)
    results = llm.generate(prompts[:10], stop=['\n'])
    agent = LLMChain(prompt=bm25_prompt, llm=llm, verbose=True)
    completions = agent.apply([
        dict(**ex, stop=['\n']) for ex in ds[test_split].select(list(range(10)))])