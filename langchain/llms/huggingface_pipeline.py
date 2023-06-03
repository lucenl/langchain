"""Wrapper around HuggingFace Pipeline APIs."""
import importlib.util
import logging
from typing import Any, List, Mapping, Optional

from pydantic import BaseModel, Extra

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

DEFAULT_MODEL_ID = "gpt2"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text2text-generation", "text-generation")

logger = logging.getLogger()


class HuggingFacePipeline(LLM, BaseModel):
    """Wrapper around HuggingFace Pipeline API.

    To use, you should have the ``transformers`` python package installed.

    Only supports `text-generation` and `text2text-generation` for now.

    Example using from_model_id:
        .. code-block:: python

            from langchain.llms import HuggingFacePipeline
            hf = HuggingFacePipeline.from_model_id(
                model_id="gpt2", task="text-generation"
            )
    Example passing pipeline in directly:
        .. code-block:: python

            from langchain.llms import HuggingFacePipeline
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_id = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
            )
            hf = HuggingFacePipeline(pipeline=pipe)
    """

    pipeline: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        task: str,
        device: int = -1,
        model_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
            from transformers import pipeline as hf_pipeline

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please it install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)

        try:
            if task == "text-generation":
                model = AutoModelForCausalLM.from_pretrained(model_id, **_model_kwargs)
            elif task == "text2text-generation":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **_model_kwargs)
            else:
                raise ValueError(
                    f"Got invalid task {task}, "
                    f"currently only {VALID_TASKS} are supported"
                )
        except ImportError as e:
            raise ValueError(
                f"Could not load the {task} model due to missing dependencies."
            ) from e

        if importlib.util.find_spec("torch") is not None:
            import torch

            cuda_device_count = torch.cuda.device_count()
            if device < -1 or (device >= cuda_device_count):
                raise ValueError(
                    f"Got device=={device}, "
                    f"device is required to be within [-1, {cuda_device_count})"
                )
            if device < 0 and cuda_device_count > 0:
                logger.warning(
                    "Device has %d GPUs available. "
                    "Provide device={deviceId} to `from_model_id` to use available"
                    "GPUs for execution. deviceId is -1 (default) for CPU and "
                    "can be a positive integer associated with CUDA device id.",
                    cuda_device_count,
                )

        pipeline = hf_pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_kwargs=_model_kwargs,
        )
        if pipeline.task not in VALID_TASKS:
            raise ValueError(
                f"Got invalid task {pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        return cls(
            pipeline=pipeline,
            model_id=model_id,
            model_kwargs=_model_kwargs,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_id": self.model_id},
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.pipeline(prompt)
        if self.pipeline.task == "text-generation":
            # Text generation return includes the starter text.
            text = response[0]["generated_text"][len(prompt) :]
        elif self.pipeline.task == "text2text-generation":
            text = response[0]["generated_text"]
        else:
            raise ValueError(
                f"Got invalid task {self.pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)
        return text

def test():
    import numpy as np
    import pandas as pd
    from functools import partial
    from pathlib import Path

    from datasets import load_dataset
    from langchain import LLMChain, FewShotPromptTemplate2
    from langchain.llms import HuggingFacePipeline
    from langchain.llms.utils import enforce_stop_tokens
    from transformers import pipeline
    from selector.bm25 import BM25ExampleSelector, BM25ExampleSelectorArgs
    from constants import Dataset as DS
    from driver import get_dataset, get_templates

    dataset, input_feature, train_split, test_split = DS.GEOQUERY, 'source', 'csl_template_1_train', 'csl_template_1_test'
    # dataset, input_feature, train_split, test_split = DS.OVERNIGHT, 'paraphrase', 'socialnetwork_template_0_train', 'socialnetwork_template_0_test'
    # dataset, input_feature, test_split = DS.BREAK, 'question_text', 'validation'
    ds = get_dataset(dataset, data_root=Path('../data'))
    candidates = ds[train_split].select(list(range(min(500, len(ds[train_split])))))
    # example_template = SemparseExampleTemplate(input_variables=['question_text', 'decomposition'])
    templates = get_templates(dataset, input_feature=input_feature)
    example_template = templates['example_template']
    n_shots = 16
    substruct = 'ngram'
    bm25_selector = BM25ExampleSelector.from_examples(
        BM25ExampleSelectorArgs(substruct, 4, n_shots, depparser='spacy'),
        candidates, example_template)
    fewshot_prompt_fn = partial(FewShotPromptTemplate2,
        input_variables=templates['example_template'].input_variables,
        example_separator='\n\n', **templates)
    bm25_prompt = fewshot_prompt_fn(example_selector=bm25_selector)

    prompts = [bm25_prompt.format(**ex) for ex in ds['train']]

    generation_kwargs = dict(
        temperature=0.0, max_new_tokens=256, top_p=1.0)
    model_id = 'EleutherAI/gpt-neo-2.7B'
    device = 4
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    if False:
        inputs = tokenizer.encode(prompts[0], return_tensors="pt").to(device)
        outputs = model.generate(inputs, **generation_kwargs)
        generation = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        completion = enforce_stop_tokens(generation, stop=['\n']).strip()
    inputs = tokenizer(prompts[:4], return_tensors="pt", padding=True, truncation=False).to(device)
    outputs = model.generate(**inputs, **generation_kwargs, eos_token_id=tokenizer.encode("\n")[0],)[:, inputs.attention_mask.shape[1]:]
    generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = enforce_stop_tokens(generation, stop=['\n']).strip()
    generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    completions = [enforce_stop_tokens(g, stop=['\n']).strip() for g in generations]



    # prompt_len = int(batch['attention_mask'].shape[1])
    # completions = [
    #     tokenizer.decode(output[prompt_len:]).strip(tokenizer.pad_token).strip()
    #     for output in outputs]
    # completion =




    pipe = pipeline(task='text-generation', device=2, model='EleutherAI/gpt-neo-2.7B', return_full_text=False)
    pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    generated = pipe(prompts[:4], batch_size=4, max_new_tokens=256, top_p=1.0, temperature=0.0, do_sample=False, num_return_sequences=1)
    completions = [enforce_stop_tokens(g[0]['generated_text'], stop=['\n'])
                   for g in generated]

    lm = HuggingFacePipeline.from_model_id(
        model_id='EleutherAI/gpt-neo-2.7B', task='text-generation', device='2',
        model_kwargs=dict(
            temperature=0.0, max_new_tokens=256, top_p=1.0,
            frequency_penalty=0.0, presence_penalty=0.0))
