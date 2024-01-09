"""Wrapper around OpenAI APIs."""
import logging
import sys
import time
import pause
import openai
import numpy as np
from typing import Generator, Mapping, Optional, Tuple, Union, Dict, Any, List

from pydantic import BaseModel, Extra, Field, root_validator

from more_itertools import chunked
from langchain.llms.base import BaseLLM
from langchain.schema import Generation, LLMResult
from langchain.utils import get_from_dict_or_env
from rich import print
from langchain.utilities.track import track

logger = logging.getLogger(__name__)


class OpenAIModel(BaseLLM, BaseModel):
    model_name: str = "code-davinci-002"
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: int = 256
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""
    frequency_penalty: float = 0
    """Penalizes repeated tokens according to frequency."""
    presence_penalty: float = 0
    """Penalizes repeated tokens."""
    n: int = 1
    """How many completions to generate for each prompt."""
    # best_of: int = 1
    # """Generates best_of completions server-side and returns the "best"."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    batch_size: int = 20
    """Batch size to use when passing multiple documents to generate."""
    base_delay: int = 15
    """Base delay for exponential backoff after every completion."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to OpenAI completion API. Default is 600 seconds."""
    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)
    """Adjust the probability of specific tokens being generated."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transfered to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        normal_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            # "best_of": self.best_of,
            "request_timeout": self.request_timeout,
            "logit_bias": self.logit_bias,
        }
        return {**normal_params, **self.model_kwargs}

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        return self._default_params

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "openai"

    def get_num_tokens(self, text: str) -> int:
        """Calculate num tokens with tiktoken package."""
        # tiktoken NOT supported for Python 3.8 or below
        if sys.version_info[1] <= 8:
            return super().get_num_tokens(text)
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please it install it with `pip install tiktoken`."
            )
        # create a GPT-3 encoder instance
        enc = tiktoken.get_encoding("gpt2")

        # encode the text using the GPT-3 encoder
        tokenized_text = enc.encode(text)

        # calculate the number of tokens in the encoded text
        return len(tokenized_text)

    def modelname_to_contextsize(self, modelname: str) -> int:
        """Calculate the maximum number of tokens possible to generate for a model.

        text-davinci-003: 4,000 tokens
        text-curie-001: 2,048 tokens
        text-babbage-001: 2,048 tokens
        text-ada-001: 2,048 tokens
        code-davinci-002: 8,000 tokens
        code-cushman-001: 2,048 tokens

        Args:
            modelname: The modelname we want to know the context size for.

        Returns:
            The maximum context size

        Example:
            .. code-block:: python

                max_tokens = openai.modelname_to_contextsize("text-davinci-003")
        """
        if modelname == "text-davinci-003":
            return 4000
        elif modelname == "text-curie-001":
            return 2048
        elif modelname == "text-babbage-001":
            return 2048
        elif modelname == "text-ada-001":
            return 2048
        elif modelname == "code-davinci-002":
            return 8000
        elif modelname == "code-cushman-001":
            return 2048
        else:
            return 4000

    def max_tokens_for_prompt(self, prompt: str) -> int:
        """Calculate the maximum number of tokens possible to generate for a prompt.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The maximum number of tokens to generate for a prompt.

        Example:
            .. code-block:: python

                max_tokens = openai.max_token_for_prompt("Tell me a joke.")
        """
        num_tokens = self.get_num_tokens(prompt)

        # get max context size for model by name
        max_size = self.modelname_to_contextsize(self.model_name)
        return max_size - num_tokens

class BaseOpenAI(OpenAIModel):
    """Wrapper around OpenAI large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain import OpenAI
            openai = OpenAI(model_name="text-davinci-003")
    """

    client: Any  #: :meta private:
    openai_api_key: Optional[str] = None
    delay: int = 15
    next_slot: int = time.time()
    keep_trying: bool = False
    verbose: bool = False

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        try:
            import openai

            openai.api_key = openai_api_key
            values["client"] = openai.Completion
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        return values

    def _get_response(self, prompts: str | List[str], params):
        while True:
            try:
                # print(f"Calling OpenAI API with {params}...")
                b = time.time()
                response = self.client.create(prompt=prompts, **params)
                if False:
                    response = self.client.create(
                        prompt=prompts,
                        engine=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        n=self.n,
                        stop=params['stop'])
                if False: print(f"OpenAI API call took {time.time() - b:.2f}s.")
                self.delay = self.base_delay
                self.next_slot = b + self.delay
                return response
            except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError) as e:
                self.delay = min(self.base_delay * 8, self.delay * 2)
                self.next_slot = b + self.delay
                # if self.keep_trying or not isinstance(e, openai.error.RateLimitError):
                if self.keep_trying:
                    pause.until(self.next_slot)
                    continue
                else:
                    raise e
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    def _agg_token_usage(self, token_usage: Dict[str, int], response) -> Dict[str, int]:
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        _keys_to_use = _keys.intersection(response["usage"])
        for _key in _keys_to_use:
            if _key not in token_usage:
                token_usage[_key] = response["usage"][_key]
            else:
                token_usage[_key] += response["usage"][_key]

    def _call_openai(self, prompts: List[str], params):
        choices = {i: [] for i in range(len(prompts))}
        token_usage = {}
        if self.batch_size == 1:
            for i, prompt in enumerate(prompts):
                response = self._get_response(prompt, params)
                choices[i].extend(response.choices)
        else:
            for batch_idx, _prompts in enumerate(chunked(prompts, self.batch_size)):
                response = self._get_response(_prompts, params)
                for choice in response.choices:
                    choices[batch_idx * self.batch_size + choice.index].append(choice)
        self._agg_token_usage(token_usage, response)
        return choices, token_usage

    def _get_ppl(self, choice):
        lens = len(choice['logprobs']['tokens'])
        ce_loss = sum(choice['logprobs']['token_logprobs'][1:])
        return ce_loss / (lens-1)  # no logprob on first token

    def _ppls(self, texts: list[str]) -> list[float]:
        params = self._invocation_params
        params['max_tokens'] = 0
        params['echo'] = True
        params['logprobs'] = 1
        completions, _ = self._call_openai(texts, params)
        ppls = np.array([self._get_ppl(completions[i][0]) for i in range(len(texts))])
        return ppls

    def _classify(self, prompts: list[str], choices: list[str]) -> list[str]:
        params = llm._invocation_params
        params['max_tokens'] = 1
        params['echo'] = False
        params['logprobs'] = 5
        completions, _ = self._call_openai(prompts, params)
        answers = []
        for i in range(len(prompts)):
            top_logprobs = completions[i][0]['logprobs']['top_logprobs'][0]
            curr_answer, curr_logprob = None, -1e9
            for tok, logprob in top_logprobs.items():
                if tok.strip() in choices and (logprob > curr_logprob):
                    curr_answer, curr_logprob = tok.strip(), logprob
            if curr_answer is None:
                print(f"Warning: could not classify prompt {prompts[i]}")
            answers.append(curr_answer)
        return answers

    def _classify_v2(
        self, prompts: list[str], choices: list[str]
    ) -> list[list[float]]:
        params = self._invocation_params
        params['max_tokens'] = 0
        params['echo'] = True
        params['logprobs'] = 1

        losses = np.empty(shape=(len(prompts), len(choices)))
        for j, choice in enumerate(choices):
            texts  = [prompt + choice for prompt in prompts]
            losses[:, j] = self._ppls(texts)
        choice_idxs = losses.argmax(axis=-1)
        return [choices[choice_idxs[i]] for i in range(len(prompts))]

    def _classify_v3(
        self, prompts: list[str], choices: list[list[str]], return_losses: bool = False
    ) -> list[str]:
        losses, answers = [], []
        for prompt, _choices in zip(prompts, choices):
            texts = [prompt + choice for choice in _choices]
            ppls = self._ppls(texts)
            choice_idxs = ppls.argmax()
            losses.append(ppls)
            answers.append(_choices[choice_idxs])
        if not return_losses: return answers
        else: return answers, losses

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Call out to OpenAI's endpoint with k unique prompts.

        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The full LLM output.

        Example:
            .. code-block:: python

                response = openai.generate(["Tell me a joke."])
        """
        # TODO: write a unit test for this
        params = self._invocation_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop

        if params["max_tokens"] == -1:
            if len(prompts) != 1:
                raise ValueError(
                    "max_tokens set to -1 not supported for multiple inputs."
                )
            params["max_tokens"] = self.max_tokens_for_prompt(prompts[0])
        choices, token_usage = self._call_openai(prompts, params)
        generations = [[Generation(text=choice["text"], generation_info=choice)
                        for choice in choices[i]]
                       for i in range(len(prompts))]
        # for i, prompt in enumerate(prompts):
        #     sub_choices = choices[i * self.n : (i + 1) * self.n]
        #     generations.append(
        #         [Generation(text=choice["text"], generation_info=choice)
        #          for choice in sub_choices]
        #     )
        return LLMResult(
            generations=generations, llm_output={"token_usage": token_usage}
        )

    def stream(self, prompt: str) -> Generator:
        """Call OpenAI with streaming flag and return the resulting generator.

        BETA: this is a beta feature while we figure out the right abstraction.
        Once that happens, this interface could change.

        Args:
            prompt: The prompts to pass into the model.

        Returns:
            A generator representing the stream of tokens from OpenAI.

        Example:
            .. code-block:: python

                generator = openai.stream("Tell me a joke.")
                for token in generator:
                    yield token
        """
        params = self._invocation_params
        if params["best_of"] != 1:
            raise ValueError("OpenAI only supports best_of == 1 for streaming")
        params["stream"] = True
        generator = self.client.create(prompt=prompt, **params)

        return generator


class BaseOpenAIChat(OpenAIModel):
    """Wrapper around OpenAI large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain import OpenAI
            openai = OpenAI(model_name="text-davinci-003")
    """

    client: Any  #: :meta private:
    openai_api_key: Optional[str] = None
    delay: int = 15
    next_slot: int = time.time()
    keep_trying: bool = False
    verbose: bool = False

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        try:
            import openai

            openai.api_key = openai_api_key
            values["client"] = openai.ChatCompletion
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        return values

    def _get_response(self, prompts: list[list[dict[str, str]]] | list[str], params):
        assert len(prompts) == 1
        messages = prompts[0]
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        while True:
            try:
                # print(f"Calling OpenAI API with {params}...")
                b = time.time()
                response = self.client.create(messages=messages, **params)
                if False:
                    response = self.client.create(
                        prompt=prompts,
                        engine=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        n=self.n,
                        stop=params['stop'])
                if False: print(f"OpenAI API call took {time.time() - b:.2f}s.")
                self.delay = self.base_delay
                self.next_slot = b + self.delay
                return response
            except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError) as e:
                self.delay = min(self.base_delay * 8, self.delay * 2)
                self.next_slot = b + self.delay
                # if self.keep_trying or not isinstance(e, openai.error.RateLimitError):
                if self.keep_trying:
                    pause.until(self.next_slot)
                    continue
                else:
                    raise e
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    def _generate(
        self, prompts: list[list[dict[str, str]]] | list[str], stop: Optional[list[str]] = None
    ) -> LLMResult:
        """Call out to OpenAI's endpoint with k unique prompts.

        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The full LLM output.

        Example:
            .. code-block:: python

                response = openai.generate(["Tell me a joke."])
        """
        # TODO: write a unit test for this
        params = self._invocation_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop

        if params["max_tokens"] == -1:
            if len(prompts) != 1:
                raise ValueError(
                    "max_tokens set to -1 not supported for multiple inputs."
                )
            params["max_tokens"] = self.max_tokens_for_prompt(prompts[0])
        sub_prompts = [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        # choices = []
        choices = {i: [] for i in range(len(prompts))}
        token_usage = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        for i, _prompts in track(enumerate(sub_prompts), total=len(sub_prompts), disable=not self.verbose):
            response = self._get_response(_prompts, params)
            for choice in response.choices:
                choices[i * self.batch_size + choice.index].append(choice)
            # choices.extend(response["choices"])
            _keys_to_use = _keys.intersection(response["usage"])
            for _key in _keys_to_use:
                if _key not in token_usage:
                    token_usage[_key] = response["usage"][_key]
                else:
                    token_usage[_key] += response["usage"][_key]
        generations = [[Generation(text=choice["message"]["content"], generation_info=choice)
                        for choice in choices[i]]
                       for i in range(len(prompts))]
        # for i, prompt in enumerate(prompts):
        #     sub_choices = choices[i * self.n : (i + 1) * self.n]
        #     generations.append(
        #         [Generation(text=choice["text"], generation_info=choice)
        #          for choice in sub_choices]
        #     )
        return LLMResult(
            generations=generations, llm_output={"token_usage": token_usage}
        )

    def stream(self, prompt: str) -> Generator:
        """Call OpenAI with streaming flag and return the resulting generator.

        BETA: this is a beta feature while we figure out the right abstraction.
        Once that happens, this interface could change.

        Args:
            prompt: The prompts to pass into the model.

        Returns:
            A generator representing the stream of tokens from OpenAI.

        Example:
            .. code-block:: python

                generator = openai.stream("Tell me a joke.")
                for token in generator:
                    yield token
        """
        params = self._invocation_params
        if params["best_of"] != 1:
            raise ValueError("OpenAI only supports best_of == 1 for streaming")
        params["stream"] = True
        generator = self.client.create(prompt=prompt, **params)

        return generator


class OpenAI(BaseOpenAI):
    """Generic OpenAI class that uses model name."""

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **super()._invocation_params}

class OpenAIChat(BaseOpenAIChat):
    """Generic OpenAI class that uses model name."""

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **super()._invocation_params}


class AzureOpenAI(BaseOpenAI):
    """Azure specific OpenAI class that uses deployment name."""

    deployment_name: str = ""
    """Deployment name to use."""

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            **{"deployment_name": self.deployment_name},
            **super()._identifying_params,
        }

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"engine": self.deployment_name}, **super()._invocation_params}

class OpenAIPooled(OpenAIModel):

    endpoints: list[OpenAI] = None #: :meta private:
    openai_api_keys: Optional[list[str]] = None
    usage: list[int] = []
    success: list[int] = []
    verbose: bool = False

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Create the endpoints"""
        del values['endpoints'], values['usage'], values['success']
        openai_api_keys = values.pop('openai_api_keys')
        endpoints = [OpenAI(**values, openai_api_key=key, keep_trying=False) for key in openai_api_keys]
        values = {**values, 'endpoints': endpoints, 'openai_api_keys': openai_api_keys,
                  'usage': [0] * len(endpoints), 'success': [0] * len(endpoints)}
        return values

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        from more_itertools import chunked
        result = LLMResult(generations=[],
                           llm_output={'token_usage': dict(completion_tokens=0, prompt_tokens=0, total_tokens=0)})
        for batch in chunked(prompts, self.batch_size):
            _res = self.complete_batch(batch, stop)
            result.generations.extend(_res.generations)
            for k in _res.llm_output['token_usage'].keys():
                result.llm_output['token_usage'][k] += _res.llm_output['token_usage'][k]
        return result

    def complete_batch(self, prompts, stop):
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
        while True:
            try:
                if False: print(f'Using endpoint {self.endpoints[0].openai_api_key} in {max(0, self.endpoints[0].next_slot - time.time()):.2f}s for {len(prompts)} prompts. Usage: {self.usage}. Success: {self.success}. ({sum(self.success)}/{sum(self.usage)})')
                idx, endpoint = min(enumerate(self.endpoints), key=lambda e: e[1].next_slot)
                if self.verbose:
                    print(f'Using endpoint {idx} in {max(0, endpoint.next_slot - time.time()):.2f}s for {len(prompts)} prompts. Usage: {self.usage}. Success: {self.success}. ({sum(self.success)}/{sum(self.usage)})')
                self.usage[idx] += 1
                slot = endpoint.next_slot
                pause.until(slot)
                result = endpoint._generate(prompts, stop)
                if self.verbose:
                    print(f'Endpoint {idx} completed {len(prompts)} prompts')
                self.success[idx] += 1
                return result
            except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError) as e:
                error_message = e._message[e._message.find('Limit:'):e._message.find(' Contact')] if isinstance(e, openai.error.RateLimitError) else e._message
                if self.verbose:
                    print(f'Endpoint [red]{idx}[/red] failed: {error_message} - will wait for [red]{endpoint.next_slot - time.time():.2f}[/red]s')
                continue
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

if __name__ == '__main__':
    if False:
        from langchain.llms.openai import OpenAIPooled
        llm = OpenAIPooled(
            # openai_api_keys=[l.strip() for l in open('../../openai_keys.txt').readlines()],
            openai_api_keys=[key], base_delay=2,
            batch_size=7, request_timeout=100,
            frequency_penalty=0, presence_penalty=0,
            model_name='code-davinci-002', max_tokens=100,
            top_p=1, temperature=0.7, verbose=True)
        llm.generate(['This is it!'] * 42, stop=['\n'])
    if True:
        # from langchain.llms.openai import OpenAI
        llm = OpenAI(
            # openai_api_keys=[l.strip() for l in open('../../openai_keys.txt').readlines()],
            openai_api_key=key, base_delay=2,
            batch_size=7, request_timeout=100,
            frequency_penalty=0, presence_penalty=0,
            model_name='code-davinci-002', max_tokens=100,
            top_p=1, temperature=0, verbose=True)
        prompts, choices = ['This is a '] * 10, ['cat', 'because']
        lp = llm.logprobs(prompts, choices)

        params = llm._invocation_params
        params['max_tokens'] = 1
        params['echo'] = False
        params['logprobs'] = 5
        res = llm._get_response(prompt, params)
        losses_l = []
        for idx, prompt in enumerate(prompts):
            _choices = [choices[idx]] if isinstance(choices[0], list) else choices
            responses = [llm._get_response(prompt + c, params) for c in _choices]
            losses = np.array([llm.extract_loss(r) for r in responses])
            losses_l.append(losses)

        params = llm._invocation_params
        params['max_tokens'] = 0
        params['echo'] = True
        params['logprobs'] = 1
        losses_l = []
        for idx, prompt in enumerate(prompts):
            _choices = [choices[idx]] if isinstance(choices[0], list) else choices
            responses = [llm._get_response(prompt + c, params) for c in _choices]
            losses = np.array([llm.extract_loss(r) for r in responses])
            losses_l.append(losses)
        print(lp)