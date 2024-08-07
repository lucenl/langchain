"""Prompt template classes."""
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate, FewShotPromptTemplate2
from langchain.prompts.loading import load_prompt
from langchain.prompts.prompt import Prompt, PromptTemplate

__all__ = [
    "BasePromptTemplate",
    "load_prompt",
    "PromptTemplate",
    "FewShotPromptTemplate",
    "FewShotPromptTemplate2",
    "Prompt",
]
