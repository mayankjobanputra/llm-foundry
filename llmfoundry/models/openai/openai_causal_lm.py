# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a Hugging Causal LM wrapped inside a :class:`.ComposerModel`."""

from typing import Any, Optional, Union
import tiktoken
import openai
# required for loading a python model into composer
from composer.metrics.nlp import (InContextLearningLMAccuracy,
                                  InContextLearningLMExpectedCalibrationError,
                                  InContextLearningMCExpectedCalibrationError,
                                  InContextLearningMultipleChoiceAccuracy,
                                  InContextLearningQAAccuracy,
                                  LanguageCrossEntropy, LanguagePerplexity)

from transformers import (PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

__all__ = ['OpenAICausalLMEvalWrapper', 'OpenAITokenizerWrapper']

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class OpenAITokenizerWrapper:
    def __init__(self, name) -> None:
        self.tokenizer = tiktoken.encoding_for_model(name)

    def __call__(self, x, add_special_tokens=False):
        if isinstance(x, str):
            return {
                "input_ids": self.tokenizer.encode(x)
            }
        else:
            return {
                "input_ids": self.tokenizer.encode_batch(x)
            }

    @property
    def pad_token_id(self):
        return self.tokenizer.eot_token
    
    @property
    def eos_token_id(self):
        return self.tokenizer.eot_token

class OpenAICausalLMEvalWrapper:
  
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        # set up training and eval metrics
        eval_metrics = [
            LanguageCrossEntropy(),
            LanguagePerplexity(),
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError()
        ]
        

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        # If the batch mode is generate, we will generate a requested number of tokens using the underlying
        # model's generate function. Extra generation kwargs can be passed in via the batch. Strings will
        # be returned from eval_forward
        breakpoint()
        chat_completion = openai.ChatCompletion.create(model=self.model_name, messages=[{"role": "user", "content": "Hello world"}])


  