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
from composer.models import ComposerModel
import torch
import os

__all__ = ['OpenAICausalLMEvalWrapper', 'OpenAITokenizerWrapper']

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenAITokenizerWrapper:
    def __init__(self, name) -> None:
        self.tokenizer = tiktoken.encoding_for_model(name)

    def __call__(self, x, add_special_tokens=False):
        return self.encode(x)

    def encode(self, x, add_special_tokens=False):
        if isinstance(x, str):
            return {
                "input_ids": self.tokenizer.encode(x)
            }
        else:
            return {
                "input_ids": self.tokenizer.encode_batch(x)
            }
    
    def decode(self, x):
        return self.tokenizer.decode(x)

    @property
    def pad_token_id(self):
        return self.tokenizer.eot_token
    
    @property
    def eos_token_id(self):
        return self.tokenizer.eot_token

class OpenAICausalLMEvalWrapper(ComposerModel):
  
    def __init__(self, model_name, tokenizer):
        self.model_name = model_name['version']
        self.tokenizer = tokenizer
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
        super(OpenAICausalLMEvalWrapper, self).__init__()
        self.mocked_layer = torch.nn.Linear(2,3)
        

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        # If the batch mode is generate, we will generate a requested number of tokens using the underlying
        # model's generate function. Extra generation kwargs can be passed in via the batch. Strings will
        # be returned from eval_forward
        for tokens, cont_idxs in zip(batch['input_ids'], batch['continuation_indices']):
            tokens = tokens.tolist()
            cont_idxs = cont_idxs.tolist()
            prompt_text = self.tokenizer.decode(tokens[:cont_idxs[0]])
            expected_cont_tokens = tokens[cont_idxs[0]:cont_idxs[-1]]
            expected_cont_string = self.tokenizer.decode(expected_cont_tokens)
            chat_completion = openai.ChatCompletion.create(
                engine=self.model_name,
                prompt=prompt_text,
                max_tokens=len(expected_cont_tokens),  
                logprobs=5
            )
            

    def forward(self):
        pass

    def loss(self):
        pass