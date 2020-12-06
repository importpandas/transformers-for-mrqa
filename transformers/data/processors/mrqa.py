import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count

#import random
import numpy as np
import sys
from tqdm import tqdm

from ...file_utils import is_tf_available, is_torch_available
from ...tokenization_bert import whitespace_tokenize
from .utils import DataProcessor
from .squad import _is_whitespace

total_mismatch_num = 0

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

class MrqaExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_tokens,
        answer_text,
        answer_position_token,
        answers=[],
        context_text="",
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = []
        self.answer_text = answer_text
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = answer_position_token[0], answer_position_token[1]

        separate_tokens = ['[DOC]', '[TLE]', ['PAR']]
        char_to_word_offset = []
        for i, doc_token in enumerate(context_tokens):
            if doc_token in separate_tokens:
                self.start_position = self.start_position - 1 if self.start_position > i else self.start_position
                self.end_position = self.end_position - 1 if self.end_position >= i else self.end_position
            else:
                self.doc_tokens.append(doc_token)
                for _ in range(len(self.doc_tokens)):
                    char_to_word_offset.append(len(self.doc_tokens) - 1)

        actual_text = "".join(context_tokens[self.start_position : (self.end_position + 1)])
        cleaned_answer_text = "".join(whitespace_tokenize(self.answer_text))
        global total_mismatch_num
        if cleaned_answer_text.lower() not in actual_text.lower():
            total_mismatch_num += 1
            #print(f'total mismatch num {total_mismatch_num}')
        assert char_to_word_offset[-1] == len(self.doc_tokens) - 1

        self.char_to_word_offset = char_to_word_offset


class MrqaProcessor(DataProcessor):
    """
    Processor for the MRQA data set.
    """

    train_file = None
    dev_file = None

    def get_train_examples(self, data_dir, filename):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: File name of training dataset

        """
        input_data = []
        with open(
            os.path.join(data_dir, filename), "r", encoding="utf-8"
        ) as reader:
            for line in reader:
                input_data.append(json.loads(line))
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """


        input_data = []
        with open(
            os.path.join(data_dir, filename), "r", encoding="utf-8"
        ) as reader:
            for line in reader:
                input_data.append(json.loads(line))
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            if 'header' in entry:
                continue
            context_tokens = [i[0] for i in entry["context_tokens"]]
            for qa in entry["qas"]:
                qas_id = qa["qid"]
                question_text = qa["question"]
                answer_text = None
                answers = []

                if is_training:
                    answer = qa["detected_answers"][0]
                    answer_text = answer["text"]
                    answers = qa["answers"]
                    answer_position_token = sorted(answer['token_spans'], key=lambda x:x[0])[0]
                else:
                    answers = qa["answers"]

                example = MrqaExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_tokens=context_tokens,
                    answer_text=answer_text,
                    answer_position_token=answer_position_token,
                    answers=answers,
                )

                examples.append(example)
        print(f' all examples num: {len(examples)}')
        return examples