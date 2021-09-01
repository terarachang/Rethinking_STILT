# coding=utf-8
# Modified from The HuggingFace Inc. team.

import csv
import glob
import json
import logging
import os
from typing import List
import pickle
import tqdm
import pdb
import numpy as np
from scipy.stats import pearsonr, spearmanr

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

class InputExample(object):
	def __init__(self, example_id, question=None, contexts=None, endings=None, 
			text_a=None, text_b=None, label=None):
		# multiple-choice classification
		self.example_id = example_id
		self.question = question
		self.contexts = contexts
		self.endings = endings
		self.label = label
		# GLUE
		self.text_a = text_a
		self.text_b = text_b


class InputFeatures(object):
	def __init__(self, example_id, choices_features, label):
		self.example_id = example_id
		self.choices_features = [
			{"input_ids": input_ids, "input_mask": input_mask}
			for input_ids, input_mask in choices_features
		]
		self.label = label


class DataProcessor(object):
	"""Base class for data converters for multiple choice data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_test_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the test set."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	def set_mode(self, mode):
		"""Sets the mode of a task if required"""
		self.task_mode = mode

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r", encoding="utf-8-sig") as f:
			return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


class COSQAProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		logger.info("LOOKING AT {} train".format(data_dir))
		return self._create_examples(self._read_csv(os.path.join(data_dir, "cosmosqa_train.csv")))

	def get_dev_examples(self, data_dir):
		logger.info("LOOKING AT {} dev".format(data_dir))
		return self._create_examples(self._read_csv(os.path.join(data_dir, "cosmosqa_val.csv")))

	def get_labels(self):
		return ['0', '1', '2', '3']

	def _read_csv(self, input_file):
		with open(input_file, "r", encoding="utf-8") as f:
			return list(csv.reader(f))

	def _create_examples(self, lines: List[List[str]]):
		examples = [
			InputExample(
				example_id=line[0],
				question=line[2],
				contexts=[line[1]]*4,
				endings=[line[3], line[4], line[5], line[6]],
				label=line[7],
			)
			for line in lines[1:]  # we skip the line with the column names
		]

		return examples


class HellaSWAGProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		logger.info("LOOKING AT {} train".format(data_dir))
		return self._create_examples(os.path.join(data_dir, "hellaswag_train.jsonl"))

	def get_dev_examples(self, data_dir):
		logger.info("LOOKING AT {} dev".format(data_dir))
		return self._create_examples(os.path.join(data_dir, "hellaswag_val.jsonl"))


	def get_labels(self):
		return [0, 1, 2, 3]

	def _create_examples(self, path):
		data = []
		with open(path, "r", encoding="utf-8") as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip()
				line = json.loads(line)
				data.append(line)

		examples = [
			InputExample(
				example_id=idx,
				question=dt["ctx_b"], 
				contexts=[dt["ctx_a"]]*4,
				endings=dt["endings"],
				label=dt["label"],
			)
			for idx, dt in enumerate(data)
		]
		return examples


class ContinueProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		logger.info("LOOKING AT {} train".format(data_dir))
		return self._create_examples(os.path.join(data_dir, "SynGPT2_train.jsonl"))

	def get_dev_examples(self, data_dir):
		logger.info("LOOKING AT {} dev".format(data_dir))
		return self._create_examples(os.path.join(data_dir, "SynGPT2_dev.jsonl"))


	def get_labels(self):
		return [0, 1, 2, 3]

	def _create_examples(self, path):
		data = []
		with open(path, "r", encoding="utf-8") as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip()
				line = json.loads(line)
				data.append(line)

		examples = [
			InputExample(
				example_id=idx,
				question=None, 
				contexts=[dt["ctx"]]*4,
				endings=dt["endings"],
				label=0,
			)
			for idx, dt in enumerate(data)
		]
		return examples


class WinograndeProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		return self._create_examples(
			self._read_jsonl(os.path.join(data_dir, f"wino_train_{self.task_mode}.jsonl")))

	def get_dev_examples(self, data_dir):
		return self._create_examples(
			self._read_jsonl(os.path.join(data_dir, "wino_dev.jsonl")))

	def get_labels(self):
		return ["1", "2"]

	def _read_jsonl(self, input_file):
		"""Reads a tab separated value file."""
		records = []
		with open(input_file, "r", encoding="utf-8-sig") as f:
			for line in f:
				records.append(json.loads(line))
			return records

	def _create_examples(self, records):
		examples = []
		for (i, record) in enumerate(records):
			guid = record['qID']
			sentence = record['sentence']

			name1 = record['option1']
			name2 = record['option2']
			if not 'answer' in record:
				# This is a dummy label for test prediction.
				# test.jsonl doesn't include the `answer`.
				label = "1"
			else:
				label = record['answer']

			conj = "_"
			idx = sentence.index(conj)
			context = sentence[:idx]
			option_str = "_ " + sentence[idx + len(conj):].strip()

			option1 = option_str.replace("_", name1)
			option2 = option_str.replace("_", name2)

			examples.append(
				InputExample(
					example_id=guid,
					question="", 
					contexts=[context]*2,
					endings=[option1, option2],
					label=label,
				)
			)
		return examples


class IQAProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		logger.info("LOOKING AT {} train".format(data_dir))
		return self._create_examples(os.path.join(data_dir, f"{self.task_mode}_iqa_train"))

	def get_dev_examples(self, data_dir):
		logger.info("LOOKING AT {} dev".format(data_dir))
		return self._create_examples(os.path.join(data_dir, "dev"))


	def get_labels(self):
		return ["1", "2", "3"]

	def _create_examples(self, path):
		data = []
		with open(path+'.jsonl', "r", encoding="utf-8") as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip()
				line = json.loads(line)
				data.append(line)

		labels = open(path+'-labels.lst', encoding="utf-8").read().splitlines()

		examples = [
			InputExample(
				example_id=idx,
				question=dt["question"], 
				contexts=[dt["context"]]*3,
				endings=[dt["answerA"], dt["answerB"], dt["answerC"]],
				label=label,
			)
			for idx, (dt, label) in enumerate(zip(data, labels))
		]
		return examples


class MRPCProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		logger.info("LOOKING AT {} train".format(data_dir))
		return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

	def get_dev_examples(self, data_dir):
		logger.info("LOOKING AT {} dev".format(data_dir))
		return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

	def get_labels(self):
		return ["0", "1"]

	def _create_examples(self, lines, mode):
		examples = []
		for idx, line in enumerate(lines):
			if idx == 0: continue # header
			examples.append(
				InputExample(
					example_id=idx,
					text_a=line[3], 
					text_b=line[4],
					label=None if mode == "test" else line[0],
				)
			)

		return examples


class SST2Processor(DataProcessor):
	def get_train_examples(self, data_dir):
		logger.info("LOOKING AT {} train".format(data_dir))
		return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

	def get_dev_examples(self, data_dir):
		logger.info("LOOKING AT {} dev".format(data_dir))
		return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

	def get_labels(self):
		return ["0", "1"]

	def _create_examples(self, lines, mode):
		examples = []
		for idx, line in enumerate(lines):
			if idx == 0: continue # header
			examples.append(
				InputExample(
					example_id=idx,
					text_a=line[0], 
					text_b=None,
					label=None if mode == "test" else line[1],
				)
			)
		if mode == 'train':
			num_data = int(self.task_mode)
			return examples[:num_data]
		else: return examples


class RTEProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		logger.info("LOOKING AT {} train".format(data_dir))
		return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

	def get_dev_examples(self, data_dir):
		logger.info("LOOKING AT {} dev".format(data_dir))
		return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

	def get_labels(self):
		return ["entailment", "not_entailment"]

	def _create_examples(self, lines, mode):
		"""Creates examples for the training and dev sets."""
		examples = []
		for idx, line in enumerate(lines):
			if idx == 0: continue # header
			examples.append(
				InputExample(
					example_id=idx,
					text_a=line[1], 
					text_b=line[2],
					label=None if mode == "test" else line[-1],
				)
			)

		return examples


class ColaProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		logger.info("LOOKING AT {} train".format(data_dir))
		return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

	def get_dev_examples(self, data_dir):
		logger.info("LOOKING AT {} dev".format(data_dir))
		return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

	def get_labels(self):
		return ["0", "1"]

	def _create_examples(self, lines, mode):
		examples = []
		for idx, line in enumerate(lines):
			examples.append(
				InputExample(
					example_id=idx,
					text_a=line[3], 
					text_b=None,
					label=None if mode == "test" else line[1],
				)
			)

		return examples


class WiCProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		logger.info("LOOKING AT {} train".format(data_dir))
		return self._create_examples(data_dir, "train")

	def get_dev_examples(self, data_dir):
		logger.info("LOOKING AT {} dev".format(data_dir))
		return self._create_examples(data_dir, "dev")

	def get_labels(self):
		return ["F", "T"]

	def _create_examples(self, data_dir, mode):
		lines = open(os.path.join(data_dir, mode, '{}.data.txt'.format(mode))).read().splitlines()
		labels = open(os.path.join(data_dir, mode, '{}.gold.txt'.format(mode))).read().splitlines()
		
		examples = []
		for idx, line in enumerate(lines):
			line = line.split('\t')
			examples.append(
				InputExample(
					example_id=idx,
					question=line[0],
					text_a=line[-2], 
					text_b=line[-1],
					label=None if mode == "test" else labels[idx],
				)
			)

		return examples


class MLIProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		logger.info("LOOKING AT {} train".format(data_dir))
		return self._create_examples(os.path.join(data_dir, "mli_train_v1.jsonl"))

	def get_dev_examples(self, data_dir):
		logger.info("LOOKING AT {} dev".format(data_dir))
		return self._create_examples(os.path.join(data_dir, "mli_dev_v1.jsonl"))

	def get_labels(self):
		return ["entailment", "contradiction", "neutral"]

	def _create_examples(self, path):
		examples = []
		with open(path, "r", encoding="utf-8") as f:
			lines = f.readlines()
			for idx, line in enumerate(lines):
				line = line.strip()
				line = json.loads(line)
				examples.append(
					InputExample(
						example_id=idx,
						text_a=line['sentence1'], 
						text_b=line['sentence2'],
						label=line['gold_label']
					)
				)
		return examples


def convert_examples_to_features(
	examples: List[InputExample],
	label_list: List[str],
	max_length: int,
	tokenizer: PreTrainedTokenizer,
	task_name,
) -> List[InputFeatures]:
	"""
	Loads a data file into a list of `InputFeatures`
	"""

	label_map = {label: i for i, label in enumerate(label_list)}

	max_seq_len = 0
	features, all_lengths = [], []
	for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))
		
		label = label_map[example.label] 

		if task_name in ['mrpc', 'rte', 'cola', 'wic', 'sst2', 'mednli']: # 1 sent (text_b=None) or 2 sents
			if task_name == 'wic':
				example.text_a = example.question + tokenizer.sep_token + example.text_a
			inp = tokenizer.encode_plus(example.text_a, example.text_b, 
				add_special_tokens=True, max_length=max_length, pad_to_max_length=True)
			choices_features = [(inp["input_ids"], inp["attention_mask"])]
		
			cur_len = len(tokenizer.encode(example.text_a, example.text_b, add_special_tokens=True))
			max_seq_len = max(max_seq_len, cur_len)
			all_lengths.append(cur_len)
			
		else: # MC datasets, iterate through all choices
			choices_features = []
			for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
				if task_name in ['winogrande', 'cont']: 
					text_a = context
					text_b = ending
				elif task_name in ['iqa', 'cosmosqa']:
					text_a = context + tokenizer.sep_token + example.question
					text_b = ending
				elif task_name == 'hellaswag':
					text_a = context
					text_b = example.question + " " + ending
				elif task_name == 'hella-p':
					text_a = ending
					text_b = None

				inp = tokenizer.encode_plus(text_a, text_b, 
					add_special_tokens=True, max_length=max_length, pad_to_max_length=True)
				input_pack = (inp["input_ids"], inp["attention_mask"])
				choices_features.append(input_pack)
			
				cur_len = len(tokenizer.encode(text_a, text_b, add_special_tokens=True))
				max_seq_len = max(max_seq_len, cur_len)
				all_lengths.append(cur_len)

		if ex_index < 2:
			logger.info("*** Example ***")
			logger.info("example id: {}".format(example.example_id))
			for choice_idx, (input_ids, attention_mask) in enumerate(choices_features):
				logger.info("choice: {}".format(choice_idx))
				logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
				logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
				logger.info("label: {}".format(label))
				logger.info("input: {}".format(tokenizer.decode(input_ids)))

		features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label,))

	logger.info('[{}] Max_seq_len: {}, Avg: {:.1f}'.format(task_name, max_seq_len, np.mean(all_lengths)))
	return features


processors = {"hellaswag": HellaSWAGProcessor, "hella-p": HellaSWAGProcessor,
		"iqa": IQAProcessor, "wic": WiCProcessor, "winogrande": WinograndeProcessor,
		"rte": RTEProcessor, "cola": ColaProcessor,
		"sst2": SST2Processor, "cont": ContinueProcessor, "mednli": MLIProcessor,
		"cosmosqa": COSQAProcessor, "mrpc": MRPCProcessor}



