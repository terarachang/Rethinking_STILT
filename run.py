# coding=utf-8
# Modified from The HuggingFace Inc. team.

import pdb
import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from typing import Dict, List, Tuple
from sklearn.metrics import matthews_corrcoef, f1_score

from transformers import (
	WEIGHTS_NAME,
	AdamW,
	RobertaConfig,
	RobertaForMultipleChoice,
	RobertaTokenizer,
	RobertaForSequenceClassification,
	get_linear_schedule_with_warmup,
	PreTrainedTokenizer,
)
from utils import convert_examples_to_features, processors

try:
	from torch.utils.tensorboard import SummaryWriter
except ImportError:
	from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
SEQ_CLASS_TASKS = set(['mrpc', 'rte', 'cola', 'wic', 'sst2', 'mednli'])



def select_field(features, field):
	return [[choice[field] for choice in feature.choices_features] for feature in features]


def simple_accuracy(preds, labels):
	return (preds == labels).mean()


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
	""" Train the model """
	if args.local_rank in [-1, 0]:
		tb_writer = SummaryWriter()

	args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
	
	train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

	t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	args.warmup_steps = int(args.warmup_ratio*t_total)
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
	)

	# multi-gpu training
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)

	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(
			model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
		)

	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
	logger.info(
		"  Total train batch size (w. parallel, distributed & accumulation) = %d",
		args.train_batch_size
		* args.gradient_accumulation_steps
		* (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
	)
	logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
	logger.info("  lr = %f", args.learning_rate)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	tr_loss, logging_mc_loss = 0.0, 0.0
	model.zero_grad()
	model.train()

	train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
	set_seed(args)	# Added here for reproductibility
	
	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
		for step, batch in enumerate(epoch_iterator):
			batch = tuple(t.to(args.device) for t in batch)
			
			inputs = {
				"input_ids": batch[0],
				"attention_mask": batch[1],
				"labels": batch[2],
			}

			outputs = model(**inputs)
			mc_loss, mc_logits = outputs[:2]

			if args.n_gpu > 1:
				mc_loss = mc_loss.mean()  # average on multi-gpu parallel training
			if args.gradient_accumulation_steps > 1:
				mc_loss = mc_loss / args.gradient_accumulation_steps

			mc_loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

			tr_loss += mc_loss.item()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1

				if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
					tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
					tb_writer.add_scalar("mc_loss", (tr_loss - logging_mc_loss) / args.logging_steps, global_step)
					logger.info(
						"Average mc_loss: %s at global step: %s",
						str((tr_loss - logging_mc_loss) / args.logging_steps),
						str(global_step),
					)
					logging_mc_loss = tr_loss

		# save for each epoch
		if False and args.local_rank in [-1, 0] and args.save_steps > 0:
			# Save model checkpoint
			output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)
			model_to_save = (
				model.module if hasattr(model, "module") else model
			)  # Take care of distributed/parallel training
			model_to_save.save_pretrained(output_dir)
			tokenizer.save_vocabulary(output_dir)
			torch.save(args, os.path.join(output_dir, "training_args.bin"))
			logger.info("Saving model checkpoint to %s", output_dir)


	if args.local_rank in [-1, 0]:
		tb_writer.close()

	return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, tokenizer, prefix="", test=False):
	eval_output_dir = args.output_dir
	if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
		os.makedirs(eval_output_dir)

	# We always run inference on a single GPU
	args.eval_batch_size = args.per_gpu_eval_batch_size
	eval_sampler = SequentialSampler(eval_dataset)
	eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

	# Eval!
	model.eval()
	logger.info("***** Running evaluation {} *****".format(prefix))
	logger.info("  Num examples = %d", len(eval_dataset))
	logger.info("  Batch size = %d", args.eval_batch_size)
	eval_loss = 0.0
	nb_eval_steps = 0
	logits_list, label_list = [], []
	for batch in tqdm(eval_dataloader, desc="Evaluating"):
		batch = tuple(t.to(args.device) for t in batch)

		with torch.no_grad():
			inputs = {
				"input_ids": batch[0],
				"attention_mask": batch[1],
				"labels": batch[2],
			}
			outputs = model(**inputs)
			tmp_eval_loss, logits = outputs[:2]

			eval_loss += tmp_eval_loss.mean().item()
			
		nb_eval_steps += 1
		logits_list.append(logits.detach().cpu().numpy())
		label_list.append(inputs["labels"].detach().cpu().numpy())

	eval_loss = eval_loss / nb_eval_steps
	labels = np.concatenate(label_list)
	preds = np.concatenate(logits_list)
#	np.save('logits_{}_{}.npy'.
#		format(args.learning_rate, args.gradient_accumulation_steps), preds)
	preds = np.argmax(preds, axis=1)
	acc = simple_accuracy(preds, labels)

	result = {"eval_acc": acc, "eval_loss": eval_loss}
	if args.task_name in ["mrpc"]:
		result["f1"] = f1_score(labels, preds)
	if args.task_name in ["cola"]:
		result["mcc"] = matthews_corrcoef(labels, preds)

	output_eval_file = os.path.join(eval_output_dir, f"{args.learning_rate}_{args.per_gpu_train_batch_size*args.gradient_accumulation_steps}_{args.warmup_ratio}_eval_results.txt")

	with open(output_eval_file, "w") as writer:
		logger.info(f"***** Eval results {prefix} *****")
		writer.write("model			  =%s\n" % str(args.model_name_or_path))
		writer.write(
			"per_gpu_train_bs * grad_acc_steps=%d\n"
			% (
				args.per_gpu_train_batch_size
				* args.gradient_accumulation_steps
			)
		)
		writer.write("max seq length  =%d\n" % args.max_seq_length)
		for key in sorted(result.keys()):
			logger.info("  %s = %s", key, str(result[key]))
			writer.write("%s = %s\n" % (key, str(result[key])))
	return result


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
	if args.local_rank not in [-1, 0]:
		torch.distributed.barrier()

	processor = processors[task]()
	processor.set_mode(args.task_mode)
	# Load data features from cache or dataset file
	if evaluate:
		cached_mode = "dev"
	elif test:
		cached_mode = "test"
	else:
		cached_mode = "train"
	assert not (evaluate and test)
	cached_features_file = os.path.join(
		args.data_dir,
		"cached_{}_{}_{}_{}".format(
			cached_mode,
			args.model_type,
			str(args.max_seq_length),
			str(task),
		),
	)
	if args.task_mode != None: cached_features_file += f'_{args.task_mode}'
	if os.path.exists(cached_features_file) and not args.overwrite_cache:
		logger.info("Loading features from cached file %s", cached_features_file)
		features = torch.load(cached_features_file)
	else:
		logger.info("Creating features from dataset file at %s", args.data_dir)
		label_list = processor.get_labels()
		if evaluate:
			examples = processor.get_dev_examples(args.data_dir)
		elif test:
			examples = processor.get_test_examples(args.data_dir)
		else:
			examples = processor.get_train_examples(args.data_dir)
		logger.info("Training number: %s", str(len(examples)))
		features = convert_examples_to_features(
			examples,
			label_list,
			args.max_seq_length,
			tokenizer,
			task_name=args.task_name
		)
		if args.local_rank in [-1, 0]:
			logger.info("Saving features into cached file %s", cached_features_file)
			torch.save(features, cached_features_file)

	if args.local_rank == 0:
		torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

	# Convert to Tensors and build dataset
	all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
	all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
	all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
	if task in SEQ_CLASS_TASKS:
		all_input_ids, all_input_mask, all_label_ids = all_input_ids.squeeze(), all_input_mask.squeeze(), all_label_ids.squeeze()


	dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)

	return dataset


def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument(
		"--data_dir",
		default=None,
		type=str,
		required=True,
		help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
	)
	parser.add_argument(
		"--model_type",
		default=None,
		type=str,
		required=True,
		help="roberta",
	)
	parser.add_argument(
		"--model_name_or_path",
		default=None,
		type=str,
		required=True,
		help="Path to pre-trained model",
	)
	parser.add_argument(
		"--task_name",
		default=None,
		type=str,
		required=True,
		help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
	)
	parser.add_argument(
		"--task_mode",
		default=None,
		type=str,
		required=False,
		help="some tasks have different modes, e.g. different sizes",
	)
	parser.add_argument(
		"--output_dir",
		default=None,
		type=str,
		required=True,
		help="The output directory where the model predictions and checkpoints will be written.",
	)

	# Other parameters
	parser.add_argument(
		"--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
	)
	parser.add_argument(
		"--tokenizer_name",
		default="",
		type=str,
		help="Pretrained tokenizer name or path if not the same as model_name",
	)
	parser.add_argument(
		"--cache_dir",
		default="",
		type=str,
		help="Where do you want to store the pre-trained models downloaded from s3",
	)
	parser.add_argument(
		"--max_seq_length",
		default=128,
		type=int,
		help="The maximum total input sequence length after tokenization. Sequences longer "
		"than this will be truncated, sequences shorter will be padded.",
	)
	parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
	parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
	parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")
	parser.add_argument(
		"--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
	)

	parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
	parser.add_argument(
		"--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
	)
	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument(
		"--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
	)
	parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Linear warmup over warmup_steps [0~1].")

	parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
	parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
	parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
	parser.add_argument(
		"--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
	)
	parser.add_argument(
		"--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
	)
	parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

	parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
	args = parser.parse_args()

	if (
		os.path.exists(args.output_dir)
		and os.listdir(args.output_dir)
		and args.do_train
		and not args.overwrite_output_dir
	):
		raise ValueError(
			"Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
				args.output_dir
			)
		)

	# Setup CUDA, GPU & distributed training
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		torch.distributed.init_process_group(backend="nccl")
		args.n_gpu = 1
	args.device = device

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
	)
	logger.warning(
		"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
		args.local_rank,
		device,
		args.n_gpu,
		bool(args.local_rank != -1),
	)

	set_seed(args)

	args.task_name = args.task_name.lower()

	# Load pretrained model and tokenizer
	if args.local_rank not in [-1, 0]:
		torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

	model_class = RobertaForSequenceClassification if args.task_name in SEQ_CLASS_TASKS else RobertaForMultipleChoice
	config_class, tokenizer_class = RobertaConfig, RobertaTokenizer
	config = config_class.from_pretrained(
		args.config_name if args.config_name else args.model_name_or_path,
		cache_dir=args.cache_dir if args.cache_dir else None,
	)
	tokenizer = tokenizer_class.from_pretrained(
		args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
		do_lower_case=args.do_lower_case,
		cache_dir=args.cache_dir if args.cache_dir else None,
	)
	if args.task_name == "mednli": config.num_labels = 3
	model = model_class.from_pretrained(
		args.model_name_or_path,
		from_tf=bool(".ckpt" in args.model_name_or_path),
		config=config,
		cache_dir=args.cache_dir if args.cache_dir else None,
	)

	if args.local_rank == 0:
		torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

	model.to(args.device)
	logger.info("Training/evaluation parameters %s", args)

	# Training
	if args.do_train:
		train_datasets = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
		global_step, tr_loss = train(args, train_datasets, model, tokenizer)
		logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

		if args.local_rank == -1 or torch.distributed.get_rank() == 0:
			logger.info("Saving model checkpoint to %s", args.output_dir)
			if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
				os.makedirs(args.output_dir)

			model_to_save = (
				model.module if hasattr(model, "module") else model
			)  # Take care of distributed/parallel training
			model_to_save.save_pretrained(args.output_dir)
			tokenizer.save_pretrained(args.output_dir)
			torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

	# Evaluation
	if (args.do_eval or args.do_test) and args.local_rank in [-1, 0]:
		checkpoint = args.output_dir
		prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
		eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, 
						evaluate=args.do_eval, test=not args.do_eval)
		results = evaluate(args, eval_dataset, model, tokenizer, prefix=prefix)

if __name__ == "__main__":
	main()
