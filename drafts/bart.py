from genericpath import exists
import os
import argparse
import random
import logging
import timeit

from tqdm import tqdm, trange
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import PretrainedConfig, AdamW, get_linear_schedule_with_warmup

from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.tokenization_bart import BartTokenizer
from transformers.models.bart.modeling_bart import BartPretrainedModel, BartModel, BartForConditionalGeneration

from ..utils import read_wrc_examples, convert_examples_to_features, RawResult, write_predictions
from ..utils_evaluate import EvalOpts, main as evaluate_on_websrc 

logger = logging.getLogger(__name__)
logger.setLevel ???

"""
A naive version of bart-finetuning on websrc with textual and html features
To do:
    - distributed training
    - common interface for backbones and methods
    - combine train and eval
    - SummaryWriter/ tensorBoardX
    - other optimizers
    - fp16, server for remote debugging...
"""

"""
All breakpoints are args to be dealt from command line
"""

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed_all(seed) # if n_gpu > 0

def to_list(tensor):
    return tensor.detach().cpu().tolist()

class StrctDataset(Dataset):
    """Dataset wrapping tensors

    Each sample will be retrieved by indexing tensors along the first dimension
    Later the dataset is sent into a sampler then a dataloader, and transfer to `input` in training phase

    Arguments:
        *tensors (*torch.Tensor): tensors having the same size of the first dimension. 
            For example (all_input_ids, all_input_masks, all_segment_ids, all_feature_index)
        page_id (list): the corresponding page_ids of the input features.
        token_to_tag (torch.Tensor): the mapping from each token to its corresponding tag id
    """

    def __init__(self, *tensors, page_ids=None, token_to_tag=None):
        tensors = tuple(tensor for tensor in tensors)
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors), "Invalid input tensors with different size in the 1st dimension, expected same size."
        self.tensors = tensors
        self.page_ids = page_ids
        self.token_to_tag = token_to_tag

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors) # return the indexed tensors if no features from other modalities is required

    def __len__(self):
        return len(self.tensors[0])


def get_bart(config):
    return BartForConditionalGeneration(config)

def train(model, targs, train_dataset, tokenizer):
    # tb_writer = SummaryWriter # from tensorboardX import SummaryWriter

    # train args and data
    targs.train_batch_size = targs.per_gpu_train_batch_size * 1 # no distribution here
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=targs.train_batch_size, sampler=train_sampler)

    if targs.max_steps > 0:
        targs.num_training_epochs = targs.max_steps // (len(train_dataloader) // targs.gradient_accumulation_steps) + 1
        num_training_steps = targs.max_steps
    else:
        num_training_steps = len(train_dataloader) // targs.gradient_accumulation_steps * targs.num_training_epochs

    # optimizer and schedule
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
        'weight_decay': targs.weight_decay},
        {'params':[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=targs.learing_rate, eps=targs.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=targs.num_warmup_steps, num_training_steps=num_training_steps)

    # Train begins
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", targs.num_training_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", targs.per_gpu_train_batch_size)
    logger.info("  Total train batch size = %d", targs.train_batch_size * targs.gradient_accumulation_steps)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #             targs.train_batch_size * args.gradient_accumulation_steps * (
    #                 torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", targs.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_training_steps)

    global_step = 0
    train_loss, logging_loss = 0.0, 0.0 # logging_loss for SummaryWriter
    model.zero_grad()
    train_trange = trange(int(targs.num_train_epochs), decs="Epoch")
    set_seed(targs.seed)
    for _ in train_trange:
        epoch_iterator = tqdm(train_dataloader, desc="Iter")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(targs.device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                ???
            }
            outputs = model(**inputs) ??? Why using a dict to copy input
            loss = outputs[0] ???

            if targs.gradient_accumulation_steps > 1:
                loss = loss / targs.gradient_accumulation_steps
            loss.backward()

            train_loss += loss.item()
            if (step + 1) % targs.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), targs.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if targs.logging_steps > 0 and global_step % targs.logging_steps == 0:
                    pass

                if targs.saving_steps > 0 and global_step % targs.saving_steps == 0:
                    # save model checkpoints
                    ckpt_dir = os.path.join(targs.output_dir, "chpt-{}".format(global_step))
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    ckpt = model.module if hasattr(model, "module") else model
                    ckpt.save_pratrained(ckpt_dir)
                    torch.save(targs, os.path.join(ckpt_dir, "training_args.bin"))
                    logging.info("Saving model checkpoint to %s", ckpt_dir)

            if 0 < targs.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < targs.max_steps < global_step:
            train_trange.close()
            break;
    # Train ends    
    # tb_writer.close()

    return global_step, train_loss / global_step

def evaluate(model, targs, eval_dataset, tokenizer, suffix=""):
    dataset, examples, features = load_and_cache_examples(targs, tokenizer, eval=True, output_examples=True) # donnot need to send in eval set as arguments?
    if not os.path.exists(targs.output_dir):
        os.makedirs(targs.output_dir)
    targs.eval_batch_size = targs.per_gpu_eval_batch_size * 1 # no distribution here
    
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, batch_size=targs.eval_batch_size, sampler=eval_sampler)

    # Eval begins
    logger.info("***** Running evaluation {} *****".format(suffix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", targs.eval_batch_size)
    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, decs="Evaluation"):
        model.eval()
        batch = tuple(t.to_device(targs.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_masks": batch[1],
                "token_type_ids": batch[2],
            }???
            feature_ids = batch[3]
            outputs = model(**inputs)
            for i, feature_id in enumerate(feature_ids):
                eval_feature = features[feature_id.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(
                    unique_id=unique_id,
                    start_logits=to_list(outputs[0][i]) ??? no use of start/end logits in QG
                )
                all_results.append(result)

    eval_time = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f secs per example)", eval_time, (eval_time / len(dataset))

    ??? all below should be modified as metrics are not the same
    output_pred_file = os.path.join(targs.output_dir, "preds_{}.json".format(suffix))
    output_tag_pred_file = os.path.join(targs.output_dir, "tag_preds_{}.json".format(suffix))
    output_nbest_file = os.path.join(targs.output_dir, "nbest_preds_{}.json".format(suffix))
    output_result_file = os.path.join(targs.output_dir, "qas_eva;_results_{}".format(suffix))
    output_file = os.path.join(targs.output_dir, "eval_matrix_results_{}".format(suffix))

    write_predictions(
        all_examples=examples,
        all_features=features,
        all_results=all_results,
        n_best_size=targs.num_best,
        max_answer_length=targs.max_answer_length, ?? unset in args
        do_lower_case=targs.do_lower_case, ?? arg not in bart model
        output_prediction_file=output_pred_file,
        output_tag_prediction_file=output_tag_pred_file,
        output_nbest_file=output_nbest_file,
        verbose_logging=targs.verbose_logging ?? unset in args
        )

    evaluate_options = EvalOpts(
        data_file=targs.eval_file,
        root_dir=targs.root_dir,
        pred_file=output_pred_file,
        tag_pred_file=output_tag_pred_file,
        result_file=output_result_file,
        outfile=output_file
    )
    results = evaluate_on_websrc(evaluate_options)
    # Eval ends
    return results

def load_and_cache_examples(args, tokenizer, do_eval=False, output_examples=False):
    file = args.eval_file if do_eval else args.train_file
    cached_feature_file = os.path.join(os.path.dirname(file), "cached_features", "cached_{}_{}_{}_{}".format(
        'eval' if do_eval else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        args.method)
        )
    if not os.path.exists(os.path.dirname(cached_feature_file)):
        os.makedirs(os.path.dirname(cached_feature_file))

    if not 

def parse_args():
    parser = argparse.ArgumentParser()

    def to_run_args(args):
        return args
    def to_model_args(args):
        return args
    def to_train_args(args):
        return args
    # Required args
    # run    
    parser.add_argument("--root_dir", default=None, type=str, required=True,
                        help="The root directory of the raw WebSRC dataset, containing all html files")
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="Json file for training, e.g. train-v1.0.json")
    parser.add_argument("--eval_file", default=None, type=str, required=True,
                        help="Json file for evaluations, e.g. dev-v1.0.json or test-v1.0.json")
    # model
    parser.add_argument("--backbone", default="baseline", type=str, required=True,
                        help="Indicate the backbone model")
    parser.add_argument("--method", default="baseline", type=str, required=True,
                        help="Indicate the method using to deal with features")
    # train
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and the predictions locate")
    
    # Optional args
    # run
    parser.add_argument("--run_train", action='store_true', required=False,
                        help="Run training")
    parser.add_argument("--run_eval", action='store_true', required=False,
                        help="Run eval on dev or test")
    parser.add_argument("--eval_when_train", action='store_true', required=False, 
                        help="Run eval during training at each logging phase")
    parser.add_argument("--eval_all_ckpts", action='store_true', required=False, 
                        help="Eval on all checkpoints with same prefix as model_name_or_path") ???
    parser.add_argument("--ckpts_min", default=0, type=int, required=False, 
                        help="Eval on checkpoints with suffix larger than or equal to it, beside the final one with no suffix")
    parser.add_argument("--ckpts_max", default=None, type=int, required=False, 
                        help="Eval on checkpoints with suffix smaller than it, beside the final one with no suffix")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA")
    # model
    parser.add_argument("--config_name_or_path", default="", type=str, required=False,
                        help="Pretrained config name or path")
    parser.add_argument("--tokenizer_name_or_path", default="", type=str, required=False,
                        help="Pretrained tokenizer name or path")
    parser.add_argument("--download_model_path", default=3000, type=int, required=False,
                        help="Path to store the downloaded pretrained model")
    # training
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, required=False,
                        help="Batch size per GPU/CPU during training")
    parser.add_argument("--per_gpu_eval_batch_size", default=None, type=int, required=False,
                        help="Batch size per GPU/CPU during eval")
    parser.add_argument("--learning_rate", default=1e-5, type=float, required=False,
                        help="The initial learning rate for AdamW")
    parser.add_argument("--adam_eps", default=1e-8, type=float, required=False,
                        help="Epsilon for AdamW")
    parser.add_argument("--weight_decay", default=0.0, type=float, required=False,
                        help="Weight decay on layers if we apply it")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, required=False,
                        help="Max gradient norm, clip it if larger than it")
    parser.add_argument("--num_training_epochs", default=3, type=int, required=False,
                        help="Total epochs during training")
    parser.add_argument("--max_steps", default=-1, type=int, required=False,
                        help="Override num_training_epochs if > 0, set total training steps")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, required=False,
                        help="Number of steps to update gradient by backward passing")
    parser.add_argument("--warmup_steps", default=0, type=int, required=False,
                        help="Linear warmup steps")
    parser.add_argument("--logging_steps", default=3000, type=int, required=False,
                        help="Log every X training steps")
    parser.add_argument("--saving_steps", default=3000, type=int, required=False,
                        help="Save checkpoint every X training steps")
    parser.add_argument("--num_nbest", default=20, type=int, required=False,
                        help="Number of n best predictions to generate in the nbest_preds.json file")
    parser.add_argument("--seed", default=42, type=int, required=False,
                        help="Random seed for initialization")
    parser.add_argument("--overwrite_output_dir", action='store_true',
                        help="Overwrite the content of the output directory (if exists)")
    parser.add_argument("--overwrite_cache", action='store_true',
                        help="Overwrite the cached training and evaluation datasets")

    args = parser.parse_args()
    rargs = to_run_args(args)
    margs = to_model_args(args)
    targs = to_train_args(args)
    return rargs, margs, targs

if __name__ == "__main__":
    rargs, margs, targs = parse_args()
    bart_config = BartConfig(...)
    bart = 