#!/mnt/xlancefs/home/yjl00/miniconda3/envs/tie/bin/python

import os
import argparse
import random
import logging
import timeit
import math
import glob

from tqdm import tqdm, trange
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import WEIGHTS_NAME, AutoConfig, AutoTokenizer, PretrainedConfig, AdamW, get_linear_schedule_with_warmup, AutoModelForSeq2SeqLM

from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.tokenization_bart import BartTokenizer
from transformers.models.bart.modeling_bart import BartPretrainedModel, BartModel, BartForConditionalGeneration

from utils import read_wrc_examples_to_qa as read_wrc_examples, convert_examples_to_features, RawResult, write_predictions
from utils_eval import EvalOpts, eval_on_websrc 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

def set_seed(seed, n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

class TextEvalMetrics:
    """
    Implementation of bleu, rouge, kl divergence, ...
    """
    def __init__(self, candidate_sentence, reference_sentence_list):
        self.candidate = candidate_sentence
        self.reference_list = reference_sentence_list

    def calculate_bleu(self, n_gram=4, fn_word_piece_tokenizer=lambda s: s.split()):
        def to_ngram_pieces(sentence, n_gram):
            single_word_pieces = fn_word_piece_tokenizer(sentence)
            ngram_word_pieces = [" ".split(single_word_pieces[i:i+n_gram]) 
                                for i in range(len(single_word_pieces)-n_gram+1)]
            return ngram_word_pieces
        def calculate_penalty(candidate, reference_list):
            lc, lr = len(candidate), min(map(len, reference_list))
            return 1 if lc > lr else math.exp(1-lr/lc)
        def n_gram_weight(n_gram):
            return 1 / n_gram

        penalty_factor = calculate_penalty(self.candidate, self.reference_list)
        bleu_scores, overall_bleu_score = [], 0
        for ref in self.reference_list:
            bleu_score = penalty_factor * math.exp()
            bleu_scores.append(bleu_score)
        
        return bleu_scores, overall_bleu_score

    def calculate_rouge_l(self):
        pass

    def calculate_meteor(self):
        pass

    def calculate_all(self):
        all_results = {
            "bleu-1": self.calculate_bleu(n_gram=1),
            "bleu-2": self.calculate_bleu(n_gram=2),
            "bleu-3": self.calculate_bleu(n_gram=3),
            "bleu-4": self.calculate_bleu(n_gram=4),
            "rouge-l": self.calculate_rouge_l(),
            "metoer": self.meteor(),
        }
        return all_results

class StructDataset(Dataset):
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

def get_pretrained_model_and_tokenizer(args):
    """
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        ...
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    """
    # attention. default config_name and tknzer_name is not `None` but `""`
    config_name_or_path = args.config_name_or_path if args.config_name_or_path else args.pretrained_model_name_or_path
    tokenizer_name_or_path = args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.pretrained_model_name_or_path

    config = BartConfig.from_pretrained(config_name_or_path, cache_dir=args.download_model_path)
    tokenizer = BartTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=args.download_model_path)
    model = BartForConditionalGeneration.from_pretrained(args.pretrained_model_name_or_path,config=config, cache_dir=args.download_model_path)
    return model, tokenizer

def generation_demo(model, tokenizer, examples):
    texts = [e.doc_tokens for e in examples]
    inputs = tokenizer(texts, max_length=1024, return_tensors="pt")
    generation_ids = model.generate(inputs["input_ids"], num_beams=5, min_length=0, max_length=32)
    qg_pairs = {}
    for g_id, text in zip(generation_ids, texts):
        question = tokenizer.decode(g_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        qg_pairs.update({text: question})
    return qg_pairs

def calculate_metrics():
    pass

def train(model, targs, tokenizer):
    # tb_writer = SummaryWriter # from tensorboardX import SummaryWriter

    train_dataset, examples, features = load_and_cache_examples(targs, tokenizer, run_eval=False)

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
    optimizer = AdamW(optimizer_grouped_parameters, lr=targs.learning_rate, eps=targs.adam_epsilon)
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
    train_trange = trange(int(targs.num_training_epochs), desc="Epoch")
    set_seed(targs.seed)
    for _ in train_trange:
        epoch_iterator = tqdm(train_dataloader, desc="Iter ")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # for t in batch:
            #     print(t.shape)
            #     print(t)
            #     exit(0)
            batch = tuple(t.to(targs.device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }
            # send in `labels` to calculate `loss`.
            # some output dimensions:
            #   - logits = lm_logits, shape = (batch_size, seq_len, config.vocab_size)
            #   - loss (1,) = loss_fct(CrossEntropy) 
            #       (lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            #       -> shape1 = (batch_size * seq_len, vocab_size), -> shape2 = (batch_size, seq_len)
            # labels are ids of tokens, either be in [0, vocab_size] or -100 if masked
            outputs = model(**inputs)
            # print(type(outputs))
            # for k, v in outputs.items():
            #     print(k, type(v))
            loss, logits = outputs['loss'], outputs['logits']

            if targs.n_gpu > 1:
                loss = loss.mean()
            if targs.gradient_accumulation_steps > 1:
                loss = loss / targs.gradient_accumulation_steps
            # print(type(loss), type(logits))
            # print(loss,)

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

def evaluate(model, targs, tokenizer, suffix=""): # read dataset in fn?? wtf
    # should consists of the stages: init -> eval -> metrics -> demonstrate
    dataset, examples, features = load_and_cache_examples(targs, tokenizer, run_eval=True) # donnot need to send in eval set as arguments?
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

    for batch in tqdm(eval_dataloader, desc="Evaluation"):
        model.eval()
        batch = tuple(t.to(targs.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_masks": batch[1],
            }
            input_ids = batch[0]
            labels, feature_ids = batch[2], batch[3]
            input_texts = [tokenizer.decode(i_id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for i_id in input_ids]
            label_texts = [tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=False) for label in labels]
            generation_ids = model.generate(inputs["input_ids"], num_beams=5, min_length=0, max_length=args.max_question_length)
            qg_pairs = {}
            for i, f_id in enumerate(feature_ids):
                g_id = generation_ids[i]
                eval_feature = features[f_id.item()]
                unique_id = int(eval_feature.unique_id)
                question = tokenizer.decode(g_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                print(input_texts[i])
                print(label_texts[i])
                print(question) # all the same???
                print("test why model.generate gives the same ids. ")
                qg_pairs.update({unique_id: question})

                # deal more with u_id to text etc...

    eval_time = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f secs per example)", eval_time, (eval_time / len(dataset)))

    demo_output_file = os.path.join(targs.output_dir, "_demo_{}".format(suffix))

    # write_predictions(
    #     all_examples=examples,
    #     all_features=features,
    #     all_results=all_results,
    #     n_best_size=targs.num_best,
    #     max_answer_length=targs.max_answer_length, ?? unset in args
    #     do_lower_case=targs.do_lower_case, ?? arg not in bart model
    #     output_prediction_file=output_pred_file,
    #     output_tag_prediction_file=output_tag_pred_file,
    #     output_nbest_file=output_nbest_file,
    #     verbose_logging=targs.verbose_logging ?? unset in args
    #     )

    # evaluate_options = EvalOpts(
    #     data_file=targs.eval_file,
    #     root_dir=targs.root_dir,
    #     pred_file=output_pred_file,
    #     tag_pred_file=output_tag_pred_file,
    #     result_file=output_result_file,
    #     outfile=output_file
    # )
    # results = eval_on_websrc(evaluate_options)
    # Eval ends
    return results

def load_and_cache_examples(args, tokenizer, run_eval=False):
    """
    Support args.method = ["text", "text-html"]
    Always return a triplet (dataset[StructDataset], examples([list(SRCExample)], features([InputFeatures]))
    """
    output_examples = True
    args.method = args.method.lower()
    file = args.eval_file if run_eval else args.train_file
    cached_feature_file = "cached_features_{}_{}_{}_{}".format(
        'eval' if run_eval else 'train',
        list(filter(None, args.pretrained_model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        args.method
        )
    cached_features_dir = os.path.join(os.path.dirname(file), "cached_features")
    if not os.path.exists(cached_features_dir):
        os.makedirs(cached_features_dir)

    cached_feature_file = os.path.join(cached_features_dir, cached_feature_file)

    if os.path.exists(cached_feature_file) and not args.overwrite_cache:
        logger.info("Loading features from cached files %s", cached_feature_file)
        features = torch.load(cached_feature_file)
        if output_examples or (args.method != "text" and not run_eval):
            examples, tag_list = read_wrc_examples(
                input_file=file,
                root_dir=args.root_dir,
                is_training=(not run_eval),
                tokenizer=tokenizer,
                method="T-PLM" if args.method=="text" else "H-PLM", # modify later
                simplify=False # why True at first?
            )
            if not run_eval and args.method != "text":
                tag_list = list(tag_list).sort()
                tokenizer.add_tokens(tag_list)
        else:
            examples = None
    else:
        logger.info("Caching features from dataset at %s", file)

        if args.method != "text" and not run_eval:
            examples, tag_list = read_wrc_examples(
                input_file=file,
                root_dir=args.root_dir,
                is_training=(not run_eval),
                tokenizer=tokenizer,
                method="T-PLM" if args.method=="text" else "H-PLM", # modify later
                simplify=False # why True at first?
            )
            tag_list = list(tag_list).sort()
            tokenizer.add_tokens(tag_list)

        examples, _ = read_wrc_examples(
            input_file=file,
            root_dir=args.root_dir,
            is_training=(not run_eval),
            tokenizer=tokenizer,
            method=args.method,
            simplify=False
            )
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_question_length, # why query? modify later
            is_training=(not run_eval)
        )
        if args.save_features:
            logger.info("Saving features into cached file %s", cached_feature_file)
            torch.save(features, cached_feature_file)

    assert examples is not None and features is not None, "else it's so stupid. bart.py line 439"
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_page_ids, all_token_to_tag = None, None

    # here, extractive qa like websrc baseline dont need text context (only the ids of the input seq)
    # start and end position are contained in `dataset`
    # while generative tasks need context to construct
    for f in features:
        print(f"tokens: {f.tokens}")
        print(f"ex_id: {f.example_index}")
        print(f"ex: {examples[f.example_index]}")
        break

    all_questions_input_tokens = [examples[f.example_index].question_text for f in features]
    print(f"type of all_q_i_tokens: {type(all_questions_input_tokens)}")
    print(f"type of 0th all_q_i_tokens: {type(all_questions_input_tokens[0])}")
    print(f"content of 0th all_q_i_tokens: {all_questions_input_tokens[0]}")
    # all_questions_input_ids = torch.tensor(
    #     [tokenizer(q_tokens, max_length=args.max_question_length, return_tensors="pt") for q_tokens in all_questions_input_tokens], 
    #     dtype=torch.long
    #     )
    all_questions_input_ids = torch.tensor(
        tokenizer(all_questions_input_tokens, padding="max_length", max_length=args.max_question_length, return_tensors="pt")["input_ids"], 
        dtype=torch.long)
    all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = StructDataset(
        all_input_ids, all_input_mask, all_questions_input_ids, all_feature_index, 
        page_ids=all_page_ids, token_to_tag=all_token_to_tag
    )

    return (dataset, examples, features)

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
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
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
                        help="Eval on all checkpoints with same prefix as pretrained_model_name_or_path") # what prefix
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
    parser.add_argument("--download_model_path", default=None, type=str, required=False,
                        help="Path to store the downloaded pretrained model")
    parser.add_argument("--max_seq_length", default=384, type=int, required=False,
                        help="The maximum total input sequence length after WordPiece tokenization. "
                                "Sequences longer than this value will be truncated, "
                                "and shorter ones will be padded")
    parser.add_argument("--doc_stride", default=128, type=int, required=False,
                        help="The stride distance to split up a long document into chunks")
    parser.add_argument("--max_question_length", default=64, type=int, required=False,
                        help="The maximum length of tokens for the question. Longer ones will be truncated to this value")
    # training
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, required=False,
                        help="Batch size per GPU/CPU during training")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, required=False,
                        help="Batch size per GPU/CPU during eval")
    parser.add_argument("--learning_rate", default=1e-5, type=float, required=False,
                        help="The initial learning rate for AdamW")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, required=False,
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
    parser.add_argument("--num_warmup_steps", default=0, type=int, required=False,
                        help="Linear warmup steps")
    parser.add_argument("--logging_steps", default=3000, type=int, required=False,
                        help="Log every X training steps")
    parser.add_argument("--saving_steps", default=3000, type=int, required=False,
                        help="Save checkpoint every X training steps")
    parser.add_argument("--num_nbest", default=20, type=int, required=False,
                        help="Number of n best predictions to generate in the nbest_preds.json file")
    parser.add_argument("--seed", default=42, type=int, required=False,
                        help="Random seed for initialization")
    parser.add_argument("--save_features", default=True, type=bool, 
                        help="Whether to save the cached feature files, default True")
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
    args = targs

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.run_train and not args.overwrite_output_dir:
        raise ValueError("Output directory {args.output_dir} already exists and non-empty. Try --overwrite_output_dir option to overwrite.")

    # set cuda without distributed training
    device = torch.device("cuda" if torch.cuda.is_available and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    args.device = device

    # setup logging for distributed training
    pass

    # set seed
    set_seed(args.seed, args.n_gpu)

    # set and load model
    model, tokenizer = get_pretrained_model_and_tokenizer(args)

    logging.info("Training parameters: %s", targs)

    # fp16
    pass

    # Training
    if args.run_train:
        # train_dataset = load_and_cache_examples(args, tokenizer, run_eval=False, output_examples=False)
        tokenizer.save_pretrained(args.output_dir)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        # global_step, tr_avg_loss = train(model, args, train_dataset, tokenizer)
        global_step, tr_avg_loss = train(model, args, tokenizer)

        logger.info(f" global step = {global_step}, average loss = {tr_avg_loss}")

        # save trained model and tokenizer
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info(f"Saving models checkpoint to {args.output_dir}")

        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save = model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.run_eval:
        ckpts = [args.output_dir]
        if args.eval_all_ckpts:
            ckpts = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)

        logger.info("Evaluate the following checkpoints: %s", ckpts)

        for ckpt in ckpts:
            global_step = ckpt.split("-")[-1] if len(ckpts) > 1 else ""
            try:
                int(global_step)
            except ValueError:
                global_step = ""
            if global_step and int(global_step) < args.ckpts_min:
                continue
            if global_step and args.ckpts_max is not None and int(global_step) >= args.ckpts_max:
                continue

            model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
            model = model.to(args.device)

            result = evaluate(model, args, tokenizer, suffix=global_step)
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

        logger.info(f"Results: {results}")