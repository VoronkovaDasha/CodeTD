# Parts of this file (generation loop, stopping-criteria setup and decoding of
# the generated code) are adapted from bigcode-evaluation-harness
# (https://github.com/bigcode-project/bigcode-evaluation-harness),
# licensed under the Apache License, Version 2.0. A copy of the license is
# in LICENSES/Apache-2.0.txt (see also http://www.apache.org/licenses/LICENSE-2.0).
#
# Modifications from the original:
#   - Generation for one prompt per call.
#   - Stopping criteria: the EOS token is added to the task's stop words once.
#   - Extraction of token-level features during generation: token
#     probabilities, merged attention maps, attention-diagonal sums over the
#     prompt and the response, and MTD_0 / MTD_1 features of the attention
#     graphs.
#   - Per-sequence feature window (prompt + generated code up to the
#     end of the post-processed output).
#   - Output: one JSON record per generated sequence
#     and one .npy file per feature and sequence.

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    set_seed
)
import torch
import argparse
from utils import HumanEvalTask, MBPPTask, EndOfFunctionCriteria
from tqdm import tqdm
import os
import copy
import json
from multiprocessing import Pool
import numpy as np
from gph import ripser_parallel


def get_gen_len(generated_tokens, gen_code_processed, tokenizer):
    for k in range(len(generated_tokens)):
        tmp = tokenizer.decode(
            generated_tokens[:k],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        if len(tmp) >= len(gen_code_processed):
            return k
    return len(generated_tokens)


def get_scores(prompt_len, outputs):
    """Probability of each generated token under the model's raw logits
    Prompt positions are 1.0."""
    scores = np.ones(outputs.sequences.shape)

    for pos in range(len(outputs.logits)):
        sm = torch.nn.Softmax(dim=1)(outputs.logits[pos])

        for seq in range(outputs.sequences.shape[0]):
            token = outputs.sequences[seq][prompt_len + pos]
            scores[seq, prompt_len + pos] = sm[seq][token].item()
    return scores


def merge_attentions(attn):
    n_layers = len(attn[0])
    n_seq, n_heads = attn[0][0].shape[:2]
    seq_len = attn[-1][0].shape[-1]
    total_attn = np.zeros((n_seq, n_layers, n_heads, seq_len, seq_len), dtype=np.float64)
    for step in attn:
        step_attn = torch.stack(step, dim=1).float().cpu().numpy()
        q_len, k_len = step_attn.shape[-2:]
        total_attn[..., k_len - q_len:k_len, :k_len] = step_attn
    return total_attn


def compute_mtd_matrix(attn_tensor, prompt_len, kind):
    # Section 6. Method (Algorithm) in the Paper.
    # attn tensor of shape (n_layers, n_heads, n_tokens, n_tokens)
    # kind = 0: prompt to 0
    # kind = 1: response to 0
    n_layers = attn_tensor.shape[0]
    n_heads = attn_tensor.shape[1]
    mtd0_mat = np.zeros((n_layers, n_heads))
    mtd1_mat = np.zeros((n_layers, n_heads))

    for i in range(n_layers):
        for j in range(n_heads):
            A = attn_tensor[i, j]
            # Construction of fully-connected undirected weighted graph
            D = 1 - (A + A.T)
            if kind == 0:
                # Block corresponding to the prompt is zeroed.
                D[:prompt_len, :prompt_len] = 0.
            elif kind == 1:
                # Block corresponding to the generation is zeroed.
                D[prompt_len:, prompt_len:] = 0.
            np.fill_diagonal(D, 0)

            r = ripser_parallel(D, metric='precomputed', n_threads=1)
            dgm0 = r['dgms'][0]
            dgm1 = r['dgms'][1]
            # MTD_0: sum of the finite H0 death times (= total weight of the
            # minimum spanning tree of D). All H0 births are 0 because the
            # diagonal of D is 0, so summing every finite entry of the diagram
            # (births and deaths) equals summing the deaths.
            assert np.all(dgm0[:, 0] == 0), "H0 births are expected to be 0"
            mtd0 = np.sum(dgm0[dgm0 < np.inf])
            # MTD_1: sum of the H1 lifespans: deaths - births.
            if dgm1.shape[0]:
                mtd1 = np.sum(dgm1[:, 1] - dgm1[:, 0])
            else:
                mtd1 = 0

            mtd0_mat[i, j] = mtd0
            mtd1_mat[i, j] = mtd1

    return mtd0_mat, mtd1_mat


def get_stopping_criteria(task, tokenizer):
    if task.stop_words and tokenizer.eos_token and tokenizer.eos_token not in task.stop_words:
        task.stop_words.append(tokenizer.eos_token)
    if hasattr(task, "check_fn"):
        return [EndOfFunctionCriteria(0, task.stop_words, tokenizer, task.check_fn)]
    if task.stop_words:
        return [EndOfFunctionCriteria(0, task.stop_words, tokenizer)]
    return []


def file_task_id(task_id, task_name):
    return task_id.replace('/', '_') if task_name == "humaneval" else task_id


def generate_task(model, tokenizer, task, dataset, sample_num, args, pool=None):
    if args.task == "humaneval":
        gen_kwargs = {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_length": args.max_length_generation,
            "max_new_tokens": None,
        }
    elif args.task == "mbpp":
        gen_kwargs = {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_length": None,
            "max_new_tokens": args.max_length_generation,
        }
    stopping_criteria = get_stopping_criteria(task, tokenizer)
    if stopping_criteria:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(stopping_criteria)
    assert tokenizer.eos_token in task.stop_words

    prompt = task.prompt_for_idx(sample_num)
    if args.task == "humaneval":
        tokenized_prompt = tokenizer(
            prompt,
            truncation=True,
            return_tensors="pt",
            max_length=args.max_length_generation
        )
    elif args.task == "mbpp":
        tokenized_prompt = tokenizer(prompt, truncation=True, return_tensors="pt")
    with torch.no_grad():
        prompt_len = tokenized_prompt['input_ids'].shape[-1]
        gen_kwargs["stopping_criteria"][0].start_length = prompt_len
        inputs = tokenized_prompt['input_ids'].to(model.device)
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=tokenized_prompt['attention_mask'].to(model.device),
            num_return_sequences=args.batch_size,
            output_logits=args.compute_features,
            output_attentions=args.compute_features,
            return_dict_in_generate=args.compute_features,
            **gen_kwargs,
        )

    generated_examples = []
    generated_tokens = outputs['sequences'].cpu().numpy() if args.compute_features else outputs.cpu().numpy()
    for i, s in enumerate(generated_tokens):
        if s[0] == tokenizer.bos_token_id:
            s = s[1:]
        gen_code = tokenizer.decode(s, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # The prompt is removed by character count in postprocess_generation /
        # get_valid_output; this holds when decoding reproduces the prompt exactly
        # (verified for all models in the paper).
        if not gen_code.startswith(prompt):
            print(f"Warning: decoded text does not start with the prompt "
                  f"(task {sample_num}, sequence {i}); generation may be misaligned.")
        gen_code_processed = task.postprocess_generation(gen_code, sample_num)

        new_example = copy.deepcopy(dataset[sample_num])
        new_example['prompt'] = prompt
        new_example['output'] = tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
        new_example['prompt_len'] = prompt_len
        new_example['generation'] = gen_code_processed
        if args.task == "humaneval":
            new_example['gen_len'] = get_gen_len(generated_tokens[i], gen_code_processed, tokenizer)
        elif args.task == "mbpp":
            valid_code = task.get_valid_output(gen_code, sample_num)
            new_example['valid_code'] = valid_code
            new_example['gen_len'] = get_gen_len(generated_tokens[i], valid_code, tokenizer)
        generated_examples.append(new_example)

    if args.compute_features:
        scores = get_scores(prompt_len, outputs)
        merged_attn = merge_attentions(outputs['attentions'])

        mtd0_mat_kind0_list = []
        mtd1_mat_kind0_list = []
        mtd0_mat_kind1_list = []
        mtd1_mat_kind1_list = []
        diag_prompt = []
        diag_response = []
        for i in range(args.batch_size):
            gen_len = generated_examples[i]['gen_len']
            # merged_attn has S - 1 positions: the last sampled token is never fed
            # back to the model, so it has no attention row. If gen_len == S, the
            # slice below is clipped to S - 1 positions.
            # Alternatively, make a separate forward pass over the full generation
            # to compute the full matrix but at extra cost.
            attn_i = merged_attn[i, :, :, :gen_len, :gen_len]

            # MTD features; layers are split across worker processes.
            splitted_attn = np.array_split(attn_i, args.num_of_workers)
            for kind, mtd0_list, mtd1_list in (
                (0, mtd0_mat_kind0_list, mtd1_mat_kind0_list),
                (1, mtd0_mat_kind1_list, mtd1_mat_kind1_list),
            ):
                parts = pool.starmap(compute_mtd_matrix, [(a, prompt_len, kind) for a in splitted_attn])
                mtd0_list.append(np.vstack([m[0] for m in parts]))
                mtd1_list.append(np.vstack([m[1] for m in parts]))

            # Sum of attention-diagonal entries over prompt / response positions,
            # one value per (layer, head), flattened layerwise.
            diag = np.diagonal(attn_i, axis1=-2, axis2=-1)          # (n_layers, n_heads, gen_len)
            diag_prompt.append(np.ascontiguousarray(diag[..., :prompt_len]).sum(-1).ravel())
            diag_response.append(np.ascontiguousarray(diag[..., prompt_len:]).sum(-1).ravel())

        del outputs
        torch.cuda.empty_cache()

        return (
            generated_examples, scores, diag_prompt, diag_response,
             mtd0_mat_kind0_list, mtd1_mat_kind0_list,
              mtd0_mat_kind1_list, mtd1_mat_kind1_list,
        )

    del outputs
    torch.cuda.empty_cache()

    return generated_examples


def generate_main(args):
    dict_precisions = {"fp16": torch.float16, "bf16": torch.bfloat16,}
    model_kwargs = {}
    print(f"Loading model in {args.precision}")
    model_kwargs["torch_dtype"] = dict_precisions[args.precision]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.compute_features:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    # Use EOS as the padding token.
    tokenizer.pad_token = tokenizer.eos_token
    new_line_arg = 'qwen' in args.model.lower()
    print('New line: ', new_line_arg)
    if args.task == "humaneval":
        task = HumanEvalTask(new_line_arg)
    elif args.task == "mbpp":
        task = MBPPTask(new_line_arg, args.two_shot)
    dataset = task.get_dataset()
    n_tasks = len(dataset)

    gen_dir = os.path.dirname(args.gen_path)
    if gen_dir:
        os.makedirs(gen_dir, exist_ok=True)

    if args.compute_features:
        pool = Pool(args.num_of_workers)
        os.makedirs(args.scores_path, exist_ok=True)
        os.makedirs(args.diag_prompt_path, exist_ok=True)
        os.makedirs(args.diag_response_path, exist_ok=True)
        os.makedirs(args.mtd0_kind0_path, exist_ok=True)
        os.makedirs(args.mtd1_kind0_path, exist_ok=True)
        os.makedirs(args.mtd0_kind1_path, exist_ok=True)
        os.makedirs(args.mtd1_kind1_path, exist_ok=True)

    for sample_num in tqdm(range(n_tasks), desc='Generating'):
        for i in range(args.start_batch, args.start_batch + args.n_batches):
            set_seed(i)
            if args.compute_features:
                (
                    gen_examples, scores, diag_prompt, diag_response,
                    mtd0_mat_kind0_list, mtd1_mat_kind0_list,
                    mtd0_mat_kind1_list, mtd1_mat_kind1_list,
                ) = generate_task(model, tokenizer, task, dataset, sample_num, args, pool)

                for j in range(args.batch_size):
                    gen_num = i * args.batch_size + j
                    task_id = file_task_id(gen_examples[j]['task_id'], args.task)
                    filename = 'task_id_{}_seed_{}_gen_num_{}'.format(task_id, i, gen_num)

                    # save scores
                    np.save(os.path.join(args.scores_path, filename), scores[j])
                    # save diag_prompt
                    np.save(os.path.join(args.diag_prompt_path, filename), diag_prompt[j])
                    # save diag_response
                    np.save(os.path.join(args.diag_response_path, filename), diag_response[j])

                    # save mtd0_kind0
                    np.save(os.path.join(args.mtd0_kind0_path, filename), mtd0_mat_kind0_list[j])
                    # save mtd1_kind0
                    np.save(os.path.join(args.mtd1_kind0_path, filename), mtd1_mat_kind0_list[j])

                    # save mtd0_kind1
                    np.save(os.path.join(args.mtd0_kind1_path, filename), mtd0_mat_kind1_list[j])
                    # save mtd1_kind1
                    np.save(os.path.join(args.mtd1_kind1_path, filename), mtd1_mat_kind1_list[j])

            else:
                gen_examples = generate_task(model, tokenizer, task, dataset, sample_num, args)

            for j, gen_ex in enumerate(gen_examples):
                gen_num = i * args.batch_size + j
                gen_examples[j]['gen_num'] = gen_num
                gen_examples[j]['seed'] = i

            # save gen_examples
            with open(args.gen_path, 'a+', encoding='utf-8') as fw:
                for ex in gen_examples:
                    fw.write(json.dumps(ex) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_sample', action="store_true")
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.95)

    parser.add_argument('--model', required=True, type=str, help="model name or path")
    parser.add_argument('--task', type=str, default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument('--max_length_generation', type=int, default=512)
    parser.add_argument('--precision', type=str, default='fp16', choices=["fp16", "bf16"])

    parser.add_argument('--compute_features', action="store_true")
    parser.add_argument('--gen_path', required=True, type=str, help="output path of your generation")
    parser.add_argument('--scores_path', type=str, help="token probabilities (softmax of raw logits)")
    parser.add_argument('--diag_prompt_path', type=str, help="output path of generation diag prompt sum")
    parser.add_argument('--diag_response_path', type=str, help="output path of generation diag response sum")
    parser.add_argument('--mtd0_kind0_path', type=str, help="output path of generation mtd0 kind 0")
    parser.add_argument('--mtd1_kind0_path', type=str, help="output path of generation mtd1 kind 0")
    parser.add_argument('--mtd0_kind1_path', type=str, help="output path of generation mtd0 kind 1")
    parser.add_argument('--mtd1_kind1_path', type=str, help="output path of generation mtd1 kind 1")

    parser.add_argument('--batch_size', default=1, type=int, help="number of sequences to generate")
    parser.add_argument('--n_batches', default=1, type=int, help="number of batches per task")
    parser.add_argument('--start_batch', default=0, type=int, help="number of start batch")
    parser.add_argument('--num_of_workers', type=int, help="number of workers")
    parser.add_argument('--two_shot', action="store_true", help="two-shot for MBPP")
    args = parser.parse_args()

    if not args.do_sample and args.batch_size > 1:
        parser.error("Greedy decoding (no --do_sample) requires --batch_size 1")

    feature_args = ["scores_path", "diag_prompt_path", "diag_response_path",
                    "mtd0_kind0_path", "mtd1_kind0_path", "mtd0_kind1_path",
                    "mtd1_kind1_path", "num_of_workers"]
    if args.compute_features:
        missing = [a for a in feature_args if getattr(args, a) is None]
        if missing:
            parser.error("--compute_features requires: " + ", ".join("--" + a for a in missing))

    print(args)

    generate_main(args)
