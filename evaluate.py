import json
import argparse
import torch
import difflib

from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        choices=['cp', 'ss'],
                        help='Path to evaluation dataset.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to result text file')
    parser.add_argument('--model', type=str, required=True,
                        choices=['bert', 'roberta', 'albert'])
    parser.add_argument('--method', type=str, required=True,
                        choices=['aula', 'aul', 'cps', 'sss'])
    args = parser.parse_args()

    return args


def load_tokenizer_and_model(args):
    '''
    Load tokenizer and model to evaluate.
    '''
    if args.model == 'bert':
        pretrained_weights = 'bert-base-cased'
    elif args.model == "roberta":
        pretrained_weights = 'roberta-large'
    elif args.model == "albert":
        pretrained_weights = 'albert-large-v2'
    model = AutoModelForMaskedLM.from_pretrained(pretrained_weights,
                                                 output_hidden_states=True,
                                                 output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

    model = model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    return tokenizer, model


def get_span(seq1, seq2, operation):
    '''
    Extract spans that are shared or diffirent between two sequences.

    Parameters
    ----------
    operation: str
        You can select "equal" which extract spans that are shared between
        two sequences or "diff" which extract spans that are diffirent between
        two sequences.
    '''
    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        if (operation == 'equal' and op[0] == 'equal') \
                or (operation == 'diff' and op[0] != 'equal'):
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2


def get_rank_for_gold_token(log_probs, token_ids):
    '''
    Get rank for gold token from log probability.
    '''
    sorted_indexes = torch.sort(log_probs, dim=1, descending=True)[1]
    ranks = torch.where(sorted_indexes == token_ids)[1] + 1
    ranks = ranks.tolist()

    return ranks


def calculate_aul(model, token_ids, log_softmax, attention):
    '''
    Given token ids of a sequence, return the averaged log probability of
    unmasked sequence (AULA or AUL).
    '''
    output = model(token_ids)
    logits = output.logits.squeeze(0)
    log_probs = log_softmax(logits)
    token_ids = token_ids.view(-1, 1).detach()
    token_log_probs = log_probs.gather(1, token_ids)[1:-1]
    if attention:
        attentions = torch.mean(torch.cat(output.attentions, 0), 0)
        averaged_attentions = torch.mean(attentions, 0)
        averaged_token_attentions = torch.mean(averaged_attentions, 0)
        token_log_probs = token_log_probs.squeeze(1) * averaged_token_attentions[1:-1]
    sentence_log_prob = torch.mean(token_log_probs)
    score = sentence_log_prob.item()

    ranks = get_rank_for_gold_token(log_probs, token_ids)

    return score, ranks


def calculate_cps(model, token_ids, spans, mask_id, log_softmax):
    '''
    Given token ids of a sequence, return the summed log probability of
    masked shared tokens between sequence pair (CPS).
    '''
    spans = spans[1:-1]
    masked_token_ids = token_ids.repeat(len(spans), 1)
    masked_token_ids[range(masked_token_ids.size(0)), spans] = mask_id
    hidden_states = model(masked_token_ids)
    hidden_states = hidden_states[0]
    token_ids = token_ids.view(-1)[spans]
    log_probs = log_softmax(hidden_states[range(hidden_states.size(0)), spans, :])
    span_log_probs = log_probs[range(hidden_states.size(0)), token_ids]
    score = torch.sum(span_log_probs).item()

    ranks = get_rank_for_gold_token(log_probs, token_ids.view(-1, 1))

    return score, ranks


def calculate_sss(model, token_ids, spans, mask_id, log_softmax):
    '''
    Given token ids of a sequence, return the averaged log probability of
    masked diffirent tokens between sequence pair (SSS).
    '''
    masked_token_ids = token_ids.clone()
    masked_token_ids[:, spans] = mask_id
    hidden_states = model(masked_token_ids)
    hidden_states = hidden_states[0].squeeze(0)
    token_ids = token_ids.view(-1)[spans]
    log_probs = log_softmax(hidden_states)[spans]
    span_log_probs = log_probs[:,token_ids]
    score = torch.mean(span_log_probs).item()

    if log_probs.size(0) != 0:
        ranks = get_rank_for_gold_token(log_probs, token_ids.view(-1, 1))
    else:
        ranks = [-1]

    return score, ranks


def main(args):
    '''
    Evaluate the bias in masked language models.
    '''
    tokenizer, model = load_tokenizer_and_model(args)
    total_score = 0
    stereo_score = 0

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    mask_id = tokenizer.mask_token_id
    log_softmax = torch.nn.LogSoftmax(dim=1)
    vocab = tokenizer.get_vocab()
    count = defaultdict(int)
    scores = defaultdict(int)
    all_ranks = []
    data = []

    with open(f'data/paralled_{args.data}.json') as f:
        inputs = json.load(f)
        total_num = len(inputs)
        for input in tqdm(inputs):
            bias_type = input['bias_type']
            count[bias_type] += 1

            pro_sentence = input['stereotype']
            pro_token_ids = tokenizer.encode(pro_sentence, return_tensors='pt')
            anti_sentence = input['anti-stereotype']
            anti_token_ids = tokenizer.encode(anti_sentence, return_tensors='pt')

            with torch.no_grad():
                if args.method == 'aula' or args.method == 'aul':
                    attention = True if args.method == 'aula' else False
                    pro_score, pro_ranks = calculate_aul(model, pro_token_ids, log_softmax, attention)
                    anti_score, anti_ranks = calculate_aul(model, anti_token_ids, log_softmax, attention)
                elif args.method == 'cps':
                    pro_spans, anti_spans = get_span(pro_token_ids[0],
                                                     anti_token_ids[0], 'equal')
                    pro_score, pro_ranks = calculate_cps(model, pro_token_ids, pro_spans,
                                              mask_id, log_softmax)
                    anti_score, anti_ranks = calculate_cps(model, anti_token_ids, anti_spans,
                                               mask_id, log_softmax)
                    pro_score = round(pro_score, 3)
                    anti_score = round(anti_score, 3)
                    data.append([anti_sentence, pro_sentence, anti_score, pro_score])
                elif args.method == 'sss':
                    pro_spans, anti_spans = get_span(pro_token_ids[0],
                                                     anti_token_ids[0], 'diff')
                    pro_score, anti_ranks = calculate_sss(model, pro_token_ids, pro_spans,
                                              mask_id, log_softmax)
                    anti_score, pro_ranks = calculate_sss(model, anti_token_ids, anti_spans,
                                               mask_id, log_softmax)

            all_ranks += anti_ranks
            all_ranks += pro_ranks
            total_score += 1
            if pro_score > anti_score:
                stereo_score += 1
                scores[bias_type] += 1

    fw = open(args.output, 'w')
    bias_score = round((stereo_score / total_score) * 100, 2)
    print('Bias score:', bias_score)
    fw.write(f'Bias score: {bias_score}\n')
    for bias_type, score in sorted(scores.items()):
        bias_score = round((score / count[bias_type]) * 100, 2)
        print(bias_type, bias_score)
        fw.write(f'{bias_type}: {bias_score}\n')
    all_ranks = [rank for rank in all_ranks if rank != -1]
    accuracy = sum([1 for rank in all_ranks if rank == 1]) / len(all_ranks)
    accuracy *= 100
    print(f'Accuracy: {accuracy:.2f}')
    fw.write(f'Accuracy: {accuracy:.2f}\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)
