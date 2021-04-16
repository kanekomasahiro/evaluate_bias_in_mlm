import json
import argparse
import csv
import ast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    return args


def preprocess_crows_pairs():
    '''
    Extract stereotypical and anti-stereotypical sentences from crows-paris.
    '''
    data = []

    with open('data/cp.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            example = {}
            direction = row['stereo_antistereo']
            example['direction'] = direction
            example['bias_type'] = row['bias_type']

            example['stereotype'] = row['sent_more']
            example['anti-stereotype'] = row['sent_less']
            data.append(example)

    return data


def preprocess_stereoset():
    '''
    Extract stereotypical and anti-stereotypical sentences from StereoSet.
    '''
    data = []
    data = []

    with open('data/ss.json') as f:
        input = json.load(f)
        for annotations in input['data']['intrasentence']:
            example = {}
            example['bias_type'] = annotations['bias_type']
            for annotation in annotations['sentences']:
                gold_label = annotation['gold_label']
                sentence = annotation['sentence']
                example[gold_label] = sentence
            data.append(example)

    return data


def main(args):

    if args.input == 'crows_pairs':
        data = preprocess_crows_pairs()
    elif args.input == 'stereoset':
        data = preprocess_stereoset()

    with open(args.output, 'w') as fw:
        json.dump(data, fw, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
