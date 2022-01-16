import argparse
import os
import sys
from scripts.naive.naive_baseline import run_all_naive_baselines
from scripts.static.birch_clusterting import run_birch_baseline
from scripts.static.jamsic_method import run_jamsic_baseline
from scripts.context.bert_clustering import run_bert_baseline


def main():
    parser = argparse.ArgumentParser(description='Arguments for SEMCLUSTERS baselines')
    parser.add_argument('method', type=str, help='Method type (birch / jamsic / naive)')
    parser.add_argument('dataset', type=str, help='Path to dataset')
    parser.add_argument('--model', type=str, help='Path to model or name of huggingface BERT')
    args = parser.parse_args()
    for path in [args.dataset, args.model]:
        if (args.method == 'naive' or args.method == 'bert') and path == args.model:
            continue
        if not os.path.exists(path):
            print(f"No such file or directory: {path}", file=sys.stderr)
            exit(-1)

    if args.method == 'naive':
        run_all_naive_baselines(args.dataset)
    elif args.method == 'jamsic':
        run_jamsic_baseline(args.dataset, args.model)
    elif args.method == 'birch':
        run_birch_baseline(args.dataset, args.model)
    elif args.method == 'bert':
        run_bert_baseline(args.dataset, args.model)
    else:
        print(f"No such method: {args.method}", file=sys.stderr)
        exit(-1)


if __name__ == '__main__':
    main()
