import argparse
import os
import sys
from scripts.naive.naive_baseline import run_all_naive_baselines
from scripts.static.birch_clustering import run_birch_baseline
from scripts.graph.egvi_clustering import run_egvi_baseline
from scripts.static.jamsic_method import run_jamsic_baseline
from scripts.context.bert_clustering import run_bert_baseline
from scripts.utils import make_picture, make_html_picture


def main():
    parser = argparse.ArgumentParser(description='Arguments for SEMCLUSTERS baselines')
    parser.add_argument('method', type=str, help='Method type (birch / jamsic / naive / egvi)')
    parser.add_argument('dataset', type=str, help='Path to dataset')
    parser.add_argument('--model', type=str, help='Path to model or name of huggingface BERT')
    parser.add_argument('--visualize', action='store_true', help='If passed returns the graph')
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
        out = run_jamsic_baseline(args.dataset, args.model)
    elif args.method == 'birch':
        out = run_birch_baseline(args.dataset, args.model)
    elif args.method == 'bert':
        out = run_bert_baseline(args.dataset, args.model)
    elif args.method == 'egvi':
        out = run_egvi_baseline(args.dataset, args.model)
    else:
        print(f"No such method: {args.method}", file=sys.stderr)
        exit(-1)
    if args.visualize:
        if args.method == 'naive':
            print(f"Can't visualize for naive", file=sys.stderr)
            exit(-1)
        make_html_picture(out, args.method)
        # out.to_csv('result_example.csv')


if __name__ == '__main__':
    main()
