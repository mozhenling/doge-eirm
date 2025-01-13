# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved



import os
import sys
import argparse
import numpy as np
import copy
from datautils import diag_datasets as datasets
from algorithms import alg_selector
from losses import lss_selector
from oututils import model_selection, reporting, os_utils
from oututils.query import Q

def print_results_tables(records, selection_method, latex):
    """Given all records, print a results table for each dataset."""
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    sub_alg_names = Q(records).select("args.erm_loss").unique()
    alg_names = ([n for n in alg_selector.ALGORITHMS if n in alg_names]) # re-arrange order
    # alg_names = ([n for n in alg_selector.ALGORITHMS if n in alg_names] +
    #              [n for n in alg_names if n not in alg_selector.ALGORITHMS])
    sub_alg_names = ([n for n in lss_selector.Losses if n in sub_alg_names]) # re-arrange order and kick out None or []
    is_DYJA = True if 'EIRM' in alg_names else False
    is_DYJ = True if 'ERM' in alg_names else False
    if sub_alg_names !=[] and ( is_DYJA or is_DYJ):
        sub_alg_names_plus = [('EIRM', n) for n in sub_alg_names ] if is_DYJA else [('ERM', n) for n in sub_alg_names ]
        sub_alg_names_plus_copy = copy.deepcopy(sub_alg_names_plus)
        non_doyojo = [n for n in alg_names if n !='EIRM' ] if is_DYJA else [n for n in alg_names if n !='ERM' ]
        non_doyojo_copy = copy.deepcopy(non_doyojo)
        alg_names_same = []
        for sub, alg in zip(sub_alg_names, non_doyojo):
            if sub == alg:
                alg_names_same.append(alg)
                if is_DYJA:
                    alg_names_same.append(('EIRM', sub))
                    sub_alg_names_plus_copy.pop(('EIRM', sub) == sub_alg_names_plus)
                else:
                    alg_names_same.append(('ERM',sub))
                    sub_alg_names_plus_copy.pop(('ERM',sub)==sub_alg_names_plus)
                non_doyojo_copy.pop(alg==non_doyojo)
        alg_names = alg_names_same + sub_alg_names_plus_copy + non_doyojo_copy

    else:
        sub_alg_names_plus = []
    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]
    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                sub_alg_now = algorithm[1] if algorithm in sub_alg_names_plus else 'None'
                algorithm_now = algorithm[0] if algorithm in sub_alg_names_plus else algorithm # correct the algorithm
                trial_accs = (grouped_records
                              .filter_equals(
                    "dataset, algorithm, erm_loss,  test_env",
                    (dataset, algorithm_now, sub_alg_now, test_env)
                ).select("sweep_acc"))

                mean, err, table[i ][j] = os_utils.format_mean(trial_accs, latex)
                means.append(mean)
            if None in means:
                table[i ][-1] = "X"
            else:
                table[i ][-1] = "{:.1f}".format(sum(means) / len(means))

        col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        os_utils.print_table(table, header_text, alg_names,sub_alg_names_plus, list(col_labels),
            colwidth=20, latex=latex)

    # Print an "averages" table over datasets
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means=[]
        for j, dataset in enumerate(dataset_names):
            sub_alg_now = algorithm[1] if algorithm in sub_alg_names_plus else 'None'
            algorithm_now = algorithm[0] if algorithm in sub_alg_names_plus else algorithm # correct the algorithm
            trial_averages = (grouped_records
                .filter_equals("algorithm, erm_loss, dataset", (algorithm_now, sub_alg_now, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                )
            )
            mean, err, table[i][j] = os_utils.format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    os_utils.print_table(table, header_text, alg_names,sub_alg_names_plus, col_labels, colwidth=25,
        latex=latex)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, default=r'./outputs/sweep_outs')
    parser.add_argument("--result_dir", type=str, default=r'./outputs')
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    results_file = "0_sweep_results.tex" if args.latex else "0_sweep_results.txt"

    os.makedirs(args.result_dir, exist_ok=True)
    sys.stdout = os_utils.Tee(os.path.join(args.result_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full DomainBed results}")
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))

    SELECTION_METHODS = [
        # model_selection.IIDAccuracySelectionMethod, # training domains are corrupted by label noises
        # model_selection.LeaveOneOutSelectionMethod, # show results when args.single_test_envs = False (costly)
        model_selection.OracleSelectionMethod, # test domains are not corrupted by label noises
    ]

    for selection_method in SELECTION_METHODS:
        if args.latex:
            print()
            print("\\subsection{{Model selection: {}}}".format(
                selection_method.name))
        print_results_tables(records, selection_method, args.latex)

    if args.latex:
        print("\\end{document}")
