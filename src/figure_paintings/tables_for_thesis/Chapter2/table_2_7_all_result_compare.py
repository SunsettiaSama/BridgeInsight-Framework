import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_BASE = PROJECT_ROOT / "results" / "training_result"
ECC_RESULT_PATH  = PROJECT_ROOT / "results" / "ecc_results"  / "ecc_search_results.json"
MECC_RESULT_PATH = PROJECT_ROOT / "results" / "mecc_results" / "mecc_search_results.json"

ML_METHODS = [
    {
        "name": "SVM",
        "path": RESULTS_BASE / "machine_learning_module" / "svm" / "svm_result.json",
    },
    {
        "name": "Naive Bayes",
        "path": RESULTS_BASE / "machine_learning_module" / "bayes" / "bayes_result.json",
    },
    {
        "name": "Cellular Automata",
        "path": RESULTS_BASE / "machine_learning_module" / "cellunar_automata" / "ca_result.json",
    },
]

DL_METHODS = [
    {
        "name": "MLP",
        "path": RESULTS_BASE / "deep_learning_module" / "mlp" / "mlp_train_result.json",
    },
    {
        "name": "RNN",
        "path": RESULTS_BASE / "deep_learning_module" / "rnn" / "rnn_train_result.json",
    },
    {
        "name": "LSTM",
        "path": RESULTS_BASE / "deep_learning_module" / "lstm" / "lstm_train_result.json",
    },
    {
        "name": "CNN",
        "path": RESULTS_BASE / "deep_learning_module" / "cnn" / "cnn_train_result.json",
    },
    {
        "name": "ResCNN",
        "path": RESULTS_BASE / "deep_learning_module" / "res_cnn" / "res_cnn_train_result.json",
    },
]

COL_WIDTHS = [20, 12, 10, 10]
TOTAL_WIDTH = sum(COL_WIDTHS) + len(COL_WIDTHS) - 1


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_dl_best_metrics_by_f1(result_data):
    epoch_states = result_data['training_metadata']['epoch_states']
    best_idx = max(range(len(epoch_states)), key=lambda i: epoch_states[i]['val_metrics']['f1'])
    vm = epoch_states[best_idx]['val_metrics']
    return vm['precision'], vm['recall'], vm['f1']


def get_ml_metrics(result_data):
    vm = result_data['val_metrics']
    return vm['precision'], vm['recall'], vm['f1']


def get_ecc_best_metrics(result_data):
    best_threshold = result_data['best_params']['threshold']
    wm = result_data['param_metrics'][str(best_threshold)]['weighted']
    return wm['Precision'], wm['Recall'], wm['F1']


def get_mecc_best_metrics(result_data):
    bp = result_data['best_params']
    key = f"{bp['k_viv']}_{bp['C_viv']}"
    wm = result_data['param_metrics'][key]['weighted']
    return wm['Precision'], wm['Recall'], wm['F1']


def fmt_header():
    return (
        f"{'Method':<{COL_WIDTHS[0]}} "
        f"{'Precision':>{COL_WIDTHS[1]}} "
        f"{'Recall':>{COL_WIDTHS[2]}} "
        f"{'F1':>{COL_WIDTHS[3]}}"
    )


def fmt_row(name, precision, recall, f1):
    return (
        f"{name:<{COL_WIDTHS[0]}} "
        f"{precision:>{COL_WIDTHS[1]}.4f} "
        f"{recall:>{COL_WIDTHS[2]}.4f} "
        f"{f1:>{COL_WIDTHS[3]}.4f}"
    )


def fmt_missing(name, path):
    return f"{'[MISSING] ' + name:<{COL_WIDTHS[0]}}  (run script first: {path})"


def main():
    ml_rows = []
    for method in ML_METHODS:
        data = load_json(method['path'])
        precision, recall, f1 = get_ml_metrics(data)
        ml_rows.append({'name': method['name'], 'precision': precision, 'recall': recall, 'f1': f1})
    ml_rows.sort(key=lambda x: x['f1'], reverse=True)

    dl_rows = []
    for method in DL_METHODS:
        data = load_json(method['path'])
        precision, recall, f1 = get_dl_best_metrics_by_f1(data)
        dl_rows.append({'name': method['name'], 'precision': precision, 'recall': recall, 'f1': f1})
    dl_rows.sort(key=lambda x: x['f1'], reverse=True)

    ecc_row  = None
    mecc_row = None

    if ECC_RESULT_PATH.exists():
        data = load_json(ECC_RESULT_PATH)
        precision, recall, f1 = get_ecc_best_metrics(data)
        ecc_row = {'name': 'ECC', 'precision': precision, 'recall': recall, 'f1': f1}

    if MECC_RESULT_PATH.exists():
        data = load_json(MECC_RESULT_PATH)
        precision, recall, f1 = get_mecc_best_metrics(data)
        mecc_row = {'name': 'MECC', 'precision': precision, 'recall': recall, 'f1': f1}

    sep   = '-' * TOTAL_WIDTH
    thick = '=' * TOTAL_WIDTH

    print(thick)
    print('All Methods Result Comparison (Val Set, Best F1 Epoch for DL)')
    print(thick)
    print(fmt_header())
    print(sep)

    print('[Machine Learning]')
    for row in ml_rows:
        print(fmt_row(row['name'], row['precision'], row['recall'], row['f1']))

    print(sep)

    print('[Deep Learning]')
    for row in dl_rows:
        print(fmt_row(row['name'], row['precision'], row['recall'], row['f1']))

    print(sep)

    print('[Physical Analysis]')
    if ecc_row:
        print(fmt_row(ecc_row['name'], ecc_row['precision'], ecc_row['recall'], ecc_row['f1']))
    else:
        print(fmt_missing('ECC', 'src/train_eval/ecc/ecc.py'))

    if mecc_row:
        print(fmt_row(mecc_row['name'], mecc_row['precision'], mecc_row['recall'], mecc_row['f1']))
    else:
        print(fmt_missing('MECC', 'src/train_eval/ecc/mecc.py'))

    print(thick)


if __name__ == "__main__":
    main()
