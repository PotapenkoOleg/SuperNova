import argparse

import requests
from tabulate import tabulate

DEFAULT_BASE_URL = "http://localhost:80"

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

CLASS_NAMES = {
    0: 'int',
    1: 'float',
    2: 'boolean',
    3: 'time',
    4: 'date',
    5: 'datetime',
    6: 'uuid',
    7: 'string'
}


def post(base_url: str, path: str, payload: dict) -> dict | list:
    try:
        response = requests.post(f"{base_url}{path}", headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")


def predict(base_url: str, input_string: str) -> dict:
    return post(base_url, "/predict/", {"input_str": input_string})


def bulk_predict(base_url: str, input_strings: list[str]) -> list[dict]:
    return post(base_url, "/bulk_predict/", {"input_strs": input_strings})


def vote_predict(base_url: str, input_strings: list[str], soft_vote: bool) -> dict:
    return post(base_url, "/vote_predict/", {"input_strs": input_strings, "soft_vote": soft_vote})


def read_examples(filename: str) -> list[tuple[str, ...]] | None:
    try:
        with open(filename, 'r') as file:
            return [tuple(line.strip().split('=')) for line in file if '=' in line]
    except FileNotFoundError:
        raise Exception(f"File {filename} not found")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")


def print_predictions(examples: list[tuple[str, ...]], results: list[dict]) -> None:
    results_table = [("Input", "Actual Class", "Predicted Class", "Probability")]
    for (key, value), result in zip(examples, results):
        results_table.append((value, CLASS_NAMES[int(key)], result['predicted-class'], result['probability']))
    print(tabulate(results_table, headers="firstrow", tablefmt="simple_grid"))


def print_vote(result: dict) -> None:
    print(f"Predicted class : {result['predicted-class']}")
    print(f"Soft vote       : {result['soft-vote']}")
    print(f"Samples         : {result['sample-count']}")
    print(f"Tie-break used  : {result['tie-break-used']}")
    print()
    results_table = [("Class", "Votes", "Probability Sum")]
    for class_name in CLASS_NAMES.values():
        results_table.append((class_name, result['votes'][class_name],
                              round(result['probability-sums'][class_name], 4)))
    print(tabulate(results_table, headers="firstrow", tablefmt="simple_grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SuperNova prediction CLI")
    parser.add_argument('--mode', choices=['predict', 'bulk_predict', 'vote_predict', 'all'],
                        default='predict', help="which endpoint to exercise (default: predict)")
    parser.add_argument('--soft-vote', action='store_true',
                        help="use soft voting in vote_predict mode")
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL,
                        help=f"API base URL (default: {DEFAULT_BASE_URL})")
    args = parser.parse_args()

    try:
        examples = read_examples("examples.csv")
        input_strings = [value for _, value in examples]

        if args.mode in ('predict', 'all'):
            print("== /predict/ ==")
            print_predictions(examples, [predict(args.base_url, value) for value in input_strings])

        if args.mode in ('bulk_predict', 'all'):
            print("== /bulk_predict/ ==")
            print_predictions(examples, bulk_predict(args.base_url, input_strings))

        if args.mode in ('vote_predict', 'all'):
            print("== /vote_predict/ ==")
            print_vote(vote_predict(args.base_url, input_strings, args.soft_vote))
    except Exception as e:
        print(f"Error: {str(e)}")
