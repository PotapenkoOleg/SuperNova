import argparse

import requests
from tabulate import tabulate

DEFAULT_BASE_URL = "http://localhost:80"
DEFAULT_INPUT_FILE = "examples.csv"

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json"
}


def post(base_url: str, path: str, payload: dict) -> dict | list:
    try:
        response = requests.post(f"{base_url}{path}", headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")


def bulk_predict(base_url: str, input_strings: list[str]) -> list[dict]:
    return post(base_url, "/bulk_predict/", {"input_strs": input_strings})


def vote_predict(base_url: str, input_strings: list[str], soft_vote: bool) -> dict:
    return post(base_url, "/vote_predict/", {"input_strs": input_strings, "soft_vote": soft_vote})


def read_input_strings(filename: str) -> list[str]:
    try:
        with open(filename, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        raise Exception(f"File {filename} not found")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")


def print_predictions(results: list[dict]) -> None:
    results_table = [("Input", "Predicted Class", "Probability")]
    for result in results:
        results_table.append((result['input-str'], result['predicted-class'], result['probability']))
    print(tabulate(results_table, headers="firstrow", tablefmt="simple_grid"))


def print_vote(result: dict) -> None:
    print(f"Predicted class : {result['predicted-class']}")
    print(f"Soft vote       : {result['soft-vote']}")
    print(f"Samples         : {result['sample-count']}")
    print(f"Tie-break used  : {result['tie-break-used']}")
    print()
    results_table = [("Class", "Votes", "Probability Sum")]
    for class_name, votes in result['votes'].items():
        results_table.append((class_name, votes, round(result['probability-sums'][class_name], 4)))
    print(tabulate(results_table, headers="firstrow", tablefmt="simple_grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SuperNova prediction CLI")
    parser.add_argument('-m', '--mode', choices=['bulk', 'vote', 'all'],
                        default='bulk', help="which endpoint to exercise (default: bulk)")
    parser.add_argument('-s', '--soft', action='store_true',
                        help="use soft voting in vote mode")
    parser.add_argument('-u', '--base-url', default=DEFAULT_BASE_URL,
                        help=f"API base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument('-f', '--input-file', default=DEFAULT_INPUT_FILE,
                        help=f"file of raw input strings, one per line (default: {DEFAULT_INPUT_FILE})")
    args = parser.parse_args()

    try:
        input_strings = read_input_strings(args.input_file)

        if args.mode in ('bulk', 'all'):
            print("== /bulk_predict/ ==")
            print_predictions(bulk_predict(args.base_url, input_strings))

        if args.mode in ('vote', 'all'):
            print("== /vote_predict/ ==")
            print_vote(vote_predict(args.base_url, input_strings, args.soft))
    except Exception as e:
        print(f"Error: {str(e)}")
