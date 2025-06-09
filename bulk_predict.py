import requests
from tabulate import tabulate

API_ENDPOINT = "http://localhost:8090/predict/"

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

def predict(input_string: str) -> dict:
    url = API_ENDPOINT
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "input_str": input_string
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")


def read_examples(filename: str) -> list[tuple[str, ...]] | None:
    try:
        with open(filename, 'r') as file:
            return [tuple(line.strip().split('=')) for line in file if '=' in line]
    except FileNotFoundError:
        raise Exception(f"File {filename} not found")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")


if __name__ == "__main__":
    try:
        examples = read_examples("examples.txt")
        results_table = []
        results_table.append(("Input", "Actual Class", "Predicted Class", "Probability"))
        for key, value in examples:
            actual_class = CLASS_NAMES[int(key)]
            result = predict(value)
            results_table.append((value, actual_class, result['predicted-class'], result['probability']))
        print(tabulate(results_table, headers="firstrow", tablefmt="simple_grid"))
    except Exception as e:
        print(f"Error: {str(e)}")
