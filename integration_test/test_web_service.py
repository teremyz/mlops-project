import json

import requests
from deepdiff import DeepDiff

cetli_id = "hW93qGPJug"
top_n = 3
url = f"http://localhost/predict?cetli_id={cetli_id}&top_n={top_n}"
actual_response = requests.get(url).json()
print("actual response:")
print(json.dumps(actual_response, indent=4))

expected_response = {
    "product_id": {"109": 176.0, "167": 5.0, "47": 194.0},
    "product_name": {
        "109": "sajt",
        "167": "avocado",
        "47": "sz\u00e1raz t\u00f6rl\u0151",
    },
    "score": {"109": 9.42, "167": 9.26, "47": 9.09},
}

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f"diff={diff}")

assert "type_changes" not in diff
assert "values_changed" not in diff
