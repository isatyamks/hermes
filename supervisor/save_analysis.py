import os
import json
from datetime import datetime


def save_analysis(analysis, output_dir="analyses"):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(output_dir, f"analysis_{timestamp}.json")

    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)

    print("Analysis saved to:", path)
    return path
