
import json
from pathlib import Path

def debug_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("Empty data")
        return
    
    task = data[0]
    print(f"Task keys: {list(task.keys())}")
    
    annotations = task.get('annotations', [])
    print(f"Number of annotations: {len(annotations)}")
    
    if annotations:
        ann = annotations[0]
        print(f"Annotation keys: {list(ann.keys())}")
        result = ann.get('result', [])
        print(f"Number of results: {len(result)}")
        for r in result:
            print(f"Result entry: from_name={r.get('from_name')}, to_name={r.get('to_name')}, type={r.get('type')}")
            print(f"Value: {r.get('value')}")

if __name__ == "__main__":
    debug_json('labels_storage/export/train_done.json')
