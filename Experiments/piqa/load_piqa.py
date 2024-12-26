from datasets import load_dataset
import json
import os

def load_and_save_piqa():
    # Load PIQA dataset from Huggingface
    dataset = load_dataset("ybisk/piqa", trust_remote_code=True)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save each split (train, validation) to separate JSON files
    for split in dataset.keys():
        output_file = f"data/piqa_{split}.json"
        
        # Convert to list of dictionaries
        data_list = dataset[split].to_dict()
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {split} split to {output_file}")

if __name__ == "__main__":
    load_and_save_piqa()
