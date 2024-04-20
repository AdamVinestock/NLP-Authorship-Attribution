from datasets import load_dataset

def load_datasets():
    dataset_names = {
        "news": "alonkipnis/news-chatgpt-long",
        "research_abstracts": "NicolaiSivesind/ChatGPT-Research-Abstracts",
        "wikipedia": "alonkipnis/wiki-intro-long"
    }
    
    datasets = {}
    for key, dataset_path in dataset_names.items():
        # Load the dataset
        datasets[key] = load_dataset(dataset_path)
        print(f"Loaded {key} dataset with {len(datasets[key]['train'])} training samples.")
    
    return datasets

load_datasets()
