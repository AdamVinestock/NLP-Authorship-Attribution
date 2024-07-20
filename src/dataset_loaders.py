from datasets import load_dataset
import pandas as pd

SEED = 42

def get_dataset(file_path: str, shuffle: bool = False):
    dataset = pd.read_csv(file_path)
    if 'id' not in dataset.columns:
        dataset['id'] = list(range(len(dataset)))
    if shuffle:
        dataset = dataset.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return dataset

def get_text_from_wiki_dataset(file_path: str, shuffle: bool = False):
    return get_dataset(file_path="src/wiki_dataset", shuffle=shuffle)

def get_text_from_news_dataset(file_path: str, shuffle: bool = False):
    return get_dataset(file_path="src/news_dataset", shuffle=shuffle)

def get_text_from_abstracts_dataset(file_path: str, shuffle: bool = False):
    return get_dataset(file_path="src/abstracts_dataset", shuffle=shuffle)

# def get_dataset(name: str, machine_field, human_field, iterable=False,
#                 text_field=None, shuffle=False):
#     dataset = load_dataset(name)['train']
#     ds = dataset.rename_columns({human_field: 'human_text', machine_field: 'machine_text'})
#     if 'id' not in ds.features:
#         ids = list(range(len(ds)))
#         ds = ds.add_column("id", ids)
#     if text_field:
#         ds = ds.rename_columns({text_field: 'text'})
#     if iterable:
#         ds = ds.to_iterable_dataset()
#     if shuffle:
#         return ds.shuffle(seed=SEED)
#     else:
#         return ds
    

def get_text_from_wiki_long_dataset(shuffle=False, text_field=None):
    return get_dataset(name="alonkipnis/wiki-intro-long", machine_field='generated_intro',
                       human_field="wiki_intro", shuffle=shuffle, text_field=text_field)


def get_text_from_wiki_long_dataset_local(shuffle=False, text_field=None, iterable=False):
    """
    A version of wiki_intro dataset with at least 15 sentences per generated article
    """
    dataset = load_dataset("alonkipnis/wiki-intro-long")
    ds = dataset.rename_columns({"wiki_intro": 'human_text', "generated_intro": 'machine_text'})
    if text_field:
        ds = ds.rename_columns({text_field: 'text'})
    if iterable:
        ds = ds.to_iterable_dataset()
    if shuffle:
        return ds.shuffle(seed=SEED)
    else:
        return ds

def get_text_from_chatgpt_abstracts_dataset(shuffle=False, text_field=None):
    return get_dataset(name="NicolaiSivesind/ChatGPT-Research-Abstracts", machine_field="generated_abstract",
                       human_field="real_abstract", shuffle=shuffle, text_field=text_field)

def get_text_from_chatgpt_news_long_dataset(shuffle=False, text_field=None):
    return get_dataset(name="alonkipnis/news-chatgpt-long", machine_field='chatgpt',
                       human_field="article", shuffle=shuffle, text_field=text_field)


def get_text_from_chatgpt_news_dataset(shuffle=False, text_field=None):
    return get_dataset(name="isarth/chatgpt-news-articles", machine_field='chatgpt',
                       human_field="article", shuffle=shuffle, text_field=text_field)


def get_text_from_wikibio_dataset(shuffle=False, text_field=None):
    return get_dataset(name="potsawee/wiki_bio_gpt3_hallucination", machine_field='gpt3_text',
                       human_field="wiki_bio_text", shuffle=shuffle, text_field=text_field)

## New datasets (22/5/2023)
def get_text_from_alpaca_gpt4_dataset(shuffle=False, text_field=None):
    return get_dataset(name="polyware-ai/alpaca-gpt4-cleaned", machine_field='output',
                       human_field="instruction", shuffle=shuffle, text_field=text_field)