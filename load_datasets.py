import pandas as pd
from datasets import load_dataset

pd.set_option('display.max_columns', None)  # Show all columns

# Load the Hugging Face dataset
hf_wiki_dataset = load_dataset('alonkipnis/wiki-intro-long', split='train')
df = pd.DataFrame(hf_wiki_dataset)
df['Llama2'], df['Llama2_len'] = None, 0
df['Falcon7B'], df['Falcon7B_len'] = None, 0

df.rename(columns={
    'wiki_intro': 'human_wiki_intro',
    'wiki_intro_len': 'human_wiki_intro_len',
    'generated_intro': 'gpt_intro',
    'generated_intro_len': 'gpt_intro_len'
    }, inplace=True)

columns_to_drop = ['prompt_tokens', 'generated_text']
df.drop(columns=columns_to_drop, inplace=True)

new_order = [
    'id', 'url', 'title', 'title_len', 'prompt',
    'human_wiki_intro', 'human_wiki_intro_len',
    'gpt_intro', 'gpt_intro_len',
    'Llama2', 'Llama2_len',
    'Falcon7B', 'Falcon7B_len']

df = df[new_order]
print(df.columns)


