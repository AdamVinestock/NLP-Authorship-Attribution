# %%
# !pip install transformers
# !pip install torch  
# !pip install datasets  

# %%
import pandas as pd
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import notebook_login

nltk.download('punkt')
pd.set_option('display.max_columns', None)  # Show all columns

# %%
notebook_login()

# %%
# Load the HuggingFace wiki_intro_long dataset
hf_wiki_dataset = load_dataset('alonkipnis/wiki-intro-long', split='train')
df_wiki = pd.DataFrame(hf_wiki_dataset)

# Add columns for Llama2 and Falcon7B model outputs
df_wiki['human_len'] = None
df_wiki['gpt_len'] = None
df_wiki['Llama2'], df_wiki['Llama2_len'] = None, None
df_wiki['Falcon'], df_wiki['Falcon_len'] = None, None

df_wiki.rename(columns={
    'wiki_intro': 'human_text',
    # 'wiki_intro_len': 'human_len',
    'generated_intro': 'gpt'
    }, inplace=True)

columns_to_drop = ['prompt_tokens', 'generated_text', 'generated_intro_len']
df_wiki.drop(columns=columns_to_drop, inplace=True)

new_order = [
    'id', 'url', 'title', 'title_len', 'prompt',
    'human_text', 'human_len',
    'gpt', 'gpt_len',
    'Llama2', 'Llama2_len',
    'Falcon', 'Falcon_len']

df_wiki = df_wiki[new_order]

# %%
print(df_wiki.columns)
print(df_wiki.shape[0])

# %%
# Load the HuggingFace news dataset
hf_news_dataset = load_dataset('alonkipnis/news-chatgpt-long', split='train')
df_news = pd.DataFrame(hf_news_dataset)

df_news.rename(columns={
    'article': 'human_text',
    'chatgpt': 'gpt'
}, inplace=True)

df_news['human_len'], df_news['gpt_len'] = None, None
df_news['Llama2'], df_news['Llama2_len'] = None, None
df_news['Falcon'], df_news['Falcon_len'] = None, None
df_news['prompt'] = None

new_order = [
    'id', 'highlights', 'prompt',
    'human_text', 'human_len',
    'gpt', 'gpt_len',
    'Llama2', 'Llama2_len',
    'Falcon', 'Falcon_len'
]

df_news = df_news[new_order]

# %%
print(df_news.columns)
print(df_news.shape[0])

# %%
# Load the HuggingFace research absracts dataset
hf_abstracts_dataset = load_dataset('NicolaiSivesind/ChatGPT-Research-Abstracts', split='train')
df_abstracts = pd.DataFrame(hf_abstracts_dataset)

df_abstracts.rename(columns={
    'real_abstract': 'human_text',
    'real_word_count': 'human_len',
    'generated_abstract': 'gpt',
    'generated_word_count': 'gpt_len'
}, inplace=True)

df_abstracts['Llama2'], df_abstracts['Llama2_len'] = None, None
df_abstracts['Falcon'], df_abstracts['Falcon_len'] = None, None
df_abstracts['prompt'] = None

new_order = [
    'title', 'prompt',
    'human_text', 'human_len',
    'gpt', 'gpt_len',
    'Llama2', 'Llama2_len',
    'Falcon', 'Falcon_len'
]

df_abstracts = df_abstracts[new_order]

# %%
print(df_abstracts.columns)
print(df_abstracts.shape[0])

# %%
def count_words_and_sentences(text):
    """
    Tokenizes the text into words and sentences using nltk 
    Returns a tuple of (n_words,n_sentences)
    """
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return (len(words), len(sentences))

def create_wiki_prompt(row):
    """
    Creates the wiki dataset prompt using the title and first 7 words written by humans
    """
    first_few_words = ' '.join(row['human_text'].split()[:7]) 
    # prompt = f"Write a Wikipedia-style intro covering the topic '{row['title']}', it should be detailed and span approximately {row['human_len'][1]} sentences long. {first_few_words}"
    prompt = (
        f"Compose a Wikipedia-style introduction for the topic '{row['title']}'. Start with a clear definition, "
        f"followed by key details and context that is essential for understanding the subject. "
        f"Ensure the introduction is detailed and spans approximately {row['human_len'][1]} sentences. "
        f"Begin with these words: {first_few_words}"
    )
    return prompt

def create_news_prompt(row):
    """
    Creates the news dataset prompt using the first 15 words written by humans, and the article highlights
    """
    first_few_words = ' '.join(row['human_text'].split()[:15]) 
    highlights = row['highlights'] 
    # prompt = f"Complete the news article, make sure to be detailed, the article should span approximately {row['human_len'][1]} sentences long.\nArticle highlights: {highlights}\nArticle:{first_few_words}"
    prompt = (
        f"Complete the news article based on the given highlights. Ensure the article is detailed and spans approximately {row['human_len'][1]} sentences long. "
        f"Incorporate the following key points:\nHighlights: {highlights}\n\n"
        f"Article begins: {first_few_words}"
    )
    return prompt

def create_abstracts_prompt(row):
    """
    Creates the abstracts dataset prompt using the title and first 15 words written by humans
    """
    first_few_words = ' '.join(row['human_text'].split()[:15]) 
    # prompt = f"Write a research abstract on the paper '{row['title']}'. Make sure to be detailed and span approximately {row['human_len'][1]} sentences long.\n{first_few_words}"
    prompt = (
        f"Write a research abstract for the paper titled '{row['title']}'."
        f"Ensure the abstract is detailed, clear, and spans approximately {row['human_len'][1]} sentences."
        f"\n{first_few_words}"
    )
    return prompt

def generate_text_gpt2xl(prompt, model, tokenizer, max_length=1024):
    """
    Encodes the prompt using the model tokenizer - max context windows of 1024, left padding
    Returns the generated text, word count and sentence count
    """

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length) # Prepare input encoding with padding and truncation
    attention_mask = torch.fliplr(inputs['attention_mask'])                                                       # Adjust attention mask for left padding 
    max_new_tokens = 1024 - inputs['input_ids'].shape[1]                                                          # gpt2 is limited to generating 1024 token including prompt

    output_ids = model.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=attention_mask,  
        max_new_tokens=max_new_tokens,
        num_return_sequences=1
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    n_words, n_sentences = count_words_and_sentences(generated_text)
    return generated_text, n_words, n_sentences

def generate_text_gpt2xl_v2(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
    output_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=1024,  # Set to the maximum length of the model
        num_return_sequences=1
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    n_words, n_sentences = count_words_and_sentences(generated_text)
    return generated_text, n_words, n_sentences



def generate_text_llama2(prompt, model, tokenizer, max_length=500):
    """
    Encodes the prompt using the model tokenizer
    Returns the generated text, word count and sentence count
    """
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_length)
    output_ids = model.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        do_sample=True,      # Enable sampling to generate more diverse responses
        temperature=0.9,     # Slightly randomize the outputs to prevent repetition
        top_k=50,            # Consider top 50 tokens for sampling at each step
        top_p=0.95,          # Use nucleus sampling with p=0.95
        num_return_sequences=1
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    n_words, n_sentences = count_words_and_sentences(generated_text)
    return generated_text, n_words, n_sentences


def generate_text_falcon(prompt, model, tokenizer, max_length=500):
    """
    Encodes the prompt using the model tokenizer
    Returns the generated text, word count and sentence count
    """
    # Adjust tokenizer padding_side for decoding
    tokenizer.padding_side = 'left'

    # Encode the prompt to tensor of input ids
    inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    # Generate response using the quantized model
    output_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length + 500, 
        num_return_sequences=1
    )
    # Decode the output ids to text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Count words and sentences in the generated text
    n_words, n_sentences = count_words_and_sentences(generated_text)
    return generated_text, n_words, n_sentences


# %%
# populate length of human text with tuple(word_count, sentence_count)
df_wiki['human_len'] = df_wiki['human_text'].apply(count_words_and_sentences)
df_news['human_len'] = df_news['human_text'].apply(count_words_and_sentences)
df_abstracts['human_len'] = df_abstracts['human_text'].apply(count_words_and_sentences)

# %%
# create prompts
df_wiki['prompt'] = df_wiki.iloc[0:10].apply(create_wiki_prompt, axis=1)
df_news['prompt'] = df_news.iloc[0:10].apply(create_news_prompt, axis=1)
df_abstracts['prompt'] = df_abstracts.iloc[0:10].apply(create_abstracts_prompt, axis=1)

# %%
# Load gpt2-xl model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
tokenizer.padding_side = 'left'  # Ensure padding from the left for gpt2
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
model.eval()

for index, row in df_wiki.head(1).iterrows():  
    prompt = row['prompt'] 
    generated_text, word_count, sent_count = generate_text_gpt2xl_v2(prompt, model, tokenizer)
    df_wiki.at[index, 'gpt'] = generated_text
    df_wiki.at[index, 'gpt_len'] = [(word_count, sent_count)]


# %%
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model.eval()

# generate text for wiki dataset
for index, row in df_wiki.head(1).iterrows():
    prompt = row['prompt']
    generated_text = generate_text_llama2(prompt, model, tokenizer)
    df_wiki.at[index, 'Llama2'] = generated_text
    df_wiki.at[index, 'Llama2_len'] = [(word_count, sent_count)]

 # generate text for news dataset
for index, row in df_news.head(1).iterrows():
    prompt = row['prompt']
    generated_text = generate_text_llama2(prompt, model, tokenizer)
    df_news.at[index, 'Llama2'] = generated_text
    df_news.at[index, 'Llama2_len'] = [(word_count, sent_count)]

# generate text for abstracts dataset
for index, row in df_abstracts.head(1).iterrows():
    prompt = row['prompt']
    generated_text = generate_text_llama2(prompt, model, tokenizer)
    df_abstracts.at[index, 'Llama2'] = generated_text
    df_abstracts.at[index, 'Llama2_len'] = [(word_count, sent_count)]

# %%
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)

# generate text for wiki dataset
for index, row in df_wiki.head(1).iterrows():
    prompt = row['prompt']
    generated_text = generate_text_llama2(prompt, model, tokenizer)
    df_wiki.at[index, 'Llama2'] = generated_text
    df_wiki.at[index, 'Llama2_len'] = [(word_count, sent_count)]

 # generate text for news dataset
for index, row in df_news.head(1).iterrows():
    prompt = row['prompt']
    generated_text = generate_text_llama2(prompt, model, tokenizer)
    df_news.at[index, 'Llama2'] = generated_text
    df_news.at[index, 'Llama2_len'] = [(word_count, sent_count)]

# generate text for abstracts dataset
for index, row in df_abstracts.head(1).iterrows():
    prompt = row['prompt']
    generated_text = generate_text_llama2(prompt, model, tokenizer)
    df_abstracts.at[index, 'Llama2'] = generated_text
    df_abstracts.at[index, 'Llama2_len'] = [(word_count, sent_count)]


# %%
# Export to CSV for external analysis
df_wiki.iloc[0:2].to_csv('generated_text_comparison.csv', index=False)

# %%
# # Apply dynamic quantization
# quantized_model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# )





