from many_atomic_detections import process_text, iterate_over_texts
from PerplexityEvaluator import PerplexityEvaluator
from PrepareSentenceContext import PrepareSentenceContext
from src.dataset_loaders import get_text_from_wiki_dataset, get_text_from_news_dataset, get_text_from_abstracts_dataset
import pandas as pd

class ResponseClass():
    def __init__(self, dataset_name, model, model_name, tokenizer,context_policies, from_sample=0, to_sample=10):
        self.model = model
        self.model_name = model_name
        self.range = "[{}, {}]".format(from_sample, to_sample)
        self.dataset_name = dataset_name
        self.context_policy = context_policies  ## 'previous-3-sentences'
        self.context = ''
        self.from_sample = from_sample
        self.to_sample = to_sample
        self.sentence_detector = PerplexityEvaluator(model, tokenizer)
        self.human_dataset, self.llama_dataset, self.falcon_dataset = self.SplitDataset()
        self.parser = PrepareSentenceContext(context_policy=self.context_policy, context=self.context)
        self.datasets_dict = {'human': self.human_dataset, 'llama': self.llama_dataset, 'falcon': self.falcon_dataset}
        self.responses = self.CalculatePerplexity()

    def SplitDataset(self):
        human_dataset = None
        llama_dataset = None
        falcon_dataset = None
        if self.dataset_name == "wiki":
            human_dataset = get_text_from_wiki_dataset(shuffle=False, text_field='human_text')
            llama_dataset = get_text_from_wiki_dataset(shuffle=False, text_field='Llama2')
            falcon_dataset = get_text_from_wiki_dataset(shuffle=False, text_field='Falcon')
        elif self.dataset_name == "news":
            human_dataset = get_text_from_news_dataset(shuffle=False, text_field='human_text')
            llama_dataset = get_text_from_news_dataset(shuffle=False, text_field='Llama2')
            falcon_dataset = get_text_from_news_dataset(shuffle=False, text_field='Falcon')
        elif self.dataset_name == "abstracts":
            human_dataset = get_text_from_abstracts_dataset(shuffle=False, text_field='human_text')
            llama_dataset = get_text_from_abstracts_dataset(shuffle=False, text_field='Llama2')
            falcon_dataset = get_text_from_abstracts_dataset(shuffle=False, text_field='Falcon')

        truncated_human_dataset = human_dataset.select(range(self.from_sample, self.to_sample + 1))
        truncated_llama_dataset = llama_dataset.select(range(self.from_sample, self.to_sample + 1))
        truncated_falcon_dataset = falcon_dataset.select(range(self.from_sample, self.to_sample + 1))

        return truncated_human_dataset, truncated_llama_dataset, truncated_falcon_dataset
    
    def CalculatePerplexity(self):
        responses = {}
        for author in self.datasets_dict:  # human, llama, or falcon
            csv_name = f"{self.dataset_name}_{author}_{self.model_name}_{self.context_policy}_{self.range}.csv"
            iterate_over_texts(self.datasets_dict[author], self.sentence_detector, self.parser, csv_name)
            df = pd.read_csv(f"Responses/{csv_name}")
            responses[author] = df
        return responses