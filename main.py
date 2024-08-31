import pandas as pd
from SentenceParser import SentenceParser
from PrepareSentenceContext import PrepareSentenceContext

def load_and_parse_csv(file_path):
    df = pd.read_csv(file_path)
    sentence_parser = SentenceParser()
    df['parsed_sentences'] = df['Falcon'].apply(lambda x: list(sentence_parser(x)))
    return df

def main():
    csv_file_path = 'wiki_dataset.csv'
    df = load_and_parse_csv(csv_file_path)
    print(df.head())

if __name__ == "__main__":
    main()

