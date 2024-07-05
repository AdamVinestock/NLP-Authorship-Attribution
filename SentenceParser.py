import re
import spacy


class Sentence(object):
    def __init__(self, text):
        self.text = text
        self.tokens = text.split()

    def __len__(self):
        return len(self.tokens)

class Sentences(object):
    def __init__(self, text):
        def iterate(text):
            ls = [s.strip() for s in re.split(r"(\.[^0-9]|\?|\!)", text)]
            for i in range(len(ls) // 2):  # split by ".", "?", or "!" and add to sentence
                yield ls[2 * i] + ls[2 * i + 1]
        self.sents = iterate(text)

    def __len__(self):
        return len(self.sents)

class SentenceParser(object):
    """
    Iterate over the text column of a dataframe
    """

    def __init__(self):
        self.sents = None

    def __call__(self, text):
        return Sentences(text)