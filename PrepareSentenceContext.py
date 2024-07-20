import logging
import spacy
import re
from SentenceParser import SentenceParser


class PrepareSentenceContext(object):
    """
    Parse text and extract length and context information

    This information is needed for evaluating log-perplexity of the text with respect to a language model
    and later on to test the likelihood that the sentence was sampled from the model with the relevant context.
    """

    def __init__(self, engine='spacy', context_policy=None, context=None):
        if engine == 'spacy':
            self.nlp = spacy.load("en_core_web_sm")
        if engine == 'regex':
            logging.warning("Regex-based parser is not good at breaking sentences like 'Dr. Stone', etc.")
            self.nlp = SentenceParser()

        self.context_policy = context_policy
        self.context = context

    def __call__(self, text):
        return self.parse_sentences(text)

    def parse_sentences(self, text):
        texts = []
        contexts = []
        lengths = []
        tags = []
        num_in_par = []
        previous = None
        previous_3 = []

        text = re.sub("(</?[a-zA-Z0-9 ]+>)\s+", r"\1. ", text)  # to make sure that tags are in separate sentences
        text = re.sub(r'<ref.*?>.*?</ref>', ' ', text)          # Remove <ref> tags and their content
        text = re.sub(r'{{.*?}}', ' ', text)                    # Remove {{...}} templates
        text = re.sub(r'\s+', ' ', text)                        # Replace multiple spaces with a single space
        parsed = self.nlp(text)


        running_sent_num = 0
        tag = None
        for i, sent in enumerate(parsed.sents):
            # Here we try to track HTML-like tags. There might be
            # some issues because spacy sentence parser has unexpected behavior
            all_tags = re.findall(r"(</?[a-zA-Z0-9 ]+>)", str(sent))
            if len(all_tags) > 0:
                if all_tags[0][:2] == '</': # a closing tag
                    if tag is None:
                        logging.warning(f"Closing tag without opening in sentence {i}: {sent}")
                    else:
                        tag = None
                else: # an opening tag
                    if tag is not None:
                        logging.warning(f"Opening tag without closing in sentence {i}: {sent}")
                    else:
                        tag = all_tags[0]
            else:  # if text is not a tag
                running_sent_num += 1
                num_in_par.append(running_sent_num)
                tags.append(tag)
                lengths.append(len(sent))
                sent_text = str(sent)
                texts.append(sent_text)

                if self.context_policy == 'previous-sentence':
                    if self.context:
                        if previous is not None:
                            context = self.context + ' ' + previous
                        else:
                            context = self.context
                    else:
                        context = previous
                    previous = sent_text
                elif self.context_policy == 'previous-3-sentences':
                    if i==0:
                        context = self.context
                        previous = sent_text
                    else:
                        if i<4:
                            previous_3.append(previous)
                            if self.context:
                                context = self.context + ' ' + "".join(previous_3)
                            else:
                                context = " ".join(previous_3)
                            previous = sent_text
                        else:
                            previous_3.pop(0)
                            previous_3.append(previous)
                            if self.context:
                                context = self.context + ' ' + "".join(previous_3)
                            else:
                                context = " ".join(previous_3)
                            previous = sent_text
                else:
                    context = self.context

                contexts.append(context)


        return {'text': texts, 'length': lengths, 'context': contexts, 'tag': tags,
                'number_in_par': num_in_par}