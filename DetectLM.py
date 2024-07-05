import numpy as np
import pandas as pd
from multitest import MultiTest
from tqdm import tqdm
import logging


def truncate_to_max_no_tokens(text, max_no_tokens):
    return " ".join(text.split()[:max_no_tokens])


class DetectLM(object):
    def __init__(self, sentence_detection_function, survival_function_per_length,
                 min_len=1, max_len=100, HC_type="stbl",
                 length_limit_policy='truncate', ignore_first_sentence=False):
        """
        Test for the presence of sentences of irregular origin as reflected by the
        sentence_detection_function. This function can be assisted by a context, which we
        determine using the context_policy argument.

        :param sentence_detection_function:  a function returning the log-perplexity of the text
        based on a candidate language model
        :param survival_function_per_length:  survival_function_per_length(l, x) is the probability of the language
        model to produce a sentence of log-perplexity as extreme as x or more, for an input sentence s
        of length l or a for an input pair (s, c) with sentence s of length l under context c.
        :param length_limit_policy: what should we do if a sentence is too long. Options are:
            'truncate':  truncate sentence to the maximal length :max_len
             'ignore':  do not evaluate the response and P-value for this sentence
             'max_available':  use the log-perplexity function of the maximal available length
        :param ignore_first_sentence:  whether to ignore the first sentence in the document or not. Useful when assuming
        context of the form previous sentence.
        """

        self.survival_function_per_length = survival_function_per_length
        self.sentence_detector = sentence_detection_function
        self.min_len = min_len
        self.max_len = max_len
        self.length_limit_policy = length_limit_policy
        self.ignore_first_sentence = ignore_first_sentence
        self.HC_stbl = True if HC_type == 'stbl' else False

    def _logperp(self, sent: str, context=None) -> float:
        return float(self.sentence_detector(sent, context))

    def _test_sentence(self, sentence: str, context=None):
        return self._logperp(sentence, context)
    
    def _get_length(self, sentence: str):
        return len(sentence.split())

    def _test_response(self, response: float, length: int):
        """
        Returns:
          response:  sentence log-perplexity
          pval:      P-value of atomic log-perplexity test
        """
        if self.min_len <= length:
            comment = "OK"
            if length > self.max_len:  # in case length exceeds specifications...
                if self.length_limit_policy == 'truncate':
                    length = self.max_len
                    comment = f"truncated to {self.max_len} tokens"
                elif self.length_limit_policy == 'ignore':
                    comment = "ignored (above maximum limit)"
                    return np.nan, np.nan, comment
                elif self.length_limit_policy == 'max_available':
                    comment = "exceeding length limit; resorted to max-available length"
                    length = self.max_len
            pval = self.survival_function_per_length(length, response)
            assert pval >= 0, "Negative P-value. Something is wrong."
            return dict(response=response, 
                        pvalue=pval, 
                        length=length,
                        comment=comment)
        else:
            comment = "ignored (below minimal length)"
            return dict(response=response, 
                        pvalue=np.nan, 
                        length=length,
                        comment=comment)

    def _get_pvals(self, responses: list, lengths: list) -> tuple:
        pvals = []
        comments = []
        for response, length in zip(responses, lengths):
            r = self._test_response(response, length)
            pvals.append(float(r['pvalue']))
            comments.append(r['comment'])
        return pvals, comments


    def _get_responses(self, sentences: list, contexts: list) -> list:
        """
        Compute response and length of a text sentence 
        """
        assert len(sentences) == len(contexts)

        responses = []
        lengths = []
        for sent, ctx in tqdm(zip(sentences, contexts)):
            length = self._get_length(sent)
            if self.length_limit_policy == 'truncate':
                sent = truncate_to_max_no_tokens(sent, self.max_len)
            responses.append(self._test_sentence(sent, ctx))
            lengths.append(length)
        return responses, lengths

    def get_pvals(self, sentences: list, contexts: list) -> tuple:
        """
        Log-perplexity test of every (sentence, context) pair
        """
        assert len(sentences) == len(contexts)

        responses, lengths = self._get_responses(sentences, contexts)
        pvals, comments = self._get_pvals(responses, lengths)
        
        return pvals, responses, comments



    # def _test_sent(self, sent: str, context=None):
    #     """
    #     Returns:
    #       response:  sentence log-perplexity
    #       pval:      P-value of atomic log-perplexity test
    #     """
    #     length = len(sent.split())  # This is the approximate length as the true length is determined by the tokenizer
    #     if self.min_len <= length:
    #         comment = "OK"
    #         if length > self.max_len:  # in case length exceeds specifications...
    #             if self.length_limit_policy == 'truncate':
    #                 sent = truncae_to_max_no_tokens(sent, self.max_len)
    #                 length = self.max_len
    #                 comment = f"truncated to {self.max_len} tokens"
    #             elif self.length_limit_policy == 'ignore':
    #                 comment = "ignored (above maximum limit)"
    #                 return np.nan, np.nan, comment
    #             elif self.length_limit_policy == 'max_available':
    #                 comment = "exceeding length limit; resorted to max-available length"
    #                 length = self.max_len
    #         response = self._logperp(sent, context)
    #         pval = self.survival_function_per_length(length, float(response))
    #         assert pval >= 0, "Negative P-value. Something is wrong."
    #         return dict(response=response, 
    #                     pvalue=pval, 
    #                     length=length,
    #                     comment=comment)
    #     else:
    #         comment = "ignored (below minimal length)"
    #         if len(sent) > 100:
    #             logging.warning(f"Sentence is too long ({sent})")
    #         return dict(response=np.nan, 
    #                     pvalue=np.nan, 
    #                     length=np.nan,
    #                     comment=comment)

    # def get_pvals(self, sentences: list, contexts: list) -> tuple:
    #     """
    #     Log-perplexity test of every (sentence, context) pair
    #     """
    #     assert len(sentences) == len(contexts)

    #     pvals = np.zeros(len(sentences))
    #     responses = np.zeros(len(sentences))
    #     comments = []
    #     for i, (sent, ctx) in tqdm(enumerate(zip(sentences, contexts))):
    #         r = self._test_sent(sent, ctx)
    #         pvals[i] = r['pvalue']
    #         responses[i] = r['response']
    #         comments.append(r['comment'])
    #     return pvals, responses, comments

    def testHC(self, sentences: list) -> float:
        pvals = np.array(self.get_pvals(sentences)[1])
        mt = MultiTest(pvals, stbl=self.HC_stbl)
        return mt.hc()[0]

    def testFisher(self, sentences: list) -> dict:
        pvals = np.array(self.get_pvals(sentences)[1])
        print(pvals)
        mt = MultiTest(pvals, stbl=self.HC_stbl)
        return dict(zip(['Fn', 'pvalue'], mt.fisher()))

    def _test_chunked_doc(self, lo_chunks: list, lo_contexts: list) -> tuple:
        pvals, responses, comments = self.get_pvals(lo_chunks, lo_contexts)
        if self.ignore_first_sentence:
            pvals[0] = np.nan
            logging.info('Ignoring the first sentence.')
            comments[0] = "ignored (first sentence)"
        
        df = pd.DataFrame({'sentence': lo_chunks, 'response': responses, 'pvalue': pvals,
                           'context': lo_contexts, 'comment': comments},
                          index=range(len(lo_chunks)))
        df_test = df[~df.pvalue.isna()]
        if df_test.empty:
            logging.warning('No valid chunks to test.')
            return None, df
        return MultiTest(df_test.pvalue, stbl=self.HC_stbl), df

    def test_chunked_doc(self, lo_chunks: list, lo_contexts: list, dashboard=False) -> dict:
        mt, df = self._test_chunked_doc(lo_chunks, lo_contexts)
        if mt is None:
            hc = np.nan
            fisher = (np.nan, np.nan)
            df['mask'] = pd.NA
        else:
            hc, hct = mt.hc(gamma=0.4)
            fisher = mt.fisher()
            df['mask'] = df['pvalue'] <= hct
        if dashboard:
            mt.hc_dashboard(gamma=0.4)
        return dict(sentences=df, HC=hc, fisher=fisher[0], fisher_pvalue=fisher[1])

    def __call__(self, lo_chunks: list, lo_contexts: list, dashboard=False) -> dict:
        return self.test_chunked_doc(lo_chunks, lo_contexts, dashboard=dashboard)