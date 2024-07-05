import torch

class PerplexityEvaluator(object):
    def __init__(self, model, tokenizer, ignore_index= -100):
        self.model = model
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, text, context=None):
        return self.log_perplexity(text, context)

    def log_perplexity(self, text, context=None):
        """
        Evaluate log perplexity of text with respect to the language model
        based on the context

        :param text:
        :param context:
        :return:
        """
        device = self.model.device
        text_ids = self.tokenizer(text, return_tensors='pt')
        if context:
            context_ids = self.tokenizer(context, return_tensors='pt')
            # input_ids = torch.concatenate([context_ids['input_ids'], text_ids['input_ids']], axis=1)
            # labels = torch.concatenate([torch.ones_like(context_ids['input_ids']) * self.ignore_index,
            #                             text_ids['input_ids']], axis=1)
            input_ids = torch.cat([context_ids['input_ids'], text_ids['input_ids']], dim=1)
            labels = torch.cat([torch.ones_like(context_ids['input_ids']) * self.ignore_index,
                                        text_ids['input_ids']], dim=1)
        else:
            input_ids = text_ids['input_ids']
            labels = input_ids
        loss = self.model(input_ids=input_ids.to(device), labels=labels.to(device)).loss

        # print(f"model float size: {next(self.model.parameters()).dtype}")
        # print(f"input float size: {input_ids.dtype}")
        # print(f"labels float size: {labels.dtype}")

        return loss.cpu().detach().numpy()