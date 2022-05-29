import math

import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse

# context, relation, answer

BATCH_SIZE = {"distilgpt2": 64,
              "openai-gpt": 64,
              "gpt2": 64,
              "gpt2-medium": 32,
              "gpt2-large": 32,
              "gpt2-xl": 16,
              "xlnet-base-cased": 32,
              "xlnet-large-cased": 16}

class ScoreComputer():
    def __init__(self, comet_instance):
        self.device = comet_instance.device
        tokenizer = comet_instance.tokenizer
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token = 0
        model = comet_instance.model
        model.to(self.device)
        model.eval()        
        self.model = model
        self.tokenizer = tokenizer

    def get_lm_score(self, batch, pad_token_id):
        """
        Get the lowest cross entropy loss for each instance (list of clarifications) in the batch
        using the langage model
        """
        # Batch: [num_clarifications, max_length]
        with torch.no_grad():
            num_clarifications, max_length = batch.shape
            shift_labels = batch[..., 1:].contiguous().view(-1).clone()
            shift_labels[shift_labels == pad_token_id] = -100
            lm_logits = self.model(batch).logits
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss.view(num_clarifications, -1).mean(1).min().cpu().item()

        return loss

    def get_score(self, context_with_choice_and_clarifications):
        # context_with_choice_and_clarifications = " ".join((context,relation,answer))
        tokenized = self.tokenizer(context_with_choice_and_clarifications, return_tensors="pt", padding=True)["input_ids"].to(self.device)
        # batch_size = BATCH_SIZE[self.lm]
        # num_batches = int(math.ceil(len(tokenized[0]) / batch_size))
        return 1/self.get_lm_score(tokenized, self.tokenizer.pad_token_id)

def main():
    pass

if __name__ == "__main__":
    main()
