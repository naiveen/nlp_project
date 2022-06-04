import os
import re
import json
import tqdm
import math
import torch
import logging
import argparse
import numpy as np

from overrides import overrides
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from comet2.comet_model import PretrainedCometModel
from score import ScoreComputer

import warnings
warnings.filterwarnings('ignore')

from generate_clarifications_from_comet import get_clarifications_socialiqa, get_clarifications_winogrande, get_clarifications_commonsenseqa

import spacy
from graph import *
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = {"distilgpt2": 64,
              "openai-gpt": 64,
              "gpt2": 64,
              "gpt2-medium": 32,
              "gpt2-large": 32,
              "comet":32,
              "gpt2-xl": 16,
              "xlnet-base-cased": 32,
              "xlnet-large-cased": 16}



class InstanceReader(object):
    def to_uniform_fields(self, fields):
        pass

    def fields_to_instance(self, fields):
        pass



class SocialIQAInstanceReader(InstanceReader):
    """
    Reads the SocialIQa dataset into a unified format with context, question, label, choices and clarifications.
    """
    def __init__(self):
        super(SocialIQAInstanceReader).__init__()
        self.QUESTION_TO_ANSWER_PREFIX = {
              "What will (.*) want to do next?": r"As a result, [SUBJ] wanted to",
              "What will (.*) want to do after?": r"As a result, [SUBJ] wanted to",
              "How would (.*) feel afterwards?": r"As a result, [SUBJ] felt",
              "How would (.*) feel as a result?": r"As a result, [SUBJ] felt",
              "What will (.*) do next?": r"[SUBJ] then",
              "How would (.*) feel after?": r"[SUBJ] then",
              "How would you describe (.*)?": r"[SUBJ] is seen as",
              "What kind of person is (.*)?": r"[SUBJ] is seen as",
              "How would you describe (.*) as a person?": r"[SUBJ] is seen as",
              "Why did (.*) do that?": r"Before, [SUBJ] wanted",
              "Why did (.*) do this?": r"Before, [SUBJ] wanted",
              "Why did (.*) want to do this?": r"Before, [SUBJ] wanted",
              "What does (.*) need to do beforehand?": r"Before, [SUBJ] needed to",
              "What does (.*) need to do before?": r"Before, [SUBJ] needed to",
              "What does (.*) need to do before this?": r"Before, [SUBJ] needed to",
              "What did (.*) need to do before this?": r"Before, [SUBJ] needed to",
              "What will happen to (.*)?": r"[SUBJ] then",
              "What will happen to (.*) next?": r"[SUBJ] then"
        }

    @overrides
    def to_uniform_fields(self, fields):
        context = fields['context']
        if not context.endswith("."):
            context += "."

        question = fields['question']
        label = fields['correct']
        choices = [fields['answerA'], fields['answerB'], fields['answerC']]
        choices = [c + "." if not c.endswith(".") else c for c in choices]

        if ("None", "None") in fields['clarifications']:
            fields['clarifications'].append(("None", "None"))

        clarifications = [c[1] if len(c[1].split()) > 1 else " ".join((c)) for c in fields['clarifications']]
        clarifications = [c[0].upper() + c[1:] for c in clarifications] + [""]

        label = ord(label) - 65
        return context, question, label, choices, clarifications

    @overrides
    def fields_to_instance(self, fields):

        context, question, label, choices, clarifications = self.to_uniform_fields(fields)

        answer_prefix = ""
        for template, ans_prefix in self.QUESTION_TO_ANSWER_PREFIX.items():
            m = re.match(template, question)
            if m is not None:
                answer_prefix = ans_prefix.replace("[SUBJ]", m.group(1))
                break

        if answer_prefix == "": 
            answer_prefix = question.replace("?", "is")

        answers = choices.copy()
        choices = [
            " ".join((answer_prefix, choice[0].lower() + choice[1:])).replace(
                "?", "").replace("wanted to wanted to", "wanted to").replace(
                "needed to needed to", "needed to").replace("to to", "to") for choice in choices]

        context_with_clarifications = [f"{context} [choice] {clarification}" for clarification in clarifications]
        context_with_choice_and_clarifications = [
            [context_with_clar.replace("[choice]", choice).strip()
             for context_with_clar in context_with_clarifications]
            for choice in choices]

        return context, question, label, choices, clarifications, context_with_choice_and_clarifications, answers


class WinograndeInstanceReader(InstanceReader):
    """
    Reads the WinoGrande dataset into a unified format with context, question, label, choices and clarifications.
    """
    @overrides
    def to_uniform_fields(self, fields):
        context = fields['sentence']
        if not context.endswith("."):
            context += "."

        label = fields['answer']
        choices = [fields['option1'], fields['option2']]

        if ("None", "None") in fields['clarifications']:
            fields['clarifications'].append(("None", "None"))

        clarifications = [c[1] if len(c[1].split()) > 1 else " ".join((c)) for c in fields['clarifications']]
        clarifications = [c[0].upper() + c[1:] for c in clarifications] + [""]

        label = int(label) - 1
        question = ''
        return context, question, label, choices, clarifications

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices, clarifications = self.to_uniform_fields(fields)
        context_with_clarifications = [f"{context} {clarification}" for clarification in clarifications]
        context_with_choice_and_clarifications = [
            [context_with_clar.replace("_", choice).strip() for context_with_clar in context_with_clarifications]
             for choice in choices]

        return context, question, label, choices, clarifications, context_with_choice_and_clarifications


class CommonsenseqaInstanceReader(InstanceReader):
    """
    Reads the CommonsenseQA dataset into a unified format with context, question, label, choices and clarifications.
    """
    @overrides
    def to_uniform_fields(self, fields):
        context = ''

        question = fields['question']['stem']
        label = ['A','B','C','D','E'].index(fields['answerKey']) if "answerKey" in fields else None
        choices = [c['text'] for c in fields['question']['choices']]

        clarifications = [c[1] if len(c[1].split()) > 1 else " ".join((c)) for c in fields['clarifications']]
        clarifications = [c[0].upper() + c[1:] for c in clarifications] + [""]
        
        return context, question, label, choices, clarifications

    @overrides
    def fields_to_instance(self, fields):
        context, question, label, choices, clarifications = self.to_uniform_fields(fields)
        context_with_clarifications = [f"{context} {question} [choice] {clarification}"
                                       for clarification in clarifications]
        context_with_choice_and_clarifications = [
            [context_with_clar.replace("[choice]", choice[0].lower() + choice[1:]).strip()
             for context_with_clar in context_with_clarifications]
             for choice in choices]

        return context, question, label, choices, clarifications, context_with_choice_and_clarifications


INSTANCE_READERS = {"socialiqa": SocialIQAInstanceReader,
                    "winogrande": WinograndeInstanceReader,
                    "commonsenseqa":CommonsenseqaInstanceReader}

CLARIFICATION_FUNCTION = {
                    "socialiqa": get_clarifications_socialiqa,
                    "winogrande": get_clarifications_winogrande,
                    "commonsenseqa":get_clarifications_commonsenseqa}
                    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="comet", type=str, required=False, help="language model to use")
    parser.add_argument("--file", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--dataset", default="socialiqa", type=str, required=True, help="Jsonl file")
    parser.add_argument("--out_dir", default=None, type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=0, type=int, required=False, help="GPU device")
    parser.add_argument("--lhops", default=2, type=int, required=False, help="Number of hops")

    args = parser.parse_args()
    logger.info(args)

    # Load the language model
    model, tokenizer = init_model(args.lm,args.device)
    comet_model = PretrainedCometModel(device=args.device)

    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")

    # Load the dataset original dataset without any clarifications
    
    with open(args.dataset) as f_in:
        with open(args.out_file, "w") as f_out:
            data_examples = [json.loads(line.strip()) for line in f_in]
            for ex in tqdm.tqdm(data_examples):
                # single instance of dataset
                # init graph
                # extend graph
                # predict output
                pass


    instance_reader = INSTANCE_READERS[args.dataset]()
    set_name = os.path.basename(args.file).replace(".jsonl", "")
    out_file = os.path.join(args.out_dir, f"{args.lm}_{set_name}_predictions.jsonl")
    gold = []
    predictions = []
    nlp = spacy.load('en_core_web_sm')
    scoreComputer = ScoreComputer(comet_model)
        # Predict instances
    with open(out_file, "w") as f_out:
        with open(args.file) as f_in:
            for line in tqdm.tqdm(f_in):
                fields = json.loads(line.strip())
                G, gold_label, predicted_label= create_graph_get_prediction(fields,instance_reader, comet_model, nlp,scoreComputer,get_clarification_func=CLARIFICATION_FUNCTION[args.dataset], lhops = args.lhops)
                gold.append(gold_label)
                predictions.append(predicted_label)


                #print(gold_label,predicted_label)
                """
                # Tokenize and pad
                tokenized = [tokenizer(per_clar, return_tensors="pt", padding=True)["input_ids"].to(device)
                             for per_clar in context_with_choice_and_clarifications]

                # Compute in batches***
                batch_size = BATCH_SIZE[args.lm]
                num_choices = len(tokenized)
                num_batches = int(math.ceil(len(tokenized[0]) / batch_size))
                per_choice_score = [1000] * num_choices

                for batch_index in range(0, num_batches):
                    curr_batch = [tokenized[i][batch_index*batch_size:(batch_index+1)*batch_size]
                                  for i in range(num_choices)]
                    curr_scores = [get_lm_score(model, clars_choice, tokenizer.pad_token_id)
                                   for clars_choice in curr_batch]
                    per_choice_score = [min(per_choice_score[i], curr_scores[i]) for i in range(num_choices)]

                prediction = int(np.argmin(per_choice_score))
                fields["prediction"] = prediction
                """
                
                #f_out.write(json.dumps(fields) + "\n")

        # Don't report accuracy if we don't have the labels
        if None not in gold:
            accuracy = accuracy_score(gold, predictions)
            print(f"Accuracy: {accuracy:.3f}")


def get_lm_score(model, batch, pad_token_id):
    """
    Get the lowest cross entropy loss for each instance (list of clarifications) in the batch
    using the langage model
    """
    # Batch: [num_clarifications, max_length]
    with torch.no_grad():
        num_clarifications, max_length = batch.shape
        shift_labels = batch[..., 1:].contiguous().view(-1).clone()
        shift_labels[shift_labels == pad_token_id] = -100
        lm_logits = model(batch).logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss.view(num_clarifications, -1).mean(1).min().cpu().item()

    return loss


def init_comet_model(model_name: str,device: torch.device):
    comet_model = PretrainedCometModel(device=device)
    tokenizer = comet_model.tokenizer
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = 0
    model = comet_model.model
    model.to(device)
    model.eval()
    return model,tokenizer

def init_model(model_name: str,
               device="cpu"):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :return: the model and tokenizer
    """
    if model_name == "comet":
        return init_comet_model(model_name,device)

    logger.info(f'Initializing {model_name}')
    device = torch.device(f'cuda:{device}') if device >= 0 else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = 0

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


if __name__ == '__main__':
    main()
