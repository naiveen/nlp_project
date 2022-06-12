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
from graph_ import KnowledgeGraph

import warnings
warnings.filterwarnings('ignore')

from generate_inferences_from_comet import get_clarifications_socialiqa, get_clarifications_winogrande, get_clarifications_commonsenseqa

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


INSTANCE_READERS = {
                    "winogrande": WinograndeInstanceReader,
                    "commonsenseqa":CommonsenseqaInstanceReader}

CLARIFICATION_FUNCTION = {
                    "socialiqa": get_clarifications_socialiqa,
                    "winogrande": get_clarifications_winogrande,
                    "commonsenseqa":get_clarifications_commonsenseqa}

QUESTION_TO_ANSWER_PREFIX_OLD = {
        "What will (.*) want to do next?": r"[SUBJ] wanted to",
        "What will (.*) want to do after?": r"[SUBJ] wanted to",
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

QUESTION_TO_ANSWER_PREFIX = {
        "What will (.*) want to do next?": r"After, [SUBJ] will want to",
        "What will (.*) want to do after?": r"After, [SUBJ] will want to",
        "How would (.*) feel afterwards?": r"[SUBJ] feels",
        "How would (.*) feel as a result?": r"[SUBJ] feel",
        "What will (.*) do next?": r"After, [SUBJ] will",
        "How would (.*) feel after?": r"[SUBJ] feel",
        "How would you describe (.*)?": r"[SUBJ] is",
        "What kind of person is (.*)?": r"[SUBJ] is",
        "How would you describe (.*) as a person?": r"[SUBJ] is",
        "Why did (.*) do that?": r"[SUBJ] did that because",
        "Why did (.*) do this?": r"[SUBJ] did this because",
        "Why did (.*) want to do this?": r"[SUBJ] wanted to this because",
        "What does (.*) need to do beforehand?": r"Beforehand, [SUBJ] needs to",
        "What does (.*) need to do before?": r"Before, [SUBJ] needs to",
        "What does (.*) need to do before this?": r"Before this, [SUBJ] needs to",
        "What did (.*) need to do before this?": r"Before this, [SUBJ] needed to",
        "What will happen to (.*)?": r"The effect on [SUBJ] will be",
        "What will happen to (.*) next?": r"The effect on [SUBJ] will be"
    }

QUESTION_TO_ANSWER_PREFIX = {
        "What will (.*) want to do next?": r"[SUBJ] wanted to",
        "What will (.*) want to do after?": r"[SUBJ] wanted to",
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
    
def preprocess_commonsenseqa(ex):
    processed_output = {"context_list": [], "answers_list":[], "ground_truth":""}

    context = ex['question']['stem']
    label = ['A','B','C','D','E'].index(ex['answerKey']) if "answerKey" in ex else None
    choices = [c['text'] for c in ex['question']['choices']]
    question = ''

    processed_output['context_list'].append(context)
    
    answers = [f"{context} [choice]" for choice in choices]
    
    processed_output['answers_list'] = answers

    processed_output['ground_truth'] = answers[label]
    processed_output['option1'] = ""
    processed_output['option2'] = ""

    return processed_output


def preprocess_winogrande(ex):
    processed_output = {"context_list": [], "answers_list":[], "ground_truth":""}
    context = ex['sentence']
    if not context.endswith("."):
        context += "."
    processed_output['context_list'].append(context)
    choices = [ex['option1'], ex['option2']]
    question = ''
    label = ex['answer']
    label = int(label) - 1
    context_with_choice = [context.replace("_", choice).strip() for choice in choices]
    processed_output['answers_list'] = context_with_choice

    processed_output['ground_truth'] = context_with_choice[label]
    processed_output['option1'] = ex['option1']
    processed_output['option2'] = ex['option2']

    return processed_output


def preprocess_socialiqa(ex):
    processed_output = {"context_list": [], "answers_list":[], "ground_truth":""}
    processed_output['context_list'].append(ex['context'])
    choices = [ex['answerA'], ex['answerB'], ex['answerC']]
    choices = [c + "." if not c.endswith(".") else c for c in choices]
    question = ex['question']
    answer_prefix = ""
    for template, ans_prefix in QUESTION_TO_ANSWER_PREFIX.items():
        m = re.match(template, question)
        if m is not None:
            answer_prefix = ans_prefix.replace("[SUBJ]", m.group(1))
            break

    if answer_prefix == "": 
        answer_prefix = question.replace("?", "is")
    
    answers = choices.copy()
    answers = [
        " ".join((answer_prefix, choice[0].lower() + choice[1:])).replace(
            "?", "").replace("wanted to wanted to", "wanted to").replace(
            "needed to needed to", "needed to").replace("to to", "to") for choice in choices]
    processed_output['answers_list'] = answers

    processed_output['ground_truth'] = answers[ord(ex['correct']) - 65]

    return processed_output

def preprocess_storycs(ex):
    emotions = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
    processed_output = {"context_list": [], "answers_list":[], "ground_truth":""}
    processed_output['context_list'].append(ex['context'])
    choices = emotions
    choices = [c + "." if not c.endswith(".") else c for c in choices]
    answers = choices.copy()
    processed_output['answers_list'] = answers
    processed_output['ground_truth'] = ex['label']

    return processed_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", default="comet", type=str, required=False, help="language model to use")
    parser.add_argument("--file", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--dataset", default="socialiqa", type=str, required=True, help="Jsonl file")
    parser.add_argument("--out_file", default=None, type=str, required=True, help="Out directory for the predictions")
    parser.add_argument("--device", default=0, type=int, required=False, help="GPU device")
    parser.add_argument("--lhops", default=2, type=int, required=False, help="Number of hops")

    args = parser.parse_args()
    logger.info(args)

    # Load the language model
    model, tokenizer = init_model(args.lm,args.device)
    comet_model = PretrainedCometModel(device=args.device)

    nlp = spacy.load('en_core_web_sm')
    scoreComputer = ScoreComputer(comet_model)
    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")

    # Load the dataset original dataset without any clarifications
    preprocess_func_dict={"storycs": preprocess_storycs,
                          "commonsenseqa": preprocess_commonsenseqa,
                          "winogrande": preprocess_winogrande,
                            }
    gold = []
    predictions = []
    with open(args.file) as f_in:
        with open(args.out_file, "w") as f_out:
            data_examples = [json.loads(line.strip()) for line in f_in]
            for ex in tqdm.tqdm(data_examples):
                kg = KnowledgeGraph(nlp, comet_model, scoreComputer, lhops=args.lhops)
                # single instance of dataset
                preprocess_func=preprocess_func_dict.get(args.dataset,preprocess_socialiqa)
                processed_input = preprocess_func(ex)
                G, predicted_answer = kg.get_prediction(processed_input)
                gold.append(processed_input['ground_truth'])
                predictions.append(predicted_answer)
        if None not in gold:
            accuracy = accuracy_score(gold, predictions) * 100
            print(f"Accuracy: {accuracy:.3f}")
    return

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
