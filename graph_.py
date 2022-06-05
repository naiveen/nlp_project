import networkx as nx
import re
import tqdm
import json
import spacy
import textacy
import logging
import argparse

import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
				

from overrides import overrides
from comet2.comet_model import PretrainedCometModel
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level=logging.INFO)
logger = logging.getLogger(__name__)


from generate_inferences_from_comet import get_clarifications_socialiqa_, get_clarifications_socialiqa, get_personx

CATEGORY_TO_QUESTION = {"xIntent": "What was the intention of PersonX?",
						"xNeed": "Before that, what did PersonX need?",
						"oEffect": "What happens to others as a result?",
						"oReact": "What do others feel as a result?",
						"oWant": "What do others want as a result?",
						"xEffect": "What happens to PersonX as a result?",
						"xReact": "What does PersonX feel as a result?",
						"xWant": "What does PersonX want as a result?",
						"xAttr": "How is PersonX seen?"}

CATEGORY_TO_PREFIX = {"xIntent": "Because PersonX wanted",
					  "xNeed": "Before, PersonX needed",
					  "oEffect": "Others then",
					  "oReact": "As a result, others feel",
					  "oWant": "As a result, others want",
					  "xEffect": "PersonX then",
					  "xReact": "As a result, PersonX feels",
					  "xWant": "As a result, PersonX wants",
					  "xAttr": "PersonX is seen as"}


question_to_comet_relation = {
		  "What will [NAME] want to do next?": "xWant",
		  "What will [NAME] want to do after?": "xWant",
		  "How would [NAME] feel afterwards?": "xReact",
		  "How would [NAME] feel as a result?": "xReact",
		  "What will [NAME] do next?": "xReact",
		  "How would [NAME] feel after?": "xReact",
		  "How would you describe [NAME]?": "xAttr",
		  "What kind of person is [NAME]?": "xAttr",
		  "How would you describe [NAME] as a person?": "xAttr",
		  "Why did [NAME] do that?": "xIntent",
		  "Why did [NAME] do this?": "xIntent",
		  "Why did [NAME] want to do this?": "xIntent",
		  "What does [NAME] need to do beforehand?": "xNeed",
		  "What does [NAME] need to do before?": "xNeed",
		  "What does [NAME] need to do before this?": "xNeed",
		  "What did [NAME] need to do before this?": "xNeed",
		  "What will happen to [NAME]?": "xEffect",
		  "What will happen to [NAME] next?": "xEffect"
	}

QUESTION_TO_ANSWER_PREFIX = {
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


class KnowledgeGraph():
	def __init__(self, nlp, comet_model, scoreComputer, lhops):
		self.G = nx.DiGraph()
		self.lhops = lhops
		self.comet_model = comet_model
		self.scoreComputer = scoreComputer
		self.nlp = nlp
		pass

	def extend_graph(self, context_list, answers):
		output = []
		for context in context_list:
			# print("context: ", context)
			inferences = get_clarifications_socialiqa_(context, self.nlp, self.comet_model, self.scoreComputer)
			for relation, out_event, score, _  in inferences:
				# print("relation: ", relation)
				# print("inference/edge: ", out_event)
				# print("edge score: ",score)
				self.G.add_node(out_event)
				self.G.add_edge(context, out_event, weight = score)
				output.append(out_event)
				for answer in answers:
					inference_answer = " ".join([out_event, answer])
					# print("answer_edge: ", inference_answer)
					answer_score = self.scoreComputer.get_score(inference_answer)
					# print("edge_score: ", answer_score)
					self.G.add_edge(out_event, answer, weight = answer_score)
		return output

	def get_prediction(self, input):
		# context_list = [ex['context']]
		# # answers = ex['answers']
		# answers = []
		# choices = [ex['answerA'], ex['answerB'], ex['answerC']]
		# choices = [c + "." if not c.endswith(".") else c for c in choices]
		# question = ex['question']
		# answer_prefix = ""
		# for template, ans_prefix in QUESTION_TO_ANSWER_PREFIX.items():
		# 	m = re.match(template, question)
		# 	if m is not None:
		# 		answer_prefix = ans_prefix.replace("[SUBJ]", m.group(1))
		# 		break

		# if answer_prefix == "": 
		# 	answer_prefix = question.replace("?", "is")
		
		# answers = choices.copy()
		# answers = [
        #     " ".join((answer_prefix, choice[0].lower() + choice[1:])).replace(
        #         "?", "").replace("wanted to wanted to", "wanted to").replace(
        #         "needed to needed to", "needed to").replace("to to", "to") for choice in choices]

		context_list = input['context_list']
		answers_list = input['answers_list']

		self.G.add_node(context_list[0])

		# print("answers: ", answers)
		for i in range(self.lhops):
			# print("lhop: ", i)
			context_list = self.extend_graph(context_list, answers_list)
		
		longest_path = nx.dag_longest_path(self.G)

		# predicted_label = answers.index(longest_path[-1])
		predicted_answer = longest_path[-1]
		# return self.G, predicted_label
		return self.G, predicted_answer



def create_graph_get_prediction(fields , instance_reader, comet_model, nlp,scoreComputer, get_clarification_func, lhops =2, num_beams = 3):
	context = fields["context"]
	fields_= fields.copy()
	context_list = [context]
	G, gold_label, answers= init_graph(fields_,instance_reader,scoreComputer,lhops )
	#nx.draw_networkx(G,with_labels=True)
	#plt.savefig("init.png")
	fields_= fields.copy()
	for _ in range(lhops-1):
		context_list = extend_graph(G, fields_, context_list,instance_reader, comet_model,nlp,scoreComputer, get_clarification_func, num_beams)
	longest_path = nx.dag_longest_path(G)
	predicted_label = answers.index(longest_path[-1])

	return G, gold_label,predicted_label

def init_graph(fields, instance_reader,scoreComputer, lhops):
	G = nx.DiGraph()
	context  = fields["context"]
	G.add_nodes_from([context])
	fields["clarifications"] =[]
	context, question, label, choices, clarifications, context_with_choice_and_clarifications, answers = \
					instance_reader.fields_to_instance(fields)
	G.add_nodes_from(answers)
	for i, answer in enumerate(answers):
		G.add_edge(context,answer, weight = scoreComputer.get_score(context_with_choice_and_clarifications[i]))
	

	return G, label, answers

def extend_graph(G, fields,context_list,instance_reader,comet_model,nlp,scoreComputer,get_clarification_func,num_beams=3):
	outputs =[]
	for context in context_list:
		fields_ = fields.copy()
		fields_["context"] = context
		clarifications = get_clarification_func(fields_, nlp, comet_model)
		for clarification in clarifications:
			fields_["clarifications"] = [clarification]
			_, question, _, _, _, context_with_choice_and_clarifications, answers = \
				instance_reader.fields_to_instance(fields_)
			
			G.add_nodes_from([clarification[1]])
			outputs.append(clarification[1])
			G.add_edge(context,clarification[1], weight = scoreComputer.get_score(context+clarification[1]))
			
			for i,answer in enumerate(answers):
				G.add_edge(clarification[1],answer, weight = scoreComputer.get_score(context_with_choice_and_clarifications[i][0]))
			plt.clf()

			#nx.draw_networkx(G,with_labels=True)
			#plt.savefig(clarification[1].replace(" ","")+".png")
	return outputs

