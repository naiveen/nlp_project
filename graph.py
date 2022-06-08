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


from generate_clarifications_from_comet import get_clarifications_socialiqa, get_personx

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

def get_graph_node(hops,val):
	return (hops,val)

def create_graph_get_prediction(fields , instance_reader, comet_model, nlp,scoreComputer, get_clarification_func, lhops =2, num_beams = 3):
	if "context" in fields.keys():
		context = fields["context"]
	else:
		context = fields["sentence"]
	fields_= fields.copy()
	context_list = [context]
	hops = 0
	G, gold_label, answers= init_graph(fields_,instance_reader,scoreComputer,lhops, hops)
	#nx.draw_networkx(G,with_labels=True)
	#plt.savefig("init.png")
	fields_= fields.copy()
	for i in range(lhops-1):
		context_list = extend_graph(G, fields_, context_list,instance_reader, comet_model,nlp,scoreComputer, get_clarification_func, num_beams, i+1)
	longest_path = nx.dag_longest_path(G)
	predicted_label = answers.index(longest_path[-1])
	return G, gold_label,predicted_label

def init_graph(fields, instance_reader,scoreComputer, lhops, hops):
	G = nx.DiGraph()
	if "context" in fields.keys():
		context = fields["context"]
	else:
		context = fields["sentence"]
	context_node_list = [get_graph_node(hops,context)]
	G.add_nodes_from(context_node_list)
	fields["clarifications"] =[]
	context, question, label, choices, clarifications, context_with_choice_and_clarifications, answers = \
					instance_reader.fields_to_instance(fields)
	G.add_nodes_from(answers)
	for i, answer in enumerate(answers):
		G.add_edge(context_node_list,answer, weight = scoreComputer.get_score(context_with_choice_and_clarifications[i]))
	

	return G, label, answers

def extend_graph(G, fields,context_node_list,instance_reader,comet_model,nlp,scoreComputer,get_clarification_func,num_beams, hops):
	outputs =[]
	for context_node in context_node_list:
		fields_ = fields.copy()
		fields_["context"] = context_node[1]
		clarifications = get_clarification_func(fields_, nlp, comet_model)
		for clarification in clarifications:
			fields_["clarifications"] = [clarification]
			_, question, _, _, _, context_with_choice_and_clarifications, answers = \
				instance_reader.fields_to_instance(fields_)
			
			G.add_nodes_from(get_graph_node(hops, clarification[1]))
			outputs.append(get_graph_node(hops, clarification[1]))
			G.add_edge(context_node,clarification[1], weight = scoreComputer.get_score(context_node[1]+clarification[1]))
			
			for i,answer in enumerate(answers):
				G.add_edge(clarification[1],answer, weight = scoreComputer.get_score(context_with_choice_and_clarifications[i][0]))
			plt.clf()
			#nx.draw_networkx(G,with_labels=True)
			#plt.savefig(clarification[1].replace(" ","")+".png")
	return outputs

