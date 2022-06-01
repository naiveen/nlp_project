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


from generate_clarifications_from_comet import get_clarifications_socialiqa, get_clarifications_winogrande, get_personx

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



def create_graph_get_prediction(fields , instance_reader, comet_model, nlp,scoreComputer, lhops =2, num_beams = 3):
	# context = fields["context"]
	context, _, _, _,_, _ = instance_reader.fields_to_instance(fields)

	fields_= fields.copy()
	context_list = [context]
	G, gold_label, answers= init_graph(fields_,instance_reader,scoreComputer,lhops )
	#nx.draw_networkx(G,with_labels=True)
	#plt.savefig("init.png")
	fields_= fields.copy()
	for _ in range(lhops-1):
		context_list = extend_graph(G, fields_, context_list,instance_reader, comet_model,nlp,scoreComputer, num_beams)
	longest_path = nx.dag_longest_path(G)
	predicted_label = answers.index(longest_path[-1])
	return G, gold_label,predicted_label

def init_graph(fields, instance_reader,scoreComputer, lhops):
	G = nx.DiGraph()
	# context, _, _, _,_, _ = instance_reader.fields_to_instance(fields)
	# context  = fields["context"]
	context, question, label, choices, clarifications, context_with_choice_and_clarifications, answers = instance_reader.fields_to_instance(fields)

	G.add_nodes_from([context])
	fields["clarifications"] =[]

	G.add_nodes_from(answers)
	for i, answer in enumerate(answers):
		G.add_edge(context,answer, weight = scoreComputer.get_score(context_with_choice_and_clarifications[i]))
	

	return G, label, answers

def extend_graph(G, fields,context_list,instance_reader,comet_model,nlp,scoreComputer, num_beams=3):
	outputs =[]
	for context in context_list:
		fields_ = fields.copy()
		fields_["context"] = context
		clarifications = get_clarifications_winogrande(fields_, nlp, comet_model)
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

