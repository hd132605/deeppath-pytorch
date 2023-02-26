import torch

import random
from collections import namedtuple, Counter
from functools import lru_cache
import numpy as np
from typing import List, Dict, Tuple

from BFS.KB import KB
from BFS.BFS import bfs

# hyperparameters
state_dim = 200
state_dim_bert = 768 * 2  # BERT encoding concatenated
state_dim_transr_kgbert_concat = 868 * 2  # TransR + KGBERT concatenated
action_space = 400
eps_start = 1
eps_end = 0.1
epe_decay = 1000
replay_memory_size = 10000
batch_size = 128
embedding_dim = 100
embedding_dim_bert = 768
embedding_dim_transr_kgbert_concat = 868
gamma = 0.99
target_update_freq = 1000
max_steps = 50
max_steps_test = 50
learning_rate = 0.001
weight_decay = 0.01

cuda = "cuda:0"
dataPath = '../NELL-995/'

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

relations = [
	"concept_agentbelongstoorganization",
	"concept_athletehomestadium",
	"concept_athleteplaysforteam",
	"concept_athleteplaysinleague",
	"concept_athleteplayssport",
	"concept_organizationheadquarteredincity",
	"concept_organizationhiredperson",
	"concept_personborninlocation",
	"concept_personleadsorganization",
	"concept_teamplaysinleague",
	"concept_teamplayssport",
	"concept_worksfor"
]


def get_path_stats_path(mode: str, relation: str):
	paths_dict = {
		"RL": dataPath + "tasks/" + relation + "/" + "path_stats.txt",
		"bert": dataPath + "tasks/" + relation + "/" + "path_stats_bert.txt",
		"kgbert": dataPath + "tasks/" + relation + "/" + "path_stats_kgbert.txt",
		"kgbert_triple": dataPath + "tasks/" + relation + "/" + "path_stats_kgbert_triple.txt",
		"transr_kgbert_concat": dataPath + "tasks/" + relation + "/" + "path_stats_transr_kgbert_concat.txt"
	}
	return paths_dict[mode]


def get_path_to_use_path(mode: str, relation: str):
	paths_dict = {
		"RL": dataPath + "tasks/" + relation + "/" + "path_to_use.txt",
		"bert": dataPath + "tasks/" + relation + "/" + "path_to_use_bert.txt",
		"kgbert": dataPath + "tasks/" + relation + "/" + "path_to_use_kgbert.txt",
		"kgbert_triple": dataPath + "tasks/" + relation + "/" + "path_to_use_kgbert_triple.txt",
		"transr_kgbert_concat": dataPath + "tasks/" + relation + "/" + "path_to_use_transr_kgbert_concat.txt"
	}
	return paths_dict[mode]


def distance(e1, e2):
	return np.sqrt(np.sum(np.square(e1 - e2)))


def compare(v1, v2):
	return sum(v1 == v2)


# @lru_cache(maxsize=512)
def teacher(e1, e2, num_paths, entity2vec, entity2id:dict, relation2id:dict, kb: KB):
	intermediates = kb.pickRandomIntermediatesBetween(e1, e2, num_paths)
	res_entity_lists = []
	res_path_lists = []
	for i in range(num_paths):
		suc1, entity_list1, path_list1 = bfs(kb, e1, intermediates[i])
		suc2, entity_list2, path_list2 = bfs(kb, intermediates[i], e2)
		if suc1 and suc2:
			res_entity_lists.append(entity_list1 + entity_list2[1:])
			res_path_lists.append(path_list1 + path_list2)
	# print('BFS found paths:', len(res_path_lists))
	
	# ---------- clean the path --------
	res_entity_lists_new = []
	res_path_lists_new = []
	for entities, relations in zip(res_entity_lists, res_path_lists):
		rel_ents = []
		for i in range(len(entities)+len(relations)):
			if i%2 == 0:
				rel_ents.append(entities[int(i/2)])
			else:
				rel_ents.append(relations[int(i/2)])

		#print(rel_ents)

		entity_stats = list(Counter(entities).items())
		duplicate_ents = [item for item in entity_stats if item[1]!=1]
		duplicate_ents.sort(key = lambda x:x[1], reverse=True)
		for item in duplicate_ents:
			ent = item[0]
			ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
			if len(ent_idx)!=0:
				min_idx = min(ent_idx)
				max_idx = max(ent_idx)
				if min_idx!=max_idx:
					rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
		entities_new = []
		relations_new = []
		for idx, item in enumerate(rel_ents):
			if idx%2 == 0:
				entities_new.append(item)
			else:
				relations_new.append(item)
		res_entity_lists_new.append(entities_new)
		res_path_lists_new.append(relations_new)
	
	# print(res_entity_lists_new)
	# print(res_path_lists_new)

	good_episodes = []
	targetID = entity2id[e2]
	for path in zip(res_entity_lists_new, res_path_lists_new):
		good_episode = []
		for i in range(len(path[0]) -1):
			currID = entity2id[path[0][i]]
			nextID = entity2id[path[0][i+1]]
			state_curr = [currID, targetID, 0]
			state_next = [nextID, targetID, 0]
			actionID = relation2id[path[1][i]]
			good_episode.append(Transition(
				state=idx_state(entity2vec, state_curr),
				action=actionID,
				next_state=idx_state(entity2vec, state_next),
				reward=1))
		good_episodes.append(good_episode)
	return good_episodes  # collection of Transition lists. Transition includes 'state'. 'state' vector is concatenation of (curr, targ-curr).


# remove same entities in path, and join with '->'
def path_clean(path):
	rel_ents = path.split(' -> ')
	relations = []
	entities = []
	for idx, item in enumerate(rel_ents):
		if idx % 2 == 0:
			relations.append(item)
		else:
			entities.append(item)
	entity_stats = list(Counter(entities).items())
	duplicate_ents = [item for item in entity_stats if item[1]!=1]
	duplicate_ents.sort(key = lambda x:x[1], reverse=True)
	for item in duplicate_ents:
		ent = item[0]
		ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
		if len(ent_idx)!=0:
			min_idx = min(ent_idx)
			max_idx = max(ent_idx)
			if min_idx!=max_idx:
				rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
	return ' -> '.join(rel_ents)


def prob_norm(probs):
	return probs/sum(probs)


def open_entity2id_and_relation2id() -> Tuple[Dict, Dict, List]:
	entity2id = {}
	relation2id = {}
	relations = []
	with open(dataPath + "entity2id.txt") as entity2id_file:
		lines = entity2id_file.readlines()
		for line in lines:
			entity2id[line.split()[0]] = int(line.split()[1])
	with open(dataPath + "relation2id.txt") as relation2id_file:
		lines = relation2id_file.readlines()
		for line in lines:
			relation2id[line.split()[0]] = int(line.split()[1])
			relations.append(line.split()[0])
	return entity2id, relation2id, relations


def open_entity2vec_and_relation2vec(mode:str) -> Tuple[np.ndarray, np.ndarray]:
	if mode == "bert":
		entity2vec = torch.load(dataPath + "entity2vec_bert.pkl").numpy()  # loads (75492, 768) entity2vec torch tensor
		relation2vec = torch.load(
			dataPath + "relation2vec_bert.pkl").numpy()  # loads (400, 768) relation2vec torch tensor
	elif mode == "kgbert":
		entity2vec = torch.vstack(
			torch.load(dataPath + "entity2kgbert")).cpu().numpy()  # entity2kgbert is list of (768,) tensor
		relation2vec = torch.vstack(torch.load(dataPath + "relation2kgbert")).cpu().numpy()  # so vstack needed
	elif mode == "kgbert_triple":
		entity2vec = torch.load(dataPath + "triple2kgbert_entity.pkl").numpy()
		relation2vec = torch.load(dataPath + "triple2kgbert_relation.pkl").numpy()
	elif mode == "transr_kgbert_concat":
		entity2vec = torch.load(dataPath + "transr_kgbert_concat_entity.pkl")
		relation2vec = torch.load(dataPath + "transr_kgbert_concat_relation.pkl")
	else:
		raise ValueError
	return entity2vec, relation2vec


def relation_to_kb(relation: str) -> KB:
	graphpath = dataPath + "tasks/" + relation + "/" + "graph.txt"
	with open(graphpath) as f:
		graph = f.readlines()
		kb = KB()
		for line in graph:
			ent1, rel, ent2 = line.rsplit()
			kb.addRelation(ent1, rel, ent2)
	return kb


def idx_state(entity2vec, idx_list):  # idx_list: [currID, targetID, 0] or [nextID, targetID, 0]
	if idx_list is not None:
		curr = entity2vec[idx_list[0], :]
		targ = entity2vec[idx_list[1], :]
		return np.expand_dims(np.concatenate((curr, targ - curr)), axis=0)
	else:
		return None


if __name__ == '__main__':
	print(prob_norm(np.array([1,1,1])))
	#path_clean('/common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/01d34b -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/0lfyx -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/01y67v -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/028qyn -> /people/person/nationality -> /m/09c7w0')





