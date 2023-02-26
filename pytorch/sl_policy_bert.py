# python
import multiprocessing

import torch
from torch.nn.functional import one_hot
from itertools import count, repeat
from multiprocessing import Pool
import sys
from setproctitle import setproctitle
import logging

# framework
from networks import PolicyNeuralNetTorchForBert
from env import Env
from utils import *


relation = sys.argv[1]
if relation.isnumeric():
    relation = relations[int(relation)]  # enables using number instead of 'concept_agentbelongstoorganizaition'
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

max_num_samples = 10  # original code : 500

device = torch.device(cuda) if torch.cuda.is_available() else torch.device("cpu")

mode = "transr_kgbert_concat"
# mode = "kgbert_triple"
# mode = "kgbert"
# mode = "bert"


def work(episode, entity2vec, entity2id, relation2id, train_data, kb):
    print("work function:", episode)
    sample = train_data[episode].split()
    teacher(sample[0], sample[1], 5, entity2vec, entity2id, relation2id, kb)


def train():
    f = open(relationPath)
    train_data = f.readlines()
    f.close()

    num_samples = min(len(train_data), max_num_samples)

    # creating initial supervised-trained model and saving that
    if mode in ["bert", "kgbert", "kgbert_triple"]:
        applied_state_dim = state_dim_bert
    elif mode == "transr_kgbert_concat":
        applied_state_dim = state_dim_transr_kgbert_concat
    else:
        raise ValueError
    model = PolicyNeuralNetTorchForBert(applied_state_dim).to(device)
    # implements L2 regularization by weight_decay (TODO needs verification)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # TODO deleted weight_decay

    logger.info("relation: " + relation)
    logger.info("Loading entity2vec and relation2vec...")
    entity2id, relation2id, relations = open_entity2id_and_relation2id()
    entity2vec, relation2vec = open_entity2vec_and_relation2vec(mode)
    logger.info("Loading completed.")

    with open(dataPath + 'kb_env_rl.txt') as f:
        """kb_all은 eliminated_kb를 만들기 위해 쓰이고 다음부터 쓰이지 않음. kb_all에는 inverse relation도 포함되어 있음."""
        kb_all = f.readlines()
    kb = relation_to_kb(relation)  # kb: every kb_env_all's triples except current working relation.
    eliminated_kb = []
    concept = "concept:" + relation
    for line in kb_all:
        rel = line.split()[2]
        if rel != concept and rel != concept + "_inv":
            eliminated_kb.append(line)

    # Nightly optimizing code which use multiprocessing
    # logger.info("multiprocessing...")
    # pool = Pool(3)
    # pool.starmap(work, zip(range(num_samples), repeat(entity2vec), repeat(entity2id), repeat(relation2id), repeat(train_data), repeat(kb)))
    # pool.close()
    # pool.join()
    # logger.info("multi end")

    # original code
    for episode in range(num_samples):
        print(f"Episode {episode}")
        print("Training Sample:", train_data[episode][:-1])

        # env = Env(dataPath, entity2vec, relation2vec, kb_all, eliminated_kb, entity2id, relation2id, relations, train_data[episode])
        sample = train_data[episode].split()

        try:
            good_episodes = teacher(sample[0], sample[1], 5, entity2vec, entity2id, relation2id, kb)
        except Exception as e:
            print("Cannot find a path")
            print("exception: ", e)
            continue

        for item in good_episodes:
            state_batch = []
            action_batch = []
            for t, transition in enumerate(item):
                state_batch.append(transition.state)  # state: vector
                action_batch.append(transition.action)  # action: ID
            state_batch = np.squeeze(state_batch)
            state_batch = np.reshape(state_batch, [-1, applied_state_dim])  # (3, 1536)

            # ported code
            state_batch = torch.tensor(state_batch, dtype=torch.float).to(device)  # (3, 1536)
            action_batch = torch.tensor(action_batch).to(device)  # (3,)
            action_prob = model.forward(state_batch)  # (3, 400)
            onehot_action = one_hot(action_batch, torch.tensor(action_space))  # (3, 400)
            action_mask = onehot_action.type(torch.bool)  # (3, 400)
            picked_action_prob = torch.masked_select(action_prob, action_mask)  # (3,)
            loss = torch.sum(-torch.log(picked_action_prob))

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if mode == "bert":
        model_save_dir = "forbert"
    elif mode == "kgbert":
        model_save_dir = "kgbert"
    elif mode == "kgbert_triple":
        model_save_dir = "kgbert_triple"
    elif mode == "transr_kgbert_concat":
        model_save_dir = mode
    else:
        raise ValueError
    # torch.save(model.state_dict(), 'torchmodels/' + model_save_dir + '/policy_supervised_' + relation + ".pth")
    # torch.save(optimizer.state_dict(), 'torchmodels/' + model_save_dir + '/policy_supervised_' + relation + "_opt.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, "torchmodels/" + model_save_dir + "/policy_supervised_" + relation + ".pth")
    print('model saved')


if __name__ == "__main__":
    setproctitle("kimjw sl_policy")
    graphpath = dataPath + "tasks/" + relation + "/" + "graph.txt"
    relationPath = dataPath + "tasks/" + relation + "/" + "train_pos"
    logger.info("mode == " + mode)
    train()
