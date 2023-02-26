# python
import torch
from torch.nn.functional import one_hot
from itertools import count
import sys
from setproctitle import setproctitle

# framework
from networks import PolicyNeuralNetTorch
from env import Env
from utils import *


# relation = sys.argv[1]
relation = "concept_agentbelongstoorganization"
relationPath = dataPath + "tasks/" + relation + "/" + "train_pos"

max_num_samples = 500  # original code : 500

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train():
    f = open(relationPath)
    train_data = f.readlines()
    f.close()

    num_samples = min(len(train_data), max_num_samples)

    # creating initial supervised-trained model and saving that
    model = PolicyNeuralNetTorch().to(device)
    # implements L2 regularization by weight_decay (TODO needs verification)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # TODO deleted weight_decay

    entity2vec = np.loadtxt(dataPath + "entity2vec.bern")

    relation2vec = np.loadtxt(dataPath + "relation2vec.bern")
    entity2id, relation2id, relations = open_entity2id_and_relation2id()

    with open(dataPath + 'kb_env_rl.txt') as f:
        kb_all = f.readlines()
    kb = relation_to_kb(relation)
    eliminated_kb = []
    concept = "concept:" + relation
    for line in kb_all:
        rel = line.split()[2]
        if rel != concept and rel != concept + "_inv":
            eliminated_kb.append(line)

    # original code
    for episode in range(num_samples):
        print(f"Episode {episode}")
        print("Training Sample:", train_data[episode][:-1])

        env = Env(entity2vec, relation2vec, kb_all, eliminated_kb, entity2id, relation2id, relations, train_data[episode])
        sample = train_data[episode].split()

        try:
            good_episodes = teacher(sample[0], sample[1], 5, env, kb)
        except Exception as e:
            print("Cannot find a path")
            continue

        for item in good_episodes:
            state_batch = []
            action_batch = []
            for t, transition in enumerate(item):
                state_batch.append(transition.state)  # state: vector
                action_batch.append(transition.action)  # action: ID
            state_batch = np.squeeze(state_batch)
            state_batch = np.reshape(state_batch, [-1, state_dim])  # (3, 200)

            # ported code
            state_batch = torch.tensor(state_batch, dtype=torch.float).to(device)  # (3, 200)
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

    torch.save(model, 'torchmodels/policy_supervised_' + relation + ".pth")
    torch.save(optimizer, 'torchmodels/policy_supervised_' + relation + "_opt.pth")
    print('model saved')


if __name__ == "__main__":
    setproctitle("kimjw supervised")
    train()












