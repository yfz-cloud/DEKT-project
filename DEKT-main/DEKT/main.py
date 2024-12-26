import torch
import logging
import numpy as np
from load_data import DATA
import pickle
import os, random
from dekt_train import DEKT

def set_random_seed(seed: int) -> None:
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


# seed = 454523  auc: 0.839844, accuracy: 0.768629,rmse: 0.395355,r2: 0.333623
# seed = 2451521  auc: 0.840092, accuracy: 0.769173,rmse: 0.395511,r2: 0.333097  epoch=25,
seed = 545194
set_random_seed(seed)


def generate_q_matrix(path, n_skill, n_problem, gamma=0.0):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            problem2skill = eval(line)
    q_matrix = np.zeros((n_problem , n_skill )) + gamma
    for p in problem2skill.keys():
        q_matrix[p][problem2skill[p]] = 1
    return q_matrix

batch_size = 32
n_at = 1305
n_it = 2844 
n_question = 102
n_exercise = 3156
seqlen = 500
d_m = 5000
d_k = 128
d_a = 50
d_e = 128
q_gamma = 0.03
dropout = 0.2

q_matrix = generate_q_matrix(
    '../data/anonymized_full_release_competition_dataset/problem2skill',
    n_question, n_exercise,
    q_gamma
)
dat = DATA(seqlen=seqlen, separate_char=',')

logging.getLogger().setLevel(logging.INFO)


# # k-fold cross validation
# k, train_auc_sum, valid_auc_sum = 5, .0, .0
# for i in range(k):
#     dekt = DEKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout)
#     train_data = dat.load_data('../data/anonymized_full_release_competition_dataset/train' + str(i) + '.txt')
#     valid_data = dat.load_data('../data/anonymized_full_release_competition_dataset/valid' + str(i) + '.txt')
#     best_train_auc, best_valid_auc = dekt.train(train_data, valid_data, epoch=20, lr=0.003, lr_decay_step=10)
#     print('fold %d, train auc %f, valid auc %f' % (i, best_train_auc, best_valid_auc))
#     train_auc_sum += best_train_auc
#     valid_auc_sum += best_valid_auc
# print('%d-fold validation: avg of best train auc %f, avg of best valid auc %f' % (k, train_auc_sum / k, valid_auc_sum / k))


# train and pred
train_data = dat.load_data('../data/anonymized_full_release_competition_dataset/train0.txt')
valid_data = dat.load_data('../data/anonymized_full_release_competition_dataset/valid0.txt')
test_data = dat.load_data('../data/anonymized_full_release_competition_dataset/test.txt')
dekt = DEKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, d_m, q_matrix, batch_size, dropout)
dekt.train(train_data, valid_data, epoch=25, lr=0.003, lr_decay_step=10)


dekt.load("params/dekt.params")
_, auc, accuracy,rmse,r2 = dekt.eval(test_data)
print("auc: %.6f, accuracy: %.6f,rmse: %.6f,r2: %.6f" % (auc, accuracy,rmse,r2))

