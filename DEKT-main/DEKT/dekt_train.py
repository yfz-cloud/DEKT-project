# coding: utf-8


import math
import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import tqdm
from scipy.stats import pearsonr
from DEKTNet import DEKTNet
from sklearn.metrics import r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def etl(*args, **kwargs) -> ...:  # pragma: no cover
    """
    extract - transform - load
    """
    pass


def train(*args, **kwargs) -> ...:  # pragma: no cover
    pass


def evaluate(*args, **kwargs) -> ...:  # pragma: no cover
    pass


class KTM(object):
    def __init__(self, *args, **kwargs) -> ...:
        pass

    def train(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def eval(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def save(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def load(self, *args, **kwargs) -> ...:
        raise NotImplementedError

def compute_accuracy_multi_class(all_target, all_pred):
    # # 将模型输出的每个样本的概率最大的类别作为预测类别
    # predicted_labels = np.argmax(all_pred, axis=1)
    # 计算准确率
    accuracy = metrics.accuracy_score(all_target, all_pred)
    return accuracy


def custom_cross_entropy_loss_multi_class(predictions, targets):
    epsilon = 1e-10  # 用于防止对数中的除零错误
    # 将预测概率通过 softmax 转换
    predictions_exp = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    predictions_softmax = predictions_exp / np.sum(predictions_exp, axis=1, keepdims=True)
    # 防止概率为0的情况
    predictions_softmax = np.maximum(epsilon, predictions_softmax)
    predictions_softmax = np.minimum(1 - epsilon, predictions_softmax)
    # 使用 numpy 的广播进行计算
    loss = -np.sum(np.log(predictions_softmax[np.arange(len(targets)), targets]))
    # 对所有样本取平均
    loss /= len(targets)
    return loss


def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)

def compute_rmse(all_target, all_pred):
    # 计算均方根误差
    return np.sqrt(np.mean((all_target - all_pred)**2))

def compute_mse(all_target, all_pred):
    # 计算均方根误差
    return np.mean((all_target - all_pred)**2)




def train_one_epoch(net, optimizer, criterion, criterion_mse,criterion_cr,batch_size, a_data, e_data,s_data, it_data, at_data ,bor_data,conc_data,conf_data,fru_data,qd_data,sd_data,tp_data,stu_data,pre_data,att_data):
    net.train()
    n = int(math.ceil(len(e_data) / batch_size))
    shuffled_ind = np.arange(e_data.shape[0])
    np.random.shuffle(shuffled_ind)
    e_data = e_data[shuffled_ind]
    s_data = s_data[shuffled_ind]
    at_data = at_data[shuffled_ind]
    a_data = a_data[shuffled_ind]
    it_data = it_data[shuffled_ind]
    bor_data = bor_data[shuffled_ind]
    conc_data = conc_data[shuffled_ind]
    conf_data = conf_data[shuffled_ind]
    fru_data = fru_data[shuffled_ind]
    # emo_data = emo_data[shuffled_ind]
    sd_data = sd_data[shuffled_ind]
    qd_data = qd_data[shuffled_ind]
    tp_data = tp_data[shuffled_ind]
    stu_data = stu_data[shuffled_ind]
    pre_data = pre_data[shuffled_ind]
    att_data = att_data[shuffled_ind]
    


    pred_list = []
    target_list = []
    pred_rmse = []



    for idx in tqdm.tqdm(range(n), 'Training'):
        optimizer.zero_grad()

        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        s_one_seq = s_data[idx * batch_size: (idx + 1) * batch_size, :]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]
        bor_one_seq = bor_data[idx * batch_size: (idx + 1) * batch_size, :]
        conc_one_seq = conc_data[idx * batch_size: (idx + 1) * batch_size, :]
        conf_one_seq = conf_data[idx * batch_size: (idx + 1) * batch_size, :]
        fru_one_seq = fru_data[idx * batch_size: (idx + 1) * batch_size, :]
        # emo_one_seq = emo_data[idx * batch_size: (idx + 1) * batch_size, :]
        sd_one_seq = sd_data[idx * batch_size: (idx + 1) * batch_size, :]
        qd_one_seq = qd_data[idx * batch_size: (idx + 1) * batch_size, :]
        tp_one_seq = tp_data[idx * batch_size: (idx + 1) * batch_size, :]
        stu_one_seq = stu_data[idx * batch_size: (idx + 1) * batch_size, :]
        pre_one_seq = pre_data[idx * batch_size: (idx + 1) * batch_size, :]
        att_one_seq = att_data[idx * batch_size: (idx + 1) * batch_size, :]
        


        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_s = torch.from_numpy(s_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)
        input_bor = torch.from_numpy(bor_one_seq).double().to(device)
        input_conc = torch.from_numpy(conc_one_seq).double().to(device)
        input_conf = torch.from_numpy(conf_one_seq).double().to(device)
        input_fru = torch.from_numpy(fru_one_seq).double().to(device)
        input_sd = torch.from_numpy(sd_one_seq).long().to(device)
        input_qd = torch.from_numpy(qd_one_seq).long().to(device)
        input_tp = torch.from_numpy(tp_one_seq).long().to(device)
        input_stu = torch.from_numpy(stu_one_seq).long().to(device)
        input_pre = torch.from_numpy(pre_one_seq).long().to(device)
        input_att = torch.from_numpy(att_one_seq).long().to(device) 


        pred , pred_bor, pred_conc, pred_conf, pred_fru  = net(input_e,input_s, input_at, target, input_it,input_bor,input_conc,input_conf,input_fru,input_qd,input_sd,input_tp,input_stu,input_pre,input_att)
        
        mask = input_e[:, 1:] > 0
        masked_pred = pred[:, 1:][mask]
        masked_truth = target[:, 1:][mask]

        mask_pred_bor = pred_bor[:,1:,][mask]
        mask_truth_bor = input_bor[:,1:][mask]

        mask_pred_conc = pred_conc[:,1:,][mask]
        mask_truth_conc = input_conc[:,1:][mask]

        mask_pred_conf = pred_conf[:,1:,][mask]
        mask_truth_conf = input_conf[:,1:][mask]

        mask_pred_fru = pred_fru[:,1:,][mask]
        mask_truth_fru = input_fru[:,1:][mask]


        loss1 = criterion(masked_pred, masked_truth).sum()

        
        loss21 = criterion_mse(mask_pred_bor,mask_truth_bor.float()).sum()  
        loss22 = criterion_mse(mask_pred_conc,mask_truth_conc.float()).sum()    
        loss23 = criterion_mse(mask_pred_conf,mask_truth_conf.float()).sum()         
        loss24 = criterion_mse(mask_pred_fru,mask_truth_fru.float()).sum()   

        loss2 = loss21+loss22+loss23+loss24

        loss = loss1 + 20*loss2
        # 0.1  auc: 0.838198, accuracy: 0.767201,rmse: 0.397254,r2: 0.327207
        # 0.5  auc: 0.839944, accuracy: 0.768986,rmse: 0.396121,r2: 0.331040
        # 1.5  auc: 0.840640, accuracy: 0.769507,rmse: 0.395222,r2: 0.334073
        # 2.0  auc: 0.840902, accuracy: 0.769501,rmse: 0.395317,r2: 0.333753
        # 10   auc: 0.840541, accuracy: 0.769676,rmse: 0.395291,r2: 0.333838
        # 20   auc: 0.839223, accuracy: 0.767909,rmse: 0.395844,r2: 0.331975
        # 50   auc: 0.837054, accuracy: 0.765902,rmse: 0.397381,r2: 0.326776
        loss.backward()
        optimizer.step()

        # y loss
        masked_pred  = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()
        pred_list.append(masked_pred)  # 多个array
        target_list.append(masked_truth)
        pred_rmse += list(masked_pred)


    # y
    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    all_pred_rmse = np.array(pred_rmse)

    # y
    loss11 = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    rmse = compute_rmse(all_target,all_pred_rmse)
    r2 = r2_score(all_target, all_pred_rmse)


    return loss11, auc, accuracy, rmse, r2


def test_one_epoch(net, batch_size, a_data, e_data,s_data, it_data, at_data,bor_data,conc_data,conf_data,fru_data,qd_data,sd_data,tp_data,stu_data,pre_data,att_data):
    net.eval()
    n = int(math.ceil(len(e_data) / batch_size))

    pred_list = []
    target_list = []
    pred_rmse = []

    
    for idx in tqdm.tqdm(range(n), 'Testing'):

        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        s_one_seq = s_data[idx * batch_size: (idx + 1) * batch_size, :]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]
        bor_one_seq = bor_data[idx * batch_size: (idx + 1) * batch_size, :]
        conc_one_seq = conc_data[idx * batch_size: (idx + 1) * batch_size, :]
        conf_one_seq = conf_data[idx * batch_size: (idx + 1) * batch_size, :]
        fru_one_seq = fru_data[idx * batch_size: (idx + 1) * batch_size, :]
        sd_one_seq = sd_data[idx * batch_size: (idx + 1) * batch_size, :]
        qd_one_seq = qd_data[idx * batch_size: (idx + 1) * batch_size, :]
        tp_one_seq = tp_data[idx * batch_size: (idx + 1) * batch_size, :]
        stu_one_seq = stu_data[idx * batch_size: (idx + 1) * batch_size, :]
        pre_one_seq = pre_data[idx * batch_size: (idx + 1) * batch_size, :]
        att_one_seq = att_data[idx * batch_size: (idx + 1) * batch_size, :]
  

        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_s = torch.from_numpy(s_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)
        input_bor = torch.from_numpy(bor_one_seq).double().to(device)
        input_conc = torch.from_numpy(conc_one_seq).double().to(device)
        input_conf = torch.from_numpy(conf_one_seq).double().to(device)
        input_fru = torch.from_numpy(fru_one_seq).double().to(device)
        input_sd = torch.from_numpy(sd_one_seq).long().to(device)
        input_qd = torch.from_numpy(qd_one_seq).long().to(device)
        input_tp = torch.from_numpy(tp_one_seq).long().to(device)
        input_stu = torch.from_numpy(stu_one_seq).long().to(device)
        input_pre = torch.from_numpy(pre_one_seq).long().to(device)
        input_att = torch.from_numpy(att_one_seq).long().to(device)


        with torch.no_grad():
            pred , pred_bor, pred_conc, pred_conf, pred_fru = net(input_e,input_s, input_at, target, input_it,input_bor,input_conc,input_conf,input_fru,input_qd,input_sd,input_tp,input_stu,input_pre,input_att)
 
            mask = input_e[:, 1:] > 0
            masked_pred = pred[:, 1:][mask].detach().cpu().numpy()
            masked_truth = target[:, 1:][mask].detach().cpu().numpy()

            
            pred_list.append(masked_pred)
            target_list.append(masked_truth)
            pred_rmse += list(masked_pred)


    # y
    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    all_pred_rmse = np.array(pred_rmse)


    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    rmse = compute_rmse(all_target,all_pred_rmse )
    r2 = r2_score(all_target, all_pred_rmse)

    return loss, auc, accuracy, rmse, r2 

class DEKT(KTM):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, d_m, q_matrix, batch_size, dropout=0.2):
        super(DEKT, self).__init__()
        q_matrix = torch.from_numpy(q_matrix).float().to(device)
        self.dekt_net = DEKTNet(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, d_m, q_matrix, dropout).to(device)
        self.batch_size = batch_size

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.002, lr_decay_step=15, lr_decay_rate=0.5) -> ...:
        optimizer = torch.optim.Adam(self.dekt_net.parameters(), lr=lr, eps=1e-8, betas=(0.1, 0.999), weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=lr_decay_rate)

        criterion = nn.BCELoss(reduction='none')
        criterion_cr = nn.CrossEntropyLoss(reduction='none')
        criterion_mse = nn.MSELoss(reduction='none')       
        best_train_auc, best_test_auc = .0, .0

        for idx in range(epoch):
            
            train_loss, train_auc, train_accuracy,train_rmse,r2  = train_one_epoch(self.dekt_net, optimizer, criterion, criterion_mse,criterion_cr,
                                                                    self.batch_size, *train_data)

            print("[Epoch %d] LogisticLossa: %.6f  " % (idx, train_loss))
            if train_auc > best_train_auc:
                best_train_auc = train_auc

            scheduler.step()

            if test_data is not None:
                test_loss, test_auc, test_accuracy ,test_rmse,test_r2 = self.eval(test_data)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f, r2: %.6f" % (idx, test_auc, test_accuracy, test_rmse,test_r2))
                if test_auc > best_test_auc:
                    torch.save(self.dekt_net.state_dict(), "params/dekt.params")
                    print(f"此时的valida auc:{test_auc}")
                    print(f"目前最好的epoch是{idx+1}")
                    best_test_auc = test_auc

        return best_train_auc, best_test_auc



    def eval(self, test_data) -> ...:
        self.dekt_net.eval()
        return test_one_epoch(self.dekt_net, self.batch_size, *test_data)

    def save(self, filepath) -> ...:
        torch.save(self.dekt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.dekt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
