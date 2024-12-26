# coding: utf-8

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DEKTNet(nn.Module):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, d_m, q_matrix, dropout=0.2):
        super(DEKTNet, self).__init__()
        self.d_m = d_m
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        self.q_matrix = q_matrix
        self.n_question = n_question


        self.at_embed = nn.Embedding(n_at + 10, d_k)
        torch.nn.init.xavier_uniform_(self.at_embed.weight)
        self.it_embed = nn.Embedding(n_it + 10, d_k)
        torch.nn.init.xavier_uniform_(self.it_embed.weight)
        self.e_embed = nn.Embedding(n_exercise + 10, d_k)
        torch.nn.init.xavier_uniform_(self.e_embed.weight)
        self.s_embed = nn.Embedding(n_question + 10, d_k)
        torch.nn.init.xavier_uniform_(self.s_embed.weight)

        self.att_embed = nn.Embedding(1000, d_k)
        torch.nn.init.xavier_uniform_(self.att_embed.weight)

        self.fru_embed = nn.Embedding(self.d_m, d_k)
        torch.nn.init.xavier_uniform_(self.fru_embed.weight)
        self.conf_embed = nn.Embedding(self.d_m, d_k)
        torch.nn.init.xavier_uniform_(self.conf_embed.weight)
        self.conc_embed = nn.Embedding(self.d_m, d_k)
        torch.nn.init.xavier_uniform_(self.conc_embed.weight)
        self.bor_embed = nn.Embedding(self.d_m, d_k)
        torch.nn.init.xavier_uniform_(self.bor_embed.weight)


        self.sd_embed = nn.Embedding(50, d_k)
        torch.nn.init.xavier_uniform_(self.sd_embed.weight)
        self.qd_embed = nn.Embedding(80, d_k)
        torch.nn.init.xavier_uniform_(self.qd_embed.weight)
        self.tp_embed = nn.Embedding(17, d_k)
        torch.nn.init.xavier_uniform_(self.tp_embed.weight)


        self.linear_1 = nn.Linear(4*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_1.weight) 
        self.linear_2 = nn.Linear(2*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_3 = nn.Linear(3*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        self.linear_4 = nn.Linear(3 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)
        self.linear_5 = nn.Linear(6*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)

        self.linear_a = nn.Linear(4*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_a.weight)
        self.linear_d = nn.Linear(2*d_k, 3)
        torch.nn.init.xavier_uniform_(self.linear_d.weight)
        self.linear_e = nn.Linear(2*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_e.weight)

        self.linear_attblock = nn.Linear(d_k*5, d_k)
        torch.nn.init.xavier_uniform_(self.linear_attblock.weight)


        self.linear_emo = nn.Linear(4*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_emo.weight)


        self.linear_bor = nn.Linear(32, 1)
        torch.nn.init.xavier_uniform_(self.linear_bor.weight)
        self.linear_conc = nn.Linear(32, 1)
        torch.nn.init.xavier_uniform_(self.linear_conc.weight)
        self.linear_conf = nn.Linear(32, 1)
        torch.nn.init.xavier_uniform_(self.linear_conf.weight)
        self.linear_fru = nn.Linear(32, 1)
        torch.nn.init.xavier_uniform_(self.linear_fru.weight)


        self.linear_sm = nn.Linear(2*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_sm.weight)

        self.linear_siga = nn.Linear(4*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_siga.weight)
        self.linear_tana = nn.Linear(4*d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_tana.weight)


        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.rulu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        


    def forward(self, e_data,s_data, at_data, a_data, it_data ,bor_data,conc_data,conf_data,fru_data,qd_data,sd_data,tp_data,stu_data,pre_data,att_data):
        batch_size, seq_len = e_data.size(0), e_data.size(1)

        e_embed_data = self.e_embed(e_data)
        s_embed_data = self.s_embed(s_data)
        at_embed_data = self.at_embed(at_data)
        it_embed_data = self.it_embed(it_data)

        qd_embed_data = self.qd_embed(qd_data)
        sd_embed_data = self.sd_embed(sd_data)
        tp_embed_data = self.tp_embed(tp_data)
        att_embed_data = self.att_embed(att_data)

        # 定义映射函数
        def map_to_class(tensor):
            return (tensor * 5000).to(torch.int)

        ## 将连续情绪值进行离散并嵌入
        fru_datas = map_to_class(fru_data)
        conf_datas = map_to_class(conf_data)
        conc_datas = map_to_class(conc_data)
        bor_datas = map_to_class(bor_data)
        fru_embed_data = self.fru_embed(fru_datas)
        conf_embed_data = self.conf_embed(conf_datas)
        conc_embed_data = self.conc_embed(conc_datas)
        bor_embed_data = self.bor_embed(bor_datas)



        a_embedd_data = a_data.view(-1, 1).repeat(1, self.d_k).view(batch_size, -1, self.d_k)
        emo_embed_data = self.linear_emo(torch.cat((fru_embed_data,conf_embed_data,conc_embed_data,bor_embed_data), 2))
        attblock = self.linear_attblock(torch.cat((e_embed_data,s_embed_data,qd_embed_data,sd_embed_data,tp_embed_data), 2))
        all_learning = self.linear_1(torch.cat((at_embed_data,s_embed_data,a_embedd_data,e_embed_data), 2))
        

        h_pre = nn.init.xavier_uniform_(torch.zeros(self.n_question , self.d_k)).repeat(batch_size, 1, 1).to(device)
        h_tilde_pre = None
        affect_h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size,self.d_k)).to(device)


        pred      = torch.zeros(batch_size, seq_len).to(device)
        pred_bor  = torch.zeros(batch_size, seq_len).to(device)
        pred_conc = torch.zeros(batch_size, seq_len).to(device)
        pred_conf = torch.zeros(batch_size, seq_len).to(device)
        pred_fru  = torch.zeros(batch_size, seq_len).to(device)


        for t in range(0, seq_len - 1):
            e = e_data[:, t]
            q_e = self.q_matrix[e].view(batch_size, 1, -1)

            a = a_embedd_data[:,t]
            it = it_embed_data[:, t]
            at = at_embed_data[:,t]
            block_a = attblock[:,t] 
            block_b = attblock[:,t+1]        
            emo = emo_embed_data[:, t]
            a_e_d = att_embed_data[:,t]

            # es_t block
            relation_matirx = torch.stack([a,a_e_d,at,it],axis=1)
            correlation_matrix = torch.sum(emo.unsqueeze(1) * relation_matirx, dim=2)
            softmax_result = F.softmax(correlation_matrix, dim=1)
            result = torch.matmul(softmax_result.unsqueeze(1), relation_matirx).squeeze(dim=1)


            # 1. Knowledge State Boosting Module 
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k)
            learning = all_learning[:, t]
            learning_gain = self.linear_2(torch.cat((learning, h_tilde_pre), 1))
            learning_gain = self.tanh(learning_gain)
            gamma_l = self.linear_3(torch.cat(( learning,h_tilde_pre,emo), 1))
            gamma_l = self.sig(gamma_l)
            LG = gamma_l * ((learning_gain + 1) / 2)
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))


            n_skill1 = LG_tilde.size(1)
            gamma_f = self.sig(self.linear_4(torch.cat((
                h_pre,
                LG.repeat(1, n_skill1).view(batch_size, -1, self.d_k),
                it.repeat(1, n_skill1).view(batch_size, -1, self.d_k)
            ), 2)))
            h = LG_tilde + gamma_f * h_pre


            # 2. Emotional State Tracing Module

            affect = self.sig(self.linear_a(torch.cat((emo,block_a,result,a), 1)))
            fa = self.linear_tana(torch.cat((affect,result,a,affect_h_pre), 1))
            fa_gain = self.tanh(fa)
            gt = self.linear_siga(torch.cat(( affect,result,a,affect_h_pre), 1))
            gt_l = self.sig(gt)
            FLG = gt_l * fa_gain
            w1 = F.softmax(LG*FLG, dim=1)
            affect_h = (1-w1)*FLG + w1*affect_h_pre


            # 3. Emotion Prediction Based on Personalized Emotional State
            x = self.sig(self.linear_e(torch.cat(( affect_h , block_b ), 1)))

            x_four = x.view(batch_size,4,-1)
            bor_x  = torch.squeeze( x_four[:,0,:], dim=1)
            conc_x = torch.squeeze( x_four[:,1,:], dim=1)
            conf_x = torch.squeeze( x_four[:,2,:], dim=1)
            fru_x  = torch.squeeze( x_four[:,3,:], dim=1)

           
            x_bor = self.sig(self.linear_bor(bor_x)).squeeze(1)
            x_conc= self.sig(self.linear_conc(conc_x)).squeeze(1)
            x_conf= self.sig(self.linear_conf(conf_x)).squeeze(1)
            x_fru = self.sig(self.linear_fru(fru_x)).squeeze(1)


            fru = self.fru_embed(map_to_class(x_fru))
            conf = self.conf_embed(map_to_class(x_conf))
            conc = self.conc_embed(map_to_class(x_conc))
            bor = self.bor_embed(map_to_class(x_bor))


            # 3. Emotion-Boosted Response Prediction
            h_tilde = self.q_matrix[e_data[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, self.d_k)
            condition = (x > 0.90) & (x< 0.1)  # 创建一个布尔张量，标记极端情绪
            updata_h = torch.where(condition,  h_tilde, h_tilde*x)  
            y = self.sig(self.linear_5(torch.cat((fru,conf ,conc,bor , updata_h,block_b*x), 1))).sum(1) / self.d_k 

            pred[:, t + 1]   =  y
            pred_bor[:,t+1]  = x_bor 
            pred_conc[:,t+1] = x_conc
            pred_conf[:,t+1] = x_conf
            pred_fru[:,t+1]  = x_fru

            # prepare for next prediction
            h_pre = h
            h_tilde_pre = h_tilde

        return pred , pred_bor, pred_conc, pred_conf, pred_fru