import numpy as np
import math


class DATA(object):
    def __init__(self, seqlen, separate_char):
        self.separate_char = separate_char
        self.seqlen = seqlen

    '''
    data format:
    length
    KC sequence
    answer sequence
    exercise sequence
    it sequence
    at sequence
    '''

    def load_data(self, path):
        f_data = open(path, 'r')
        a_data = []
        s_data = []
        e_data = []
        it_data = []
        at_data = []

        fru_data=[]
        conf_data=[]
        conc_data=[]
        bor_data=[]

        sd_data =[]
        qd_data =[]

        tp_data=[]

        stu_data=[]
        pre_data=[]
        att_data=[]

        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 16 != 0:
                line_data = line.split(self.separate_char)
                if len(line_data[len(line_data) - 1]) == 0:
                    line_data = line_data[:-1]

            if lineID % 16 == 2:
                A = line_data
            elif lineID % 16 ==1:
                S =line_data
            elif lineID % 16 == 3:
                E = line_data
            elif lineID % 16 == 4:
                IT = line_data
            elif lineID % 16 == 5:
                AT = line_data
            elif lineID % 16 ==6:
                BOR =line_data
            elif lineID % 16 == 7:
                CONC = line_data
            elif lineID % 16 == 8:
                CONF = line_data
            elif lineID % 16 == 9:
                FRU = line_data
            elif lineID % 16 == 10:
                QD = line_data
            elif lineID % 16 == 11:
                SD = line_data
            elif lineID % 16 == 12:
                TP = line_data
            elif lineID % 16 == 13:
                STU = line_data
            elif lineID % 16 == 14:
                PRE = line_data
            elif lineID % 16 == 15:
                ATT = line_data

                # start split the data
                n_split = 1
                total_len = len(A)
                if total_len > self.seqlen:
                    n_split = math.floor(len(A) / self.seqlen)
                    if total_len % self.seqlen:
                        n_split = n_split + 1

                for k in range(n_split):
                    answer_sequence = []
                    exercise_sequence = []
                    skill_sequence =[]
                    it_sequence = []
                    at_sequence = []

                    fru_sequence = []
                    conf_sequence = []
                    conc_sequence = []
                    bor_sequence = []

                    sd_sequence = []
                    qd_sequence = []
                    tp_sequence = []

                    stu_sequence = []
                    pre_sequence = []
                    att_sequence = []


                    if k == n_split - 1:
                        end_index = total_len
                    else:
                        end_index = (k + 1) * self.seqlen
                    # choose the sequence length is larger than 2
                    if end_index - k * self.seqlen > 2:
                        for i in range(k * self.seqlen, end_index):
                            answer_sequence.append(int(A[i]))
                            exercise_sequence.append(int(E[i]))
                            skill_sequence.append(int(S[i]))
                            it_sequence.append(int(IT[i]))
                            at_sequence.append(int(AT[i]))

                            bor_sequence.append(float(BOR[i]))
                            conc_sequence.append(float(CONC[i]))
                            conf_sequence.append(float(CONF[i]))
                            fru_sequence.append(float(FRU[i]))
                            sd_sequence.append(int(SD[i]))
                            qd_sequence.append(int(QD[i]))
                            tp_sequence.append(int(TP[i]))
                            stu_sequence.append(int(STU[i]))

                            pre_sequence.append(int(PRE[i]))
                            att_sequence.append(int(ATT[i]))


                        # print('instance:-->', len(instance),instance)
                        a_data.append(answer_sequence)
                        e_data.append(exercise_sequence)
                        s_data.append(skill_sequence)
                        it_data.append(it_sequence)
                        at_data.append(at_sequence)
                        bor_data.append(bor_sequence)
                        conc_data.append(conc_sequence)
                        conf_data.append(conf_sequence)
                        fru_data.append(fru_sequence)
                        qd_data.append(qd_sequence)  
                        sd_data.append(sd_sequence)
                        tp_data.append(tp_sequence)   
                        stu_data.append(stu_sequence)
                        pre_data.append(pre_sequence)
                        att_data.append(att_sequence)



        f_data.close()
        # data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        a_dataArray = np.zeros((len(a_data), self.seqlen))
        for j in range(len(a_data)):
            dat = a_data[j]
            a_dataArray[j, :len(dat)] = dat

        e_dataArray = np.zeros((len(e_data), self.seqlen))
        for j in range(len(e_data)):
            dat = e_data[j]
            e_dataArray[j, :len(dat)] = dat

        it_dataArray = np.zeros((len(it_data), self.seqlen))
        for j in range(len(it_data)):
            dat = it_data[j]
            it_dataArray[j, :len(dat)] = dat

        s_dataArray = np.zeros((len(s_data), self.seqlen))
        for j in range(len(s_data)):
            dat = s_data[j]
            s_dataArray[j, :len(dat)] = dat

        at_dataArray = np.zeros((len(at_data), self.seqlen))
        for j in range(len(at_data)):
            dat = at_data[j]
            at_dataArray[j, :len(dat)] = dat


        fru_dataArray = np.zeros((len(fru_data), self.seqlen))
        for j in range(len(fru_data)):
            dat = fru_data[j]
            fru_dataArray[j, :len(dat)] = dat 
        conf_dataArray = np.zeros((len(conf_data), self.seqlen))
        for j in range(len(conf_data)):
            dat = conf_data[j]
            conf_dataArray[j, :len(dat)] = dat
        conc_dataArray = np.zeros((len(conc_data), self.seqlen))
        for j in range(len(conc_data)):
            dat = conc_data[j]
            conc_dataArray[j, :len(dat)] = dat
        bor_dataArray = np.zeros((len(bor_data), self.seqlen))
        for j in range(len(bor_data)):
            dat = bor_data[j]
            bor_dataArray[j, :len(dat)] = dat


        sd_dataArray = np.zeros((len(sd_data), self.seqlen))
        for j in range(len(sd_data)):
            dat = sd_data[j]
            sd_dataArray[j, :len(dat)] = dat
        qd_dataArray = np.zeros((len(qd_data), self.seqlen))
        for j in range(len(qd_data)):
            dat = qd_data[j]
            qd_dataArray[j, :len(dat)] = dat   
        tp_dataArray = np.zeros((len(tp_data), self.seqlen))
        for j in range(len(tp_data)):
            dat = tp_data[j]
            tp_dataArray[j, :len(dat)] = dat 

        stu_dataArray = np.zeros((len(stu_data), self.seqlen))
        for j in range(len(stu_data)):
            dat = stu_data[j]
            stu_dataArray[j, :len(dat)] = dat 

        pre_dataArray = np.zeros((len(pre_data), self.seqlen))
        for j in range(len(pre_data)):
            dat = pre_data[j]
            pre_dataArray[j, :len(dat)] = dat 
        att_dataArray = np.zeros((len(att_data), self.seqlen))
        for j in range(len(att_data)):
            dat = att_data[j]
            att_dataArray[j, :len(dat)] = dat 

        return a_dataArray, e_dataArray,s_dataArray ,it_dataArray, at_dataArray,bor_dataArray,conc_dataArray,conf_dataArray,fru_dataArray,qd_dataArray,sd_dataArray,tp_dataArray,stu_dataArray,pre_dataArray,att_dataArray
