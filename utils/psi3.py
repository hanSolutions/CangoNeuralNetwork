# -*- coding: utf8 -*-
import sys
import operator
import pandas as pd
import numpy as np 

class Bin:
    def __init__(self, ng1 = 0.5, nb1 = 0.5, ng2 = 0.5, nb2 = 0.5, \
                 lo = 0.0, hi = 0.0):
        self.pg1 = 0.0
        self.pb1 = 0.0
        self.pg2 = 0.0 
        self.pb2 = 0.0
        self.ng1 = ng1
        self.nb1 = nb1
        self.ng2 = ng2
        self.nb2 = nb2
        self.lo = lo
        self.hi = hi
        
    def shouldInclude(self, probability):
        return self.lo <= probability < self.hi
    
    def inc(self, label, f1f2):
        if label == 0:
            if f1f2 == 1:
                self.ng1 += 1
            elif f1f2 == 2:
                self.ng2 += 1
            else:
                print ("File Error %d" %f1f2)
        elif label == 1:
            if f1f2 == 1:
                self.nb1 += 1
            elif f1f2 == 2:
                self.nb2 += 1
            else:
                print ("File Error %d" %f1f2)
        else:
            print ("Label Error %d" %label )    
            
    def psig(self, tg1, tg2):
        if self.ng1 == 0.5 and self.ng2 == 0.5:
            return 0
        self.pg1 = float(self.ng1) / tg1
        self.pg2 = float(self.ng2) / tg2
        return (self.pg1 - self.pg2) * np.log(float(self.pg1) / self.pg2)
    
    def psib(self, tb1, tb2):
        if self.nb1 == 0.5 and self.nb2 == 0.5:
            return 0
        self.pb1 = float(self.nb1) / tb1
        self.pb2 = float(self.nb2) / tb2
        return (self.pb1 - self.pb2) * np.log(float(self.pb1) / self.pb2)
    
class LastBin(Bin):
    def shouldInclude(self, value):
        return self.lo <= value <= self.hi


class PSI:
    def __init__(self, BIN_INIT_VAL = 0.5, BIN_PROB_LEN = 0.05):
        self.BIN_INIT_VAL = 0.5
        self.BIN_PROB_LEN = 0.05
        self.NUM_BINS = int(1.0 / float(BIN_PROB_LEN))
        
        self.bins = []
        val = 0.0
        for i in range(0, self.NUM_BINS - 1):
            self.bins.append(Bin(self.BIN_INIT_VAL, self.BIN_INIT_VAL, self.BIN_INIT_VAL, self.BIN_INIT_VAL, 
                                 val, val + self.BIN_PROB_LEN))
            val += BIN_PROB_LEN
        self.bins.append(LastBin(self.BIN_INIT_VAL, self.BIN_INIT_VAL, self.BIN_INIT_VAL, self.BIN_INIT_VAL, 
                                 val, val + self.BIN_PROB_LEN))
             
        self.tg = [0.0, 0.0]
        self.tb = [0.0, 0.0] 
            
#     def calcPSIDelegate(self, fName1 = None, f1c1 = None, f1c2 = None, fName2 = None, f2c1 = None, f2c2 = None):
#             
#         df1 = pd.read_csv(fName1)
#         df2 = pd.read_csv(fName2)
#         label1 = df1[f1c1].tolist()
#         prob1 = df1[f1c2].tolist()
#         label2 = df2[f2c1].tolist()
#         prob2 = df2[f2c2].tolist()
#         assert len(label1) == len(prob1), "Length Error, File 1 %d, %d" % (len(label1), len(prob1))
#         assert len(label2) == len(prob2), "Length Error, File 2 %d, %d" % (len(label2), len(prob2))
#         self.calcPSI(label1, prob1, label2, prob2)

        
    def calcPSI(self, label1, prob1, label2, prob2):
        assert len(label1) == len(prob1), "Length Error label1 %d, prob1 %d" % (len(label1), len(prob1))
        assert len(label2) == len(prob2), "Length Error label2 %d, prob2 %d" % (len(label2), len(prob2))
        
        for i in range(0, len(label1)):
            l = float(label1[i])
            p = float(prob1[i])
            if l == 0:
                self.tg[0] += 1
            elif l == 1:
                self.tb[0] += 1
            else:
                print ("Label Error %d" % l )
            for bin in self.bins:
                if bin.shouldInclude(p):
                    bin.inc(l, 1)
                    break     
                   
        for i in range(0, len(label2)):
            l = float(label2[i])
            p = float(prob2[i])
            if l == 0:
                self.tg[1] += 1
            elif l == 1:
                self.tb[1] += 1
            else:
                print ("Label Error %d" % l )
            for bin in self.bins:
                if bin.shouldInclude(p):
                    bin.inc(l, 2)
                    break
        return self.psi_f()

    def psi_f(self):
        psi = 0.0
        for b in self.bins:
            psig = b.psig(self.tg[0], self.tg[1])
            psib = b.psib(self.tb[0], self.tb[1])
#             print ("=================================================================================================\n")
#             print("From %f to %f: PSI Good: %f, PSI Bad: %f\n" %(float(b.lo), float(b.hi), float(psig), float(psib)) )
#             print("File1 TotalGood: %f, Good: %f, GoodProb: %f, TotalBad: %f, Bad: %f, BadProb: %f\n" %(float(self.tg[0]), float(b.ng1), float(b.pg1), float(self.tb[0]), float(b.nb1), float(b.pb1)) )
#             print("File2 TotalGood: %f, Good: %f, GoodProb: %f, TotalBad: %f, Bad: %f, BadProb: %f" %(float(self.tg[1]), float(b.ng2), float(b.pg2), float(self.tb[1]), float(b.nb2), float(b.pb2)) )
                  
            psi += psig
            psi += psib
            
        print ("Total PSI: %f" % psi )
        return psi
    

