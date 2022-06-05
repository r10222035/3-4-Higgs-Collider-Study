#!/usr/bin/env python
# coding: utf-8
# python chisq_method.py <test h5 file>

import h5py
import itertools
import numpy as np
from tqdm import tqdm
import sys

def all_pairs(lst):
    if len(lst) < 2:
        yield []
        return
    if len(lst) % 2 == 1:
        # Handle odd length list
        for i in range(len(lst)):
            for result in all_pairs(lst[:i] + lst[i+1:]):
                yield result
    else:
        a = lst[0]
        for i in range(1,len(lst)):
            pair = (a,lst[i])
            for rest in all_pairs(lst[1:i]+lst[i+1:]):
                yield [pair] + rest
                
def chi2_diHiggs(m1,m2):
    mh = 125.10
    return (m1-mh)**2 + (m2-mh)**2

def Mjj(j1,j2):
    pt1, eta1, phi1, m1 = j1[0], j1[1], j1[2], j1[3]
    pt2, eta2, phi2, m2 = j2[0], j2[1], j2[2], j2[3]
    
    px1, py1, pz1 = pt1*np.cos(phi1), pt1*np.sin(phi1), np.sqrt(m1**2+pt1**2)*np.sinh(eta1)
    e1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    px2, py2, pz2 = pt2*np.cos(phi2), pt2*np.sin(phi2), np.sqrt(m2**2+pt2**2)*np.sinh(eta2)
    e2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    
    return np.sqrt((e1+e2)**2-(px1+px2)**2-(py1+py2)**2-(pz1+pz2)**2)

def same_diHiggs(pair1, pair2):
    # pair = [1,2,3,4]
    
    h1_true = {pair1[0],pair1[1]}
    h2_true = {pair1[2],pair1[3]}
    
    h1_test = {pair2[0],pair2[1]}
    h2_test = {pair2[2],pair2[3]}
    
    test_h = [h1_test, h2_test]
    
    same = False
    for id1, id2 in itertools.permutations([0,1]):
        h1 = test_h[id1]
        h2 = test_h[id2]
        if h1_true == h1:
            if h2_true == h2:
                same = True
                break
    
    return same

def n_correct_diHiggs(pair1, pair2):
    # pair = [1,2,3,4]
    
    h1_true = {pair1[0],pair1[1]}
    h2_true = {pair1[2],pair1[3]}
    
    h1_test = {pair2[0],pair2[1]}
    h2_test = {pair2[2],pair2[3]}
    
    test_h = [h1_test, h2_test]
    
    nh = 0
    for id1, id2 in itertools.permutations([0,1]):
        h1 = test_h[id1]
        h2 = test_h[id2]
        if h1_true == h1:
            nh = 1
            if h2_true == h2:
                nh = 2
                break
    
    return nh

def print_result(total_2h_event, correct_2h_event, total_2h_Higgs, correct_2h_Higgs):
    print("Nj=4, event fraction: {:.3}".format(total_2h_event[4]/total_2h_event.sum()))
    print("Nj=5, event fraction {:.3}".format(total_2h_event[5]/total_2h_event.sum()))
    print("Nj>=6, event fraction {:.3}".format(total_2h_event[6:].sum()/total_2h_event.sum()))
    print("total, event fraction {:.3}".format(total_2h_event.sum()/total_2h_event.sum()))

    print("Nj=4, event efficiency {:.3}".format(correct_2h_event[4]/total_2h_event[4]))
    print("Nj=5, event efficiency {:.3}".format(correct_2h_event[5]/total_2h_event[5]))
    print("Nj>=6, event efficiency {:.3}".format(correct_2h_event[6:].sum()/total_2h_event[6:].sum()))
    print("total, event efficiency {:.3}".format(correct_2h_event.sum()/total_2h_event.sum()))

    print("Nj=4, Higgs efficiency {:.3}".format(correct_2h_Higgs[4]/total_2h_Higgs[4]))
    print("Nj=5, Higgs efficiency {:.3}".format(correct_2h_Higgs[5]/total_2h_Higgs[5]))
    print("Nj>=6, Higgs efficiency {:.3}".format(correct_2h_Higgs[6:].sum()/total_2h_Higgs[6:].sum()))
    print("total, Higgs efficiency {:.3}".format(correct_2h_Higgs.sum()/total_2h_Higgs.sum()))
    
def main(file_path):
#     file = "/home/r10222035/CPVDM/100k_diHiggs_testing.h5"
    with h5py.File(file_path,'r') as f:
        total_2h_event = np.zeros(11)
        correct_2h_event = np.zeros(11)

        total_2h_Higgs = np.zeros(11)
        correct_2h_Higgs = np.zeros(11)

        nevent = f["source/pt"].shape[0]

        for event in tqdm(range(nevent)):
            # 是否有兩個 h
            check = f["h1/mask"][event] and f["h2/mask"][event]
            if not check: continue

            nj = f["source/mask"][event].sum()
            pt = f["source/pt"][event]
            eta = f["source/eta"][event]
            phi = f["source/phi"][event]
            mass = f["source/mass"][event]

            total_2h_event[nj] += 1
            total_2h_Higgs[nj] += 2

            h1_b1 = f["h1/b1"][event]
            h1_b2 = f["h1/b2"][event]
            h2_b1 = f["h2/b1"][event]
            h2_b2 = f["h2/b2"][event]

            true_pair = [h1_b1,h1_b2,h2_b1,h2_b2]

            chisq = -1 
            pair = []
            for combination in itertools.combinations(range(nj), 4):
                for (i1,i2), (i3,i4) in all_pairs(combination):
                    j1 = [pt[i1], eta[i1], phi[i1], mass[i1]]
                    j2 = [pt[i2], eta[i2], phi[i2], mass[i2]]
                    j3 = [pt[i3], eta[i3], phi[i3], mass[i3]]
                    j4 = [pt[i4], eta[i4], phi[i4], mass[i4]]

                    mh1 = Mjj(j1,j2)
                    mh2 = Mjj(j3,j4)

                    tem = chi2_diHiggs(mh1,mh2)

                    # 
                    if chisq < 0 or tem < chisq:
                        chisq = tem
                        pair = [i1,i2,i3,i4]
            correct_2h_Higgs[nj] += n_correct_diHiggs(true_pair,pair)
            if same_diHiggs(true_pair, pair):
                correct_2h_event[nj] += 1  

        print_result(total_2h_event, correct_2h_event, total_2h_Higgs, correct_2h_Higgs)
    
if __name__ == '__main__':
    if len(sys.argv)<2: print("No input file.")
        
    main(sys.argv[1])