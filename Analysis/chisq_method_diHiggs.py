#!/usr/bin/env python
# coding: utf-8
# python chisq_method.py <test h5 file> <True/False: use b-tag>

import h5py
import itertools
import numpy as np
from tqdm import tqdm
import sys
import os
import pandas as pd

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

def Mjets(*arg):
    e_tot, px_tot, py_tot, pz_tot = 0, 0, 0, 0
    
    for jet in arg:
        pt, eta, phi, m = jet[0], jet[1], jet[2], jet[3]
        
        px, py, pz = pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)
        e = np.sqrt(m**2 + px**2 + py**2 + pz**2)
        
        px_tot += px
        py_tot += py
        pz_tot += pz
        e_tot += e
    
    return np.sqrt(e_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2)

def compare_jet_list_diHiggs(pair1, pair2, nh_max=2):
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
                
    same = True if nh==nh_max else False
    return same, nh

def get_chisq_result_diHiggs(total_event, correct_event, total_Higgs, correct_Higgs, nh):
    
    if nh=='all':
        result = {'Event type': ['Nj=4','Nj=5','Nj>=6','Total'],
                  'Event Fraction': [total_event[1:,4].sum()/total_event.sum(),
                                     total_event[1:,5].sum()/total_event.sum(),
                                     total_event[1:,6:].sum()/total_event.sum(),
                                     total_event[1:].sum()/total_event.sum()],
                  'Event Efficiency': [correct_event[1:,4].sum()/total_event[1:,4].sum(),
                                       correct_event[1:,5].sum()/total_event[1:,5].sum(),
                                       correct_event[1:,6:].sum()/total_event[1:,6:].sum(),
                                       correct_event[1:].sum()/total_event[1:].sum()],
                  'Higgs Efficiency': [correct_Higgs[1:,4].sum()/total_Higgs[1:,4].sum(),
                                       correct_Higgs[1:,5].sum()/total_Higgs[1:,5].sum(),
                                       correct_Higgs[1:,6:].sum()/total_Higgs[1:,6:].sum(),
                                       correct_Higgs[1:].sum()/total_Higgs[1:].sum()],}
    else:
        result = {'Event type': ['Nj=4','Nj=5','Nj>=6','Total'],
                  'Event Fraction': [total_event[nh,4]/total_event.sum(),
                                     total_event[nh,5]/total_event.sum(),
                                     total_event[nh,6:].sum()/total_event.sum(),
                                     total_event[nh].sum()/total_event.sum()],
                  'Event Efficiency': [correct_event[nh,4]/total_event[nh,4],
                                       correct_event[nh,5]/total_event[nh,5],
                                       correct_event[nh,6:].sum()/total_event[nh,6:].sum(),
                                       correct_event[nh].sum()/total_event[nh].sum()],
                  'Higgs Efficiency': [correct_Higgs[nh,4]/total_Higgs[nh,4],
                                       correct_Higgs[nh,5]/total_Higgs[nh,5],
                                       correct_Higgs[nh,6:].sum()/total_Higgs[nh,6:].sum(),
                                       correct_Higgs[nh].sum()/total_Higgs[nh].sum()],}
    df = pd.DataFrame(result)
    
    return df
    
def main(file_path, use_btag=False):
    use_btag = True if use_btag=='True' else False
    
    with h5py.File(file_path,'r') as f:
        total_event = np.zeros((3,11))
        total_Higgs = np.zeros((3,11))
        correct_event = np.zeros((3,11))
        correct_Higgs = np.zeros((3,11))

        nevent = f["source/pt"].shape[0]

        for event in tqdm(range(9535,nevent)):

            nj = f["source/mask"][event].sum()
            pt = f["source/pt"][event]
            eta = f["source/eta"][event]
            phi = f["source/phi"][event]
            mass = f["source/mass"][event]
            btag = f["source/btag"][event]

            h1_mask, h2_mask = f["h1/mask"][event], f["h2/mask"][event]
            
            h1_b1 = f["h1/b1"][event] if h1_mask else -1
            h1_b2 = f["h1/b2"][event] if h1_mask else -1
            h2_b1 = f["h2/b1"][event] if h2_mask else -1
            h2_b2 = f["h2/b2"][event] if h2_mask else -1
            
            is_2h_event = h1_mask and h2_mask
            is_1h_event = h1_mask != h2_mask
            is_0h_event = ~h1_mask and ~h2_mask
            
            if is_2h_event:
                total_event[2,nj] += 1
                total_Higgs[2,nj] += 2 
            if is_1h_event:
                total_event[1,nj] += 1
                total_Higgs[1,nj] += 1
            if is_0h_event:
                total_event[0,nj] += 1
                continue
                
            true_pair = [h1_b1,h1_b2, h2_b1,h2_b2]

            chisq = -1 
            pair = []

            jets_index = np.where(btag)[0] if use_btag else range(nj)

            for combination in itertools.combinations(jets_index, 4):
                for (i1,i2), (i3,i4) in all_pairs(combination):
                    jet = [[pt[i], eta[i], phi[i], mass[i]] for i in [i1,i2,i3,i4]]

                    mh1, mh2 = Mjets(jet[0],jet[1]), Mjets(jet[2],jet[3])

                    tem = chi2_diHiggs(mh1,mh2)

                    if chisq<0 or tem<chisq:
                        chisq = tem
                        pair = [i1,i2,i3,i4]
            
            if is_2h_event:
                same, nh = compare_jet_list_diHiggs(true_pair, pair, nh_max=2)
                correct_Higgs[2,nj] += nh
                if same: 
                    correct_event[2,nj] += 1
                    
            if is_1h_event:
                same, nh = compare_jet_list_diHiggs(true_pair, pair, nh_max=1)
                correct_Higgs[1,nj] += nh
                if same: 
                    correct_event[1,nj] += 1

        df_all = get_chisq_result_diHiggs(total_event, correct_event, total_Higgs, correct_Higgs, nh='all')
        print('All Events:')
        print(df_all)
        df_1h = get_chisq_result_diHiggs(total_event, correct_event, total_Higgs, correct_Higgs, nh=1)
        print('1 Higgs Events:')
        print(df_1h)
        df_2h = get_chisq_result_diHiggs(total_event, correct_event, total_Higgs, correct_Higgs, nh=2)
        print('2 Higgs Events:')
        print(df_2h)

        file_name = os.path.splitext(os.path.basename(file_path))[0]

        df_all.to_csv(f"{file_name}--chisq_result_all.csv")
        df_1h.to_csv(f"{file_name}--chisq_result_1h.csv")
        df_2h.to_csv(f"{file_name}--chisq_result_2h.csv")
    
if __name__ == '__main__':
    if len(sys.argv)<2: print("No input file.")
        
    main(sys.argv[1], sys.argv[2])