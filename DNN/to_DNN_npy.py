#!/usr/bin/env python
# coding: utf-8
# python to_DNN_npy.py <main_file_path> <predict_file_path> <pairing_method>
# <main_file_path>: contain event infomattion

import h5py
import os
import sys

import numpy as np
from tqdm import tqdm

def DeltaR(eta1,phi1, eta2,phi2):
    dEta = eta1-eta2
    dPhi = abs(phi1-phi2)
    if dPhi > np.pi:
        dPhi = 2*np.pi - dPhi

    dR = (dPhi**2 + dEta**2)**0.5

    return dR

def FourMomentum(pt, eta, phi, m):
    px, py, pz = pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)
    e = np.sqrt(m**2 + px**2 + py**2 + pz**2)
    return e, px, py, pz

def PtEtaPhiM(px, py, pz, e):
    E, px ,py, pz = e, px, py, pz  
    P = np.sqrt(px**2 + py**2 + pz**2)
    pt = np.sqrt(px**2 + py**2)
    eta = 1./2.*np.log((P + pz)/(P - pz))
    phi = np.arctan2(py, px)
    m = np.sqrt(np.sqrt((E**2 - px**2 - py**2 - pz**2)**2))

    return pt, eta, phi, m

def min_dR_method(PT, Eta, Phi, Mass):
    # get h-jets pairing
    pairing = []
    for (i1,i2), (i3,i4) in all_pairs([0,1,2,3]):

        PT1 = PT[[i1,i2]] 
        Eta1 = Eta[[i1,i2]]
        Phi1 = Phi[[i1,i2]]
        Mass1 = Mass[[i1,i2]]
        
        e1, px1, py1, pz1 = FourMomentum(PT1, Eta1, Phi1, Mass1)
        PTH1, EtaH1, PhiH1, _ = PtEtaPhiM(np.sum(px1), np.sum(py1), np.sum(pz1), np.sum(e1))

        PT2 = PT[[i3,i4]]
        Eta2 = Eta[[i3,i4]]
        Phi2 = Phi[[i3,i4]]
        Mass2 = Mass[[i3,i4]]

        e2, px2, py2, pz2 = FourMomentum(PT2, Eta2, Phi2, Mass2)
        PTH2, EtaH2, PhiH2, _ = PtEtaPhiM(np.sum(px2), np.sum(py2), np.sum(pz2), np.sum(e2))
        
        # sorted by PT vector sum
        if PTH1 < PTH2:
            i1, i2, i3, i4 = i3, i4, i1, i2
            PTH1, EtaH1, PhiH1, PTH2, EtaH2, PhiH2 = PTH2, EtaH2, PhiH2, PTH1, EtaH1, PhiH1

        dR1 = DeltaR(Eta1[0], Phi1[0], Eta1[1], Phi1[1])
        pairing.append([i1,i2,i3,i4, dR1])
    
    h_candidate = pairing[0][0:4]
    min_dR = pairing[0][4]
    for i1,i2,i3,i4, dR in pairing:
        if dR < min_dR:
            min_dR = dR
            h_candidate = [i1,i2,i3,i4]
            
    return h_candidate

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

def SPANet_pairing(PT, Eta, Phi, Mass):
    # 已經過 preseleciton PT, Eta, Phi, Mass只有4個jets的資料
    i1, i2, i3, i4 = 0,1,2,3
    
    PT1 = PT[[i1,i2]] 
    Eta1 = Eta[[i1,i2]]
    Phi1 = Phi[[i1,i2]]
    Mass1 = Mass[[i1,i2]]

    e1, px1, py1, pz1 = FourMomentum(PT1, Eta1, Phi1, Mass1)
    PTH1, _, _, _ = PtEtaPhiM(np.sum(px1), np.sum(py1), np.sum(pz1), np.sum(e1))

    PT2 = PT[[i3,i4]]
    Eta2 = Eta[[i3,i4]]
    Phi2 = Phi[[i3,i4]]
    Mass2 = Mass[[i3,i4]]

    e2, px2, py2, pz2 = FourMomentum(PT2, Eta2, Phi2, Mass2)
    PTH2, _, _, _ = PtEtaPhiM(np.sum(px2), np.sum(py2), np.sum(pz2), np.sum(e2))

    # sorted by PT vector sum
    if PTH1 > PTH2:
        return [0, 1, 2, 3]
    else:
        return [2, 3, 0, 1] 

def main(main_file, predict_file, pairing_method):
    data_array = []

    with h5py.File(main_file, 'r') as f, h5py.File(predict_file, 'r') as f_pre:
        nevent = f['source/pt'].shape[0]
        
        for event in tqdm(range(nevent)):
            
            nj = f['source/mask'][event].sum()

            jet_PT = f['source/pt'][event][:nj]
            jet_Eta = f['source/eta'][event][:nj]
            jet_Phi = f['source/phi'][event][:nj]
            jet_Mass = f['source/mass'][event][:nj]
            jet_BTag = f['source/btag'][event][:nj]

            MET = f['Missing_ET/MET'][event]
            MET_phi = f['Missing_ET/Phi'][event]

            n_e = f['Lepton/electron'][event]
            n_mu = f['Lepton/muon'][event]           
            
            if pairing_method == 'min-dR':
                # |eta| < 2.5 & PT > 40 GeV & b-tagged
                eta_pt_bTag_cut = np.where((np.abs(jet_Eta) < 2.5) & (jet_PT > 40) & (jet_BTag == 1))[0]
                
                # choose 4 highest pt b-jets
                h_jets = eta_pt_bTag_cut[0:4]

                PT = jet_PT[h_jets]
                Eta = jet_Eta[h_jets]
                Phi = jet_Phi[h_jets]
                Mass = jet_Mass[h_jets]
                h_candidate = min_dR_method(PT, Eta, Phi, Mass)
            else:
                h_jets = [f_pre['h1/b1'][event], f_pre['h1/b2'][event], 
                          f_pre['h2/b1'][event], f_pre['h2/b2'][event],]

                PT = jet_PT[h_jets]
                Eta = jet_Eta[h_jets]
                Phi = jet_Phi[h_jets]
                Mass = jet_Mass[h_jets]
                h_candidate = SPANet_pairing(PT, Eta, Phi, Mass)
                
            BTag = jet_BTag[h_jets]
                          

            # Total invariant mass: mhh
            e, px, py, pz = FourMomentum(PT, Eta, Phi, Mass)
            PTHH, _, _, mHH = PtEtaPhiM(np.sum(e), np.sum(px), np.sum(py), np.sum(pz))

            # Get Higgs candidates information
            i1, i2, i3, i4 = h_candidate[0:4] 

            PT1 = PT[[i1,i2]]
            Eta1 = Eta[[i1,i2]]
            Phi1 = Phi[[i1,i2]]
            Mass1 = Mass[[i1,i2]]

            e1, px1, py1, pz1 = FourMomentum(PT1, Eta1, Phi1, Mass1)
            PTH1, EtaH1, PhiH1, MassH1 = PtEtaPhiM(np.sum(px1), np.sum(py1), np.sum(pz1), np.sum(e1))

            PT2 = PT[[i3,i4]]
            Eta2 = Eta[[i3,i4]]
            Phi2 = Phi[[i3,i4]]
            Mass2 = Mass[[i3,i4]]

            e2, px2, py2, pz2 = FourMomentum(PT2, Eta2, Phi2, Mass2)
            PTH2, EtaH2, PhiH2, MassH2 = PtEtaPhiM(np.sum(px2), np.sum(py2), np.sum(pz2), np.sum(e2))

            dR1 = DeltaR(Eta[i1], Phi[i1], Eta[i2], Phi[i2])
            dR2 = DeltaR(Eta[i3], Phi[i3], Eta[i4], Phi[i4])

            data_array.append([PTH1, EtaH1, PhiH1, MassH1, PTH2, EtaH2, PhiH2, MassH2,
                               dR1, dR2,
                               MET, MET_phi,
                               n_e, n_mu,
                               *BTag,
                               PTHH, mHH])
            
    root, _ = os.path.splitext(main_file)
    np.save(f'{root}_{pairing_method}.npy', np.array(data_array))
        
if __name__ == '__main__':

    main(sys.argv[1], sys.argv[2], sys.argv[3])