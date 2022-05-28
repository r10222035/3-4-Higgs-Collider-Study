#!/usr/bin/env python
# coding: utf-8
# python to_Training_data_diHiggs.py <root file path> <output file name>

import uproot
import numpy as np
import h5py
from tqdm import tqdm
import sys
import os
import shutil

class BranchGenParticles:
    def __init__(self,file):
        print("Initialize GenParticles")
        self.file = file
        self.length = len(file["Particle.Status"].array())
        print("Initialize GenParticles: Status")
        self.Status = file["Particle.Status"].array()
        print("Initialize GenParticles: PID")
        self.PID = file["Particle.PID"].array()
        print("Initialize GenParticles: M1")
        self.M1 = file["Particle.M1"].array()
        print("Initialize GenParticles: M2")
        self.M2 = file["Particle.M2"].array()
        print("Initialize GenParticles: D1")
        self.D1 = file["Particle.D1"].array()
        print("Initialize GenParticles: D2")
        self.D2  = file["Particle.D2"].array()
        print("Initialize GenParticles: PT")
        self.PT = file["Particle.PT"].array()
        print("Initialize GenParticles: Eta")
        self.Eta =  file["Particle.Eta"].array()
        print("Initialize GenParticles: Phi")
        self.Phi = file["Particle.Phi"].array()
        print("Initialize GenParticles: Mass")
        self.Mass = file["Particle.Mass"].array()
        print("Initialize GenParticles: Charge")
        self.Charge = file["Particle.Charge"].array()
        self.Labels = ["Status", "PID" , "M1", "M2", "D1", "D2", "PT", "Eta", "Phi", "Mass","Charge"]
        
    def length_At(self, i):
        return len(self.Status[i])
    def Status_At(self, i):
        return self.Status[i]
    def PID_At(self, i):
        return self.PID[i]
    def M1_At(self, i):
        return self.M1[i]
    def M2_At(self, i):
        return self.M2[i]
    def D1_At(self, i):
        return self.D1[i]
    def D2_At(self, i):
        return self.D2[i]
    def PT_At(self, i):
        return self.PT[i]
    def Eta_At(self, i):
        return self.Eta[i]
    def Phi_At(self, i):
        return self.Phi[i]
    def Mass_At(self, i):
        return self.Mass[i]
    def Charge_At(self, i):
        return self.Charge[i]
    
def DeltaR(eta1,phi1, eta2,phi2):
    dEta = eta1-eta2
    dPhi = abs(phi1-phi2)
    if dPhi > np.pi:
        dPhi = 2*np.pi - dPhi

    dR = (dPhi**2 + dEta**2)**0.5
    return dR
    
def create_diHiggs_dataset(f, nevent, MAX_JETS):
    f.create_dataset('source/mask', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='|b1')
    f.create_dataset('source/pt', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('source/eta', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('source/phi', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('source/mass', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')

    f.create_dataset('h1/mask', (nevent,), maxshape=(None,), dtype='|b1')
    f.create_dataset('h1/b1', (nevent,), maxshape=(None,), dtype='<i8')
    f.create_dataset('h1/b2', (nevent,), maxshape=(None,), dtype='<i8')

    f.create_dataset('h2/mask', (nevent,), maxshape=(None,), dtype='|b1')
    f.create_dataset('h2/b1', (nevent,), maxshape=(None,), dtype='<i8')
    f.create_dataset('h2/b2', (nevent,), maxshape=(None,), dtype='<i8')
    
def get_particle_mask(quarks_Jet, quarks_index):
    # quarks_index: 粒子對應的夸克編號
    # 若某粒子的 每個夸克都有對應到 jet 且 每個夸克對應的 jet 都沒有重複，則返回 true
    mask = True
    for i in quarks_index:
        if quarks_Jet[i] == -1:
            mask = False
        else:
            for j in range(len(quarks_Jet)):
                if j == i: continue
                if quarks_Jet[i] == quarks_Jet[j]: mask = False
    return mask
    
def resize_h5(file_path, nevent):
    with h5py.File(file_path,'r+') as f:
        for group in f:
            for dataset in f[group]:
                shape = list(f[group][dataset].shape)
                shape[0] = nevent
                f[group][dataset].resize(shape)
    print(f"resize to {nevent}")
    
def main(file_path, output_name, nevent_max=1000000):

    root_file = uproot.open(file_path)["Delphes;1"]
    GenParticle = BranchGenParticles(root_file)
    
    MAX_JETS = 10

    jet_PT = root_file["Jet.PT"].array()
    jet_Eta = root_file["Jet.Eta"].array()
    jet_Phi = root_file["Jet.Phi"].array()
    jet_Mass = root_file["Jet.Mass"].array()

    nevent = min(len(jet_PT), nevent_max)

    r = 0.9
    n_train = int(nevent*r)
    n_test = nevent - n_train

    train_index = 0
    test_index = 0

    train_file_path = f"{output_name}_training.h5"
    test_file_path = f"{output_name}_testing.h5"

    with h5py.File(train_file_path, "w") as f_train, h5py.File(test_file_path, "w") as f_test:

        create_diHiggs_dataset(f_train, n_train, MAX_JETS)
        create_diHiggs_dataset(f_test, n_test, MAX_JETS)

        for event in tqdm(range(nevent)):

            # 夸克資料
            # b夸克 衰變前的編號
            quarks_id = []
            quarks_Eta = []
            quarks_Phi = []
            quarks_Jet = [-1,-1,-1,-1]

            # 2 個 Higgs
            for j in np.where(GenParticle.PID_At(event) == 25)[0]:
                # h > b b~
                b1 = GenParticle.D1_At(event)[j]
                b2 = GenParticle.D2_At(event)[j]

                # 找出 b 衰變前的編號
                d1 = GenParticle.D1_At(event)[b1]
                while abs(GenParticle.PID_At(event)[d1]) == 5:
                    b1 = d1
                    d1 = GenParticle.D1_At(event)[b1]

                # 找出 b~ 衰變前的編號
                d2 = GenParticle.D1_At(event)[b2]
                while abs(GenParticle.PID_At(event)[d2]) == 5:
                    b2 = d2
                    d2 = GenParticle.D1_At(event)[b2]

                quarks_id.extend([b1,b2])

            quarks_Eta.extend(GenParticle.Eta_At(event)[quarks_id])
            quarks_Phi.extend(GenParticle.Phi_At(event)[quarks_id])

            # Jet 資料
            # |eta| < 2.5
            eta_cut_index = np.array(np.abs(jet_Eta[event])<2.5)

            nj = eta_cut_index.astype("int").sum()
            # 至少要 4 jet
            if nj<4: continue

            PT = np.array(jet_PT[event][eta_cut_index])
            Eta = np.array(jet_Eta[event][eta_cut_index])
            Phi = np.array(jet_Phi[event][eta_cut_index])
            Mass = np.array(jet_Mass[event][eta_cut_index])

            # 找出每個夸克配對的 jet
            for quark in range(len(quarks_Jet)):
                for i in range(min(nj,MAX_JETS)):
                    dR = DeltaR(Eta[i], Phi[i], quarks_Eta[quark], quarks_Phi[quark])
                    if dR < 0.4 and quarks_Jet[quark] == -1:
                        quarks_Jet[quark] = i

            # 準備寫入資料
            source_mask = np.arange(MAX_JETS)<nj

            if nj > MAX_JETS:
                source_pt = PT[:MAX_JETS]
                source_eta = Eta[:MAX_JETS]
                source_phi = Phi[:MAX_JETS]
                source_mass = Mass[MAX_JETS]
            else:
                source_pt = np.pad(PT, (0,MAX_JETS-nj))
                source_eta = np.pad(Eta, (0,MAX_JETS-nj))
                source_phi = np.pad(Phi, (0,MAX_JETS-nj))
                source_mass = np.pad(Mass, (0,MAX_JETS-nj))

            h1_mask = get_particle_mask(quarks_Jet, quarks_index=(0,1))
            h2_mask = get_particle_mask(quarks_Jet, quarks_index=(2,3))

            h1_b1 = quarks_Jet[0]
            h1_b2 = quarks_Jet[1]

            h2_b1 = quarks_Jet[2]
            h2_b2 = quarks_Jet[3]

            if event<n_train:
                f_train["source/mask"][train_index] = source_mask
                f_train["source/pt"][train_index] = source_pt
                f_train["source/eta"][train_index] = source_eta
                f_train["source/phi"][train_index] = source_phi
                f_train["source/mass"][train_index] = source_mass

                f_train["h1/mask"][train_index] = h1_mask
                f_train["h1/b1"][train_index] = h1_b1
                f_train["h1/b2"][train_index] = h1_b2

                f_train["h2/mask"][train_index] = h2_mask
                f_train["h2/b1"][train_index] = h2_b1
                f_train["h2/b2"][train_index] = h2_b2
                
                train_index += 1       
            else:
                f_test["source/mask"][test_index] = source_mask
                f_test["source/pt"][test_index] = source_pt
                f_test["source/eta"][test_index] = source_eta
                f_test["source/phi"][test_index] = source_phi
                f_test["source/mass"][test_index] = source_mass

                f_test["h1/mask"][test_index] = h1_mask
                f_test["h1/b1"][test_index] = h1_b1
                f_test["h1/b2"][test_index] = h1_b2

                f_test["h2/mask"][test_index] = h2_mask
                f_test["h2/b1"][test_index] = h2_b1
                f_test["h2/b2"][test_index] = h2_b2
                
                test_index += 1
              
    resize_h5(train_file_path, train_index)
    resize_h5(test_file_path, test_index)
    
if __name__ == '__main__':
    if len(sys.argv)<2: print("No input file.")
        
    main(sys.argv[1], sys.argv[2])