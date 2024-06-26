#!/usr/bin/env python
# coding: utf-8
# generate di-Higgs HDF5 data for SPANet
# jets are required PT > 40 GeV
# with correct jet assignment

# python to_diHiggs_h5_PT40_w_pairing.py <root file path> <output file name> <minimum b-jet>

import uproot
import numpy as np
import h5py
from tqdm import tqdm
import sys
import math
import os

class BranchGenParticles:
    def __init__(self,file,start,end):
        print('Initialize GenParticles')
        self.file = file
        self.length = len(file['Particle.Status'].array(entry_start=start, entry_stop=end))
        print('Initialize GenParticles: Status')
        self.Status = file['Particle.Status'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: PID')
        self.PID = file['Particle.PID'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: M1')
        self.M1 = file['Particle.M1'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: M2')
        self.M2 = file['Particle.M2'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: D1')
        self.D1 = file['Particle.D1'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: D2')
        self.D2  = file['Particle.D2'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: PT')
        self.PT = file['Particle.PT'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: Eta')
        self.Eta =  file['Particle.Eta'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: Phi')
        self.Phi = file['Particle.Phi'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: Mass')
        self.Mass = file['Particle.Mass'].array(entry_start=start, entry_stop=end)
        print('Initialize GenParticles: Charge')
        self.Charge = file['Particle.Charge'].array(entry_start=start, entry_stop=end)
        self.Labels = ['Status', 'PID' , 'M1', 'M2', 'D1', 'D2', 'PT', 'Eta', 'Phi', 'Mass', 'Charge']
        
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
    
def create_diHiggs_dataset_b(f, nevent, MAX_JETS):
    # with b-tagging information
    f.create_dataset('source/mask', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='|b1')
    f.create_dataset('source/pt', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('source/eta', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('source/phi', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('source/mass', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='<f4')
    f.create_dataset('source/btag', (nevent, MAX_JETS), maxshape=(None, MAX_JETS), dtype='|b1')

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

def write_dataset(file, index, data):
    # data: dictionary
    for key, value in data.items():
        file[key][index] = value
    
def resize_h5(file_path, nevent):
    with h5py.File(file_path,'r+') as f:
        for group in f:
            for dataset in f[group]:
                shape = list(f[group][dataset].shape)
                shape[0] = nevent
                f[group][dataset].resize(shape)
    print(f'{file_path} resize to {nevent}')
    
def main(file_path, output_path, nbj_min=2, nevent_max=100000):

    nbj_min = int(nbj_min)
    MAX_JETS = 10
    
    root_file = uproot.open(file_path)['Delphes;1']
    num_entries =  root_file.num_entries
    
    for i in range(math.ceil(num_entries/nevent_max)):
        
        start = i * nevent_max
        end = min(start + nevent_max, num_entries)
        
        GenParticle = BranchGenParticles(root_file, start, end)

        jet_PT = root_file['Jet.PT'].array(entry_start=start, entry_stop=end)
        jet_Eta = root_file['Jet.Eta'].array(entry_start=start, entry_stop=end)
        jet_Phi = root_file['Jet.Phi'].array(entry_start=start, entry_stop=end)
        jet_Mass = root_file['Jet.Mass'].array(entry_start=start, entry_stop=end)
        jet_BTag = root_file['Jet.BTag'].array(entry_start=start, entry_stop=end)

        nevent = len(jet_PT)
        event_index = 0

#         event_file_path = f'{output_name}-{i:02}.h5'
        
        root, ext = os.path.splitext(output_path)
        event_file_path = f'{root}-{i:02}.h5'

        with h5py.File(event_file_path, 'w') as f_out:

            create_diHiggs_dataset_b(f_out, nevent, MAX_JETS)

            for event in tqdm(range(nevent)):

                # 夸克資料
                # b夸克 衰變前的編號
                quarks_id = []
                quarks_Eta = []
                quarks_Phi = []
                quarks_Jet = [-1,-1,-1,-1]
                
                PID = GenParticle.PID_At(event)
                D1 = GenParticle.D1_At(event)
                D2 = GenParticle.D2_At(event)
                h_list = []
                h_count = 0

                # 找出衰變前的 b 編號
                for i in range(len(PID)):    
                    if PID[i] == 25 and i not in h_list:  
                        h = i
                        h_list.append(h)
                        # 找出 h 衰變前的編號
                        while PID[D1[h]] == PID[h]:
                            h = D1[h]
                            h_list.append(h)
                        
                        # 找出 b 衰變前的編號
                        b1, b2 = D1[h], D2[h]
                        while PID[D1[b1]] == PID[b1]:
                            b1 = D1[b1]
                        while PID[D1[b2]] == PID[b2]:
                            b2 = D1[b2]
                            
                        quarks_id.extend([b1,b2])
                        
                        h_count += 1
                    # 2 個 Higgs    
                    if h_count == 2:
                        break

                quarks_Eta.extend(GenParticle.Eta_At(event)[quarks_id])
                quarks_Phi.extend(GenParticle.Phi_At(event)[quarks_id])
                
                # Jet 資料
                # |eta| < 2.5 & PT > 40 GeV
                eta_pt_cut = np.array((np.abs(jet_Eta[event]) < 2.5) & (jet_PT[event] > 40))

                nj = eta_pt_cut.sum()

                # 至少要 4 jet
                if nj < 4: continue

                nbj = np.array(jet_BTag[event][eta_pt_cut][:MAX_JETS]).sum()
                # 在前 MAX_JETS jets 中，至少要 nbj_min 個 b-jet
                if nbj < nbj_min: continue

                PT = np.array(jet_PT[event][eta_pt_cut])
                Eta = np.array(jet_Eta[event][eta_pt_cut])
                Phi = np.array(jet_Phi[event][eta_pt_cut])
                Mass = np.array(jet_Mass[event][eta_pt_cut])
                BTag = np.array(jet_BTag[event][eta_pt_cut])
                            
                # 找出每個夸克配對的 jet
                more_than_1_jet = False
                for quark in range(len(quarks_Jet)):
                    for i in range(min(nj,MAX_JETS)):
                        dR = DeltaR(Eta[i], Phi[i], quarks_Eta[quark], quarks_Phi[quark])
                        if dR < 0.4 and quarks_Jet[quark]==-1:
                            quarks_Jet[quark] = i
                        elif dR < 0.4:
                            more_than_1_jet = True

                if more_than_1_jet: continue

                # 準備寫入資料
                data_dict = {
                    'source/mask': np.arange(MAX_JETS)<nj,
                    'source/pt': PT[:MAX_JETS] if nj>MAX_JETS else np.pad(PT, (0,MAX_JETS-nj)),
                    'source/eta': Eta[:MAX_JETS] if nj>MAX_JETS else np.pad(Eta, (0,MAX_JETS-nj)),
                    'source/phi': Phi[:MAX_JETS] if nj>MAX_JETS else np.pad(Phi, (0,MAX_JETS-nj)),
                    'source/mass': Mass[:MAX_JETS] if nj>MAX_JETS else np.pad(Mass, (0,MAX_JETS-nj)),
                    'source/btag': BTag[:MAX_JETS] if nj>MAX_JETS else np.pad(BTag, (0,MAX_JETS-nj)),

                    'h1/mask': get_particle_mask(quarks_Jet, quarks_index=(0,1)),
                    'h2/mask': get_particle_mask(quarks_Jet, quarks_index=(2,3)),

                    'h1/b1': quarks_Jet[0],
                    'h1/b2': quarks_Jet[1],
                    'h2/b1': quarks_Jet[2],
                    'h2/b2': quarks_Jet[3],
                }

                write_dataset(f_out, event_index, data_dict)
                event_index += 1

        resize_h5(event_file_path, event_index) 
    
if __name__ == '__main__':
    if len(sys.argv)<4: print('Wrong input format.')
        
    main(sys.argv[1], sys.argv[2], sys.argv[3])