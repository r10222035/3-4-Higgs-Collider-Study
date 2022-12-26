#!/usr/bin/env python
# coding: utf-8
# python to_DNN_data.py <root_file_path> <output_file_path>

import uproot
import numpy as np
from tqdm import tqdm
import h5py
import math
import os
import sys

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
    phi = np.arctan(py/px)
    m = np.sqrt(np.sqrt((E**2 - px**2 - py**2 - pz**2)**2))

    return pt, eta, phi, m

def create_diHiggs_dataset(f, nevent, MAX_JETS):
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
    
    f.create_dataset('Missing_ET/MET', (nevent,), maxshape=(None,), dtype='<f4')
    f.create_dataset('Missing_ET/Phi', (nevent,), maxshape=(None,), dtype='<f4')
    
    f.create_dataset('Lepton/electron', (nevent,), maxshape=(None,), dtype='<i8')
    f.create_dataset('Lepton/muon', (nevent,), maxshape=(None,), dtype='<i8')


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
    # from root file to HDF5 data. For SPANet predict. For DNN training
    # do not contain correct jet matching, only contain jets infomation

    nbj_min = int(nbj_min)
    MAX_JETS = 10
    
    root_file = uproot.open(file_path)['Delphes;1']
    num_entries =  root_file.num_entries
    
    for i in range(math.ceil(num_entries/nevent_max)):      
        
        start = i * nevent_max
        end = min(start + nevent_max, num_entries)

        jet_PT = root_file['Jet.PT'].array(entry_start=start, entry_stop=end)
        jet_Eta = root_file['Jet.Eta'].array(entry_start=start, entry_stop=end)
        jet_Phi = root_file['Jet.Phi'].array(entry_start=start, entry_stop=end)
        jet_Mass = root_file['Jet.Mass'].array(entry_start=start, entry_stop=end)
        jet_BTag = root_file['Jet.BTag'].array(entry_start=start, entry_stop=end)
        
        n_electron = root_file['Electron_size'].array(entry_start=start, entry_stop=end)
        n_muon = root_file['Muon_size'].array(entry_start=start, entry_stop=end)
        
        Missing_ET = root_file['MissingET.MET'].array(entry_start=start, entry_stop=end)
        Missing_Phi = root_file['MissingET.Phi'].array(entry_start=start, entry_stop=end)

        nevent = len(jet_PT)
        event_index = 0
        
        name, _ = os.path.splitext(output_path)
        event_file_path = f'{name}-{i:02}.h5'

        with h5py.File(event_file_path, 'w') as f_out:

            create_diHiggs_dataset(f_out, nevent, MAX_JETS)

            for event in tqdm(range(nevent)):
                
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

                # 準備寫入資料
                data_dict = {
                    'source/mask': np.arange(MAX_JETS)<nj,
                    'source/pt': PT[:MAX_JETS] if nj>MAX_JETS else np.pad(PT, (0,MAX_JETS-nj)),
                    'source/eta': Eta[:MAX_JETS] if nj>MAX_JETS else np.pad(Eta, (0,MAX_JETS-nj)),
                    'source/phi': Phi[:MAX_JETS] if nj>MAX_JETS else np.pad(Phi, (0,MAX_JETS-nj)),
                    'source/mass': Mass[:MAX_JETS] if nj>MAX_JETS else np.pad(Mass, (0,MAX_JETS-nj)),
                    'source/btag': BTag[:MAX_JETS] if nj>MAX_JETS else np.pad(BTag, (0,MAX_JETS-nj)),

                    'h1/mask': False,
                    'h2/mask': False,

                    'h1/b1': -1,
                    'h1/b2': -1,
                    'h2/b1': -1,
                    'h2/b2': -1,
                    
                    'Missing_ET/MET': Missing_ET[event],
                    'Missing_ET/Phi': Missing_Phi[event],
                    
                    'Lepton/electron': n_electron[event],
                    'Lepton/muon': n_muon[event],
                }

                write_dataset(f_out, event_index, data_dict)
                event_index += 1

        resize_h5(event_file_path, event_index)
        
if __name__ == '__main__':

    main(sys.argv[1], sys.argv[2], 4)