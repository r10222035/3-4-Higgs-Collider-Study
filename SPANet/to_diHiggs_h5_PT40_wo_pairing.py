#!/usr/bin/env python
# coding: utf-8
# generate di-Higgs HDF5 data for SPANet 
# jets are required PT > 40 GeV
# without jet assignment

# python to_diHiggs_h5_PT40_wo_pairing.py <root file path> <output file name> <minimum b-jet>

import uproot
import numpy as np
import h5py
from tqdm import tqdm
import sys
import math
    
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
    
def main(file_path, output_name, nbj_min=2, nevent_max=100000):
    # from root file to HDF5 data. For SPANet predict.
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

        nevent = len(jet_PT)
        event_index = 0

        event_file_path = f'{output_name}-{i:02}.h5'

        with h5py.File(event_file_path, 'w') as f_out:

            create_diHiggs_dataset_b(f_out, nevent, MAX_JETS)

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
                }

                write_dataset(f_out, event_index, data_dict)
                event_index += 1

        resize_h5(event_file_path, event_index)

if __name__ == '__main__':
    if len(sys.argv)<4: print('Wrong input format.')
        
    main(sys.argv[1], sys.argv[2], sys.argv[3])