# coding: utf-8
# python from_h5_to_npy.py <true_file> <predict_file> <output_path> <pairing_method>
# python from_h5_to_npy.py SPANET

# extract data from .h5 file, compute high-level features, and save as .npy file
# .h5 file: SPANet 2 format

import h5py
import numpy as np
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

def DeltaR(eta1, phi1, eta2, phi2):
    dEta = eta1 - eta2
    dPhi = np.abs(phi1 - phi2)
    dPhi = np.where(dPhi > np.pi, 2 * np.pi - dPhi, dPhi)
    return np.sqrt(dEta ** 2 + dPhi ** 2)

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
    m = np.sqrt(E**2 - px**2 - py**2 - pz**2)

    return pt, eta, phi, m

def min_dR_pairing(PT, Eta, Phi, Mass):
    # 假設 PT, Eta, Phi, Mass 的形狀為 (n_events, 4)
    
    # 生成所有可能的配對 (n_pairs, 4)
    pairs = np.array(list(all_pairs([0, 1, 2, 3]))).reshape(-1, 4)

    # PT shape: (n_events, 4) -> (n_events, 1, 4), pairs shape: (n_pairs, 4) -> (1, n_pairs, 4)
    # output array shape: (n_events, n_pairs, 4)
    PT_pairs = np.take_along_axis(PT[:, None, :], pairs[None, :, :], axis=2)
    Eta_pairs = np.take_along_axis(Eta[:, None, :], pairs[None, :, :], axis=2)
    Phi_pairs = np.take_along_axis(Phi[:, None, :], pairs[None, :, :], axis=2)
    Mass_pairs = np.take_along_axis(Mass[:, None, :], pairs[None, :, :], axis=2)

    # e1 shape: (n_events, n_pairs, 2)
    # PTH1 shape: (n_events, n_pairs)
    e1, px1, py1, pz1 = FourMomentum(PT_pairs[..., :2], Eta_pairs[..., :2], Phi_pairs[..., :2], Mass_pairs[..., :2])
    PTH1, _, _, _ = PtEtaPhiM(np.sum(px1, axis=2), np.sum(py1, axis=2), np.sum(pz1, axis=2), np.sum(e1, axis=2))

    e2, px2, py2, pz2 = FourMomentum(PT_pairs[..., 2:], Eta_pairs[..., 2:], Phi_pairs[..., 2:], Mass_pairs[..., 2:])
    PTH2, _, _, _ = PtEtaPhiM(np.sum(px2, axis=2), np.sum(py2, axis=2), np.sum(pz2, axis=2), np.sum(e2, axis=2))
    
    # possible_pairs shape: (2, n_pairs, 4) -> (1, 2, n_pairs, 4)
    # greater_PT shape: (n_events, n_pairs) -> (n_events, 1, n_pairs, 1)  
    # indices shape: (n_events, 1, n_pairs, 4) -> (n_events, n_pairs, 4)
    possible_pairs = np.array([pairs, pairs[:,[2,3,0,1]]])
    greater_PT = np.where(PTH1 > PTH2, 0, 1)
    indices = np.take_along_axis(possible_pairs[None, ...] , greater_PT[:,None, :, None] , axis=1)[:,0,:,:]

    # Eta1, Phi1 shape: (n_events, n_pairs, 2)
    Eta1 = np.take_along_axis(Eta[:, None, :], indices[:,:,[0,1]], axis=2)
    Phi1 = np.take_along_axis(Phi[:, None, :], indices[:,:,[0,1]], axis=2)

    # dR shape: (n_events, n_pairs)
    dR = DeltaR(Eta1[:,:,0], Phi1[:,:,0], Eta1[:,:,1], Phi1[:,:,1])

    # 找到每個事件的最小 dR (n_events,)
    min_dR_indices = np.argmin(dR, axis=1)

    # output array shape: (n_events, 1, 4)
    # 返回最小 dR 的配對 (n_events, 4)
    return np.take_along_axis(indices, min_dR_indices[:, None, None], axis=1)[:,0,:]

def SPANet_pairing(PT, Eta, Phi, Mass):
    # 假設 PT, Eta, Phi, Mass 的形狀為 (n_events, 4)
    i1, i2, i3, i4 = 0,1,2,3
    
    PT1 = PT[:, [i1,i2]] 
    Eta1 = Eta[:, [i1,i2]]
    Phi1 = Phi[:, [i1,i2]]
    Mass1 = Mass[:, [i1,i2]]

    e1, px1, py1, pz1 = FourMomentum(PT1, Eta1, Phi1, Mass1)
    PTH1, _, _, _ = PtEtaPhiM(np.sum(px1, axis=1), np.sum(py1, axis=1), np.sum(pz1, axis=1), np.sum(e1, axis=1))

    PT2 = PT[:, [i3,i4]]
    Eta2 = Eta[:, [i3,i4]]
    Phi2 = Phi[:, [i3,i4]]
    Mass2 = Mass[:, [i3,i4]]

    e2, px2, py2, pz2 = FourMomentum(PT2, Eta2, Phi2, Mass2)
    PTH2, _, _, _ = PtEtaPhiM(np.sum(px2, axis=1), np.sum(py2, axis=1), np.sum(pz2, axis=1), np.sum(e2, axis=1))

    # sorted by PT vector sum
    return np.where(PTH1[:, None] > PTH2[:, None], [0, 1, 2, 3], [2, 3, 0, 1])
    
def main(true_file, predict_file, output_path, pairing_method):
    # true_file: h5 file has true label
    # predict_file: h5 file has predict label. For min_dR pairing, it would not be used. 
    # output_path: npy file
    # pairing_method: min_dR or SPANet

    with h5py.File(predict_file, 'r') as f_pre, h5py.File(true_file, 'r') as f_true:
        nevent = f_pre['INPUTS/Source/pt'].shape[0]
        
        jet_PT = f_pre['INPUTS/Source/pt'][...]
        jet_Eta = f_pre['INPUTS/Source/eta'][...]
        jet_Phi = f_pre['INPUTS/Source/phi'][...]
        jet_Mass = f_pre['INPUTS/Source/mass'][...]
        jet_BTag = f_pre['INPUTS/Source/btag'][...]

        # |eta| < 2.5 & PT > 40 GeV & b-tagged
        eta_pt_bTag_cut = ((np.abs(jet_Eta) < 2.5) & (jet_PT > 40) & (jet_BTag == 1))

        if pairing_method == 'min_dR':
            # choose 4 highest pt b-jets
            h_jets = np.array([np.where(row)[0][:4] for row in eta_pt_bTag_cut])

            PT = np.take_along_axis(jet_PT, h_jets, axis=1)
            Eta = np.take_along_axis(jet_Eta, h_jets, axis=1)
            Phi = np.take_along_axis(jet_Phi, h_jets, axis=1)
            Mass = np.take_along_axis(jet_Mass, h_jets, axis=1)
            h_candidate = min_dR_pairing(PT, Eta, Phi, Mass)


        elif pairing_method == 'SPANET':
            h_jets = np.vstack([f_pre['TARGETS/h1/b1'][...],
                                f_pre['TARGETS/h1/b2'][...],
                                f_pre['TARGETS/h2/b1'][...],
                                f_pre['TARGETS/h2/b2'][...]]).transpose()

            PT = np.take_along_axis(jet_PT, h_jets, axis=1)
            Eta = np.take_along_axis(jet_Eta, h_jets, axis=1)
            Phi = np.take_along_axis(jet_Phi, h_jets, axis=1)
            Mass = np.take_along_axis(jet_Mass, h_jets, axis=1)
            h_candidate = SPANet_pairing(PT, Eta, Phi, Mass)
        else:
            print('Wrong pairing method')
            return 

        BTag = jet_BTag[np.arange(nevent)[:,None],h_jets][np.arange(nevent)[:,None],h_candidate]

        # Total invariant mass: mhh
        e, px, py, pz = FourMomentum(PT, Eta, Phi, Mass)
        PTHH, _, _, mHH = PtEtaPhiM(np.sum(px, axis=1), np.sum(py, axis=1), np.sum(pz, axis=1), np.sum(e, axis=1))

        # Get Higgs candidates information
        i1, i2, i3, i4 = h_candidate[:,0], h_candidate[:,1], h_candidate[:,2], h_candidate[:,3]
        i1i2, i3i4 = h_candidate[:,[0,1]], h_candidate[:,[2,3]]

        PT1 = PT[np.arange(nevent)[:,None],i1i2]
        Eta1 = Eta[np.arange(nevent)[:,None],i1i2]
        Phi1 = Phi[np.arange(nevent)[:,None],i1i2]
        Mass1 = Mass[np.arange(nevent)[:,None],i1i2]

        e1, px1, py1, pz1 = FourMomentum(PT1, Eta1, Phi1, Mass1)
        PTH1, EtaH1, PhiH1, MassH1 = PtEtaPhiM(np.sum(px1, axis=1), np.sum(py1, axis=1), np.sum(pz1, axis=1), np.sum(e1, axis=1))

        PT2 = PT[np.arange(nevent)[:,None],i3i4]
        Eta2 = Eta[np.arange(nevent)[:,None],i3i4]
        Phi2 = Phi[np.arange(nevent)[:,None],i3i4]
        Mass2 = Mass[np.arange(nevent)[:,None],i3i4]

        e2, px2, py2, pz2 = FourMomentum(PT2, Eta2, Phi2, Mass2)
        PTH2, EtaH2, PhiH2, MassH2 = PtEtaPhiM(np.sum(px2, axis=1), np.sum(py2, axis=1), np.sum(pz2, axis=1), np.sum(e2, axis=1))

        dR1 = DeltaR(Eta[np.arange(nevent),i1], Phi[np.arange(nevent),i1], Eta[np.arange(nevent),i2], Phi[np.arange(nevent),i2])
        dR2 = DeltaR(Eta[np.arange(nevent),i3], Phi[np.arange(nevent),i3], Eta[np.arange(nevent),i4], Phi[np.arange(nevent),i4])

        data_array = np.vstack([PTH1, EtaH1, PhiH1, MassH1, PTH2, EtaH2, PhiH2, MassH2,
                                dR1, dR2,
                                # MET, MET_phi,
                                # n_e, n_mu,
                                BTag.transpose(),
                                PTHH, mHH]).transpose()

        label = f_true['CLASSIFICATIONS/EVENT/signal'][...]

        npy_dict = {
                'data': np.array(data_array),
                'label': np.array(label),       
            }
        
        np.save(output_path, npy_dict)

        print(f'Finish saving npy file to {output_path}')

if __name__ == '__main__':

    true_file = sys.argv[1]
    predict_file = sys.argv[2]
    output_path = sys.argv[3]
    pairing_method = sys.argv[4]
    main(true_file, predict_file, output_path, pairing_method)