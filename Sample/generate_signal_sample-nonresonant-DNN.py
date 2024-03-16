#!/usr/bin/env python
# coding: utf-8

# python generate_signal_sample-nonresonant-DNN.py <kappa> 

# 生成指定 kappa 數值的 root 事件，並轉成 h5 檔案，其中包含 DNN 所需的資料
# .h5 file would save in h5_dir/1M_diHiggs_4b_PT40-<run>.h5

import os
import re
import sys

def main():
    kappa = sys.argv[1]
    
    # Generate 1M event
    mg5_dir = f'./MG5/pphh_DNN_{kappa}'

    if os.path.isdir(mg5_dir):
        # generate event
        os.system(f'''
        ./MG5/pphh_DNN_{kappa}/bin/generate_events << eof
        analysis=off
        eof''')
    else:
        # generate process card
        mg5_card_origin_path = './Cards/proc_pphh_DNN.txt'
        mg5_card_path = f'./MG5/proc_pphh_DNN_{kappa}.txt'

        cmd = f'cp {mg5_card_origin_path} {mg5_card_path}'
        os.system(cmd)

        cmd = f'sed -i -e s/kappavalue/{kappa}/g {mg5_card_path}'
        os.system(cmd)
        # generate event
        cmd = f'~/Software/MG5_aMC_v3_3_1/bin/mg5_aMC {mg5_card_path}'
        os.system(cmd)

    # select the events in mg5_dir    
    h5_dir = f'./h5_data/pphh_DNN_{kappa}/'
    if not os.path.isdir(h5_dir):
        os.makedirs(h5_dir)
    
    # select all run
    for f in os.listdir(os.path.join(mg5_dir, 'Events')):
        match = re.match(f'run_(\d\d)_decayed_1', f)
        if match:
            run = match.group(1)

            root_file = os.path.join(mg5_dir, f'Events/run_{run}_decayed_1/tag_1_delphes_events.root')
            if not os.path.isfile(root_file):
                continue
            output_file = f'1M_diHiggs_4b_PT40-{run}.h5'
            output_path = os.path.join(h5_dir, output_file)

            output_file_exist = False
            name, _ = os.path.splitext(output_file)
            for file in os.listdir(h5_dir):
                if name in file:
                    output_file_exist = True

            if not output_file_exist:
                cmd = f'python ./from_root_to_h5-DNN.py {root_file} {output_path} 4'
                print(cmd)
                os.system(cmd)
                # remove pythia and delphes file
                os.remove(root_file)
                os.remove(os.path.join(mg5_dir, f'Events/run_{run}_decayed_1/tag_1_pythia8_events.hepmc.gz'))
           
if __name__ == '__main__':        
    main()