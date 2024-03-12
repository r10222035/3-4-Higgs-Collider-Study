#!/usr/bin/env python
# coding: utf-8

# python generate_background_sample-resonant.py <randomseed>

import os
import re
import sys

def main():
    # Generate 1M event
    rnd = sys.argv[1]
    
    mg5_dir = f'./MG5/resonant/pp4b_{rnd}'

    if os.path.isdir(mg5_dir):
        # generate event
        os.system(f'''
        {mg5_dir}/bin/generate_events << eof
        analysis=off
        eof''')
    else:
        # generate process card
        mg5_card_origin_file = './MG5/resonant/proc_pp4b.txt'
        mg5_card_file = f'./MG5/resonant/proc_pp4b_{rnd}.txt'

        cmd = f'cp {mg5_card_origin_file} {mg5_card_file}'
        os.system(cmd)

        cmd = f'sed -i -e s/randomseed/{rnd}/g {mg5_card_file}'
        os.system(cmd)
        # generate event
        cmd = f'~/Software/MG5_aMC_v3_3_1/bin/mg5_aMC {mg5_card_file}'
        os.system(cmd)

    # select the events in mg5_dir    
    h5_dir = f'./h5_data/resonant/pp4b_{rnd}/'
    if not os.path.isdir(h5_dir):
        os.makedirs(h5_dir)

    # selection
    for f in os.listdir(os.path.join(mg5_dir, 'Events')):
        match = re.match(f'run_(\d\d)', f)
        if match:
            run = match.group(1)

            root_file = os.path.join(mg5_dir, f'Events/run_{run}/tag_1_delphes_events.root')
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
                cmd = f'python ../SPANet2/to_diHiggs_h5_PT40_wo_pairing.py {root_file} {output_path} 4'
                print(cmd)
                os.system(cmd)
                # remove pythia and delphes file
                os.remove(os.path.join(mg5_dir, f'Events/run_{run}/tag_1_pythia8_events.hepmc.gz'))
                os.remove(root_file)              
            
if __name__ == '__main__':        
    main()