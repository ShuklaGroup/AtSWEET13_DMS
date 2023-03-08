import numpy as np
import glob
import os
pwd=os.getcwd()

## This part is for your protein!
## TM4 residues in AtSWEET13 are actually 94-126, but because of pdb renumbering I had to shift it down
TM4_resids = [93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125]
TM4_resname = ["A","N","K","K","T","R","I","S","T","L","K","V","L","G","L","L","N","F","L","G","F","A","A","I","V","L","V","C","E","L","L","T","K"]

len_TM4 = len(TM4_resids)

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

for pdb in glob.glob('SUC*.pdb'):
    name=pdb.strip('.pdb')
    for resid in range(len_TM4):
        pos = np.str(TM4_resids[resid])
        wt_res = TM4_resname[resid]
        for aa in amino_acids:
            f=open(name + '_' + TM4_resname[resid] + pos + aa + '_MPddG.py','w+')
            f.write('''#!/usr/bin/env python
## This script is written using PyRosetta-4 2022. 
## To obtain PyRosetta, you must first get an academic license from:
## https://els2.comotion.uw.edu/
##
## With your PyRosetta license (different from Rosetta), you can find download instructions at:
## https://rosettacommons.github.io/PyRosetta.notebooks/
## https://www.pyrosetta.org/downloads
##
## PyRosetta was built using `pip install <whl> .
## The wheel file I used was:
## pyrosetta-2022.47+release.d2aee95-cp36-cp36m-linux_x86_64.whl ;
## this corresponds to the pyrosetta release of:
## PyRosetta4.Debug.python36.linux 2022.47+release.d2aee95a6b7bf6ee70c5e2c7b29d0915e9112fa7 
##
## PyRosetta works for a specific version of python. I used a Python 3.6 environment here for this .whl build
##
## You can get PyRosetta from here (once you have a user and password):
## https://graylab.jhu.edu/download/PyRosetta4/archive/release/
##
## This script assumes you have already developed a membrane for your membrane protien span file.
## There is a good tutorial for this found here: http://carbon.structbio.vanderbilt.edu/index.php/rosetta-tutorials
## Download the tutorial zip labeled "New RosettaMP membrane framework" and follow the "tutorial_mp_span_from_pdb" docx
##
## To do anything in Rosetta, nonstandard residues and atoms should be removed/renamed. PDB atoms and resids MUST
## start from the number `1`
## Once you have your span file made and validated and you trust your input pdb conformations, you can perform calculations
## like this. I tweaked the code presented in:
## https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/15.02-Membrane-Protein-ddG-of-mutation.ipynb
## Code had to be tweaked because the `predict_ddG.py` script in captured_protocols resulted in segmentation faults due to 
## deprecation issues.
##
## Austin Weigle, Diwakar Shukla Group, University of Illinois
## December 7th, 2022

import os
import time
startTime = time.time()

## Import general dependencies
import numpy as np
import pyrosettacolabsetup; pyrosettacolabsetup.install_pyrosetta()
import pyrosetta
from pyrosetta import *
init( extra_options="-score:weights franklin2019  -pH_mode true -value_pH 7.4 -mp:lipids:has_pore false -mp:thickness 10.5 -mp:lipids:composition DOPC -multithreading true")

file_exists = os.path.exists('test.relax.repack.pdb')

if file_exists == True:
    ## Import membrane-related packages
    from pyrosetta.rosetta.protocols.membrane import *

    pose = pose_from_pdb('test.relax.repack.pdb')

    ## Add membrane
    add_memb = AddMembraneMover("SW13_SUC.span")
    add_memb.apply(pose)
    init_mem_pos = pyrosetta.rosetta.protocols.membrane.MembranePositionFromTopologyMover()
    init_mem_pos.apply(pose)

    ## Create a scoring function. As of 2020, this is the up-to-date scoring function for membrane proteins
    sfxn = create_score_function("franklin2019")

    ## Perform the ddG calculation. In the actual monomer_ddG executable, Rosetta will perform the calculation 50 times.
    ## It will then return the difference of the mean of the top 3 (most stable) WT versus mean of the top 3 (most stable) mutant scores
    ## Because for some reason this version of PyRosetta mutate_residue function will not yield any output which can be stored
    ## to a variable, the purpose of this script is to simply get the data to print on the screen, which will then be saved to an output
    ## file using bash commands. The output will be parsed afterwards using 'grep', and processed from there.
    ## It is best practice to have a repacking radius of 8 Å per Rosetta documentation (assuming row3 minimization from Kellogg2011).

    print("------------------------------------------------")
    print("resid: {pos} \t wt_resname: {wt_res} \t repack_aa: {aa}")
    count = 0
    while count < 50:
        print("Iteration: " + np.str(count+1))
        pyrosetta.toolbox.mutants.mutate_residue(pose, {pos}, '{aa}', 8.0, sfxn)
        count+=1

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

elif file_exists != True:
    pose = pose_from_pdb('{pdb}')

    ## Import membrane-related packages
    from pyrosetta.rosetta.protocols.membrane import *

    ## Add membrane
    add_memb = AddMembraneMover("SW13_SUC.span")
    add_memb.apply(pose)
    init_mem_pos = pyrosetta.rosetta.protocols.membrane.MembranePositionFromTopologyMover()
    init_mem_pos.apply(pose)

    ## If you want to check that everything is good
    #print(pose.conformation())
    #print(pose.conformation().membrane_info())

    ## Create a scoring function. As of 2020, this is the up-to-date scoring function for membrane proteins
    sfxn = create_score_function("franklin2019")

    ## Add constraints prior to minimization
    constraint = pyrosetta.rosetta.protocols.constraint_generator.AtomPairConstraintGenerator()
    res_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    res_sel.set_index_range(1,220)
    constraint.set_residue_selector(res_sel)
    constraint.set_ca_only(True)
    constraint.set_use_harmonic_function(True)
    constraint.set_max_distance(5.0)
    constraint.set_sd(0.5)
    constraint.set_weight(1.0)
    add_csts = pyrosetta.rosetta.protocols.constraint_generator.AddConstraints()
    add_csts.add_generator(constraint)
    add_csts.apply(pose)
    stm = pyrosetta.rosetta.core.scoring.ScoreTypeManager()
    atom_pair_constraint = stm.score_type_from_name("atom_pair_constraint")
    sfxn.set_weight(atom_pair_constraint, 1.0)

    ## Relax your structure using minimal minimization ... save the resulting PDB
    relax = pyrosetta.rosetta.protocols.relax.membrane.MPFastRelaxMover(sfxn)
    #relax.constrain_relax_to_start_coords(True) # this is only for non-MP fast relax
    relax.apply(pose)
    pose.dump_pdb('./test.relax.repack.pdb')

    ## Load and initialize the relaxed pdb
    relaxPose = pose_from_pdb('test.relax.repack.pdb')
    add_memb = AddMembraneMover("SW13_SUC.span")
    add_memb.apply(relaxPose)
    init_mem_pos = pyrosetta.rosetta.protocols.membrane.MembranePositionFromTopologyMover()
    init_mem_pos.apply(relaxPose)

    ## Perform the ddG calculation. In the actual monomer_ddG executable, Rosetta will perform the calculation 50 times.
    ## It will then return the difference of the mean of the top 3 (most stable) WT versus mean of the top 3 (most stable) mutant scores
    ## Because for some reason this version of PyRosetta mutate_residue function will not yield any output which can be stored
    ## to a variable, the purpose of this script is to simply get the data to print on the screen, which will then be saved to an output
    ## file using bash commands. The output will be parsed afterwards using 'grep', and processed from there.
    ## It is best practice to have a repacking radius of 8 Å per Rosetta documentation (assuming row3 minimization from Kellogg2011).

    print("------------------------------------------------")
    print("resid: {pos} \t wt_resname: {wt_res} \t repack_aa: {aa}")
    count = 0
    while count < 50:
        print("Iteration: " + np.str(count+1))
        pyrosetta.toolbox.mutants.mutate_residue(pose, {pos}, '{aa}', 8.0, sfxn)
        count+=1

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

'''.format(pos=pos,wt_res=wt_res,aa=aa,pdb=pdb))
    f.close()
