## Install and perform ProSE embeddings

Input files for ProSE embeddings are AtSW13_TM4_muts.fa and MPTherm_database.fa .

Follow the instructions found at https://github.com/tbepler/prose.
Specifically:
	- Download the pre-trained embedding models
	- Setup python environment

Once installation is done, perform the ProSE embedding as follows:

1. Prepare your sequences in a single fasta file (with .fa extension, NOT .fasta)
2. Use the embed_sequences file:

	`python embed_sequences.py --pool avg -o output.h5 input.fa`

   One can type `python embed_sequences.py --help` for more options. Default model is `prose_mt` .

3. The output file is in .h5 format. You can convert parts of the .h5 file into a .npy array 
with the embedding valus as follows:

	```
	## install h5py
	## https://docs.h5py.org/en/stable/build.html
	import h5py
	import numpy as np
	
	# Load the .h5 file and visualize keys
	file = h5py.File('AtSW13_TM4_muts_avg.h5','r')
	file.keys()

	# consider example of just the wildtype sequence key
	print(type(file['Q9FGQ2_WT']))

	# provided key, convert embedding into an array using either option
	# note that this will have to be repeated for all desired keys
	data_list = list(file['Q9FGQ2_WT'])
	data_array = (file['Q9FGQ2_WT'])[()] #this creates a numpy array

	# save array as .npy file
	np.save('example.npy', data_array)
	```
