# divde_data for a single subject

def divide_data(subject):
	#input parameter is the data of one subject
	
	#output parameter is the cleaned matrix for the subject - intialize below
	big_mat = []
	
	import numpy as np

	#create a list just to iterate through
	chunks = list(range(0,11))
	
	for chunk in chunks:
		#pull out chunk based on the chunk number there
		current_chunk = split2chunk(subject['labels'],subject['features'],chunk)
	
		#create a list to iterate through
		labels = ['face','cat','bottle','shoe','scissor','house','chair','scramblepix']	
	
		for label in labels:
			#for each chunk, pull out subset based on the label
			current_label = current_chunk[current_chunk[:,0]==label]

			#transform each category matrix into a numpy array
    			current_label = np.array(current_label)
		
    			#average each numpy array into a single row
    			current_label = np.mean(current_label, axis=0)
			
			#add to big matrix holding averaged trials for each label from each chunk
			big_mat.append(current_label)
		
	#subtract the averaged rest values from every trial
	avg_rest = get_rest(subject['labels'],subject['features'])
	
	#convert final output to numpy array
	big_mat =np.array(big_mat)
	
	return big_mat
	
