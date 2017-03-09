# divde_data for a single subject
def divide_data(labels, features):
	#INPUT parameters are the data for one subject
	#subject = np.hstack((labels, features)) #combine into one matrix 
	
	#OUTPUT parameter is the cleaned matrix for the subject 
	#this final output is a matrix where the first two columns are the label and the chunk number (ie, "y")
	#the rest of the columns are feature values (ie, "X")
	#each row in the matrix corresponds to a single trial in a chunk that has been averaged from all timepoints for that trial
	#a trial is basically one attempt at a categorization
	
	#intialize a temp variable below
	big_mat = []
	
	import numpy as np

	#create a list just to iterate through
	chunks = list(range(0,11)) #the number of possible chunks
	
	for chunk in chunks: #repeat for every chunk
		#pull out chunk based on the current chunk number being looked at
		current_chunk = split2chunk(labels,features,chunk)
		#current_chunk = subject[subject[:,1] == chunk]
	
		#create a list of the categories to iterate through 
		categories = ['face','cat','bottle','shoe','scissor','house','chair','scramblepix']	
	
		for category in categories: #repeat for every category within each chunk
			#for each chunk, pull out subset based on the category label
			current_cat = split2cat(category,current_chunk)
			#current_cat = current_chunk[current_chunk[:,0]==category]

			#transform each category matrix into a numpy array
    			current_cat = np.array(current_cat)
		
    			#average each numpy array into a single row
    			current_cat = np.mean(current_cat, axis=0)
			
			#add to big matrix holding averaged trials for each category from each chunk
			big_mat.append(current_cat)
		
	#convert final output to numpy array
	big_mat =np.array(big_mat)
	
	#subtract the averaged rest values from every trial
	avg_rest = get_rest(labels,features) #get vector of average rest values
	big_mat_features = big_mat[:,2:] #get only the features from our final cleaned data
	big_mat_labels = big_mat[:,0:2] #also store only the labels to put together with final features later
	final_features = big_mat_features - avg_rest
	
	#put final labels and features together
	final_data = np.hstack((big_mat_labels, final_features))
	
	return final_data
	
