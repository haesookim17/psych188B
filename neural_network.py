#PREPROCESS BEFORE THIS
#INSTALL AND CONFIGURE KERAS BEFORE THIS

####################################################################################
#
# (thanks Jeff)                                                                    
# This neural_network function takes in our cleaned data set and runs a neural network model on it    
#
####################################################################################

def neural_network(labels,features):
  # @ param labels takes in the labels of one subject's cleaned data set
  # @ param features takes in the features of one subject's cleaned data set
  
  #split into training and testing set
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(features, labels[;,0], test_size=.2)

  # specify the hidden layer sizes: --> this is up to you to decide specifications
  layer_sizes = [10, 5]

  # Keras uses the Sequential model for linear stacking of layers.
  # That is, creating a neural network is as easy as (later) defining the layers!
  from keras.models import Sequential
  model = Sequential()
  
  # Use the dropout regularization method
  from keras.layers import Dropout

  # Now that we have the model, let's add some layers:
  from keras.layers.core import Dense, Activation
  # Everything we've talked about in class so far is referred to in 
  # Keras as a "dense" connection between layers, where every input 
  # unit connects to a unit in the next layer

  # First a fully-connected (Dense) hidden layer with appropriate input
  # dimension, 10 outputs, and ReLU activation
  #THIS IS THE INPUT LAYER
  model.add(Dense(
      input_dim=X_train.shape[1], output_dim=layer_sizes[0]
  ))
  model.add(Activation('relu'))

  #ADD DROPOUT --> MUST DECIDE PERCENTAGE OF INPUT UNITS TO DROPOUT
  model.add(Dropout(.2))

  # Now our second hidden layer with 10 inputs (from the first
  # hidden layer) and 5 outputs. Also with ReLU activation
  #THIS IS HIDDEN LAYER
  model.add(Dense(
      input_dim=layer_sizes[0], output_dim=layer_sizes[1]
  ))
  model.add(Activation('relu'))

  #ADD DROPOUT
  model.add(Dropout(.2))

  # Finally, add a readout layer, mapping the 5 hidden units
  # to two output units using the softmax function
  #THIS IS OUR OUTPUT LAYER
  model.add(Dense(output_dim=np.unique(y_train).shape[0], init='uniform'))
  model.add(Activation('softmax'))

  # Next we let the network know how to learn
  from keras.optimizers import SGD
  sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
  model.compile(loss='categorical_crossentropy', optimizer=sgd)

  # Before we can fit the network, we have to one-hot vectorize our response.
  # Fortunately, there is a keras method for that.
  from keras.utils.np_utils import to_categorical
  #for each of our 8 categories, map an output
  y_train_vectorized = to_categorical(y_train)

  #print out shape
  y_train_vectorized.shape

  #remember that the bigger the nb_epoch the better the fit (so go bigger than 50)
  model.fit(X_train, y_train_vectorized, nb_epoch=50, batch_size=50, verbose = 0)

  #now our neural network works like a scikit-learn classifier
  proba = model.predict_proba(X_test, batch_size=32)

  # Print the accuracy:
  from sklearn.metrics import accuracy_score
  classes = np.argmax(proba, axis=1)
  accuracy_score(y_test, classes)
  
  return 
