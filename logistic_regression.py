###################################################################################
#
# By Bryce
# Run a logistic regression model
##################################################################################
def logistic_regression(labels,features):
  #@ param labels are the labels of one subject's cleaned/preprocessed dataset
  #@ param features are the features of one subject's cleaned/preprocessed dataset
  
  #split into training and testing set
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(features, labels[;,0], test_size=.2)
    
  # Import our scikit-learn function
  from sklearn.linear_model import LogisticRegression
  
  # Create our logistic regression model
  logreg = LogisticRegression()
  
  # Train our model on our data
  logreg.fit(X_train, y_train)
  y_pred_lr = logreg.decision_function(test_X)
  
  # Test our model and score it
  score3 = logreg.score(test_X, test_y)
  print("The logistic regression model has an accuracy score of:", score3)
  
  return
