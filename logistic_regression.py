###################################################################################
#
# By Bryce
# Run a logistic regression 
##################################################################################
def logistic_regression(labels,features):
  #@ param labels are the labels of one subject's cleaned/preprocessed dataset
  #@ param features are the features of one subject's cleaned/preprocessed dataset
    
  # Import our scikit-learn functions
  from sklearn.linear_model import LogisticRegression
  
  # Create our logistic regression model
  logreg = LogisticRegression()
  
  # Train our model on our data
  logreg.fit(preProc_X, y)
  y_pred_lr = logreg.decision_function(test_X)
  
  # Test our model and score it
  score3 = logreg.score(test_X, test_y)
  print("The logistic regression model has an accuracy score of:", score3)
  
  return
