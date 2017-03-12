##################################################################################
#
# By Bryce Wong
# Implements an SVM model with RBF kernels
#
#################################################################################
def SVM_rbf_kernel(labels, features):
  #@ param labels are the labels for one subject's cleaned/preprocessed dataset
  #@ param features are the features for one subject's cleaned/preprocessed dataset
  
  # Import our scikit-learn stuff
  from sklearn.svm import SVC
  from sklearn.grid_search import GridSearchCV
  from sklearn.cross_validation import StratifiedKFold
  
  # Set a range of possible values for our C parameter and gamma parameter to iterate through
  possible_C = np.logspace(-3, 9, 13)
  possible_gamma = np.logspace(-7, 4, 12)
  
  # Fill a grid with our possibe combinations of C and gamma values
  param_grid = dict(gamma=possible_gamma, C=possible_C)
  
  # Create our cross-validation function
  cv = StratifiedKFold(labels[:,0], 10) # Uses our labels as our y-vector, makes 10 folds
  
  # Create our svm model
  svc = SVC()
  
  # Cross-validate our parameters in our grid to find best combination of the params
  grid = GridSearchCV(svc, param_grid=param_grid, cv=cv)
  grid.fit(features, labels[:,0])
  #print(grid.best_params_)

  # Create our svm model with rbf kernels using our optimal params
  svc_rbf = SVC(**grid.best_params_, kernel="rbf")
  svc_rbf.fit(features, labels[:,0])

  # Turn it into a pipeline if we want?
  svc_rbf_model = pp.make_pipeline(preprocessor, SVC(**grid.best_params_, kernel="rbf"))

  # Fit our pipelined SVM model if we want?
  svc_rbf_model.fit(features, labels[:,0])
  y_pred_svc = svc_rbf_model.decision_function(test_X)
  score1 = svc_rbf_model.score(test_X, test_y)
  print("The SVC model with RBF kernals has an accuracy score of:", score1)
  
  return
