# iprc
IMPLEMENT MULTILAYER PERCEPTRON:
  import numpy as np
  from sklearn.neural_network import MLPClassifier
  
  X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  y = np.array([[0], [1], [1], [0]])
  
  clf = MLPClassifier(hidden_layer_sizes=(3), activation='relu')
  
  clf.fit(X, y)
  
  y_pred = clf.predict(X)
  
  print(y_pred)
