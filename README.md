# iprc
IMPLEMENT MULTILAYER PERCEPTRON:
  import numpy as np
  from sklearn.neural_network import MLPClassifier
  
  # Generate XOR training data
  X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  y = np.array([[0], [1], [1], [0]])
  
  # Create a multilayer perceptron classi3ier
  clf = MLPClassifier(hidden_layer_sizes=(3), activation='relu')
  
  # Train the classifier
  clf.fit(X, y)
  
  # Make predictions on the training data
  y_pred = clf.predict(X)
  
  # Print the predictions
  print(y_pred)
