import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = [
[10.0, 0.046, 0.918],
[10.0, 0.041, 0.918],
[ 1.0, 0.036, 0.913],
[ 0.1, 0.001, 0.612],
[ 0.1, 0.006, 0.765],
[ 0.1, 0.011, 0.819],
[ 0.1, 0.016, 0.821],
[ 0.1, 0.021, 0.769],
[ 0.1, 0.026, 0.694],
[ 0.1, 0.031, 0.633],
[ 0.1, 0.036, 0.612],
[ 0.1, 0.041, 0.612],
[ 0.1, 0.046, 0.612],
[ 0.1, 0.051, 0.612],
[ 0.1, 0.056, 0.612],
[ 0.1, 0.061, 0.612],
[ 0.1, 0.066, 0.612],
[ 0.1, 0.071, 0.612],
[ 0.1, 0.076, 0.612],
[ 0.1, 0.081, 0.612],
[ 0.1, 0.086, 0.612],
[ 0.1, 0.091, 0.612],
[ 0.1, 0.096, 0.612],
[ 0.5, 0.001, 0.810],
[ 0.5, 0.006, 0.881],
[ 0.5, 0.011, 0.904],
[ 0.5, 0.016, 0.906],
[ 0.5, 0.021, 0.907],
[ 0.5, 0.026, 0.906],
[ 0.5, 0.031, 0.902],
[ 0.5, 0.036, 0.899],
[ 0.5, 0.041, 0.897],
[ 0.5, 0.046, 0.885],
[ 0.5, 0.051, 0.866],
[ 0.5, 0.056, 0.812],
[ 0.5, 0.061, 0.744],
[ 0.5, 0.066, 0.689],
[ 0.5, 0.071, 0.649],
[ 0.5, 0.076, 0.633],
[ 0.5, 0.081, 0.630],
[ 0.5, 0.086, 0.630],
[ 0.5, 0.091, 0.625],
[ 0.5, 0.096, 0.621],
[ 1.0, 0.001, 0.862],
[ 1.0, 0.006, 0.901],
[ 1.0, 0.011, 0.902],
[ 1.0, 0.016, 0.907],
[ 1.0, 0.021, 0.911],
[ 1.0, 0.026, 0.911],
[ 1.0, 0.031, 0.911],
[ 1.0, 0.041, 0.911],
[ 1.0, 0.046, 0.911],
[ 1.0, 0.051, 0.911],
[ 1.0, 0.056, 0.911],
[ 1.0, 0.061, 0.899],
[ 1.0, 0.066, 0.890],
[ 1.0, 0.071, 0.875],
[ 1.0, 0.076, 0.854],
[ 1.0, 0.081, 0.828],
[ 1.0, 0.086, 0.789],
[ 1.0, 0.091, 0.751],
[ 1.0, 0.096, 0.715],
[10.0, 0.001, 0.883],
[10.0, 0.006, 0.907],
[10.0, 0.011, 0.911],
[10.0, 0.016, 0.911],
[10.0, 0.021, 0.918],
[10.0, 0.026, 0.920],
[10.0, 0.031, 0.916],
[10.0, 0.036, 0.916],
[10.0, 0.051, 0.914],
[10.0, 0.056, 0.909],
[10.0, 0.061, 0.909],
[10.0, 0.066, 0.902],
[10.0, 0.071, 0.885],
[10.0, 0.076, 0.864],
[10.0, 0.081, 0.843],
[10.0, 0.086, 0.819],
[10.0, 0.091, 0.789],
[10.0, 0.096, 0.756],
[10.0, 0.086, 0.819],
[10.0, 0.091, 0.789],
]

Cs = [0.1, 0.5, 1, 10]
gammas = [0.001, 0.006, 0.011, 0.016, 0.021, 0.026, 0.031, 0.036, 0.041, 0.046, 0.051, 0.056, 0.061, 0.066, 0.071, 0.076, 0.081, 0.086, 0.091, 0.096]

table = np.zeros((len(gammas), len(Cs)))
for C, gamma, acc in data:
    table[gammas.index(gamma), Cs.index(C)] = acc

plt.title('SVM RBF Accuracy')
sns.heatmap(data=table, cmap='YlOrRd', annot=True, xticklabels=Cs, yticklabels=gammas)
plt.xlabel('Regularization term C')
plt.ylabel('RBF Gamma')
plt.savefig('svm_rbf.pdf')
plt.show()
