import pickle
import numpy as np
import matplotlib.pyplot as plt

# For plotting accuracies over many n (used for sparse vector)

with open('outputs.pkl', 'rb') as pkl:
    outputs_brute = pickle.load(pkl)

with open('outputs_large.pkl', 'rb') as pkl:
    outputs_large = pickle.load(pkl)

names = {
        0: 'Lichess Dataset',
        1: 'FICS Dataset',
        'Log-Reg': 'Logistic Regression',
        'SVM RBF': 'SVM with RBF Kernel'
        }

for ix, outputs in enumerate([outputs_brute, outputs_large]):
    xs = sorted(outputs.keys())
    for model in ['Log-Reg', 'SVM RBF']:
        ys_train = [outputs[x][model][0] for x in xs]
        ys_test = [outputs[x][model][1] for x in xs]
        plt.title('{} on the {}'.format(
                  names[model], names[ix]))
        plt.ylabel('Model Score')
        plt.xlabel('First X moves')
        plt.plot(xs, ys_train, 'b--', xs, ys_test, 'bo')
        plt.axis([0, 36, 0.5, 1.05])
        plt.savefig('{} {}'.format(names[ix], names[model]))
        plt.show()
