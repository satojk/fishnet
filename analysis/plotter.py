import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('outputs.pkl', 'rb') as pkl:
    outputs_brute = pickle.load(pkl)

with open('outputs_coords.pkl', 'rb') as pkl:
    outputs_coord = pickle.load(pkl)

names = {
        0: 'Sparse vector representation',
        1: 'Coordinate representation',
        'Log-Reg': 'Logistic Regression',
        'SVM Poly': 'SVM with Polynomial Kernel'
        }

for ix, outputs in enumerate([outputs_brute, outputs_coord]):
    xs = sorted(outputs.keys())
    for model in ['Log-Reg', 'SVM Poly']:
        ys_train = [outputs[x][model][0] for x in xs]
        ys_test = [outputs[x][model][1] for x in xs]
        plt.title('{} classifier using {}'.format(
                  names[model], names[ix]))
        plt.ylabel('Model Score')
        plt.xlabel('First X moves')
        plt.plot(xs, ys_train, 'b--', xs, ys_test, 'bo')
        plt.axis([0, 40, 0.5, 1.05])
        plt.savefig('{} {}'.format(names[ix], names[model]))
        plt.show()
