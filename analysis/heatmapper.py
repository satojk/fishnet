import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For plotting coefficient matrices generated with num_pieces extractor (number 
# of pieces at the end of game)

names = {
        0: 'Lichess Dataset',
        1: 'FICS Dataset',
        }

for ix, name in enumerate(['lichess', 'fics']):
    df = pd.read_csv('coeff_numpieces_{}.csv'.format(name),
                      header=None,
                      names=list(range(1, 17)))
    df = df.reindex(index=df.index[::-1])
    plt.title('Endgame pieces coefficient matrix on the {}'.format(names[ix]))
    plt.ylabel('Black\'s remaining pieces at endgame')
    plt.xlabel('White\'s remaining pieces at endgame')
    sns.heatmap(data=df,
                center=0.0,
                cmap='RdBu',
                yticklabels=list(range(1, 17))[::-1])
    plt.savefig('NumPieces Coefficient Heatmap {}'.format(names[ix]))
    plt.show()
