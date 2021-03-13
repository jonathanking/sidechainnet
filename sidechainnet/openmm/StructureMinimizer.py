import sidechainnet as scn
import sidechainnet.utils.minimize as minimize
from sidechainnet.openmm.openmmlayer import OpenMMLayer
import random
import numpy as np
import warnings

# warnings.filterwarnings('ignore')


class StructureMinimizer(object):

    def __init__(self, sequence, coords, lr=0.00001):
        self.optimum_coords = minimize.minimize_entry_coords_only(sequence, coords)
        self.openmmlayer = OpenMMLayer(sequence, coords)
        self.lr = lr

    def minimize(
        self,
        max_Iterations=100,
        tolerance=30
    ):  # Tolerance : Number of times it tries to decrease loss once loss stops decreasing
        np.random.seed(100)
        if max_Iterations == -1:
            tillbest_loss = float("inf")
            counter = 0
            while True:
                loss = self.openmmlayer()
                if loss.item() < tillbest_loss:
                    counter = 0
                    tillbest_loss = loss.item()
                    self.optimum_coords = self.openmmlayer.coords
                    print('Till Minimum Loss', tillbest_loss)
                else:
                    counter += 1
                    print('Loss did not decline for', counter, 'time', sep=' ')
                    if counter >= tolerance:
                        return
                loss.backward()
                self.openmmlayer.step(self.lr)

        else:
            for i in range(max_Iterations):
                loss = self.openmmlayer()
                loss.backward()
                self.openmmlayer.step(self.lr)
                print(loss.item())

    def get_optimum_coords(self):
        return self.optimum_coords


def inject_noise(coords):
    """
    :param coords: The co-ordinates of sidechainnet. Dimension L x 14 x 3, where L is the number of residues
    :return: The co-ordinates altered with random noise. All zero co-ordinates means missing atoms. Those are not altered.
    """
    nonzero = np.nonzero(coords)
    for i in range(len(nonzero[0])):
        coords[nonzero[0][i]][nonzero[1][i]] += random.rand(-1, 1) * 0.2
    return coords


if __name__ == '__main__':
    data = scn.load("debug")
    seq = data['train']['seq'][1]
    coords = data['train']['crd'][1]
    sm = StructureMinimizer(sequence=seq, coords=coords)
    sm.minimize(
        max_Iterations=100
    )  # When set to -1 it finds the local minima irrespective of the number of iterations
    print(sm.get_optimum_coords())