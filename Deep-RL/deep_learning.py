import numpy as np


class Network:
    def __init__(
        self,
        nInputUnits,
        nLayers=None,
        nOutputUnits=None,
        nHiddenUnits=None,
        initVar=None,
    ):
        if nLayers is None:
            nLayers = 5  # set number of layers for neural network

        # Number of units in each layer
        if nOutputUnits is None:
            nOutputUnits = 1
        if nHiddenUnits is None:
            nOutputUnits = 20
        if initVar is None:
            initVar = 1

        self.wtMatrix, self.nonLin = self.build_network(
            nLayers, nOutputUnits, nInputUnits, nHiddenUnits, initVar
        )

    def build_network(self, nLayers, nOutputUnits, nInputUnits, nHiddenUnits, initVar):
        initVar = 1
        nonLin = np.block(
            [False, np.ones((1, nLayers - 2), dtype=bool), True]
        ).squeeze()

        # Create initial weight matrices
        nUnits = np.block(
            [nInputUnits, np.repeat(nHiddenUnits, nLayers - 2), nOutputUnits]
        )
        wtMatrix = []
        for i in range(0, nLayers - 1):
            wtMatrix.append(np.random.normal(0, initVar, (nUnits[i], nUnits[i + 1])))
        return wtMatrix, nonLin
