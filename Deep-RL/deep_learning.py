import numpy as np
from tqdm import tqdm
from utils import Sigmoid


class Network:
    def __init__(
        self,
        nInputUnits,
        nLayers=None,
        nOutputUnits=None,
        nHiddenUnits=None,
        initVar=None,
        activation_func="sigmoid",
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

        self.activation_func = Sigmoid()
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

    def forward_pass(self, nLayers, X, i):
        """Generate model prediction (forward pass of activity through units)."""
        activity = [np.array([]) for _ in range(nLayers)]
        for j in range(nLayers):
            # Determine layer input:
            if j == 0:
                input = X[i, :]  # THIS WILL BE YOUR POSITION/ODOR!!!!!
            else:
                if activity[j].shape == ():  # Convert to a vector in case it is scalar
                    activity[j] = activity[j, np.newaxis]
                input = activity[j - 1] @ self.wtMatrix[j - 1]

            # Apply non-linearity
            if self.nonLin[j]:
                activity[j] = self.activation_func(input)
            else:
                activity[j] = input
        return activity

    def backward_pass(self, nLayers, Y, activity, i):
        """Backpropagate errors to compute gradients for all layers."""
        delta = [np.array([]) for _ in range(nLayers)]
        for j in reversed(range(nLayers)):
            # Determine layer input:
            if j == nLayers - 1:
                # IF there is nonlinearity, should multiply by derivative of
                # activation with respect to input (activity.*(1-activity)) here.
                delta[j] = (Y[i] - activity[j]) * (
                    self.activation_func.gradient(activity[j])
                ).T  # THIS SHOULD BE REPLACED WITH YOUR COST FUNCTION!

                # doing this in RL framework means that you'll need one RPE for
                # each output neuron -- so RPE computed above should be
                # associated with the action agent took... all other RPEs
                # should be zero.

            else:
                # OK, here is the magic -- neurons in layer j share the
                # gradient (ie. prediction errors) from the next layer
                # according to their responsibility... that is to say, if I
                # project to a unit in next layer with a strong weight,
                # then i inherit the gradient (PE) of that unit.
                if delta[j + 1].shape == ():  # Convert to a vector in case it is scalar
                    delta[j + 1] = delta[j + 1, np.newaxis]
                delta[j] = (
                    self.wtMatrix[j]
                    @ delta[j + 1]
                    * (activity[j] * (1.0 - activity[j])).T
                )
        return delta

    def gradient_descent(self, weight, learning_rate, activity, delta):
        updated_weight = (
            weight + learning_rate * np.expand_dims(activity, axis=1) * delta.T
        )
        return updated_weight

    def backprop(self, n_obs, X, Y, nLayers, learning_rate):
        allError = np.nan * np.ones(n_obs)
        catPredict = np.nan * np.ones(n_obs)

        for i in tqdm(range(n_obs)):
            activity = self.forward_pass(nLayers=nLayers, X=X, i=i)

            # Take an action! softmax over actions or similar

            # incorporate your model of the task,
            # to determine where agent actually goes.

            # Now you need to do another forward pass, to see how good the new
            # state is so that you can compute the RPE below.

            # your cost function will differ from the one below,
            # should look something like this:
            # C =  R - X(S)*W+ DISCOUNT*max(X(S')*W)

            delta = self.backward_pass(nLayers=nLayers, Y=Y, activity=activity, i=i)

            # Update weight matrices according to gradients and activities:
            for j in range(len(self.wtMatrix) - 1):
                # nn.wtMatrix[j] = (
                #     nn.wtMatrix[j]
                #     + p.learning_rate * np.expand_dims(
                #     activity[j], axis=1) * delta[j + 1].T
                # )
                self.wtMatrix[j] = self.gradient_descent(
                    weight=self.wtMatrix[j],
                    learning_rate=learning_rate,
                    activity=activity[j],
                    delta=delta[j + 1],
                )

            # store error:
            allError[i] = delta[-1]
            catPredict[i] = activity[-1] > 0.5
        return allError, catPredict, delta, activity
