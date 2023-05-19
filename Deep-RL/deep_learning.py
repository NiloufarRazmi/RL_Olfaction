import numpy as np
from utils import Sigmoid


class Network:
    def __init__(
        self,
        nInputUnits,
        nLayers,
        nOutputUnits,
        nHiddenUnits,
        initVar=1,
        activation_func="sigmoid",
    ):
        """
        Initialises the neural network.

        Parameters
        ----------
        nLayers: int
            set number of layers for neural network
        nInputUnits: int
            Number of units in the first layer
        nOutputUnits: int
            Number of units in the last layer
        nHiddenUnits: int
            Number of units in each layer
        initVar: float
            Variance for the weight initialization
            By default, the weights for the network are initialized randomly,
            using a Gaussian distribution with mean 0, and variance 1.

        Returns
        -------
        object
            a `Network` instance
        """
        self.nLayers = nLayers
        self.activation_func = Sigmoid()
        self.wtMatrix, self.nonLin = self.build_network(
            nLayers, nOutputUnits, nInputUnits, nHiddenUnits, initVar
        )

    def build_network(self, nLayers, nOutputUnits, nInputUnits, nHiddenUnits, initVar):
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

    def forward_pass(self, x_obs):
        """Generate model prediction (forward pass of activity through units)."""
        activity = [np.array([]) for _ in range(self.nLayers)]
        for layer in range(self.nLayers):
            # Determine layer input:
            if layer == 0:
                input = x_obs  # THIS WILL BE YOUR POSITION/ODOR!!!!!
            else:
                if (
                    activity[layer].shape == ()
                ):  # Convert to a vector in case it is scalar
                    activity[layer] = activity[layer, np.newaxis]
                # import ipdb; ipdb.set_trace()
                input = activity[layer - 1] @ self.wtMatrix[layer - 1]

            # Apply non-linearity
            if self.nonLin[layer]:
                activity[layer] = self.activation_func(input)
            else:
                activity[layer] = input
        return activity

    def backward_pass(self, nLayers, y_obs, activity):
        """Backpropagate errors to compute gradients for all layers."""
        delta = [np.array([]) for _ in range(nLayers)]
        for layer in reversed(range(nLayers)):
            # Determine layer input:
            if layer == nLayers - 1:
                # IF there is nonlinearity, should multiply by derivative of
                # activation with respect to input (activity.*(1-activity)) here.
                # delta[layer] = (y_obs - activity[layer]) * (
                #     self.activation_func.gradient(activity[layer])
                # ).T  # THIS SHOULD BE REPLACED WITH YOUR COST FUNCTION!

                delta[layer] = (
                    self.cost_derivative(output_activations=activity[layer], y=y_obs)
                    * (self.activation_func.gradient(activity[layer])).T
                )

                # doing this in RL framework means that you'll need one RPE for
                # each output neuron -- so RPE computed above should be
                # associated with the action agent took... all other RPEs
                # should be zero.

            else:
                # OK, here is the magic -- neurons in layer j share the
                # gradient (ie. prediction errors) from the next layer
                # according to their responsibility... that is to say, if I
                # project to a unit in next layer with a strong weight,
                # then I inherit the gradient (PE) of that unit.
                if (
                    delta[layer + 1].shape == ()
                ):  # Convert to a vector in case it is scalar
                    delta[layer + 1] = delta[layer + 1, np.newaxis]
                delta[layer] = (
                    self.wtMatrix[layer]
                    @ delta[layer + 1]
                    * (activity[layer] * (1.0 - activity[layer])).T
                )
        return delta

    def gradient_descent(self, weight, learning_rate, activity, delta):
        updated_weight = (
            weight + learning_rate * np.expand_dims(activity, axis=1) * delta.T
        )
        return updated_weight

    def backprop(self, X, y, nLayers, learning_rate):
        n_obs = len(y)
        # allError = np.nan * np.ones_like(y)
        allError = np.array([])
        # y_hat = np.nan * np.ones_like(y)
        y_hat = np.array([])

        for obs in range(n_obs):
            activity = self.forward_pass(x_obs=X[obs, :])

            # Take an action! softmax over actions or similar

            # incorporate your model of the task,
            # to determine where agent actually goes.

            # Now you need to do another forward pass, to see how good the new
            # state is so that you can compute the RPE below.

            # your cost function will differ from the one below,
            # should look something like this:
            # C =  R - X(S)*W+ DISCOUNT*max(X(S')*W)

            delta = self.backward_pass(nLayers=nLayers, y_obs=y[obs], activity=activity)

            # Update weight matrices according to gradients and activities:
            for layer in range(len(self.wtMatrix) - 1):
                # nn.wtMatrix[layer] = (
                #     nn.wtMatrix[layer]
                #     + p.learning_rate * np.expand_dims(
                #     activity[layer], axis=1) * delta[layer + 1].T
                # )
                self.wtMatrix[layer] = self.gradient_descent(
                    weight=self.wtMatrix[layer],
                    learning_rate=learning_rate,
                    activity=activity[layer],
                    delta=delta[layer + 1],
                )

            # import ipdb; ipdb.set_trace()
            # store error:
            # allError[obs] = delta[-1]
            # y_hat[obs] = activity[-1] > 0.5
            # y_hat[obs] = activity[-1]

            if allError.size == 0:
                allError = delta[-1]
            else:
                allError = np.append(allError, delta[-1])
            if y_hat.size == 0:
                y_hat = activity[-1]
            else:
                y_hat = np.append(y_hat, activity[-1])
        return allError, y_hat, delta, activity

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives $\partial C_x /
        \partial a$ for the output activations."""
        return y - output_activations
