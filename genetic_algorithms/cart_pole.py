import gym
import time
import numpy as np
import pickle

from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

INPUTS = 4
HIDDEN_LAYER = 4
OUTPUTS = 1


class CartPole:

    def __init__(self, random_seed=None):

        self.env = gym.make('CartPole-v1')

        if random_seed is not None:
            self.env.seed(random_seed)

    def __len__(self):
        return INPUTS * HIDDEN_LAYER + HIDDEN_LAYER * OUTPUTS + HIDDEN_LAYER + OUTPUTS

    @ignore_warnings(category=ConvergenceWarning)
    def init_mlp(self, net_params):
        '''
        initializes a Multilayer Perceptron (MLP) Regressor with the desired network architecture (layers)
        and network parameters (weights and biases).
        :param net_params: a list of floats representing the network parameters (weights and biases) of the MLP
        :return: initialized MLP Regressor
        '''

        # create the initial MLP:
        mlp = MLPRegressor(hidden_layer_sizes=(HIDDEN_LAYER,), max_iter=1)

        # This will initialize input and output layers, and nodes weights and biases:
        # we are not otherwise interested in training the MLP here, hence the settings max_iter=1 above
        mlp.fit(np.random.uniform(low=-1, high=1, size=INPUTS).reshape(1, -1), np.ones(OUTPUTS))

        # weights are represented as a list of 2 ndarrays:
        # - hidden layer weights: INPUTS x HIDDEN_LAYER
        # - output layer weights: HIDDEN_LAYER x OUTPUTS

        num_weights = INPUTS * HIDDEN_LAYER + HIDDEN_LAYER * OUTPUTS
        weights = np.array(net_params[:num_weights])

        mlp.coefs_ = [
            weights[0:INPUTS * HIDDEN_LAYER].reshape((INPUTS, HIDDEN_LAYER)),
            weights[INPUTS * HIDDEN_LAYER:].reshape((HIDDEN_LAYER, OUTPUTS))
        ]

        # biases are represented as a list of 2 ndarrays:
        # - hidden layer biases: HIDDEN_LAYER x 1
        # - output layer biases: OUTPUTS x 1
        biases = np.array(net_params[num_weights:])
        mlp.intercepts_ = [biases[:HIDDEN_LAYER], biases[HIDDEN_LAYER:]]

        return mlp

    def get_score(self, net_params):
        '''
        calculates the score of a given solution, represented by the list of float-valued network parameters,
        by creating a corresponding MLP Regressor, initiating an episode of the Cart-Pole environment and
        running it with the MLP controlling the actions, while using the observations as inputs.
        Higher score is better
        :param net_params: a list of floats representing the network parameters (weights and biases) of the MLP
        :return: the calculated score value
        '''

        mlp = self.init_mlp(net_params)

        self.env.reset()

        action_counter = 0
        total_reward = 0
        observation = self.env.reset()
        action = int(mlp.predict(observation.reshape(1, -1)) > 0)

        while True:
            action_counter += 1
            observation, reward, done, info = self.env.step(action)
            total_reward += reward

            if done:
                break
            else:
                action = int(mlp.predict(observation.reshape(1, -1)) > 0)
                # print(action)

        return total_reward

    def saved_params(self, net_params):
        '''
        serializes and saves a list of network parameters using pickle
        :param net_params: a list of floats representing the network parameters (weights and biases) of the MLP
        '''

        saved_params = []
        for param in net_params:
            saved_params.append(param)

        pickle.dump(saved_params, open('cart_pole_data.pickle', 'wb'))

    def replay_with_saved_params(self):
        '''
        deserializes a saved list of network parameters and uses it to replay an episode
        '''

        saved_params = pickle.load(open('cart_pole_data.pickle', 'rb'))
        self.replay(saved_params)

    def replay(self, net_params):
        '''
        renders the environment and uses the given network parameters to replay an episode, to visualize a give solution
        :param net_params: a list of floats representing the network parameters (weights and biases) of the MLP
        '''
        mlp = self.init_mlp(net_params)

        self.env.render()

        action_counter = 0
        total_reward = 0
        observation = self.env.reset()
        action = int(mlp.predict(observation.reshape(1, -1)) > 0)

        while True:
            action_counter += 1
            self.env.render()
            observation, reward, done, info = self.env.step(action)
            total_reward += reward

            print(action_counter, ':-------------------')
            print('action = ', action)
            print('observation = ', observation)
            print('reward = ', reward)
            print('total_reward = ', total_reward)
            print('done = ', done)
            print()

            if done:
                break
            else:
                time.sleep(0.03)
                action = int(mlp.predict(observation.reshape(1, -1)) > 0)

        self.env.close()


def main():
    cart = CartPole()
    cart.replay_with_saved_params()

    exit()


if __name__ == '__main__':
    main()
