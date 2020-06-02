import gym
import time
import pickle

MAX_STEPS = 200
FLAG_LOCATION = 0.5


class MountainCar:

    def __init__(self, random_seed):

        self.env = gym.make('MountainCar-v0')
        self.env.seed(random_seed)

    def __len__(self):
        return MAX_STEPS

    def get_score(self, actions):
        '''
        Calculates the score of a given solution, represented by the list of actions, for the Mountain-Car environment,
        by initiating an episode of the Mountain-Car environment and running it with the provided actions.
        Lower score is better.
        :param actions: a list of actions (values 0, 1, or 2) to be fed into the mountain cart environment
        :return: the calculated score value
        '''

        # start a new episode
        self.env.reset()
        action_counter = 0

        # feed the actions to the environment:
        for action in actions:
            action_counter += 1
            # provide an action and get feedback

            observation, reward, done, info = self.env.step(action)

            # episode over - either the car hit the flag, or 200 actions processed:
            if done:
                break

        # evaluate the results to produce the score:
        if action_counter < MAX_STEPS:
            # the car hit the flag
            # start from a score of 0
            # reward further for a smaller amount of steps

            score = 0 - (MAX_STEPS - action_counter) / MAX_STEPS
        else:
            # the car did not hit the flag:
            # reward according to distance from flag
            score = abs(observation[0] - FLAG_LOCATION)  # we want to minimize that

        return score

    def save_actions(self, actions):
        '''
        serializes and saves a list of actions using pickle
        :param actions: a list of actions (values 0, 1, or 2) to be fed into the mountain cart environment
        :return:
        '''
        saved_actions = []

        for action in actions:
            saved_actions.append(action)

        pickle.dump(saved_actions, open('mountain_car_data.pickle', 'wb'))

    def replay_saved_actions(self):
        '''
        deserializes a saved list of actions and replays it
        '''
        saved_actions = pickle.load(open('mountain_car_data.pickle', 'rb'))
        self.replay(saved_actions)

    def replay(self, actions):
        '''
        renders the environment and replays the list of actions into it, to visualize a given solution
        :param actions: a list of actions (values 0, 1, or 2) to be fed into the mountain cart environment
        '''

        # start a new episode
        observation = self.env.reset()

        # start rendering:
        self.env.render()
        action_counter = 0

        # replay the given actions by feeding them into the environment:

        for action in actions:

            action_counter += 1
            self.env.render()
            observation, reward, done, info = self.env.step(action)
            print(action_counter, ': -----------------------')
            print('action = ', action)
            print('observation = ', observation)
            print('distance from flag = ', abs(observation[0] - 0.5))
            print()

            if done:
                break
            else:
                time.sleep(0.02)

        self.env.close()


def main():
    RANDOM_SEED = 42
    car = MountainCar(RANDOM_SEED)
    car.replay_saved_actions()


if __name__ == '__main__':
    main()
