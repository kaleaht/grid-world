import numpy

def make_random_policy(robot):
    def random_policy(state):
        return numpy.random.choice(robot.A, 1)[0]

    return random_policy


def q_learning(robot, policy, init_state, discount_factor=1.0, num_iterations=10000):
    """
    Q-Learning Algorithm (Algorithm 3 in the lecture slides)
    :param robot: instance of Robot class
    :param policy: policy of rumba, mapping from state space to action space
    :param init_state: Initial state of rumba
    :param discount_factor: Gamma discount factor
    :param num_iterations: number of iterations
    :return: Q, the optimal action-value function, mapping from states to action values
    """

    # Initially your Q is zero matrix of shape (# states, # actions)
    Q = numpy.zeros((robot.nSp, robot.nA))

    i = 0

    # Start from the initial state
    cur_state = init_state
    while i <= num_iterations:
        i += 1

        ## STUDENT TASK ##
        # Choose the next action according to the policy
        # Policy is a function that takes state and returns an action
        action = ...

        ## STUDENT TASK ##
        # Get the state and reward by taking action from the current state
        new_state, reward = ...

        ## STUDENT TASK ##
        # Update current Q value for current state and action
        # using equation (9) in Algorithm 3 from the lecture slides
        Q[cur_state][action] = ...

        cur_state = new_state

    return Q


def get_optimal_path(robot, Q):
    """
    Extract the optimal path (list of states) from the initial state to the terminal state
    according to action-value function Q.
    :param robot: instance of Robot class
    :param Q: action-value function, mapping from states to action values
    :return: path (list of states) from starting point to the charging station (terminal state)
    """

    path = []
    init_state = robot.S_start[0]

    ## STUDENT TASK ##
    # Starting from the initial state find the optimal path given your Q function to the terminal state.
    # You can test if your state is terminal by calling robot.is_terminal(your_state)
    # Iterate until you will reach the terminal state.
    while ...

    return path
