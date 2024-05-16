from typing import List, Dict, Optional, Tuple, Callable
from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmGridWorld
from visualizer.farm_visualizer import InteractiveFarm
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch import optim
import random

#Neural Net Class Structure
class NNet(nn.Module):
    #Takes in a total of 4 layers - Input, 2 Hidden, 1 Output
    #Uses Dropout for optimization
    def __init__(self, input_dim: int, hidden_dim: int, hidden_dim_2: int, output_dim: int):
        super(NNet, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)  # Example dropout layer

        self.lin2 = nn.Linear(hidden_dim, hidden_dim_2)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)  # Example dropout layer

        self.lin3 = nn.Linear(hidden_dim_2, output_dim)
    
    def forward(self, x):
        x = self.drop1(self.act1(self.lin1(x)))
        x = self.drop2(self.act2(self.lin2(x)))
        x = self.lin3(x)
        return x
        


def update_dp(viz: InteractiveFarm, state_values, policy):
    viz.set_state_values(state_values)
    viz.set_policy(policy)
    viz.window.update()


def update_model_free(viz: InteractiveFarm, state, action_values):
    viz.set_action_values(action_values)
    viz.board.delete(viz.agent_img)
    viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]
    viz.window.update()


def evaluate_nnet(nnet: nn.Module, data_input_np, data_labels_np):
    nnet.eval()
    criterion = nn.CrossEntropyLoss()

    val_input = torch.tensor(data_input_np).float()
    val_labels = torch.tensor(data_labels_np).long()
    nnet_output: Tensor = nnet(val_input)

    loss = criterion(nnet_output, val_labels)

    nnet_label = np.argmax(nnet_output.data.numpy(), axis=1)
    acc: float = 100 * np.mean(nnet_label == val_labels.data.numpy())

    return loss.item(), acc


def train_nnet(train_input_np: np.ndarray, train_labels_np: np.array, val_input_np: np.ndarray,
               val_labels_np: np.array) -> nn.Module:
    """

    :param train_input_np: training inputs
    :param train_labels_np: training labels
    :param val_input_np: validation inputs
    :param val_labels_np: validation labels
    :return: the trained neural network
    """
    #Creates a neural network with the input dimensions as the starting layer then custom for hidden, then 10 for output 0-9
    nnet = NNet(784, 256, 128, 10)
    #Custom learning rate and discount
    lr = 0.0001
    lr_d = 0.99
    device = "cpu"

    nnet.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nnet.parameters(), lr=lr)

    epochs = 2
    #Runs for 2 epochs to keep in time constraint, more would result in better neural net
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        #Shuffles the data in unison for each epoch
        train_input_np, train_labels_np = shuffle_in_unison(train_input_np, train_labels_np)

        #Implements Stochastic Gradient Descent using mini_batches of size 15
        mini_batches = create_mini_batches(train_input_np, train_labels_np, 15)

        itr = 0
        #Trains the neural net
        for mini_batch in mini_batches:
            nnet = train_nnet_mini_batch(nnet, mini_batch, criterion, optimizer, lr, lr_d)

        #lp, ap = evaluate_nnet(nnet, val_input_np, val_labels_np)
        #print("Loss: ", lp, " Accuracy: ", ap)

    return nnet


def train_nnet_mini_batch(nnet, mini_batch, criterion, optimizer, lr, lr_d):
        train_itr = 0
        inputs = mini_batch[0]
        labels = mini_batch[1]
        num_itrs = len(inputs)
        #Uses Torch to train nnet
        #Based of slides
        while train_itr < num_itrs:
            optimizer.zero_grad()

            lr_itr = lr * (lr_d ** train_itr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_itr

            nnet_inputs = torch.tensor(inputs).float()
            nnet_labels = torch.tensor(labels).long()

            nnet_outputs: Tensor = nnet(nnet_inputs)
            loss = criterion(nnet_outputs, nnet_labels)

            loss.backward()

            optimizer.step()
            train_itr = train_itr + 1

        return nnet

#Function for creating mini_batches
def create_mini_batches(inputs, labels, batch_size):

    data_size = inputs.shape[0]
    
    # Create an array of indices and shuffle it
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    
    batches = []
    
    #Goes through and add's each image to a batch
    for start_idx in range(0, data_size, batch_size):
        end_idx = min(start_idx + batch_size, data_size)
        batch_indices = indices[start_idx:end_idx]
        
        batch_inputs = inputs[batch_indices]
        batch_labels = labels[batch_indices]
        
        batches.append((batch_inputs, batch_labels))
    
    return batches

#Function for shuffling up the training data
def shuffle_in_unison(train_input_np, train_labels_np):  
    indices = np.arange(len(train_input_np))
    
    np.random.shuffle(indices)

    shuffled_inputs = train_input_np[indices]
    shuffled_labels = train_labels_np[indices]
    
    return shuffled_inputs, shuffled_labels

def train_nnet_np(train_input_np: np.ndarray, train_labels_np: np.array, val_input_np: np.ndarray,
                  val_labels_np: np.array) -> Callable:
    """

    :param train_input_np: training inputs
    :param train_labels_np: training labels
    :param val_input_np: validation inputs
    :param val_labels_np: validation labels
    :return: the trained neural network
    """
    pass


def evaluate_nnet_np(nnet: Callable, data_input_np: np.ndarray, data_labels_np: np.array) -> Tuple[float, float]:
    """
    :param nnet: the trained neural network
    :param data_input_np: validation inputs
    :param data_labels_np: validation labels
    :return: the loss and the accuracy
    """
    pass


def policy_iteration(env: FarmGridWorld, states: List[State], state_values: Dict[State, float],
                     policy: Dict[State, List[float]], discount: float, policy_eval_cutoff: float,
                     viz: Optional[InteractiveFarm]) -> Tuple[Dict[State, float], Dict[State, List[float]]]:
    """
    @param env: environment
    @param states: all states in the state space
    @param state_values: dictionary that maps states to values
    @param policy: dictionary that maps states to a list of probabilities of taking each action
    @param discount: the discount factor
    @param policy_eval_cutoff: the cutoff for policy evaluation
    @param viz: optional visualizer

    @return: the state value function and policy found by policy iteration
    """
    is_converged = False

    #Based of Psuedo code
    while not is_converged:
        state_values = policy_evaluation(env, states, state_values, policy, discount, policy_eval_cutoff)
        policy, policy_stable = policy_improvement(env, states, state_values, policy, discount)
        #update_dp(viz, state_values, policy)
        if policy_stable:
            is_converged = True

    return state_values, policy


def policy_evaluation(env: FarmGridWorld, states: List[State], state_values: Dict[State, float],
                      policy: Dict[State, List[float]], discount: float, policy_eval_cutoff: float) -> Dict[State, float]:
    #Based of psuedo code, returns updated state_values
    delta = np.inf
    while delta > policy_eval_cutoff:
        delta = 0
        for state in states:
            v = state_values[state]
            new_v = 0
            actions = env.get_actions(state)
            for action in actions:
                action_prob = policy[state][action]
                exp_reward, next_states, probs = env.state_action_dynamics(state, action)
                action_value = sum(prob * (exp_reward + discount * state_values[next_state]) for next_state, prob in zip(next_states, probs))
                new_v += action_prob * action_value
            delta = max(delta, abs(v - new_v))
            state_values[state] = new_v
    return state_values

def policy_improvement(env: FarmGridWorld, states: List[State], state_values: Dict[State, float],
                       policy: Dict[State, List[float]], discount: float) -> Dict[State, List[float]]:
    #Based of equation in slides, adapted to code and fluffed out
    #Returns the updated policy and if the policy is stable which means it has converged.
    policy_stable = True
    for state in states:
        old_actions = policy[state]
        action_values = []
        for action in env.get_actions(state):
            exp_reward, next_states, probs = env.state_action_dynamics(state, action)
            action_value = sum(prob * (exp_reward + discount * state_values[next_state]) for next_state, prob in zip(next_states, probs))
            action_values.append(action_value)
        
        max_value = max(action_values)
        best_actions = [i for i, value in enumerate(action_values) if value == max_value]
        
        new_action_probs = [1.0 if i in best_actions else 0.0 for i in range(len(env.get_actions(state)))]
        
        if old_actions != new_action_probs:
            policy_stable = False
        
        policy[state] = new_action_probs
    
    return policy, policy_stable

def sarsa(env: Environment, action_values: Dict[State, List[float]], epsilon: float, learning_rate: float,
          discount: float, num_episodes: int, viz: Optional[InteractiveFarm]) -> Dict[State, List[float]]:
    
    """
    @param env: environment
    @param action_values: dictionary that maps states to their action values (list of floats)
    @param epsilon: epsilon-greedy policy
    @param learning_rate: learning rate
    @param discount: the discount factor
    @param num_episodes: number of episodes for learning
    @param viz: optional visualizer

    @return: the learned action value function
    """
    num_start_states = 20

    #Based off of pseudo code
    for episode in range(num_episodes):
        #print("New Episode!")
        goal = False
        #Initialze the state
        start_states = env.sample_start_states(num_start_states)
        selected_index = random.randint(0, num_start_states - 1)
        current_state = start_states[selected_index]

        #Get action
        actions = env.get_actions(current_state)
        current_action = epsilon_greedy(current_state, actions, action_values, epsilon)
        itr = 0
        #Loop for steps - 50 max or Goal
        while True:
            #Take Action
            next_state, reward = env.sample_transition(current_state, current_action)
            next_actions = env.get_actions(next_state)
            next_action = epsilon_greedy(next_state, next_actions, action_values, epsilon)
            
            #Q(S, A) = Q(S, A) + a[R + yQ(S', A') - Q(S, A)]
            q_current = action_values[current_state][current_action]
            q_next = action_values[next_state][next_action] if not goal else 0
            td_target = reward + discount * q_next
            td_error = td_target - q_current

            #Updates action values
            action_values[current_state][current_action] += learning_rate * td_error

            #Sets state and action to next
            current_state = next_state
            current_action = next_action

            #Checks end conditions
            goal = env.is_terminal(current_state)
            if (goal) or (itr >= 50):
                break
            itr += 1

            #update_model_free(viz, current_state, action_values)

    return action_values

#Based off of https://www.baeldung.com/cs/epsilon-greedy-q-learning
def epsilon_greedy(state, actions, action_values, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        return max(actions, key=lambda action: action_values[state][action])