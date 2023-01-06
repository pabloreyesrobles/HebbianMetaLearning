import gym
import torch

import numpy as np
#import pybullet_envs
from gym.spaces import Discrete, Box
from gym import wrappers as w
import pickle
import argparse
import sys

from typing import List, Any

from hebbian_weights_update import *
from policies import MLP_heb, CNN_heb
from wrappers import ScaledFloatFrame

import gym_icub_skin
from functions import *
import yarp

gym.logger.set_level(40)

def norm_goal(goal):
    return (goal - 250.0) / 250.0

# For arm denorm from tanh
def denorm(x, low, high):
    return (high - low) * ((x + 1) / 2) + low

def evaluate_hebb(hebb_rule : str, environment : str, init_weights = 'uni', render = True, *evolved_parameters: List[np.array]) -> None:
    """
    Copypasta function from fitness_functions::fitness_hebb
    It adds rendering of the environment and prints the cumulative episodic reward
    """
    
    
    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            if init_weights == 'xa_uni':  
                torch.nn.init.xavier_uniform(m.weight.data, 0.3)
            elif init_weights == 'sparse':  
                torch.nn.init.sparse_(m.weight.data, 0.8)
            elif init_weights == 'uni':  
                torch.nn.init.uniform_(m.weight.data, -0.1, 0.1)
            elif init_weights == 'normal':  
                torch.nn.init.normal_(m.weight.data, 0, 0.024)
            elif init_weights == 'ka_uni':  
                torch.nn.init.kaiming_uniform_(m.weight.data, 3)
            elif init_weights == 'uni_big':
                torch.nn.init.uniform_(m.weight.data, -1, 1)
            elif init_weights == 'xa_uni_big':
                torch.nn.init.xavier_uniform(m.weight.data)
            elif init_weights == 'default' or init_weights == None:
                pass
            
    # Unpack evolved parameters
    try: 
        hebb_coeffs, initial_weights_co = evolved_parameters
    except: 
        hebb_coeffs = evolved_parameters[0]
            
    # Intial weights co-evolution flag:
    coevolve_init = True if init_weights == 'coevolve' else False

    with torch.no_grad():
                    
        # Load environment
        # try:
        #     env = gym.make(environment, verbose = 0)
        # except:
        #     env = gym.make(environment)
        env = gym.make(environment)       

        # env.render()  # bullet envs
        pixel_env = False

        # Specific for icub-skin environment
        # Input dimension: current left arm position +  desired 2D touch point
        input_dim = 2 # 2 + env.observation_space['joints']['left_arm'].shape[0] 
        # Action dimension: left arm desired position
        action_dim = 7 # env.action_space['left_arm'].shape[0]
        # MLP model with hebbian coefficients
        p = MLP_heb(input_dim, action_dim)         
        
        # Initialise weights of the policy network with an specific distribution or with the co-evolved weights
        if coevolve_init:
            torch.nn.utils.vector_to_parameters( torch.tensor (initial_weights_co, dtype=torch.float32 ),  p.parameters() )
        else:       
            p.apply(weights_init)
             # Load CNN paramters
            if pixel_env:
                cnn_weights1 = initial_weights_co[:162]
                cnn_weights2 = initial_weights_co[162:]
                list(p.parameters())[0].data = torch.tensor(cnn_weights1.reshape((6,3,3,3))).float()
                list(p.parameters())[1].data = torch.tensor(cnn_weights2.reshape((8,6,5,5))).float()
        p = p.float()
        
        # Unpack network's weights
        if pixel_env:
            weightsCNN1, weightsCNN2, weights1_2, weights2_3, weights3_4 = list(p.parameters())
        else:
            weights1_2, weights2_3, weights3_4 = list(p.parameters())
            
        
        # Convert weights to numpy so we can JIT them with Numba
        weights1_2 = weights1_2.detach().numpy()
        weights2_3 = weights2_3.detach().numpy()
        weights3_4 = weights3_4.detach().numpy()
           

        # normalised_weights = True
        normalised_weights = True

        neg_count = 0
        rew_ep = 0
        t = 0

        modes = yarp.VectorInt(16)
        icontrol = [drv['driver'].viewIControlMode() for drv in env.motor_drivers]

        for ic in icontrol:
            ic.getControlModes(modes.data())
            for i in range(modes.length()):
                if modes[i] == yarp.VOCAB_CM_HW_FAULT:
                    modes[i] = yarp.VOCAB_CM_FORCE_IDLE
            ic.setControlModes(modes.data())

            for i in range(modes.length()):
                if modes[i] == yarp.VOCAB_CM_FORCE_IDLE:
                    modes[i] = yarp.VOCAB_CM_POSITION
            ic.setControlModes(modes.data())

        env.reset()

        goal = np.array(test_grid())

        for i_g, g in enumerate(goal):
            
            observation = norm_goal(g)
            o0, o1, o2, o3 = p([observation])
            o0 = o0.numpy()
            o1 = o1.numpy()
            o2 = o2.numpy()
            
            o3 = torch.tanh(o3).numpy()
            # o3 = o3.numpy()
            # o3 = np.clip(o3, env.action_space['left_arm'].low[0:7], env.action_space['left_arm'].high[0:7])

            low = env.action_space['left_arm'].low[0:7] - home_pose()[0:7]
            high = env.action_space['left_arm'].high[0:7] - home_pose()[0:7]
            delta = 0.5 * denorm(o3, low, high)

            action = action_home(env) # From functions.py of icub-skin repo

            action['left_arm'][0:7] += o3 * 40.0
            action['left_arm'][0:7] = np.clip(action['left_arm'][0:7], env.action_space['left_arm'].low[0:7], env.action_space['left_arm'].high[0:7]) 
            
            # Environment simulation step
            observation, reward, done, info  = env.step(action)  
            torso = observation['touch']['torso']

            if (torso[0] + torso[1]) > 0:
                rew_ep += 20.0 * np.exp(-12.5 * np.linalg.norm(g - np.array(torso)) / np.linalg.norm(np.array([500, 500])))
            rew_ep += reward
            
            #### Episodic/Intra-life hebbian update of the weights
            if hebb_rule == 'A': 
                weights1_2, weights2_3, weights3_4 = hebbian_update_A(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'AD':
                weights1_2, weights2_3, weights3_4 = hebbian_update_AD(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'AD_lr':
                weights1_2, weights2_3, weights3_4 = hebbian_update_AD_lr(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABC':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABC(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABC_lr':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABC_lr(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABCD':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABCD(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABCD_lr':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABCD_lr_D_in(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABCD_lr_D_out':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABCD_lr_D_out(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABCD_lr_D_in_and_out':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABCD_lr_D_in_and_out(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            else:
                raise ValueError('The provided Hebbian rule is not valid')


            # Normalise weights per layer
            if normalised_weights == True:
                (a, b, c) = (0, 1, 2) if not pixel_env else (2, 3, 4)
                list(p.parameters())[a].data /= list(p.parameters())[a].__abs__().max()
                list(p.parameters())[b].data /= list(p.parameters())[b].__abs__().max()
                list(p.parameters())[c].data /= list(p.parameters())[c].__abs__().max()
            
        env.close()
        
        print('\n Episode cumulative rewards ', int(rew_ep))

    
    
    
def main(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--environment', type=str, default='CarRacing-v0', metavar='', help='Gym environment: any OpenAI Gym may be used')
    parser.add_argument('--hebb_rule', type=str,  default = 'ABCD_lr', metavar='', help='Hebbian rule type: A, AD, AD_lr, ABC, ABC_lr, ABCD, ABCD_lr, ABCD_lr_D_out, ABCD_lr_D_in_and_out')    
    parser.add_argument('--init_weights', type=str,  default = 'uni', metavar='', help='Weight initilisation distribution used to sample from at each episode: uni, normal, default, xa_uni, sparse, ka_uni')
    parser.add_argument('--path_hebb', type=str,  default = None, metavar='', help='path to the evolved Hebbian coefficients')
    parser.add_argument('--path_coev', type=str,  default = None, metavar='', help='path to the evolved CNN parameters or the coevolve initial weights')

    args = parser.parse_args()

    hebb_coeffs = torch.load(args.path_hebb)
    coevolved_or_cnn_parameters = torch.load(args.path_coev) if args.path_coev is not None else None    
    init_weights = 'uni' 
    render = True
    
    # Run the environment
    evaluate_hebb(args.hebb_rule, args.environment, args.init_weights, render, hebb_coeffs, coevolved_or_cnn_parameters)
    
if __name__ == '__main__':
    main(sys.argv)

    
    
