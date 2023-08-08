import gym
from gym import wrappers as w
from gym.spaces import Discrete, Box, Dict
#import pybullet_envs
import numpy as np
import torch
import torch.nn as nn
from typing import List, Any

#import CoppeliaGym

from policies import MLP_heb, CNN_heb
from hebbian_weights_update import *
from wrappers import FireEpisodicLifeEnv, ScaledFloatFrame

from functions import *
import yarp
import rospy
from std_srvs.srv import Empty
import time

from gp_shape import GaussianProcessShapeRegressor
from scipy.ndimage import gaussian_filter
import h5py

def norm_goal(goal):
    return (goal - 250.0) / 250.0

# For arm denorm from tanh
def denorm(x, low, high):
    return (high - low) * ((x + 1) / 2) + low

# Define a custom function to apply the Gaussian filter at specified locations
def apply_gaussian_filter(data, center, sigma=20):
    # Create a filter with the desired size (radius)
    filter_size = int(6 * sigma)  # Adjust as needed
    
    win = np.array([center[0] - filter_size / 2, center[0] + 1 + filter_size / 2, center[1] - filter_size / 2, center[1] + 1 + filter_size / 2], dtype=np.uint16)
    filt = data[win[0] : win[1], win[2] : win[3]]

    # Apply the Gaussian filter to the data at the specified center
    filtered_data = gaussian_filter(filt, sigma=sigma, mode='constant')
    
    return filtered_data.sum()

def fitness_hebb(hebb_rule : str, environment : gym.Env, goal: List[np.array], init_weights = 'uni', *evolved_parameters: List[np.array]) -> float:
    """
    Evaluate an agent 'evolved_parameters' controlled by a Hebbian network in an environment 'environment' during a lifetime.
    The initial weights are either co-evolved (if 'init_weights' == 'coevolve') along with the Hebbian coefficients or randomly sampled at each episode from the 'init_weights' distribution. 
    Subsequently the weights are updated following the hebbian update mechanism 'hebb_rule'.
    Returns the episodic fitness of the agent.
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
            elif init_weights == 'ones':
                torch.nn.init.ones_(m.weight.data)
            elif init_weights == 'zeros':
                torch.nn.init.zeros_(m.weight.data)
            elif init_weights == 'default':
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

        retry = True
        env = environment
        env.reset_simulation()
        time.sleep(0.5)
        
        while retry:
            retry = False
            #rospy.wait_for_service('/gazebo/reset_simulation')
            #reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            #reset_simulation()
            
            #env = gym.make('icub_skin-v0')
            env.reset()

            modes = yarp.VectorInt(16)

            for ic in env.icontrol:
                ic.getControlModes(modes.data())
                for i in range(modes.length()):
                    if modes[i] == yarp.VOCAB_CM_HW_FAULT:
                        retry = True
                        modes[i] = yarp.VOCAB_CM_FORCE_IDLE
                ic.setControlModes(modes.data())

                for i in range(modes.length()):
                    if modes[i] == yarp.VOCAB_CM_FORCE_IDLE:
                        modes[i] = yarp.VOCAB_CM_POSITION
                ic.setControlModes(modes.data())
            
            if retry:
                env.reset_simulation()
                time.sleep(0.5)
                
        # env.render()  # bullet envs
        pixel_env = False

        # Specific for icub-skin environment
        # Input dimension: current left arm position +  desired 2D touch point
        effector_pose_size = 7
        gaussian_kernels = 29
        input_dim = effector_pose_size + gaussian_kernels #env.N_TAXELS_TORSO # 2 + env.observation_space['joints']['left_arm'].shape[0] 
        # Action dimension: left arm desired position
        action_dim = 7 # x, y, z
        # MLP model with hebbian coefficients
        p = MLP_heb(input_dim, action_dim)                  
        
        # Initialise weights of the policy network with an specific distribution or with the co-evolved weights
        if coevolve_init:
            nn.utils.vector_to_parameters( torch.tensor (initial_weights_co, dtype=torch.float32 ),  p.parameters() )
        else:       
            # Randomly sample initial weights from chosen distribution
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

        # Normalize weights flag for non-bullet envs
        normalised_weights = True

        # Inner loop
        neg_count = 0
        rew_ep = 0
        t = 0

        #modes = yarp.VectorInt(16)
        #icontrol = [drv['driver'].viewIControlMode() for drv in env.motor_drivers]

        #reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        #reset_simulation()

        # retry = True
        # env.set_action_cmd('motors')
        # for ic in icontrol:
        #     ic.getControlModes(modes.data())
        #     for i in range(modes.length()):
        #         modes[i] = yarp.VOCAB_CM_FORCE_IDLE
        #     ic.setControlModes(modes.data())

        #     for i in range(modes.length()):
        #         if modes[i] == yarp.VOCAB_CM_FORCE_IDLE:
        #             modes[i] = yarp.VOCAB_CM_POSITION
        #     ic.setControlModes(modes.data())
        
        env.set_action_cmd('effector')
        env.icontrol[1].getControlModes(modes.data())
        for i in range(7):
            modes[i] = yarp.VOCAB_CM_POSITION_DIRECT
        env.icontrol[1].setControlModes(modes.data())

        #steps = int((goal[0] + 1) * 5)
        steps = int(-(90)*np.exp(-0.1 * goal[0]) + 100)

        def pose_norm(data, norm_min, norm_max, original_min, original_max):
            return (data - original_min) / (original_max - original_min) * (norm_max - norm_min) + norm_min

        # Define the desired range for normalization
        min_value = np.array([0.0, 0.0])
        max_value = np.array([1.0, 1.0])

        # Calculate the original minimum and maximum values
        original_min = np.array([-0.08, 0.0])
        original_max = np.array([0.08, 0.3])

        # Generate a grid of points in the 2D plane
        pos_grid = np.linspace(0.0, 1.0, 100)
        xx, yy = np.meshgrid(pos_grid, pos_grid)
        pos_grid = np.column_stack((xx.ravel(), yy.ravel()))

        # Initialize the shape model
        gp_regressor = GaussianProcessShapeRegressor(alpha=0.002, n_restarts_optimizer=10)
        init_std = None

        kcenters = np.load('kcenters.npy')

        observation = env.get_obs()
        state = np.concatenate([observation['effector_pose'], np.zeros(29)]) #or maybe included in env.get_obs()
        no_reward_cnt = 0

        pose_arr = []

        #for i_g in range(steps):
        while True:
            # Initialize an array to store the activations of each kernel
            sigma = 20
            kernel_activations = []
            touch_data = np.zeros([515, 515])

            for j, t in enumerate(observation['skin']['torso']):
                if t != 0.0:
                    t_x, t_y = env.SKIN_COORDINATES[-1][1][j], env.SKIN_COORDINATES[-1][2][j]
                    touch_data[t_x, t_y] = t / 255.0

            # Iterate over the centers of the Gaussian kernels
            for center in kcenters:
                # Apply the Gaussian filter at the specified center
                kernel_activation = apply_gaussian_filter(touch_data, center, sigma=sigma)

                # Append the activation to the array
                kernel_activations.append(kernel_activation)

            state = np.concatenate([observation['effector_pose'], kernel_activations])
            
            def norm_cmd(cmd):
                arm_low, arm_high = env.action_space['left_arm'].low[0:7], env.action_space['left_arm'].high[0:7]
                return 2 * (cmd - arm_low) / (arm_high - arm_low) - 1

            #observation = np.concatenate([norm_cmd(env.get_obs()['joints']['left_arm'][0:7]), norm_goal(g)])
            #observation = norm_goal(g)            

            o0, o1, o2, o3 = p([state])
            o0 = o0.numpy()
            o1 = o1.numpy()
            o2 = o2.numpy()

            # TODO: check what is the best activation for the end effector pose
            o3 = torch.tanh(o3).numpy() * 0.1
            # o3 = o3.numpy()
            # o3 = np.clip(o3, env.action_space['left_arm'].low[0:7], env.action_space['left_arm'].high[0:7])

            # low = env.action_space['left_arm'].low[0:7] - home_pose()[0:7]
            # high = env.action_space['left_arm'].high[0:7] - home_pose()[0:7]
            # delta = 0.5 * denorm(o3, low, high)

            # action = action_home(env) # From functions.py of icub-skin repo

            # action['left_arm'][0:7] += o3 * 40.0
            # action['left_arm'][0:7] = np.clip(action['left_arm'][0:7], env.action_space['left_arm'].low[0:7], env.action_space['left_arm'].high[0:7]) 
            # action['left_arm'][0:7] += delta

            # TODO: action will be current pose += o3
            action = np.zeros(7)
            action += o3

            # Environment simulation step
            observation, reward, done, info = env.step(action)
            torso = observation['touch']['torso']

            if reward > 0:
                no_reward_cnt = 0
                #with open('output/effector_pos/pos_' + str(goal[0]) + '_' + str(goal[1]) + '.txt', 'a+') as f:
                pose = observation['effector_pose']
                pose_arr.append(pose)

                X_norm = pose_norm(pose[1:3], min_value, max_value, original_min, original_max)
                y_norm = pose_norm(pose[0], 0.0, 1.0, -0.2, 0.0)

                gp_regressor.add_data_point(X_norm, y_norm)
                #f.write(f'{pose[0]:f},{pose[1]:f},{pose[2]:f}\n')

                if init_std == None:
                    _, std = gp_regressor.predict_shape(pos_grid)
                    init_std = np.mean(std)
            else:                
                no_reward_cnt += 1
 
            # TODO: no nearby point search criteria needed
            # if (torso[0] + torso[1]) > 0:
            #     rew_ep += 25.0 * np.exp(-15.0 * np.linalg.norm(g - np.array(torso)) / np.linalg.norm(np.array([500, 500])))
            rew_ep += reward

            if no_reward_cnt == 10:
                break
            
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
        
        #env.close()
        if rew_ep > 0:
            _, std = gp_regressor.predict_shape(pos_grid)
            int_rew = 20 * (init_std - np.mean(std)) / init_std # check factor

            output_file = f'output/{goal[2]:d}.hdf5'
            with h5py.File(output_file, 'a') as f:
                g = f.require_group(f'gen{goal[0]:d}')
                p = g.create_dataset(f'pop{goal[1]:d}', data=np.array(pose_arr))
                p.attrs['fitness'] = rew_ep
                p.attrs['init_std'] = init_std
                p.attrs['std'] = np.mean(std)
                p.attrs['ext_rew'] = rew_ep
                p.attrs['int_rew'] = int_rew
                p.attrs['rew_ep'] = rew_ep + int_rew

            rew_ep += int_rew

    return rew_ep
    # return max(rew_ep, 0)

