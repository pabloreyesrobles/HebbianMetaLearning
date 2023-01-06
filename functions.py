import numpy as np
import rospy
from explauto import Environment

def action_random(env):
  random_action = env.action_space.sample()
  random_action['head'] = np.double(np.array([0, 0, 0, 0, 0, 0]))
  random_action['right_arm'] = np.double(np.array([-30, 30, 0, 45, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0]))
  random_action['torso'] = np.double(np.array([0, 0, 0]))
  random_action['left_arm'][7] = 60
  random_action['left_arm'][8] = 90
  for i in range(9, 16):
    random_action['left_arm'][i] = 0
  return random_action
  
def home_pose():
  return [-60, 40, 80, 105, -60, 24, 0, 60, 90, 0, 0, 0, 0, 0, 0, 0]

def action_home(env):
  action = action_random(env)
  action['left_arm'] = np.double(np.array(home_pose()))
  return action
  
def ros_reset():
  reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
  reset_simulation()

def test_grid():
  return [
    [144,  74], [144, 116], [144, 198], [179, 136], [180, 178],
    [216, 116], [216, 198], [252, 136], [254, 178], [286, 116],
    [288, 198], [321, 136], [322, 178], [356,  74], [356, 116],
    [358, 198], [110, 320], [110, 360], [144, 259], [144, 300],
    [146, 380], [178, 238], [180, 320], [182, 362], [216, 258],
    [215, 300], [217, 381], [218, 423], [249, 329], [253, 318],
    [254, 360], [254, 442], [286, 258], [286, 300], [288, 380],
    [286, 423], [320, 238], [322, 318], [322, 362], [356, 260],
    [356, 300], [358, 380], [392, 320], [392, 362],
  ]

def test_model(env, model, i):
  test_grid = [[100, 100]]

  model.mode = "exploit"
  filename = 'output/data-' + str(i) + '.txt'
  fh = open(filename, 'w')
  print('Test learnt model')
  for i in range(len(test_grid)):
    s_g = test_grid[i]
    best_dist = float("inf")
    s = None
      
    # Best of K=2, just to be sure
    for kk in range(2):
      m = model.inverse_prediction(s_g)

      action = action_home(env)
      action['left_arm'][0:7] = np.double(np.array(m))
        
      # Perform action
      observation, reward, done, info = env.step(action)
      torso = observation['touch']['torso']
        
      if (torso[0] + torso[1] > 0):
        dist = ((s_g[0] - torso[0])**2 + (s_g[1] - torso[1])**2)**0.5
        if dist < best_dist:
          s = s_g[:]
          best_dist = dist
            
    print('Expected observation: ' + ','.join(str(x) for x in s_g))
    if s is None:
      print('Actual observation:   None')
      fh.write(str(s_g[0]) + ' ' + str(s_g[1]) + ' ' + str(9999) + ' ' + str(9999) + ' ' + str(9999) + '\n')
    else:
      print('Actual observation:   ' + ','.join(str(x) for x in s))
      print('Distance:   ' + str(best_dist))
      fh.write(str(s_g[0]) + ' ' + str(s_g[1]) + ' ' + str(s[0]) + ' ' + str(s[1]) + ' ' + str(best_dist) + '\n')
  print('')
  fh.close()
  model.mode = "explore"

