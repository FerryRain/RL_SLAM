import math
import time

import numpy as np

import rospy
import torch
from cv_bridge import CvBridge

from hwt_data.msg import Hwt_ht_basic
from hwt_data.msg import Hwt_ht_position
from sensor_msgs.msg import Image

from RL_train.preprocess_vector_class import Vector
from visualization_msgs.msg import MarkerArray
import argparse

GOAL_REACHED_DIST = 30
COLLISION_DIST = 0.35
TIME_DELTA = 5


def args_set():
    """
    Initialization Parameter
    Returns:
        Args:
    """
    #Yolo Part
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default='false', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    return parser.parse_args()

# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, args):
        rospy.init_node("gym", anonymous=True)
        # self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0
        self.odom_z = 0

        self.goal_x = 3.0
        self.goal_y = 0.0

        self.pix_x = 330
        self.pix_y = 250

        self.id = 500

        # self.last_odom = None

        self.args = args
        self.Vec = Vector(self.args)



        self.action_pub = rospy.Publisher("/hwt/px4/Setmsgs/cmd", Hwt_ht_position, queue_size=1)
        # self.basic_Sub = rospy.Subscriber(
        #     "/hwt/px4/basic", Hwt_ht_basic, self.odom_callback, queue_size=1
        # )


    # def callback(self, data1):
    #     global color_image
    #     bridge = CvBridge()
    #     color_image = bridge.imgmsg_to_cv2(data1, 'bgr8')

    # def odom_callback(self, od_data):
    #     self.last_odom = od_data

    # Perform an action and read a new state
    def step(self, action):
        target = False

        cmd = Hwt_ht_position()
        cmd.id = self.id
        cmd.MOVE_FORM = 2  # 0 == ENU, 1 == vel, 2 == body
        cmd.Set_local_pose[0] = float(action[0])
        cmd.Set_local_pose[1] = float(action[1])
        cmd.Set_local_pose[2] = float(0)
        self.action_pub.publish(cmd)
        self.id += 10


        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        # get state
        # h, yaw = round(self.Vec.basic.position_now[2], 2), round(self.Vec.basic.attitude[2], 2)

        # Calculate robot heading from odometry data
        self.odom_x = self.Vec.basic.position_now[0]
        self.odom_y = self.Vec.basic.position_now[1]

        try:
            state = self.Vec.get_vector(self.Vec.color_image, self.Vec.basic)
            state_array = state.numpy()
        except:
            state = self.Vec.get_vector(self.Vec.color_image, self.Vec.basic)
            state = None
            state_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            print("step failed: data fusion faild")

        if state == None:
            state_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            state = torch.from_numpy(state_array)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        distance2 = np.linalg.norm(
            [(state_array[0] + (state_array[2] - state_array[0])/2) - self.pix_x,
             (state_array[1] + (state_array[3] - state_array[1])/2) - self.pix_y]
        )

        # Detect if the goal has been reached and give a large positive reward
        if distance2 < GOAL_REACHED_DIST:
            target = True

        done = True
        # state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, distance, distance2)
        print(reward)
        return state, reward, done, target

    def reset(self):

        cmd = Hwt_ht_position()
        cmd.id = self.id
        self.id += 1
        cmd.MOVE_FORM = 0
        cmd.Set_local_pose[0] = np.random.uniform(0.0,4.0)
        cmd.Set_local_pose[1] = np.random.uniform(-1.0, 3.0)
        cmd.Set_local_pose[2] = np.random.uniform(4.5, 6.0)
        cmd.Set_yaw = np.random.uniform(-np.pi, np.pi)
        self.action_pub.publish(cmd)
        time.sleep(TIME_DELTA)

        self.odom_x = self.Vec.basic.position_now[0]
        self.odom_y = self.Vec.basic.position_now[1]

        # get state
        h, yaw = round(self.Vec.basic.position_now[2], 2), round(self.Vec.basic.attitude[2], 2)

        self.odom_x = self.Vec.basic.position_now[0]
        self.odom_y = self.Vec.basic.position_now[1]

        try:
            state = self.Vec.get_vector(self.Vec.color_image, self.Vec.basic)
        except:
            state = None
            state = self.Vec.get_vector(self.Vec.color_image, self.Vec.basic)
            print("reset failed: data fusion faild0")
        if state == None:
            state_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            state = torch.from_numpy(state_array)

        return state



    @staticmethod
    def get_reward(target, distance, distance2):
        if target:
            return 100.0
        else:
            return -(1*distance + 0.01*distance2)

if __name__ == '__main__':
    args = args_set()
    env = GazeboEnv(args)
    i = 0
    while i<3:
        i += 1
        env.reset()
        env.step([3.0, 0.0])
