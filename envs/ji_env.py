import airsim
import numpy as np
import math
import time
import gym
import cv2
from gym import spaces
import random
from envs.airsim_env import AirSimEnv
from PIL import Image as im
from matplotlib import pyplot as plt
from random import randrange
import tensorflow as tf
class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address):
        super().__init__()

        self.state = {
            "depth": np.zeros([1,100,100]),
            "dyn_state": np.zeros(3),
            "position": np.zeros([1,2]),
            "global_pos":np.zeros(3),
            "collision": False
        }
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.start= time.time()
        
        #self._setup_flight()

    def _get_obs(self):

        
        self.drone_state = self.drone.getMultirotorState()


        vx = self.drone_state.kinematics_estimated.linear_velocity.x_val
        vy = self.drone_state.kinematics_estimated.linear_velocity.y_val
        yaw_rate = self.drone_state.kinematics_estimated.angular_velocity.z_val


        self.state['position'] = np.array([
            (self.target_pos[0]-self.drone_state.kinematics_estimated.position.x_val),
            (self.target_pos[1]-self.drone_state.kinematics_estimated.position.y_val)
            ])
        self.state["dyn_state"] = self.make_batch(np.array([vx, vy, yaw_rate]))
        self.state['collision'] = self.drone.simGetCollisionInfo().has_collided
        self.state["depth"]= self.lidar_data()
        self.state["global_pos"]= np.array([self.drone_state.kinematics_estimated.position.x_val,self.drone_state.kinematics_estimated.position.y_val])
        
        return self.state
    
    def lidar_data(self):
        nowtime=time.time()
        lidardata=[]
        lidardata = self.drone.getLidarData().point_cloud
        xy_data1=[]


        lidardata1=np.array(lidardata).reshape(len(lidardata)//3,3)
        xy_data= np.delete(lidardata1,-1,axis=1)

        for a in xy_data:
            if 0<abs(a[0])<20 and 0<abs(a[1])<20:
                xy_data1.append([int(a[0]*2.5)+50,int(a[1]*2.5)+50])
        

        xy_data1 = np.unique(xy_data1, axis=0)

        self.distance= xy_data1

        self.time+=1

        blank_img= np.zeros((100,100))
        
        for a in xy_data1:
            if 1<a[0]<99 and 1<a[1]<99:
                blank_img[100-a[0],a[1]]=255
                '''blank_img[100-a[0]-1,a[1]-1]=255
                blank_img[100-a[0],a[1]-1]=255
                blank_img[100-a[0]+1,a[1]-1]=255
                blank_img[100-a[0]+1,a[1]]=255
                blank_img[100-a[0]+1,a[1]+1]=255
                blank_img[100-a[0],a[1]+1]=255
                blank_img[100-a[0]-1,a[1]]=255
                blank_img[100-a[0]-1,a[1]+1]=255'''

        image_array= np.array(blank_img)
        cv2.imshow("Visualization", image_array)
        cv2.waitKey(1)
        return image_array

    
    def setter(self, list):
        set_done=[]
        for a in list:
            if a not in set_done:
                set_done.append(a)
        return set_done
    
    def make_batch(self, x):
        return np.expand_dims(x, axis=0)

    def __del__ (self):
        self.drone.reset() 

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        x,y,z,w = airsim.utils.to_quaternion(0, 0, np.random.randint(0,360))
        loc= [[30,40], [81,57], [38,112],[-83, -183], [-15, 43], [0,0]]

        target_positions= [[[230,240],[-170,-160]],[[-119, -143],[-119,257]],[[-162,-88],[238,-88]],
        [[117,17],[-283,17]],[[185,243],[-215,-157]],[[200,200],[-200,200]]]

        ranloc= random.choice(loc)

        x= ranloc[0]
        y= ranloc[1]
        z= -3

        index= loc.index(ranloc)
        self.target_pos= random.choice(target_positions[index])

        position = airsim.Vector3r(x, y, z)
        pose = airsim.Pose(position)
        self.drone.simSetVehiclePose(pose, ignore_collision= True)
        self.time=0
        
        self.drone.moveToPositionAsync(x,y,-5,3).join()

    def move(self, action):

        self.action = action
        vx, yaw_rate = action[0]*0.2, action[1]*9
        self.drone.moveByVelocityZBodyFrameAsync(
            vx = float(vx),
            vy = 0.0,
            z = -5.0,
            duration = 0.5,
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate= float(yaw_rate))
        )     

    def get_reward(self):
        collision=0
        
        if self.state['collision']==True:
            done = 1
            collision=-100
        elif (self.state["global_pos"])[0]>=300 or (self.state["global_pos"])[1]>=300:
            done=1
        elif abs(self.state['position'][0])<1 and abs(self.state['position'][1])<1:
            done=1
            collision=10
            print ("GOAL REACHED")
        else:
            done = 0

        APF= -self.APE()- self.RPE()

        #print (self.APE(), self.RPE())

        if self.time<3:
            self.prev_APF=APF        
        reward= APF-self.prev_APF+collision


        self.prev_APF=APF
     
        return reward, done
    
    def APE(self):
        gain= 0.5
        Euclidean_sq= (self.state['position'][0])**2+(self.state['position'][1])**2
        APE= (1/2)*gain*Euclidean_sq
        return APE
    
    def RPE(self):
        gain= 0.5
        repulse_sum=0
        prev_Euclidean=25
        for val in self.distance:
            val[0]=(val[0]-50)/2.5
            val[1]=(val[1]-50)/2.5
            Euclidean= math.sqrt(val[0]**2+val[1]**2)
            if Euclidean==0:
                Euclidean=prev_Euclidean
            if Euclidean<=5:
                repulse= ((1/(Euclidean))-(1/5))**2
            else:
                repulse=0
            repulse_sum+=repulse

            prev_Euclidean=Euclidean

        RPE= (1/2)*gain*repulse_sum       

        return RPE

        
    def step(self, action):

        self.move(action)

        self.obs= self._get_obs()

        self.reward, self.done= self.get_reward()
        return self.obs, self.reward, self.done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()