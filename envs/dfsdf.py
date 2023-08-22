# 이 부분에 PPO에 관한 action, state 등을 해 놓는다.
import numpy as np
import airsim
import time
import gym
import random
import typing
import math
import cv2

from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
 
class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address):
        super().__init__()

        self.state = {
            "depth": np.zeros([1,100,100]),
            "position" : np.zeros([1, 2]),
            "collision" : False,
            "position_state" : np.zeros([1, 2]),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.start= time.time()

        self._setup_flight()

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # start_position = [[87, 15.0], [87, 15.5], [87, 16.0], [87, 16.5], [87, 17.0], [87, 17.5], [87, 18.0], [87, 18.5]]
        start_position = [[95, 16.0]]
        # target_position = [[135.37, 25.26], [135.37, 23.26], [135.51, 26.87], [135.51, 28.69], [138.14, 28.26], [137.85, 26.71], [137.83, 23.28], [137.83, 23.26]]
        target_position = [[126.40, 38.85]]
        # start_position = [[0, 0]]
        # target_position = [[10.00,0.16]]
        random_start = random.choice(start_position)

        self.start_x = random_start[0]
        self.start_y = random_start[1]
        start_z = 1.2

        start_index = start_position.index(random_start)
        
        #여기서 설정한 target_pos는 그냥 내가 쓰기 위해 설정하는 것일 뿐이다.
        self.target_pos = target_position[start_index]

        position = airsim.Vector3r(self.start_x, self.start_y, start_z)
        pose = airsim.Pose(position)
        self.drone.simSetVehiclePose(pose, ignore_collision=True)

        self.start = time.time()

        self.drone.moveToPositionAsync(self.start_x, self.start_y, start_z, 3).join()
        self.drone.moveByVelocityAsync(1, 0.0, 1.2, 0).join()
        # 초기 드론 위치를 여기서 설정
    
    def _get_obs(self):

        self.drone_state = self.drone.getMultirotorState()

        #이 부분을 수정해야 함
        self.state["position"] = np.array([
            abs(self.target_pos[0]-self.drone_state.kinematics_estimated.position.x_val),
            abs(self.target_pos[1]-self.drone_state.kinematics_estimated.position.y_val)
            ])

        # 너무 멀리 갔을 때를 위해 설정        
        self.state["position_state"]= np.array([self.drone_state.kinematics_estimated.position.x_val,self.drone_state.kinematics_estimated.position.y_val])

        
        #충돌에 대한 정보 업데이트
        collision = self.drone.simGetCollisionInfo().has_collided
        
        #lidar 정보 업데이트    
        self.state["lidar"] = self.lidar_obs()
        self.state["collision"] = collision
        
        return self.state
    
    def lidar_obs(self):
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

    def _do_action(self, action):
        yaw_rate = action*20
        vx_action = 3.0
        #x축 속도 고정, yaw의 회전만으로 장애물 회피
        
        self.drone.moveByVelocityZBodyFrameAsync(
            vx = vx_action,
            vy = 0.0,
            z = 1.2,
            duration = 3,
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate= float(yaw_rate))
        )  

    # def make_batch(self, x):
    #     return np.expand_dims(x, axis=0)

    def AF(self):

        att_gain = 0.05
        # [0 ~ 1] nomalize
        distance = np.linalg.norm([self.state["position"][0],self.state["position"][1]]) / 63.28
        att_Force = att_gain*distance
        return att_Force
        
        # x, y로 주는 방식
        
        # distance_x = self.state["position"][0] # target - current
        # distance_y = self.state["positin"][1] # target - current
        # max_dis_x = self.target_pos[0] - self.start_x
        # max_dis_y = self.target_pos[1] - self.start_y
        
        # att_Force = att_gain*(max_dis_x + max_dis_y - distance_x - distance_y)
        return att_Force

    #nomalize [0 ~ 1]
    def RF(self):
        gain= 0.05
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
        
    def vector(self):
        vector_gain = 0.001
        yaw = airsim.to_eularian_angles(self.drone_state.kinematics_estimated.orientation)[2]
        
        x = np.cos(yaw)
        y = np.sin(yaw)

        direction_vector = np.array([x, y])

        distance = np.linalg.norm([self.state["position"][0],self.state["position"][1]])
        target_vector = self.state["position"]/distance # 현재 방향 벡터

        # 많이 다르면 0, 방향이 같으면 1
        cosine_similarity = np.dot(direction_vector, target_vector)
        
        vector_Force = vector_gain*cosine_similarity
        # print(vector_Force)

        return vector_Force

    def _compute_reward(self):
        #reward를 어떻게 주어줄 지에 대해서 작성
        goal = 0
        done = 0
        collision = False
        out = 0
        # x_dis = self.target_pos[0] - s    elf.state["position_state"][0]  
        # y_dis = self.target_pos[1] - self.state["position_state"][1]
        
        x_val = self.state["position_state"][0]
        y_val = self.state["position_state"][1]

        if self.state['collision'] == True:
            done = 1
            collision = -80
            # print("++++++++++++++++++++++++AF++++++++++++++++++++++++")
            # print(self.AF())
            # print("++++++++++++++++++++++++RF++++++++++++++++++++++++")
            # print(self.RF())
            # print("++++++++++++++++++++++++VC++++++++++++++++++++++++")
            # print(self.vector())
        
        # 일정 boundary 생성, 일정 범위 밖으로 나가게 되면 episode 끝 및 reward 낮은 값 줌
        elif x_val >= 150.0 or x_val<= 93.0 or y_val > 41.0 or y_val < 13.5:
            out = -80
            done = 1
            # print("++++++++++++++++++++++++AF++++++++++++++++++++++++")
            # print(self.AF())
            # print("++++++++++++++++++++++++RF++++++++++++++++++++++++")
            # print(self.RF())
            # print("++++++++++++++++++++++++VC++++++++++++++++++++++++")
            # print(self.vector())


        elif self.state["position"][0]<7 and self.state["position"][1]<7:
            done = 1
            goal = 100
            # print("++++++++++++++++++++++++AF++++++++++++++++++++++++")
            # print(self.AF())
            # print("++++++++++++++++++++++++RF++++++++++++++++++++++++")
            # print(self.RF())
            # print("++++++++++++++++++++++++VC++++++++++++++++++++++++")
            # print(self.vector())

            print("GOAL REACHED")

        else:
            done = 0
        
        # AF = 거리, RF = 장애물 충돌, vector = 방향
        APF = self.AF() - self.RF() + self.vector()
        # APF = -self.AF() - self.RF()

        # APF = self.AF() + self.vector()

        reward = APF + collision + goal + out
        
        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()
