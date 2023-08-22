import numpy as np
import matplotlib.pyplot as plt

# Parameters
# Parameters

#Kp => 제어 상수

Kp_att = 0.02
Kp_rel = 3.0
obstacle_bound = 2.0

def calc_attractive_force(x,y,gx,gy):
	
	# e_x와 e_y는 goal 지점과 현재 위치의 차이
	e_x, e_y = gx-x, gy-y

	# goal 지점과 로봇 간의 거리 => distance 안에 입력
	distance = np.linalg.norm([e_x,e_y])

	# 로봇이 목표 위치로 이동하기 위한 인공힘의 x 축과 y 축 성분
	att_x = Kp_att * e_x/distance
	att_y = Kp_att * e_y/distance

	return att_x, att_y

def calc_repulsive_force(x,y,obs):

	rep_x,rep_y = 0,0

	# obs.shape => obstacle의 수
	for obs_xy in np.ndindex(obs.shape[0]):
		# x, y => 로봇의 초기 위치
		# 장애물 까지의 거리를 측정
		obs_dis_x, obs_dis_y = obs[obs_xy][0]-x, obs[obs_xy][1]-y 
		
		# obs_dis의 r 값을 측정
		obs_dis = np.linalg.norm([obs_dis_x,obs_dis_y]) 

		# obs_dis가 장애물과 일정 거리 안에 존재할 때
		if obs_dis < obstacle_bound:
			# rep_x, y를 설정
			rep_x = rep_x - Kp_rel * (1/obs_dis - 1/obstacle_bound)*(1/(obs_dis*obs_dis))*obs_dis_x/obs_dis
			rep_y = rep_y - Kp_rel * (1/obs_dis - 1/obstacle_bound)*(1/(obs_dis*obs_dis))*obs_dis_y/obs_dis
		else:
			rep_x = rep_x
			rep_y = rep_y

	return rep_x, rep_y

def Artificial_Potention_Field(start_x,start_y,goal_x,goal_y,obs):

	# 초기 위치 설정
	x,y = start_x,start_y

	# 로봇의 이동 궤적 저장
	trace_x = []
	trace_y = []

	# 저장
	trace_x.append(x)
	trace_y.append(y)

	while(1):

		#
		att_x, att_y = calc_attractive_force(x,y,goal_x,goal_y)
		rep_x, rep_y = calc_repulsive_force(x,y,obs)

		#로봇의 위치 지정하는 부분. => 이 쪽 부분을 reward 지정 형식으로 변경하면 될 듯.
		
		#여기서 끝 까지는 필요 없는 부분
		pot_x = att_x+rep_x
		pot_y = att_y+rep_y

		x = x + pot_x
		y = y + pot_y

		trace_x.append(x)
		trace_y.append(y)

		# 로봇이 도착함을 알림
		error = np.linalg.norm([goal_x-x,goal_y-y])

		if error < 1:
			plt.plot(obs[:,0],obs[:,1],'bo')
			plt.plot(trace_x,trace_y,'go',goal_x,goal_y,'ro')
			plt.show()
			break

def main():
	print("Artificial Potential Field Start!!")

	# 현재 로봇의 좌표 입력
	start_x, start_y = 0.0, 0.0
	
	#목적지에 대한 좌표 입력
	goal_x, goal_y = 30.0, 30.0

	# 장애물에 대한 좌표 => lidar 배열 정보를 입력하여, obs 정보를 얻음
	obs = np.array([[15.0,14.0],
					[10.0,11.0]])

	Artificial_Potention_Field(start_x,start_y,goal_x,goal_y,obs)

if __name__=='__main__':
	print('__file__'+" start!!")
	main()
	print('__file__'+" Done!!")