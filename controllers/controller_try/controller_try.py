from controller import Robot, Supervisor,Connector
from ikpy.chain import Chain
import pandas as pd
import numpy as np
import time
import random
import cv2


import math

# 初始化 Supervisor
supervisor = Supervisor()
# robot_chain = Chain.from_urdf_file("LRMate-200iD.urdf", base_element="Base")
robot_chain = Chain.from_urdf_file("LRMate-200iD.urdf",
         base_elements=['Base','J1','J2','J3','J4','J5','J6'])
# 設定時鐘（time step）
robot_chain.active_links_mask = [False] + [True] * (len(robot_chain.links) - 1)
# robot_chain.active_links_mask = [False, True, True, True, True, True, True]
timestep = int(supervisor.getBasicTimeStep())

#利用順項運動學計算出末端軸位置
def get_endpoint_position(angles):
    endpoint_position=robot_chain.forward_kinematics(angles)
    return endpoint_position

#利用逆向運動學，根據末端軸位置以及角度推算手臂個軸角度
def get_IK_angle(target_position, target_orientation=[0, 1, 0], orientation_axis="Z"):
    # 初始化機器人鏈條
    
    for i, link in enumerate(robot_chain.links):
        print(f"Joint {i}: {link.name}")

    # 計算逆向運動學
    # ikAnglesD = robot_chain.inverse_kinematics(target_position,target_orientation=[0, 1, 0])#不限制角度
    ikAnglesD= robot_chain.inverse_kinematics(
    target_position,
    target_orientation=target_orientation,
    orientation_mode=orientation_axis)#限制角度以及末端軸位置
    return ikAnglesD

def rotation_matrix_to_euler_angles(R):#將旋轉矩陣轉換為歐拉角
    """
    Convert a rotation matrix to Euler angles (in degrees) with ZYX order.
    
    :param R: np.ndarray, a 3x3 rotation matrix.
    :return: tuple of Euler angles (rx, ry, rz) in degrees.
    """
    # Check for valid rotation matrix
    if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("Invalid rotation matrix")
    
    # Extract angles
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0
    
    # Convert radians to degrees
    rx = math.degrees(rx)
    ry = math.degrees(ry)
    rz = math.degrees(rz)
    
    return rx, ry, rz

def calculate_A_prime(R_BA, t_BA, R_B_prime, t_B_prime):#輸入工件路徑採樣點研磨時的座標平移與歐拉角，輸出工件座標研磨時的平移與歐拉角
    """
    根据 B' 座标系在世界座标系下的表示，计算 A' 在世界座标系下的表示。

    :param R_BA: np.ndarray, B 座标系在 A 座标系下的旋转矩阵 (3x3).
    :param t_BA: np.ndarray, B 座标系在 A 座标系下的平移向量 (3x1).
    :param R_B_prime: np.ndarray, B' 座标系在世界座标系下的旋转矩阵 (3x3).
    :param t_B_prime: np.ndarray, B' 座标系在世界座标系下的平移向量 (3x1).
    :return: (R_A_prime, t_A_prime), A' 座标系在世界座标系下的旋转矩阵和平移向量.
    """
    # 计算 A' 的旋转矩阵
    R_A_prime = R_B_prime @ np.linalg.inv(R_BA)
    
    # 计算 A' 的平移向量
    t_A_prime = t_B_prime - R_A_prime @ t_BA
    R_A_prime=rotation_matrix_to_euler_angles(R_A_prime)
    return R_A_prime, t_A_prime

def Rotation_matrix(rx,ry,rz):#將歐拉角轉換為旋轉矩陣
    # Convert degrees to radians
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)
    
    # Calculate the rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(rx), -math.sin(rx)],
                    [0, math.sin(rx), math.cos(rx)]])
    
    R_y = np.array([[math.cos(ry), 0, math.sin(ry)],
                    [0, 1, 0],
                    [-math.sin(ry), 0, math.cos(ry)]])
    
    R_z = np.array([[math.cos(rz), -math.sin(rz), 0],
                    [math.sin(rz), math.cos(rz), 0],
                    [0, 0, 1]])
    
    # Combine the rotation matrices
    R = np.dot(np.dot(R_z, R_y), R_x)
    return R


def euler_to_axis_angle(rx, ry, rz):#欧拉角轉换为轴-角表示(webots內的rotation以轴-角表示)
    """
    Converts Euler angles (in degrees) to axis-angle representation.
    
    Args:
    rx: Rotation around x-axis in degrees.
    ry: Rotation around y-axis in degrees.
    rz: Rotation around z-axis in degrees.
    
    Returns:
    A tuple of four values representing the axis-angle (axis_x, axis_y, axis_z, angle).
    """
    # Convert degrees to radians
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)
    
    # Calculate the rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(rx), -math.sin(rx)],
                    [0, math.sin(rx), math.cos(rx)]])
    
    R_y = np.array([[math.cos(ry), 0, math.sin(ry)],
                    [0, 1, 0],
                    [-math.sin(ry), 0, math.cos(ry)]])
    
    R_z = np.array([[math.cos(rz), -math.sin(rz), 0],
                    [math.sin(rz), math.cos(rz), 0],
                    [0, 0, 1]])
    
    # Combine the rotation matrices
    R = np.dot(np.dot(R_z, R_y), R_x)
    
    # Calculate the axis-angle representation
    angle = math.acos((np.trace(R) - 1) / 2)
    sin_angle = math.sin(angle)
    
    if sin_angle > 1e-6:  # Avoid division by zero
        axis_x = (R[2, 1] - R[1, 2]) / (2 * sin_angle)
        axis_y = (R[0, 2] - R[2, 0]) / (2 * sin_angle)
        axis_z = (R[1, 0] - R[0, 1]) / (2 * sin_angle)
    else:
        # If the angle is very small, the axis is not well-defined, return the default axis
        axis_x = 1
        axis_y = 0
        axis_z = 0

    return axis_x, axis_y, axis_z, angle
#------------------------------------------------------工件姿態反推手臂末端軸姿態

def get_transformation_matrix(rotation, translation):
    """
    根據旋轉矩陣和平移向量生成4x4的齊次轉換矩陣
    """
    T = np.eye(4)
    T[:3, :3] = rotation  # 設置旋轉部分
    T[:3, 3] = translation  # 設置平移部分
    return T

def invert_transformation_matrix(T):
    """
    反轉4x4齊次轉換矩陣
    """
    R_inv = T[:3, :3].T  # 旋轉矩陣的轉置
    t_inv = -R_inv @ T[:3, 3]  # 平移向量的反轉
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def calculate_B_prime(A_rot, A_trans, B_rot, B_trans, A_prime_rot, A_prime_trans):#從
    """
    計算B'的旋轉與平移矩陣
    給予一個A坐標系與B坐標系相對世界座標的轉移矩陣,接著給出A座標系移動後的新座標系A',假設B'相對A'位置與B相對A相同,輸入A,B,A'的旋轉以及平移矩陣,求出B'的平移與旋轉矩陣

    """
    # Step 1: 計算齊次轉換矩陣
    T_A_W = get_transformation_matrix(A_rot, A_trans)
    T_B_W = get_transformation_matrix(B_rot, B_trans)
    T_A_prime_W = get_transformation_matrix(A_prime_rot, A_prime_trans)

    # Step 2: 計算T_B_A
    T_A_W_inv = invert_transformation_matrix(T_A_W)
    T_B_A = T_A_W_inv @ T_B_W

    # Step 3: 計算T_B_prime_W
    T_B_prime_W = T_A_prime_W @ T_B_A

    # Step 4: 提取B'的旋轉和平移矩陣
    B_prime_rot = T_B_prime_W[:3, :3]
    B_prime_trans = T_B_prime_W[:3, 3]

    return B_prime_rot, B_prime_trans

#-----------------------------------------------------------------------
#控制手臂各軸馬達
motors = []
motors.append(supervisor.getDevice('J1'))
motors.append(supervisor.getDevice('J2'))
motors.append(supervisor.getDevice('J3'))
motors.append(supervisor.getDevice('J4'))
motors.append(supervisor.getDevice('J5'))
motors.append(supervisor.getDevice('J6'))
v=0
for i in range(6):  # UR5e有6個關節
    # joint_name = f"shoulder_lift_joint{i+1}"
    motors[i].setPosition(0)  # 初始位置設為0
# pose_node = supervisor.getFromDef("end_effector")
# target_position=[0.6,0,0.4]
# orientation=[1,0,0,0]


#各馬達上加入sensor測量轉角位置
sensors = []
for motor in motors:
    sensor = motor.getPositionSensor()
    sensor.enable(timestep)
    sensors.append(sensor)
# translation_field = pose_node.getField("translation")
# rotation_field = pose_node.getField("rotation")
#--------------------------------------------------------------------------------------------------------------
# 读取CSV文件
file_path = r"簡單幾何2__final_ver.csv" # 请将这里的路径替换为你的CSV文件路径
df = pd.read_csv(file_path)

# R_contactpoint_frame = Rotation_matrix(-104.57, -104.57,-75.43)#角度單為為度rx = -104.57度, ry = -90.00度, rz = -75.43度
R_contactpoint_frame = Rotation_matrix(-90,0,0)#角度單為為度
t_contactpoint_frame = np.array([0.06, 0.73, 0.39])#此處為砂帶上接觸點的座標，可由"座標轉換.ipynb"計算其歐拉角(軸-角--->歐拉)

solid_node = supervisor.getFromDef("simple_2_show")#獲得工件模型
if solid_node is None:
    raise ValueError("Solid node not found")

t_toolframes=[]
r_toolframes=[]
R_A_primes=[]
t_A_primes=[]
for index, row in df.iterrows():

    r_sample=row.iloc[5:8]
    t_sample=row.iloc[2:5]
    R_samplepoint=Rotation_matrix(r_sample[0],r_sample[1],r_sample[2])
    t_samplepoint = np.array([t_sample[0]/1000,t_sample[1]/1000,t_sample[2]/1000])


    # B' 座标系在世界座标系下的旋转矩阵和平移向量-->此處B'應設定為砂帶上的接觸點座標系

    # 计算 A' 在世界座标系下的表示-->研磨過程中工件座標系在世界座標系的位置
    R_A_prime, t_A_prime = calculate_A_prime(R_samplepoint, t_samplepoint, R_contactpoint_frame, t_contactpoint_frame)#工件坐標系相對世界座標系的旋轉矩陣與平移矩陣
    R_A_primes.append(R_A_prime)
    t_A_primes.append(t_A_prime)
    t_toolframe=[t_A_prime[0],t_A_prime[1],t_A_prime[2]]
    r_toolframe=euler_to_axis_angle(R_A_prime[0],R_A_prime[1],R_A_prime[2])
    r_toolframes.append(r_toolframe)
    t_toolframes.append(t_toolframe)
print(R_A_primes[1])  
    # solid_node.getField('translation').setSFVec3f(t_toolframe)
    # solid_node.getField('rotation').setSFRotation(r_toolframe)#讓工件根據加工路徑移動
# print(t_toolframes)
# print(r_toolframes)
#-------------------------------------------------------------------------------------
supervisor.step(int(supervisor.getBasicTimeStep()))





###########################測試將solid固定於末端軸
# 获取机器人末端节点（使用场景树中末端节点的名字）

# # 获取 Solid 节点（使用场景树中 Solid 节点的名字）
# box_node = supervisor.getFromDef('box')

# connector1 = supervisor.getFromDef("connector_1")
# connector2 = supervisor.getFromDef("connector_2")
# ################################
# if connector1 and connector2:
#     # 鎖定連接器
#     supervisor.lock(connector1.getId())
#     supervisor.lock(connector2.getId())
arm_connector = supervisor.getDevice("connector_1")
arm_connector.enablePresence(timestep)
##################
for i in range(supervisor.getNumberOfDevices()):
    device = supervisor.getDeviceByIndex(i)
    print(f"Device {i}: {device.getName()} (Type: {device.getNodeType()})")















i=0

# 控制器主循環
while supervisor.step(timestep) != -1:
    print("timestep=",timestep)
    t=i//2
    # 這裡可以加入你的控制邏輯，像是設定關節角度
    # motors[0].setPosition(i)  # 將第一個關節移動到1.0弧度位置
    # motors[1].setPosition(i)  # 將第二個關節移動到0.5弧度位置
    #    # 檢查是否可以連接


#將研磨路徑實際跑一遍以確認目標姿態----------------------------------------------------------
    # t_toolframe=[t_toolframes[int(t)]]
    # r_toolframe=[r_toolframes[int(t)]]
    # print(t_toolframe[0])
    
    # r_toolframe=r_toolframe[0]
    # r_toolframe=[r_toolframe[0],r_toolframe[1],r_toolframe[2],r_toolframe[3]]
    # print(f"第{t}行",r_toolframe)
    # solid_node.getField('translation').setSFVec3f(t_toolframe[0])
    # solid_node.getField('rotation').setSFRotation(r_toolframe)
#----------------------------------------------------------------------------------------
    #設定solid並獲取該節點之位置
    ball_node = supervisor.getFromDef("ball")
    ball_translation_field = ball_node.getField("translation")
    ball_position = ball_translation_field.getSFVec3f()
    # print("solid_position=",ball_position)
#---------------------------------------讓手臂末端軸跟隨球走動
    # ikAnglesD=get_IK_angle(ball_position)
    # # print("ikAngleD=",ikAnglesD)
    # # 設置 Webots 中的機器人關節角度

    # for n, motor in enumerate(motors):
    #     motor.setPosition(ikAnglesD[n+2])
    # joint_angles = [sensor.getValue() for sensor in sensors]#實際從webots內測量得到的
    # # print("Joint angles:", joint_angles)
    # # translation_field.setSFVec3f(target_position)
    # # rotation_field.setSFRotation(orientation)
    # joint_angles_for_get_endpoint_position=ikAnglesD
    # joint_angles_for_get_endpoint_position[2:8]=joint_angles
    # endpoint_posture=get_endpoint_position(joint_angles_for_get_endpoint_position)
    # # print("endpoint_posture=",endpoint_posture)#endpoint_posture表示ur5e最後一軸底下，DEF end_effector Pose的世界座標位置以及orientation
    # endpoint_rotation_matrix=endpoint_posture[:3,:3]
    # endpoint_translation_vector = endpoint_posture[:3, 3]
    # endpoint_euler_angles=rotation_matrix_to_euler_angles(endpoint_rotation_matrix)
    # endpoint_axis_angle=euler_to_axis_angle(endpoint_euler_angles[0],endpoint_euler_angles[1],endpoint_euler_angles[2])
#--------------------------------------------------------------------------測試直接讀取手臂末端位置-->失敗
    # # 获取末端执行器的当前位置和角度
    # end_effector_position = end_effector_translation_field.getSFVec3f()
    # print("end_effector_position=",end_effector_position)
    # end_effector_rotation = end_effector_rotation_field.getSFRotation()
    # print("endpoint_translation_vector=",endpoint_translation_vector)
#------------------------------------------------------------將工件固定於機器手臂末端
    # if not connector.isLocked():
    #     connector.lock()
    # else:
    #     # 执行其他任务，例如移动工具
    #     pass
    if arm_connector is not None:
        if arm_connector.getPresence() == 1:
            print("檢測到物體靠近機械臂")
            if not arm_connector.isLocked():
                print("嘗試鎖定...")
                arm_connector.lock()
                if arm_connector.isLocked():
                    print("物體已成功抓取")
                else:
                    print("鎖定失敗")
            else:
                print("Connector 已經處於鎖定狀態")
        else:
            print("沒有檢測到物體")
    else:
        print("Connector 未初始化")
    # # 设置 Solid 的位置和角度
    # box_node.getField('translation').setSFVec3f(list(endpoint_translation_vector))
    # box_node.getField('rotation').setSFRotation(endpoint_axis_angle)
 #--------------------------------------------------------利用逆向運動學直接讓手臂到達目標位置
    R_workpiece=[ [9.99849432e-01 ,-1.73525964e-02 , 1.23176546e-05],
                    [-1.09013912e-05  ,8.17113863e-05 , 9.99999997e-01],
                    [-1.73525974e-02 ,-9.99849429e-01,  8.15099158e-05]]
    T_workpiece=[0.00052047 , 0.234508 , 0.519081]
    R_endeffector=[[-0.99984945 ,-0.0       ,  -0.01735178],
                    [ 0.0   ,       1.0      ,   -0.0        ],
                    [ 0.01735178 , 0.0       ,  -0.99984945]]
    T_endeffector=[0.00052047 , 0.234508 , 0.519081]
    #歐拉角應該先轉成旋轉矩陣
    print(i)
    T_A_prime=[(t_A_primes[t])[0],(t_A_primes[t])[1],(t_A_primes[t])[2]]
    print((R_A_primes[t])[0],(R_A_primes[t])[1],(R_A_primes[t])[2])
    R_A_prime=Rotation_matrix((R_A_primes[t])[0],(R_A_primes[t])[1],(R_A_primes[t])[2])
    B_prime_rot, B_prime_trans = calculate_B_prime(R_workpiece, T_workpiece, R_endeffector, T_endeffector, R_A_prime, T_A_prime)#輸入工件姿態，輸出手臂末端姿態，皆以旋轉矩陣以及平移矩陣描述
    y_axis_world =(B_prime_rot[:,1])#計算手臂末端軸的方向
    print("R_A_prime=",R_A_prime)
    print("T_A_prime=",T_A_prime)
    print("y_axis_world=",y_axis_world)

    ikAnglesD=get_IK_angle(B_prime_trans,y_axis_world)
    # print("ikAngleD=",ikAnglesD)
    # 設置 Webots 中的機器人關節角度

    for n, motor in enumerate(motors):
        motor.setPosition(ikAnglesD[n+2])
 #--------------------------------------------------
    i=i+1
    if i==224:
        i=i-224
    print("t=",t)

    # if #達到目標姿態 :
    #     i=i+1#更新目標位置
#現在問題在於，如何知道末端軸or工件在世界座標中的位置?可能可以使用ikpy正向運動學