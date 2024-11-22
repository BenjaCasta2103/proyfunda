#!/usr/bin/env python3

import numpy as np
from copy import copy
from pyquaternion import Quaternion
import rbdl
cos=np.cos; sin=np.sin; pi=np.pi

def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.
    """
    # Escriba aqui la matriz de transformacion homogenea en funcion de los valores de d, theta, a, alpha
    T = np.array([[cos(theta), -cos(alpha)*sin(theta), sin(alpha)*sin(theta), a*cos(theta)],
                  [sin(theta), cos(alpha)*cos(theta), -sin(alpha)*cos(theta), a*sin(theta)],
                  [0, sin(alpha), cos(alpha), d],
                  [0, 0, 0, 1]])
    return T



def fkine_kuka(q):
    """
    Calcular la cinematica directa del robot kuka dados sus valores articulares.
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]
    """
    # Longitudes (en metros)
    L1 = 0.081105

    L21 = 0.2125
    L22 = 0.009999

    L31 = 0.192811 
    L32 = 0.025419 
    L33 = 0.010387

    L41 = 0.2235 
    L42 = 0.029806
    L43 = 0.01095

    L51 = 0.174541 
    L52 = 9.4e-05
    L53 = 0.009803

    L61 = 0.225781 
    L62 = 0.000386
    L63 = 0.069795

    L71 = 0.070816 
    L72 = 0.070981

    L81 = 0.15
    L82 = 0.001283

    # d = distancia entre z-1 origen anterior
    # theta = rotación respecto de z-1
    # a = distancia entre x
    # alpha = rotación respecto a x

    # Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
    T11 = dh(L1, 0, 0,   0)
    T12 = dh(0, q[0], 0,  0)
    T1 = T11 @ T12

    T21 = dh(L21,  0, 0,  pi/2)
    T22 = dh(-L22,  q[1], 0,   -pi/2)
    T2 = T21 @ T22

    T31 = dh(L31, 0, 0,    -pi/2)
    T32 = dh(-L33, 0, L32,    pi/2)
    T33 = dh(0, q[2], 0,    0)
    T3 = T31 @ T32 @ T33

    T41 = dh(L41,  0, 0,    pi/2)
    T42 = dh(L43,  0, L42,    0)
    T43 = dh(0,  -q[3], 0,    -pi/2)
    T4 = T41 @ T42 @ T43

    T51 = dh(L51, 0, L52,    pi/2)
    T52 = dh(-L53, 0, 0,    -pi/2)
    T53 = dh(0, q[4], 0,    0)
    T5 = T51 @ T52 @ T53

    T61 = dh(L61,  0, L62,    pi/2)
    T62 = dh(-L63,  0, 0,    0)
    T63 = dh(0,  q[5], 0,    -pi/2)
    T6 = T61 @ T62 @ T63

    T71 = dh(L71, 0, 0,    pi/2 )
    T72 = dh(L72, 0, 0,    -pi/2)
    T73 = dh(q[6], 0, 0,    0 )
    T7 = T71 @ T72 @ T73

    T81 = dh(L81, 0, 0,    pi/2 )
    T82 = dh(L82, 0, 0,    -pi/2)
    T83 = dh(0, -q[7], 0,    pi/2)
    T8 = T81 @ T82 @ T83

    # Efector final con respecto a la base
    T = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7 @ T8
    return T

def jacobian(q, delta=0.001):
    """
    Jacobiano para el kuka
    """
    # Alocar espacio: matriz de ceros del tamaño adecuado
    J = np.zeros((3,8))
    # Posición en la configuración q
    Th = fkine_kuka(q)
    x = Th[0:3,3]  # NECESITO LA CINEMÁTICA DIRECTA

    # Iteraciones para las derivadas columna por columna
    for i in range(8):
        # Copiar la configuracion articular inicial (importante)
        dq = np.copy(q)

        # Incrementar la articulacion i-esima usando un delta: qi+delta
        dq[i] = dq[i] + delta
        # Posición luego del incremento (con qi+delta)
        dTh = fkine_kuka(dq)
        dx = dTh[0:3,3]

        # Columna i usando diferencias finitas
        columna_i = 1/delta * (dx-x)
        # Almacenamiento de la columna i
        J[:,i] = columna_i

    return J

def ikine_kuka(xdes, q0):
    """
    Cinemática inversa - Metodo de newton
    """
    epsilon  = 1e-4
    max_iter = 1000
    delta    = 0.001

    q  = copy(q0)
    for i in range(max_iter):
        J = jacobian(q,delta)   # Matriz Jacobiana
        Th = fkine_kuka(q)      # Matriz Actual
        xact = Th[0:3,3]        # Posición
        e = xdes-xact           # Error
        q = q + np.dot(np.linalg.pinv(J),e)
        #Condicion de termino
        if(np.linalg.norm(e)<epsilon):
            break
        pass
    return q

def ik_gradient_kuka(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0.
    Emplear el metodo gradiente
    """
    epsilon  = 1e-4
    max_iter = 1000
    delta    = 0.001
    alpha    = 0.5

    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian(q,delta)         # Matriz Jacobiana
        Td = fkine_kuka(q)            # Matriz Actual
        xact = Td[0:3,3]              # Posicion Actual
        e = xdes-xact                 # Error
        q = q + alpha*np.dot(J.T,e)   # Metodo de la gradiente
        #Condicion de termino
        if(np.linalg.norm(e)<epsilon):
            break
        pass
    return q

def rot2quat(R):
 """
 Convertir una matriz de rotacion en un cuaternion

 Entrada:
  R -- Matriz de rotacion
 Salida:
  Q -- Cuaternion [ew, ex, ey, ez]

 """
 dEpsilon = 1e-6
 quat = 4*[0.,]

 quat[0] = 0.5*np.sqrt(R[0,0]+R[1,1]+R[2,2]+1.0)
 if ( np.fabs(R[0,0]-R[1,1]-R[2,2]+1.0) < dEpsilon ):
  quat[1] = 0.0
 else:
  quat[1] = 0.5*np.sign(R[2,1]-R[1,2])*np.sqrt(R[0,0]-R[1,1]-R[2,2]+1.0)
 if ( np.fabs(R[1,1]-R[2,2]-R[0,0]+1.0) < dEpsilon ):
  quat[2] = 0.0
 else:
  quat[2] = 0.5*np.sign(R[0,2]-R[2,0])*np.sqrt(R[1,1]-R[2,2]-R[0,0]+1.0)
 if ( np.fabs(R[2,2]-R[0,0]-R[1,1]+1.0) < dEpsilon ):
  quat[3] = 0.0
 else:
  quat[3] = 0.5*np.sign(R[1,0]-R[0,1])*np.sqrt(R[2,2]-R[0,0]-R[1,1]+1.0)

 return np.array(quat)

def TF2xyzquat(T):
 """
 Convert a homogeneous transformation matrix into the a vector containing the
 pose of the robot.

 Input:
  T -- A homogeneous transformation
 Output:
  X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
       is Cartesian coordinates and the last part is a quaternion
 """
 quat = rot2quat(T[0:3,0:3])
 res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
 return np.array(res)

def jacobian_pose(q, delta=0.001):
 """
 Jacobiano analitico para la posicion y orientacion (usando un
 cuaternion). Retorna una matriz de 7xn y toma como entrada el vector de
 configuracion articular q=[q1, q2, q3, ..., qn]
 """

 n = q.size
 J = np.zeros((7,n))
 # Posición en la configuración q
 Th = fkine_kuka(q)
 x = Th[0:3,3]  # NECESITO LA CINEMÁTICA DIRECTA
 ori = Th[0:3,0:3]  # NECESITO LA ORIENTACIÓN
 rot = Quaternion(matrix=ori)
 rot = np.array([rot.w, rot.x, rot.y, rot.z])
 des = np.hstack((x,rot))

 # Iteraciones para las derivadas columna por columna
 for i in range(8):
   # Copiar la configuracion articular inicial (importante)
   dq = np.copy(q)
   # Incrementar la articulacion i-esima usando un delta: qi+delta
   dq[i] = dq[i] + delta
   # Posición luego del incremento (con qi+delta)
   dTh = fkine_kuka(dq)
   dx = dTh[0:3,3]
   ori2 = dTh[0:3,0:3]
   drot = Quaternion(matrix=ori2)
   drot = np.array([drot.w, drot.x, drot.y, drot.z])
   act = np.hstack((dx,drot))

   col = PoseError(act,des)
   J[:,i] = col

 return J

def PoseError(x,xd):
 """
 Determine the pose error of the end effector.

 Input:
 x -- Actual position of the end effector, in the format [x y z ew ex ey ez]
 xd -- Desire position of the end effector, in the format [x y z ew ex ey ez]
 Output:
 err_pose -- Error position of the end effector, in the format [x y z ew ex ey ez]
 """
 pos_err = x[0:3]-xd[0:3]
 qact = Quaternion(x[3:7])
 qdes = Quaternion(xd[3:7])
 #qdif = qdes*qact.inverse
 qdif = qdes*(qact.w,-qact.x, -qact.y, -qact.z)
 qua_err = np.array([qdif.w,qdif.x,qdif.y,qdif.z])
 err_pose = np.hstack((pos_err,qua_err))
 return err_pose

class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.M = np.zeros([ndof, ndof])
        self.b = np.zeros(ndof)
        self.dt = dt
        self.robot = rbdl.loadModel('../urdf/lbr_iiwa7_r800.urdf')

    def send_command(self, tau):
        rbdl.CompositeRigidBodyAlgorithm(self.robot, self.q, self.M)
        rbdl.NonlinearEffects(self.robot, self.q, self.dq, self.b)
        ddq = np.linalg.inv(self.M).dot(tau-self.b)
        self.q = self.q + self.dt*self.dq
        self.dq = self.dq + self.dt*ddq

    def read_joint_positions(self):
        return self.q

    def read_joint_velocities(self):
        return self.dq
