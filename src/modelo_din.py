#!/usr/bin/env python3

import rbdl
import numpy as np

if __name__ == '__main__':

  # Lectura del modelo del robot a partir de URDF (parsing)
  modelo = rbdl.loadModel("/home/benja/proy_funda/src/kuka_iiwa_description/urdf/lbr_iiwa7_r800.urdf")
  # Grados de libertad
  ndof = modelo.q_size

  # Configuracion articular
  q = np.array([0.5, 0.2, 0.3, 0.8, 0.5, 0.6, 0.07, 0.2])
  # Velocidad articular
  dq = np.array([0.8, 0.7, 0.8, 0.6, 0.9, 1.0, 0.2, 0.0])
  # Aceleracion articular
  ddq = np.array([0.2, 0.5, 0.4, 0.3, 1.0, 0.5, 0.3, 0.0])
  
  # Arrays numpy
  zeros = np.zeros(ndof)          # Vector de ceros
  tau   = np.zeros(ndof)          # Para torque
  g     = np.zeros(ndof)          # Para la gravedad
  c     = np.zeros(ndof)          # Para el vector de Coriolis+centrifuga
  M     = np.zeros([ndof, ndof])  # Para la matriz de inercia
  e     = np.eye(8)               # Vector identidad
  mi    = np.zeros(ndof)
  
  # Torque dada la configuracion del robot
  rbdl.InverseDynamics(modelo, q, dq, ddq, tau)
  
  # Parte 1: Calcular vector de gravedad, vector de Coriolis/centrifuga,
  # y matriz M usando solamente InverseDynamics
  
  # Vector de gravedad
  rbdl.InverseDynamics(modelo, q, zeros, zeros, g)
  g = np.round(g,2)
  print('VECTOR GRAVEDAD')
  print(g)

  # Vector de Coriolis/centrifuga
  rbdl.InverseDynamics(modelo, q, dq, zeros, c)
  c = c-g
  c = np.round(c,2)
  print('VECTOR F y C')
  print(c)

  # Matriz M
  for i in range(ndof):
    rbdl.InverseDynamics(modelo, q, zeros, e[i,:], mi)
    M[i,:] = mi - g
    
  print('MATRIZ INERCIA')
  print(np.round(M,2))


  # Parte 2: Calcular M y los efectos no lineales b usando las funciones
  # CompositeRigidBodyAlgorithm y NonlinearEffects. Almacenar los resultados
  # en los arreglos llamados M2 y b2
  b2 = np.zeros(ndof)          # Para efectos no lineales
  M2 = np.zeros([ndof, ndof])  # Para matriz de inercia
  
  rbdl.CompositeRigidBodyAlgorithm(modelo, q, M2)
  rbdl.NonlinearEffects(modelo, q, dq, b2)
  print('VECTOR DE EFECTOS NO LINEALES')
  print(np.round(b2,2))

  print('MATRIZ DE INERCIA M2')
  print(np.round(M2,2))
  
  # Parte 2: Verificacion de valores
  
  e_1 = b2 - c - g
  print('1ra VERIFICACION')
  print(np.round(e_1, 2))

  # Resta de matrices de inercias obtenidas en ambos puntos
  e_2 = M2 - M
  print('2da VERIFICACION')
  print(np.round(e_2, 2))
  
  # Parte 3: Verificacion de la expresion de la dinamica
  tau2 = M.dot(ddq) + c + g
  tau3 = M2.dot(ddq) + b2
  print('Vector de torques obtenidos con la funcion InverseDynamics')
  print(np.round(tau,1))
  print('Vector de torques obtenidos en la primera parte con la funcion InverseDynamics')
  print(np.round(tau2,1))
  print('Vector de torques obtenidos en la segunda parte con las funciones CompositeRigidAlgorithm y NonlinearEffects')
  print(np.round(tau3,1))