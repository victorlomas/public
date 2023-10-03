# -*- coding: utf-8 -*-
""" Módulo para el procesamiento de nubes de puntos en imágenes binarias. """

import numpy as np
import math
import cv2 as cv
def pcd_2_image(pcd,pixels_height):
  """ Transforma una nube de puntos a imagen binaria.

  Args:
    pcd (numpy.ndarray): Nube de puntos (coordenadas cartesianas).
    pixels_height (int): Altura deseada de la imagen en pixeles. El ancho
      es determinado automaticamente mediante la relación de aspecto de la
      nube de puntos.

  Returns:
    img (numpy.ndarray): Imagen binaria que representa a la nube de puntos.
  """

  #Se agrega la validación de que al menos sea de un pixel de alto para evitar errores
  if pixels_height < 1: pixels_height=1

  # Obtiene las cotas máximas y mínimas para calcular la relación de aspecto
  # de la nube de puntos
  y_max = np.max(pcd[:,1])
  y_min = np.min(pcd[:,1])
  x_max = np.max(pcd[:,0])
  x_min = np.min(pcd[:,0])
  aspect_ratio = round((x_max - x_min) / (y_max - y_min),2)

  # Relación de aspecto mínima soportada
  if aspect_ratio < 0.016: aspect_ratio = 0.016

  # Relación de aspecto máxima soportada
  if aspect_ratio > 62.5: aspect_ratio = 62.5

  # Utilizando la relación de aspecto de la nube de puntos obtiene el ancho
  # de la imagen
  pixels_width = math.ceil(pixels_height * aspect_ratio)

  img = pcd_2_image_proportion(
    capa_x=pcd[:,0],
    capa_y=pcd[:,1],
    resolution_x=pixels_width,
    resolution_y=pixels_height,
    x_max=x_max,
    x_min=x_min,
    y_max=y_max,
    y_min=y_min)
    
  return img

def pcd_2_image_proportion(capa_x,capa_y,resolution_x,resolution_y,x_max,x_min,y_max,y_min):
  """ Método de proporciones para obtener imágenes binarias a partir de nubes de puntos.

  Args:
    capa_x (numpy.ndarray): Arreglo de numpy con las primeras coordenadas
    capa_y (numpy.ndarray): Arreglo de numpy con las segundas coordenadas
    resolution_x (int): Ancho deseado de la imagen.
    resolution_y (int): Alto deseado de la imagen.
    x_max (float): máximo de capa_x.
    x_min (float): mínimo de capa_x.
    y_max (float): máximo de capa_y.
    y_min (float): mínimo de capa_y.

  Returns:
    img_binaria (numpy.ndarray): Imagen binaria que representa a la nube de puntos.
  """

  #Estas son constantes de proporción entre la resolución (se pone -1 porque se empieza a indizar desde cero) y lo que mide la figura max-min
  cte_x=(resolution_x-1)/(x_max-x_min)
  cte_y=(resolution_y-1)/(y_max-y_min)
  
  #Obtnemos las distancias cada punto al mínimo, notemos que son positivas por lo que no hace falta valor absoluto
  distancia_x=capa_x - x_min
  distancia_y=capa_y - y_min
  
  #Al multiplicar cada distancia por la constante de proporción para obtener los valores escalados
  producto_x=distancia_x*cte_x
  producto_y=distancia_y*cte_y
  
  #para poder asignar un índice redondeamos, así obtenemos valores entre 0,1,2, ... , resolución-1 
  indices_img_x=np.around(producto_x).astype(int)    
  indices_img_y=np.around(producto_y).astype(int)
  
  img_binaria=np.zeros((resolution_y,resolution_x),dtype=np.uint8)

  #La clave está en que al escalar obtuvimos de una vez los indices de cada punto en la matriz
  img_binaria[resolution_y-1-indices_img_y, indices_img_x]=255
  
  return img_binaria


def get_contours(img,min_area_ratio):
  """ Obtiene los contornos que cumplen con una proporción mínima.

  Se seleccionan aquellos contornos cuya área es mayor a una proporción mínima.
  La proporción se calcula dividiendo el área del contorno entre el área total
  de la imagen (el producto de su resolución).

  Args:
    img (numpy.ndarray): Imagen binaria de la cual se obtendrán los contornos.
    min_area_ratio (float): Proporción mínima que los contornos deben ocupar con
      respecto al área total de la imagen.

  Returns:
    selected_contours (list): Lista que contiene todas los contornos que cumplen
      cuya área es mayor a la proporción solicitada.
  """

  # Obtiene todos los contornos de la imagen
  contours = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0]
  # El área total de la imagen es el producto de su resolución
  area = img.shape[0] * img.shape[1]
  # Se seleccionan los contornos cuya área es mayor a min_area_ratio

  selected_contours = list(
    filter(
      lambda contour: cv.contourArea(contour) / area >= min_area_ratio,
      contours
    )
  )

  return selected_contours


def get_bof(contour,angle_offset):
  """ Obtiene el descriptor BOF de un contorno.

  Args:
    contour (numpy.ndarray): Coordenadas en pixeles de todos los puntos del
      contorno.
    angle_offset (int): Rango de espaciamiento entre cada ángulo para
      almacenar las magnitudes del descriptor BOF.

  Returns:
    bof (int): Descriptor BOF, el cual consiste en un arreglo unidimensional que
      contiene las distancias del centroide al contorno de la figura cada
      "angle_offset" grados.
  """
  # Genera un vector vacio para almacenar al descriptor
  bof = np.zeros(int(360 / angle_offset))

  # Obtiene el centroide del contorno
  M = cv.moments(contour)
  centroid = ( int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]) )

  # Comprime el contorno en un tensor de orden 2 (una matriz)
  contour = np.squeeze(contour)

  difference=contour-centroid
  distance=np.linalg.norm(difference, axis=1)

  # Obtiene el índice de acuerdo al ángulo formado entre el eje horizonal
  # positivo y el vector formado del centroide al punto.
  # La operación remainder permite que los ángulos oscilen entre 0 y 359.
  idx=np.remainder(np.degrees(np.arctan2(difference[:,1],difference[:,0])),360 ) / angle_offset

  bof[idx.astype(int)]=distance
  bof = bof / np.max(bof)

  return bof

def get_image_bofs(contours,angle_offset):
  """ Obtiene los descriptores BOF de todos los contornos solicitados.

  Args:
    contours (list): Lista de contornos, de los cuales se requiere extraer su BOF.
    angle_offset (int): Rango de espaciamiento entre cada ángulo para
      almacenar las magnitudes del descriptor BOF.

  Returns:
    bofs (list): Lista de descriptores BOF.
  """
  bofs = list(
    map( get_bof, contours, np.ones(len(contours)) * angle_offset )
  )
  return bofs

def get_centroid(contour):
  """ Obtiene el centroide de un contorno.

  Args:
    contour (numpy.ndarray): Coordenadas en pixeles de todos los puntos del
      contorno.

  Returns:
    Tupla que contiene las coordenadas del centroide del contorno.
  """
  M = cv.moments(contour)
  return ( int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]) )

def soften_image(img,ksize):
  """ Suaviza la imagen binaria.

  El suavizado lo realiza mediante la operación morfológica de cierre.

  Args:
    img (numpy.ndarray): Imagen binaria.
    ksize (tuple): Tamaño del kernel a utilizar.

  Returns:
    Imagen binaria suavizada.
  """
  kernel = np.ones(ksize,np.uint8)
  return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
