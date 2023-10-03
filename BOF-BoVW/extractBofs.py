# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import PCD_Image_Processor
import cv2 as cv
import open3d as o3d

def declaraArray(n):
    """ Crea una lista que contiene n listas vacias
    Args:
        n (int): Número de elementos que requiere contener la lista
    Returns:
        Lista con n listas vacias
    """
    return [ [] for _ in range(n) ]

def cortaCapas(data,nCapasOintervalo,metodo, eje):
    """ Funcion para realizar el corte de capas del conjunto de puntos.

    Args:
        data (numpy.ndarray): Arreglo que contiene los datos de la nube de puntos.
        nCapasOintervalo (int/float): Número de capas o intervalo de separación deseado.
        metodo (int): Selección del método de corte de capas a utilizar.
        eje (int): Eje perpendicular para el corte de capas. 0 es el eje x,
            1 es el eje y, 2 es el eje z.
    Returns:
        Lista con capas obtenidas.
    """
    capas=[]

    # Si la nube de puntos está vacía, retorna un arreglo de capas vacio
    if len(data) == 0: return capas

    #Switch para escoger el metodo
    if metodo == 1:
        # Si se tienen menos puntos que nCapas, se retorna una lista vacía
        if len(data) < nCapasOintervalo: return capas
        capas=corteMetodoA(data,int(nCapasOintervalo), eje)
    elif metodo == 2:
        if len(data) < nCapasOintervalo: return capas
        capas=corteMetodoB(data,int(nCapasOintervalo), eje)
    elif metodo == 3:
        capas=corteMetodoC(data,nCapasOintervalo, eje)
    else:
        print("\n\n\n--------------->METODO SELECCIONADO INCORRECTO<------------\n\n\n\n")
        raise ValueError("Error al escoger el método")

    #Antes de avanzar, se deben de quitar algunas capas que en la declaración del metodo se pudieron quedar vacias
    capas= quitaCapasVacias(capas)
    return capas #Retornamos la Lista de capas utiles (Cada capa tiene su conjunto de puntos)


def corteMetodoA(data,nLayers,axis):
    """ Método A para partir capas con base en el numero de capas.

    Se observan las coordenadas en el eje seleccionado de los puntos y estos se asignan a una capa
    dependiendo de si su coordenada correspondiente está en un intervalo establecido
    uniformentente por el número de capas. Todos los puntos de una capa son
    proyectados a un plano cuyo vector normal es el eje seleccionado.

    Ejemplo:
        Se tienen los puntos A, B, C, D, E, cuyas coordenadas en el eje perpendicular
        son -2, -0.5, 4, 6 y 8 respectivamente. Se deseean 5 capas.
        El intervalo de separación entre capas es de (8 - (-2)) / 5 = 2.
        Las capas son:
            Capa 0: [-2 , 0)
            Capa 1: [ 0 , 2)
            Capa 2: [ 2 , 4)
            Capa 3: [ 4 , 6)
            Capa 4: [ 6 , 8]

    Args:
        data (numpy.ndarray): Arreglo que contiene los datos de la nube de puntos.
        nLayers (int): Número de capas que se desean tener.
        axis (int): Eje perpendicular para el corte de capas. 0 es el eje x,
            1 es el eje y, 2 es el eje z.

    Returns:
        Lista que contiene las capas deseadas represtadas como listas.
        Cada capa contiene listas de tamaño 2 que representan las coordeanas en los dos ejes restantes
        de los puntos pertenecientes a esa capa.
    """
    if nLayers < 1: raise ValueError("Selecciona un número de capas positivo")

    #x_axis y y_axis son las direcciones del plano perpendicular al eje, 0 es el eje x, 1 es el eje y, 2 es el eje z
    if axis == 0: # Eje x
        x_axis = 1
        y_axis = 2
    elif axis == 1: # Eje y
        x_axis = 0
        y_axis = 2
    elif axis == 2: # Eje z
        x_axis = 0
        y_axis = 1
    else:
        raise ValueError("El eje perpendicular debe ser 0, 1 o 2")

    dataXY = data[:,(x_axis,y_axis)]

    # Obtiene el máximo y mínimo del eje perpendicular
    dataAxis = data[:,axis]
    maxData = np.amax(dataAxis)
    minData = np.amin(dataAxis)

    # Obtiene el ancho de una capa
    interval = (maxData-minData)/nLayers

    distance = dataAxis - minData

    # Asigna a cada indice de los datos, la capa a la que corresponde.
    positionIndexCapa = np.floor(distance/interval).astype(int) if interval > 0 \
        else np.zeros_like(dataAxis,dtype="int") # valida si interval > 0
                                    # para evitar error de divisón entre cero

    # Existe el caso cuando dataAxis == maxDato, entonces positionIndexCapa = nLayers
    # lo cual es incorrecto porque los índices deben de ir de 0 a nLayers -1
    # Para solucionar esto, se parachan los valores positionIndexCapa == nLayers
    positionIndexCapa[positionIndexCapa==nLayers] = nLayers - 1

    layers = declaraArray(nLayers)

    #Se realiza la separación de las capas buscando los elementos que tienen como indice el número de capa.
    for i in range(nLayers):
        layers[i] = dataXY[np.where(positionIndexCapa == i)]

    return layers


def corteMetodoB(data,nLayers,axis):
    """ Método B para partir capas con base en el numero de capas, ajustando
    los intervalos de acuerdo a la media.

    Se observan las coordenadas en el eje seleccionado de los puntos y estos se asignan a una capa
    dependiendo de si su coordenada correspondiente está en un intervalo establecido
    por el número de capas por debajo y por encima de la media.
    Todos los puntos de una capa son proyectados a un plano cuyo vector normal es
    el eje seleccionado.

    Ejemplo:
        Se tienen los puntos A, B, C, D, E, cuyas coordenadas en el eje perpendicular
        son -2, 3, 8, 8 y 8 respectivamente. Se deseean 5 capas.
        La media es 5.
        Se desean 5 capas, por lo que se tienen 3 capas por debajo de la media y
        2 capas por encima de la media.
        Las capas son:
            Capas inferiores
            Capa 0: [-2     , 0.333)
            Capa 1: [ 0.333 , 2.666)
            Capa 2: [ 2.666 , 5    )
            Capas superiores
            Capa 3: [ 5     , 6.5  )
            Capa 4: [ 6.5   , 8    ]

    Args:
        data (numpy.ndarray): Arreglo que contiene los datos de la nube de puntos.
        nLayers (int): Número de capas que se desean tener.
        axis (int): Eje perpendicular para el corte de capas. 0 es el eje x,
            1 es el eje y, 2 es el eje z.

    Returns:
        Lista que contiene las capas deseadas represtadas como listas.
        Cada capa contiene listas de tamaño 2 que representan las coordeanas en los dos ejes restantes
        de los puntos pertenecientes a esa capa.
    """
    if nLayers < 1: raise ValueError("Selecciona un número de capas positivo")

    # Este método no funciona con una capa
    if nLayers == 1:
        return corteMetodoA(data,nLayers,axis)

    #x_axis y y_axis son las direcciones del plano perpendicular al eje, 0 es el eje x, 1 es el eje y, 2 es el eje z
    if axis == 0: # Eje x
        x_axis = 1
        y_axis = 2
    elif axis == 1: # Eje y
        x_axis = 0
        y_axis = 2
    elif axis == 2: # Eje z
        x_axis = 0
        y_axis = 1
    else:
        raise ValueError("El eje perpendicular debe ser 0, 1 o 2")

    dataXY = data[:,(x_axis,y_axis)]

    # Obtiene el máximo, mínimo y la media
    dataAxis = data[:,axis]
    maxData = np.amax(dataAxis)
    meanData = np.mean(dataAxis)
    minData = np.amin(dataAxis)

    #Obtenemos el numero de capas antes y despues del promedio
    nLayersLow = math.ceil(nLayers/2)
    nLayersHigh = math.floor(nLayers/2)

    # Obtenemos el intervalo de separacion de las capas, de ambos grupos, de esta forma nos 
    # quedan dos grupos con una distribucion de puntos por capa mas eficaz
    intervalLow = (meanData-minData)/nLayersLow
    intervalHigh = (maxData-meanData)/nLayersHigh

    # Se asigna a cada indice de los datos, la capa a la que corresponde.
    positionIndexCapa = np.where(
        dataAxis <= meanData,
        np.floor((dataAxis - minData)/intervalLow).astype(int),
        np.floor((dataAxis - meanData)/intervalHigh).astype(int) + nLayersLow
    # La siguiente linea valida si invervalLow e intervalHigh son mayores que cero para evitar
    # errores de división entre cero
    )  if intervalLow > 0 and intervalHigh > 0 else np.zeros_like(dataAxis,dtype="int")

    # Existe el caso cuando dataAxis == maxDato, entonces positionIndexCapa = nLayers
    # lo cual es incorrecto porque los índices deben de ir de 0 a nLayers -1
    # Para solucionar esto, se parachan los valores positionIndexCapa == nLayers
    positionIndexCapa[positionIndexCapa==nLayers] = nLayers - 1

    layers = declaraArray(nLayers)

    # Se realiza la separación de las capas buscando los elementos que tienen como indice el número de capa.
    for i in range(nLayers):
        layers[i] = dataXY[np.where(positionIndexCapa == i)]

    return layers


def corteMetodoC(data,interval,axis):
    """ Método C para partir capas a partir de un intervalo deseado.

    Se observan las coordenadas de los puntos y estos se asignan a una capa
    dependiendo de si su coordenada correspondiente está en un intervalo establecido
    por el usuario dentro del eje seleccionado.
    Todos los puntos de una capa son proyectados a un plano cuyo vector normal es
    el eje seleccionado.

    Ejemplo:
        Se tienen los puntos A, B, C, D, E, cuyas coordenadas en el eje perpendicular
        son -2, 3, 4, 7 y 8 respectivamente. Se deseea un intervalo de 4.
        Con ello se obtienen 3 capas.
        Las capas son:
            Capa 0: [-2 , 2)
            Capa 1: [ 2 , 6)
            Capa 2: [ 6 , 8]

    Args:
        data (numpy.ndarray): Arreglo que contiene los datos de la nube de puntos.
        interval (float): Intervalo de separación deseado entre capas.
        axis (int): Eje perpendicular para el corte de capas. 0 es el eje x,
            1 es el eje y, 2 es el eje z.

    Returns:
        Lista que contiene las capas deseadas represtadas como listas.
        Cada capa contiene listas de tamaño 2 que representan las coordeanas en los ejes restantes
        de los puntos pertenecientes a esa capa.
    """
    #x_axis y y_axis son las direcciones del plano perpendicular al eje, 0 es el eje x, 1 es el eje y, 2 es el eje z
    if axis == 0: # Eje x
        x_axis = 1
        y_axis = 2
    elif axis == 1: # Eje y
        x_axis = 0
        y_axis = 2
    elif axis == 2: # Eje z
        x_axis = 0
        y_axis = 1
    else:
        raise ValueError("El eje perpendicular debe ser 0, 1 o 2")

    if interval <= 0: raise ValueError("Selecciona un intervalo positivo")

    # Se separan los datos en coordenadas del plano y coordenadas de la recta perpendicular
    dataXY = data[:,(x_axis,y_axis)]
    dataAxis = data[:,axis]

    # Se obtienen el mínimo y el máximo del eje perpendicular
    maxData = np.amax(dataAxis)
    minData = np.amin(dataAxis)

    # Se obtiene el número de capas que contendrá el intervalo deseado
    nLayers = np.ceil((maxData-minData)/interval).astype(int)

    distance = dataAxis - minData

    # Se obtienen los índices de capas
    positionIndexCapa = np.floor(distance/interval).astype(int)

    # Existe el caso cuando dataAxis == maxDato, entonces positionIndexCapa = nLayers
    # lo cual es incorrecto porque los índices deben de ir de 0 a nLayers -1
    # Para solucionar esto, se parachan los valores positionIndexCapa == nLayers
    positionIndexCapa[positionIndexCapa==nLayers] = nLayers - 1

    layers = declaraArray(nLayers)

    #Se realiza la separación de las capas buscando los elementos que tienen como indice el número de capa.
    for i in range(nLayers):
        layers[i] = dataXY[np.where(positionIndexCapa == i)]

    return layers


def quitaCapasVacias(arrayDeCapas):
    """ Elimina capas vacias.

    Elimina las capas que no poseen mas de 3 puntos (La cantidad minima para
    formar un área con trayectoria cerrada), así como las capas cuyos puntos
    estan en una recta paralela a cualquiera de los ejes coordenados. Esto
    último porque no se puede obtener la relación de aspecto de una recta
    paralela a los ejes coordeandos.

    Args:
        arrayDeCapas (list): Lista que contiene las capas obtenidas.

    Returns:
        Lista con capas no vacias.
    """

    i=0
    while i<len(arrayDeCapas):
        capa = np.asarray(arrayDeCapas[i])
        
        if len(capa)<3 or len(np.unique(capa[:,0]))<2 or len(np.unique(capa[:,1]))<2:
            print("Esta capa se quitara: "+str(i))
            plt.plot(arrayDeCapas[i][:,0],arrayDeCapas[i][:,1],'o')
            arrayDeCapas.pop(i)
        else:
            i=i+1

    return arrayDeCapas


def transforma_TESTER(bof,centroide):
    """ Función para transformar el descriptor BOF en coordenadas cartesianas.

    Función para verificar que los datos de módulos calculados sean
    similares a los de la figura original.
    Transforma las coordenadas polares descritas por BOF en coordenadas cartesianas
    que pueden ser graficadas.

    Args:
        bof (numpy.ndarray): Contiene el descriptor BOF de la capa.
        centroide (tuple): Contiene las coordenadas del centroide de la capa.

    Returns:
        Arreglo de numpy que contiene los puntos descritos por BOF en coordenadas
        cartesianas.
    """
    array=np.zeros((len(bof),2))
    for i in range(len(bof)):
        coordenadas=polarToCartesiana_TESTER(i*int(360/len(bof)),bof[i])
        array[i,0]=coordenadas[0]+centroide[0]
        array[i,1]=coordenadas[1]+centroide[1]
    return array #Retorna matriz Numpy que contiene los 180 vertices (en coordenadas cartesianas) del contorno de la figura

def polarToCartesiana_TESTER(grados,r):
    """ Convierte coordenadas polares en cartesianas.

    Args:
        grados (float): Ángulo en grados.
        r (float): Distancia al centro de coordenadas.

    Returns:
        Tupla con las coordenadas cartesiandas.
    """
    rad=math.radians(grados+180)
    x=-r*math.cos(rad)
    y=r*math.sin(rad)
    return (x,y) #Regresa el valor de coordenadas cartesianas del punto ingresado por polares en el argumento r(distancia del centro al punto) y grados


def interpola(bof):
    """ Interpolación de puntos vacios en el descriptor.

    Args:
        bof (numpy.ndarray): Contiene el descriptor BOF de la capa.

    Returns:
        Descriptor BOF sin espacios vacios (módulos con valor cero).
    """
    # A continuación se realiza la interpolación para los puntos vacios en el descriptor
    #Verificando que tenemos valores en todos los datos de la matriz, sino, se rellenan con una interpolación entre disctancias

    # Se define el tamaño del descriptor
    bofSize = len(bof)

    #Caso 1, faltan al inicio
    if bof[0] == 0:
        contEspaciosVaciosContiguos=1
        #Comenzamos a contar hacia adelante cuantos vacios hay hasta encontrar un valor, ese nos servira de extremo
        j=1
        while bof[j] == 0:
            j=j+1
            contEspaciosVaciosContiguos=contEspaciosVaciosContiguos+1
        extremoB=bof[j]

        #Comenzamos a contar hacia atras cuantos vacios hay hasta encontrar un valor, ese nos servira de extremo
        k=bofSize-1
        while bof[k] == 0:
            k=k-1
            contEspaciosVaciosContiguos=contEspaciosVaciosContiguos+1
        extremoA=bof[k]

        #Comenzamos a calcular los valores
        interpolar =extremoB - extremoA #calculamos cuanto varia un extremo y otro, 
        interpolar= interpolar/(contEspaciosVaciosContiguos+1) #A esta variacion la dividimos entre el numero de espacios vacios y obtenemos un numero que sera usado para hacer incrementos

        #Para todos los valores hacia atras, a partir del extremo, su valor de distancia sera sumando el incremento "interpolar" 
        for f in range(k,bofSize):
            extremoA=extremoA+interpolar
            bof[f]=extremoA
        #Para todos los valores hacia adelante, a partir del ultimo valor del ciclo anterior, seguiran aumentandose con el incremento hasta llegar al otro extremo
        for f in range(0,j):
            extremoA=extremoA+interpolar
            bof[f]=extremoA

    #Caso 2, faltan en medio
    j=1 #iniciamos en 1 ya que comprobamos que el 0 si tiene valores
    while j<bofSize-1:
        #Verificamos si el valor actual es 0, si no es, solamente se avanza
        if bof[j]==0:
            #si fue verdadero, se guarda el valor anterior como extremo 
            contEspaciosVaciosContiguos=0
            extremoA=bof[j-1]
            k=j #Guardamos el punto en el que comenzamos a contar los espacios vacios
            while j<bofSize-1 and bof[j]==0: #contamos los vacios mientras avanzamos la variable j iterador
                contEspaciosVaciosContiguos=contEspaciosVaciosContiguos+1
                j=j+1

            #al salir del ciclo anterior, veremos si el ultimo en determinarse como vacio es el ultimo dato (el 180)
            if j==bofSize-1 and bof[bofSize-1]==0: #Si es el dato y ademas esta vacio, se cuenta y el extremo será el dato numero 0 (le damos la vuelta)
                contEspaciosVaciosContiguos=contEspaciosVaciosContiguos+1
                extremoB=bof[0]

            else: #en caso contrario, solamente se toma como el ultimo dato como extremo
                extremoB=bof[j]

            #Se obtienen los datos para interpolar como en el caso 1
            interpolar =extremoB - extremoA
            interpolar= interpolar/(contEspaciosVaciosContiguos+1)

            #Se guardan los datos de las distancias que faltan

            for f in range(k,j):
                extremoA=extremoA+interpolar
                bof[f]=extremoA
        else:
            j=j+1

    #Caso 3: falta el utlimo
    if bof[bofSize-1]==0:
        extremoA=bof[bofSize-2]
        extremoB=bof[0]
        #Se obtienen los datos para interpolar como en el caso 1
        interpolar =extremoB - extremoA
        interpolar= interpolar/2

        bof[bofSize-1]=extremoA+interpolar
    
    return bof

#Funcion principal en la que se trabaja
def extractBofs(pcd,axis,method=1,layers=1,min_ratio=0.1,binary_size=150,plotBof=False,floor_plane=None,ceil_plane=None,file=True):
    """ Extracción de BOFs.

    A partir de una nube de puntos se generan capas (curvas de nivel) de acuerdo
    al método seleccionado y se obtiene el descriptor BOF de cada capa.

    Args:
        file (str): Indica la ubicación de la nube de puntos en formato pcd.
        axis (int): Eje perpendicular al plano de proyección. 
            0 es el eje x, 1 es el eje y,  2 es el eje Z
        method (int): Selección del método de corte de capas a utilizar.
        layers (int/float): Número de capas o intervalo de separación deseado.
        plotBof (bool): True si se quiere graficar los datos de cada capa.
        floor_plane (tuple): Tupla de tamaño 4 con los coeficientes de la
            ecuación cartesiana del plano del piso. Se utiliza para filtrar/quitar
            aquellos puntos que están por debajo del plano del piso.
            Si la ecuación del plano es Ax + By + Cz + D = 0, la tupla que se
            debe enviar es (A,B,C,D).

    Returns:
        bofsFrame (list): Lista que contiene los descriptores BOF de cada capa
            obtenida.
    """
    #Primero extraemos los datos crudos
    if file:
        pcd = o3d.io.read_point_cloud(pcd)
    else:
        pcd=pcd
        
    data = np.asarray(pcd.points)

    # Si se envió un plano del piso
    if floor_plane:
        A, B, C, D = floor_plane
        # Ecuación del plano con "y" despejada
        y_plane = -A/B * data[:,0] - C/B * data[:,2] - D/B
        above_floor = data[:,1] > y_plane
        data = data[above_floor]

    # Si se envió un plano del techo
    if ceil_plane:
        A, B, C, D = ceil_plane
        # Ecuación del plano con "y" despejada
        y_plane = -A/B * data[:,0] - C/B * data[:,2] - D/B
        under_ceiling = data[:,1] < y_plane
        data = data[under_ceiling]

    #Cortamos los datos en capas-curvas de nivel
    #metodoA : 1, metodoB: 2, metodoC: 3
    #Para los métodos A y B el segundo parámetro es el número de capas, para el método C es el intervalo.
    #el último parámetro es el eje perpendicular al plano de proyección. 0 es el eje x, 1 es el eje y,  2 es el eje Z
    capas=cortaCapas(data,nCapasOintervalo=layers,metodo=method,eje=axis)

    #Declaramos la variable que contendrá todos los 180 modulos normalizados de cada capa
    bofsScene=[]

    # Se itera sobre cada capa
    for i in range(len(capas)):
        # cada capa es una matriz numpy
        pcd = np.asarray(capas[i])

        # Transforma la capa a imagen
        img = PCD_Image_Processor.pcd_2_image(pcd=pcd,pixels_height=binary_size)

        # Suaviza la imagen para eliminar huecos causados por la baja resolución
        # de la nube de puntos
        img = PCD_Image_Processor.soften_image(img,(5,5))

        # Obtiene los contornos de la imagen
        contours = PCD_Image_Processor.get_contours(img,min_area_ratio=min_ratio)

        # Si no se obtuvo contorno alguno, se salta la iteración
        if len(contours) == 0:
            continue

        # Obtención de los descriptores BOF de todos los contornos obtenidos
        angleOffset = 2
        bofsLayer = PCD_Image_Processor.get_image_bofs(contours,angleOffset)

        # Se interpolan los puntos vacios del descriptor
        interpolatedBofs = list( map( interpola, bofsLayer) )

        # Se agregan las BOF a la capa
        bofsScene.extend(interpolatedBofs)

        # Si se desean graficar los datos de la capa y su descriptor
        if plotBof:
            # Obtiene los centroides de todos los contornos encontrados
            centroids = list(
               map(PCD_Image_Processor.get_centroid,contours)
            )

            # Crea una imagen RGB
            colorImg = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
            # Dibuja todos los contornos obtenidos que cumplen con el criterio
            cv.drawContours(colorImg,contours,-1,(255,0,0),2)
            # Dibuja todos los centroides de los contornos encontrados
            for centroid in centroids:
                cv.circle(colorImg, centroid, 3, (0,0,255), -1)

            # Creación de la primer figura, la cual mostrará la nube de puntos
            # de la capa, así como su representación en imagen.
            fig, axs = plt.subplots(1, 2)
            fig.set_size_inches(12,4.5)
            fig.suptitle(f'Capa {i}: Nube de puntos e imagen binaria')

            # Despliegue de la nube de puntos de la capa
            axs[0].set_title('Nube de puntos')
            axs[0].plot(pcd[:,0],pcd[:,1],'o')

            # Despliegue de la imagen binaria
            axs[1].set_title('Imagen Binaria')
            axs[1].imshow(colorImg)

            # Muestra la figura con la nube de puntos y la imagen binaria
            plt.show()

            numBofs = len(bofsLayer)

            # Creación de la seunda figura, la cual mostrará los contornos
            # encontrados, así como sus BOF.
            fig, axs = plt.subplots(nrows=numBofs,ncols=3,squeeze=False)
            plt.subplots_adjust(wspace=0.3,hspace=0.4)
            fig.set_size_inches(12,4*numBofs)
            fig.suptitle(f'Capa {i}: Contornos y descriptores')

            for j in range(numBofs):
                # Despliegue de la imagen con el contorno j-ésimo
                colorImg = cv.cvtColor(img,cv.COLOR_GRAY2RGB)
                cv.drawContours(colorImg,contours,j,(255,0,0),2)
                cv.circle(colorImg, centroids[j], 3, (0,0,255), -1)
                axs[j,0].set_title(f'Contorno {j}')
                axs[j,0].imshow(colorImg)

                # Conteo del número de puntos que se interpolaron
                interpolatedPoints = np.count_nonzero(bofsLayer[j]==0)

                # Gráfica de la BOF interpolada
                axs[j,1].set_title(f'BOF {j} interpolada')
                axs[j,1].set_xlabel(f'Ángulo [°]\nSe interpolaron {interpolatedPoints}'+
                    f' puntos de {interpolatedBofs[j].size}')
                axs[j,1].set_ylabel('Magnitud')
                axs[j,1].grid()
                axs[j,1].plot(np.linspace(0,360,num=len(interpolatedBofs[j])),interpolatedBofs[j],'or')
                axs[j,1].plot(np.linspace(0,360,num=len(interpolatedBofs[j])),interpolatedBofs[j],'--')

                # Transformación de la BOF a coordenadas cartesianas
                testInterpolatedBof = transforma_TESTER(interpolatedBofs[j],(0,0))

                # Gráfica de la BOF en coordenadas cartesianas
                axs[j,2].set_title(f'BOF {j} interpolada\nen coordenadas cartesianas')
                axs[j,2].plot(testInterpolatedBof[:,0],testInterpolatedBof[:,1],'o')

            plt.show()


    return bofsScene

# Si se ejecuta el script actual y no está siendo importado
if __name__ == '__main__':
    begin=time.time()
    extractBofs(file='pcd_test.ply',method=1,layers=1,axis=2,min_ratio=0.1,binary_size=150,plotBof=True,
        # ceil_plane=(0,1,0,-.55)
        floor_plane=(0,1,0.11,0.57)
    )
    end=time.time()
    print("Tiempo de ejecución: "+ str(end-begin))
