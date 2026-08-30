import numpy as np
import hashlib
import math
import matplotlib.pyplot as plt
from datetime import datetime

import random
import string
import psutil
import os
import time
import threading

from .crsst21 import crsst21

from .ve_impostor import VE_keygen, VE_impostor_encrypt, VE_impostor_decrypt
from nacl.public import PublicKey


class usgpv:

    def __init__(self,n_param,N_param, cupd_param, seed=None):

        '''
        Establecemos los parámetros básicos que debe introducir el usuario, registros y parámetros
        operativos que utilizaremos en algunas funciones.
        '''

        # Definimos la época
        self.epoc_param = 0
        
        # Definimos la dimensión de salida, el número máximo de actualizaciones y el parámetro cupd
        self.n_param = n_param
        self.N_param = N_param
        self.cupd_param = cupd_param

        # Definimos las claves privadas, públicas y el token como variables para usar
        self.R_matrix = None
        self.A_matrix = None
        self.token = None
        self.sigma_G = 10.0 
        self.p0_param = 0.3

        # Definimos un registro para guardar las claves secretas, públicas, tokens y firmas por épocas
        self.secret_keys = []
        self.public_keys = []
        self.VE_secret_keys = []
        self.VE_public_keys = []
        self.tokens = []
        self.signs = []

        # Definimos una seed para reproducibilidad y un generador de números aleatorios de numpy
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    # Registra una clave secreta
    def reg_secret_key(self,sk):
        self.secret_keys.append((self.epoc_param,sk))

    # Registra una clave pública
    def reg_public_key(self,pk):
        self.public_keys.append((self.epoc_param,pk))

    # Registra una clave privada de VE
    def reg_VE_secret_key(self,VE_sk):
        self.VE_secret_keys.append((self.epoc_param,VE_sk))

    # Registra una clave pública de VE
    def reg_VE_public_key(self,VE_pk):
        self.VE_public_keys.append((self.epoc_param,VE_pk))

    # Registra un token de actualización
    def reg_token(self,token):
        self.tokens.append((self.epoc_param,token))

    # Registra una firma
    def reg_sign(self,sign):
        self.signs.append((self.epoc_param,sign))

    def show_system_params(self):

        '''
        Esta función imprime los parámetros del sistema para un instante concreto.
        '''
        print('\n===== SYSTEM PARAMETERS =====')

        print(f'e          = {self.epoc_param}')
        print(f'n          = {self.n_param}')
        print(f'N          = {self.N_param}')

        print(f'q          = {self.q_param}')
        print(f'k = log2(q)= {self.k_param}')

        print(f'm          = {self.m_param}')
        print(f'w = nk     = {self.w_param}')
        print(f'm_bar      = {self.mbar_param}')

        print(f'r          = {self.r_param}')
        print(f'c_upd      = {self.cupd_param}')
        print('='*40)
        print('A matrix')
        print(self.A_matrix)
        print('R matrix')
        print(self.R_matrix)
        print('='*40)

        print('\n--- Gaussian parameters s_i ---')
        print(f's_required_1      = {self.s_required_1}')
        print(f's_required_2      = {self.s_required_2}')
        for i, s in enumerate(self.s_params, start=1):
            print(f's_{i} = {s}')

        print('\n--- Signature bounds beta_i ---')
        for i, beta in enumerate(self.beta_params, start=1):
            print(f'beta_{i} = {beta}')

        print('\n==============================\n')

    @staticmethod
    def gram_schmidt(base: np.ndarray) -> np.ndarray:

        '''
        Esta función implementa el algoritmo de ortogonalización de Gram-Schmidt
        Input: base (base de vectores)
        Output: B_star (base de vectores ortogonales)
        '''
        # Tomamos las dimensiones de la base como matriz
        n, m = base.shape

        # Definimos una matriz de cero en la que iremos guardando las coordenadas
        # por columnas
        B_star = np.zeros((n, m), dtype=float)

        # Iteramos en cada columna de B
        for i in range(m):
            # Copiamos la columna en v
            v = base[:, i].copy()

            # Para cada coordenada de ese vector vertical
            for j in range(i):

                # Calculamos el denominador de la proyección
                denom = np.dot(B_star[:, j], B_star[:, j])

                # Si el denominador es 0 devolvemos error
                if denom == 0:
                    raise ValueError('La base tiene vectores linealmente dependientes')

                # Calculamos la proyección de v sobre todos los vectores que estén ya en B_star
                projection = (np.dot(base[:, i], B_star[:, j]) / denom) * B_star[:, j]

                # Le restamos a v la parte no ortogonal en cada una de los vectores de B_star
                v = v - projection

            # Guardamos v en su columna correspondiente
            B_star[:, i] = v

        # Devolvemos la base ortogonalizada
        return B_star

    def setup(self):

        '''
        Esta función establece el resto de parámetros calculados a partir de los aportados por el usuario.
        '''

        # Definimos el módulo q
        self.q_param = int(2**np.ceil(1.3*np.log2(self.n_param)))

        # Definimos la dimensión m del espacio de partida
        self.m_param = int(2*self.n_param*np.log2(self.q_param))

        # Definimos el parámetro k (número de bits necesarios para enteros módulo q = 2^k)
        self.k_param = int(np.log2(self.q_param))

        # Definimos el parámetro w
        self.w_param = int(self.n_param*self.k_param)

        # Definimos el parámetro m barra
        self.mbar_param = int(self.m_param - self.w_param)

        # Definimos el parámetro r para escalar dispersión y calcular cotas.
        self.r_param = max(1.0, (np.ceil(np.log2(self.w_param))))

        # Calculamos los parámetros gaussianos para todas las épocas
        self.s_params = [np.round(self.p0_param*self.w_param,2)] 
        for _ in range(self.N_param-1):
            value = self.s_params[-1]
            self.s_params.append(np.round(self.cupd_param*value + self.cupd_param,2))
        
        # Calculamos las cotas de las firmas para todas las épocas
        self.beta_params = [np.round(1.5*self.r_param*self.s_params[0]*np.sqrt(self.m_param),2)]
        for _ in range(self.N_param-1):
            value = self.beta_params[-1]
            self.beta_params.append(np.round(self.cupd_param*value + self.cupd_param*np.sqrt(self.m_param),2))



    def TrapGen(self) -> tuple[np.ndarray, np.ndarray]:

        '''
        Esta función genera una par (A,R) donde A es una clave pública y R una trapdoor secreta para generar firmas cortas
        según el esquema usgpv.
        Input: n_param (dimensión del espacio de salida), m_param (dimensión del espacio de entrada), q_param (módulo)
        Output: A_e (clave pública), R_e (trapdoor o clave privada)
        '''
        
        # Definimos el vector g (bloque de la matriz gadget G)
        self.g_vector = np.array([2**i for i in range(self.k_param)], dtype=np.int64).reshape(1, self.k_param)

        # Definimos matriz identidad de dimensión n y de dimensión m
        self.I_n = np.eye(self.n_param, dtype=np.int64)
        self.I_m = np.eye(self.m_param, dtype=np.int64)

        # Definimos la matriz G gadget como el producto de kron de las dos matrices anteriores
        self.G_matrix = np.kron(self.I_n, self.g_vector)

        # Ahora definimos la matriz S_k (bloque de la matriz S) como una matriz de ceros
        self.S_k = np.zeros((self.k_param, self.k_param), dtype=np.int64)

        # Introducimos los valores correctos en la diagonal y la subdiagonal
        for i in range(self.k_param):
            self.S_k[i,i] = 2
            
            if i < self.k_param-1:
                self.S_k[i+1,i] = -1

        # Definimos la matriz S como el producto de kron de las matriz S_k y la identidad
        self.S_matrix = np.kron(self.I_n, self.S_k)

        # Calculamos la matriz de Gram-Schmidt y la guardamos para usarla más adelante
        self.S_bar = self.gram_schmidt(self.S_matrix)

        # Ahora definimos las matrices A barra aleatoriamente con dimension n x mbar
        self.A_bar_matrix = self.rng.integers(low=0,high=self.q_param,size=(self.n_param, self.mbar_param),dtype=np.int64)

        # Definimos la matriz R = {-1,0,1}^(mbar x w) con distribución P = [0.05, 0.9, 0.05]
        # Definición provisional, revisar distribución
        self.R_matrix = self.rng.choice([-1, 0, 1],size=(self.mbar_param, self.w_param),p=[self.p0_param/2, 1-self.p0_param, self.p0_param/2]).astype(np.int64)

        # Definimos la segunda parte de la matriz A = G - Abar @ R
        self.A1 = (self.G_matrix - self.A_bar_matrix @ self.R_matrix) % self.q_param

        # Construimos la matriz A = [Abar | A_1]
        self.A_matrix = np.concatenate([self.A_bar_matrix, self.A1], axis=1)

        # Definimos la identidad de dimensión w x w para construir la matriz B = [R | I_w]^t
        self.I_w = np.eye(self.w_param, dtype=np.int64)
        self.B_matrix = np.vstack([self.R_matrix, np.eye(self.w_param)]).astype(np.int64)


        # Definimos las matrices de covarianza Sigma_G y Sigma, ya que se usan en varios algoritmos y una vez
        # fijados los parámetros gaussianos y el parámetro r, podemos calcularlas. Además, sigma_G es fijo.
        self.SIGMA_G = self.sigma_G * self.I_w.astype(float)

        self.SIGMA = (
            self.s_params[self.epoc_param]
        )**2 * np.eye(self.m_param, dtype=float)


        # Ahora vamos a comprobar si se cumplen las condiciones sobre el parámetro de dispersión con las matrices que acabamos
        # de definir. Se tienen que cumplir dos condiciones:

        # La primera es que s_e >= C*||Sbar||*sqrt(log w) con C > 1. Como tomamos provisionalmente sigma_G = 2 y C = 1.5, 
        # realizamos así el cálculo
        Sbar_max = np.max(np.linalg.norm(self.S_bar, axis=0))
        self.s_required_1 = 1.5 * Sbar_max * np.sqrt(np.log2(self.w_param))

        # Comprobamos si se cumple la primera condición
        test1 = self.r_param * np.sqrt(self.sigma_G) >= self.s_required_1
        if not test1:
            print(self.r_param * np.sqrt(self.sigma_G))
            print(self.s_required_1)
            raise ValueError('El parámetro de dispersión no supera s_required_1')
        
        print(f'Test de dispersión adecuada para la matriz G: {test1}')

        # La segunda es que s_e >= r * sqrt((2 + sigma_G)*lambda_max) donde lambda_max es el mayor valor autovalor de B @ B^t
        lambda_max = np.linalg.eigvalsh(self.B_matrix.T @ self.B_matrix).max()
        self.s_required_2 = np.sqrt((2.0+self.sigma_G)*lambda_max)

        # Comprobamos si se cumple la segunda condición
        test2 = self.s_params[self.epoc_param] >= self.s_required_2
        if not test2:
            print(self.s_params[0])
            print(self.s_required_2)
            raise ValueError('El parámetro de dispersión no supera s_required_2')
        print(f'Test de dispersión adecuada para  s_{self.epoc_param}: {test2}')
        
        return self.A_matrix, self.R_matrix
    
    def BitDecomp(self, V: np.ndarray) -> np.ndarray:

        '''
        Calcula la descomposición binaria de cada vector columna de la matriz de entrada V y 
        la devuelve en una matriz Z_0, donde la columna i de Z_0 es la descomposición binaria 
        de la columna i de V.

        Input: V (matriz entera de dimensión n x m)
        Output: Z_0 (matriz binaria de dimensión nk x m)
        '''

        # Imponemos que la entrada sea una matriz entera módulo q
        V = np.asarray(V, dtype=np.int64) % self.q_param

        # Si la entrada es un vector, la convertimos en una matriz columna
        if V.ndim == 1:
            V = V.reshape(-1, 1) # -1 calcula el número de filas automáticamente. El 1 indica la columna

        # Verificamos que la dimensión de salida coincide con la establecida en el sistema
        if V.shape[0] != self.n_param:
            raise ValueError(f'V debe tener {self.n_param} filas, pero tiene shape {V.shape}')

        # Generamos un vector con tantas coordenadas como bits necesitamos (necesitamos k bits)
        shifts = np.arange(self.k_param, dtype=np.int64)

        # Coordenada a coordenada combinamos los elementos de cada matriz desplazando i in {1,...,k} bits y 
        # tomando el primero con & 1. Concretamente, se genera bits con dimensión (V.shape[0],shifts.shape[0],V.shape[1])
        # y luego se rellena coordenada a coordenada haciendo la operación que en este caso es 
        #                      bits[i,k,j] = (V[i,j] >> shifts[k]) & 1

        bits = (V[:, None, :] >> shifts[None, :, None]) & 1

        # Luego cambiamos a la dimensión que queremos que es nk x m, y rellenamos por filas, lo cual nos devuelve los bits 
        # de las coordenadas de cada columna concatenados desde el bit menos significativo de la primera coordenada hasta 
        # el más significativo de la última coordenada
        Z0 = bits.reshape(self.w_param, V.shape[1]).astype(np.int64)

        return Z0
    
    
    def klein_sampler(self, Z0: np.ndarray, S: np.ndarray, sigma=1.0) -> np.ndarray:

        '''
        Samplea preimágenes del coset de soluciones Gz = 0 (mod q).
        Input: Z0 (preimagenes deterministas), S (base del retículo de soluciones de Gz = 0) y sigma (varianza de la distribución)
        Output: Y (soluciones muestreadas según una normal de parámetro sqrt(sigma) de Gz = 0)
        '''


        # Empezamos convirtiendo la matriz S a una matriz entera por seguridad
        S = np.asarray(S, dtype=np.int64)

        # Guardamos una versión con reales para poder calcular su Gram-Schmidt
        S_float = S.astype(float)

        # Tomamos las preimagenes deterministas calculada con binary_coordinates_matrix
        Z0 = np.asarray(Z0, dtype=float)

        # Si lo que recibimos es un vector, lo pasamos a una matriz de una columna
        if Z0.ndim == 1:
            Z0 = Z0.reshape(-1, 1) # Aquí -1 calcula el número de filas necesarias automáticamente

        # Comprobamos si ese número de filas es el adecuado según el esquema
        if Z0.shape[0] != self.w_param:
            raise ValueError(
                f'Z0 debe tener {self.w_param} filas, pero tiene shape {Z0.shape}'
            )

        # Guardamos el número de columnas, es decir, el número de preimágenes que vamos a samplear
        L = Z0.shape[1]

        # Calculamos el Gram-Schmidt de la matriz S
        Sbar = self.gram_schmidt(S_float)

        # Definimos nuestros centros 
        C = -Z0.copy()

        # Definimos la matriz de coeficientes que vamos a ir rellenando
        A_coeff = np.zeros((self.w_param, L), dtype=np.int64)

        # Vamos a ir sampleando en paralelo las coordenadas de todas las preimágenes que necesitamos 
        # desde i = w hasta i = 1
        for i in reversed(range(self.w_param)):

            # Tomamos la primera columna de la matriz S y Sbar
            sbar_i = Sbar[:, i]
            s_i = S_float[:, i]

            # Calculamos el denominador de la media para la fila i
            norm_sq = np.dot(sbar_i, sbar_i)

            # Comprobamos que la norma no sea nula
            if norm_sq == 0:
                raise ValueError('Vector Gram-Schmidt nulo')

            # Calculamos el vector medias para la fila i
            mu_i = (sbar_i @ C) / norm_sq

            # Calculamos la varianza para la fila i
            sigma_i = self.r_param * sigma / np.sqrt(norm_sq)

            # Sampleamos L enteros a la vez con los parámetros que hemos calculado
            a_i = np.rint(
                self.rng.normal(
                    loc=mu_i,
                    scale=sigma_i,
                    size=L
                )
            ).astype(np.int64) # Además los redondeamos ya que deben ser coeficientes enteros

            # Introducimos los coeficientes de la fila i en la matriz de coeficientes
            A_coeff[i, :] = a_i

            # Actualizamos todos los centros a la vez. Outer crea un vector con los productos
            # de las coordenadas de s_i y a_i
            C = C - np.outer(s_i, a_i)

        # Una vez calculados todos los coeficientes, multiplicamos por S y obtenemos todas las 
        # soluciones del coset de soluciones de Gz = 0
        Y = S @ A_coeff

        return Y.astype(np.int64)
           
    def oracle_sampler(self, V: np.ndarray) -> np.ndarray:
        
        '''
        Esta función calcula preimágenes de las columnas de la matriz V para la ecuación modular
                                        GZ = V (mod q)
        sampleando sobre el coset de soluciones con una distancia estadística del muestreo ideal
        despreciable en el número de columnas de G.
        Input: V (matriz de síndromes)
        Output: Z (matriz de preimágenes de la forma Z0 + Y)
        '''

        # Convertimos la entrada en una matriz entera por seguridad
        V = np.asarray(V, dtype=np.int64) % self.q_param

        # Si la entrada es un vector, convertimos en una matriz de una columna
        if V.ndim == 1:
            V = V.reshape(-1, 1)

        # Calculamos la parte determinista
        Z0 = self.BitDecomp(V)

        # Calculamos la parte probabilistica
        Y = self.klein_sampler(
            Z0,
            self.S_matrix,
            sigma=np.sqrt(self.sigma_G)
        )

        # Combinamos las soluciones en una matriz de preimágenes para la ecuación GZ = V (mod q)
        Z = Z0 + Y

        # Comprobamos si efectivamente verifica la ecuación
        if np.any((self.G_matrix @ Z) % self.q_param != V % self.q_param):
            raise ValueError('Las preimágenes no cumplen la ecuación GZ = V (mod q)')

        return Z.astype(np.int64)
    

    def SamplePre(self, Y: np.ndarray, SIGMA_override=False) -> np.ndarray:

        '''
        Esta función toma una matriz de sindromes y calcula preimágenes para la ecuación modular AU = y.
        Input: Y (matriz de sindromes)
        Output: U (matriz de preimágenes o firmas)
        '''
        # Convertimos la matriz de entrada en una matriz de enteros por seguridad
        Y = np.asarray(Y, dtype=np.int64) % self.q_param

        # Con esta variable detectamos si la entrada tiene una o más columnas, es decir, si vamos a calcular
        # una o más preimágenes. En general, usaremos una preimagen para firmar mensajes y varias preimagenes
        # para calcular el token de actualización
        single_input = False

        # Si la entrada es un vector, lo convertimos a una matriz de una columna
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
            single_input = True # Modificamos la variable para usarla después

        # Nos aseguramos que la matriz de entrada tenga la dimensión del espacio de salida
        if Y.shape[0] != self.n_param:
            raise ValueError(
                f'Y debe tener {self.n_param} filas, pero tiene shape {Y.shape}'
            )

        # Convertimos la matriz B a real para usarla en el cálculo de la dispersión del vector p
        B_float = self.B_matrix.astype(float)

        SIGMA = self.SIGMA

        if SIGMA_override:
            SIGMA = (self.s_params[0])**2 * np.eye(self.m_param, dtype=float)

        # Calculamos la matriz de covarianza para muestrear la propuesta de firma inicial. En este caso, sabemos 
        # que es semidefinida positiva por la segunda condición evaluada en TrapGen
        self.SIGMA_p = SIGMA - B_float @ self.SIGMA_G @ B_float.T
        self.SIGMA_p = (self.SIGMA_p + self.SIGMA_p.T) / 2

        # Sampleamos el vector p de una distribución normal multivariante centrada en 0 y con la covarianza SIGMA_p
        p_real = self.rng.multivariate_normal(
            mean=np.zeros(self.m_param),
            cov=(self.r_param**2) * self.SIGMA_p,
            size=Y.shape[1]
        ).T
        

        # Redondeamos p a un vector entero
        P = np.rint(p_real).astype(np.int64)

        # Dividimos el vector p en p1 y p2 
        P1 = P[:self.mbar_param, :]
        P2 = P[self.mbar_param:, :]

        # Calculamos wbar y wprima 
        Wbar = (
            self.A_bar_matrix @ (P1 - self.R_matrix @ P2)
        ) % self.q_param

        Wprima = (
            self.G_matrix @ P2
        ) % self.q_param

        # Calculamos V como la diferencia entre Y y la imagen del vector P. Así al calcular la preimagen de V, podremos 
        # corregir P para que verifique la ecuación modular
        V = (Y - Wbar - Wprima) % self.q_param

        # Sampleamos la preimagen correctora con el oráculo
        Z = self.oracle_sampler(V)

        # Calculamos finalmente nuestra preimagen
        X = P + self.B_matrix @ Z

        # Comprobamos que la solución verifica efectivamente la ecuación modular
        if not np.array_equal((self.A_matrix @ X) % self.q_param,Y):
            raise ValueError('Las preimágenes generadas no cumplen la ecuación matricial')

        # Si la entrada es un vector, devolvemos la primera columna en forma de vector
        if single_input:
            return X[:, 0]

        return X

    def H(self, message: str) -> np.ndarray:

        '''
        Esta función traduce un mensaje en texto plano a un vector de dimensión n con coordenadas módulo q
        Input: m (mensaje en texto plano)
        Output: y (vector n dimensional de coordenadas módulo q)
        '''

        # Verificamos que la entrada es una cadena de texto
        if isinstance(message, str):
            # La códificamos con utf-8 para tratar acentos y carácteres especiales
            message = message.encode("utf-8")

        # Calculamos el número total de bytes que sería w/8 = n*k/8
        total_bytes = math.ceil(self.w_param / 8)

        # Shake256 es una función hash que, a diferencia de sha_256, devuelve tantos bits como queramos. Estos
        # bits están generados a partir del mensaje de entrada.
        digest = hashlib.shake_256(message).digest(total_bytes) # Nos devuelve los bytes que necesitamos

        # Convertimos la cadena de bytes a un entero para después tomar la cadena de bits del tamaño que queramos
        X = int.from_bytes(digest, byteorder="little")

        # Mask va a ser siempre una cadena de k 1's
        mask = self.q_param - 1

        # Definimos una lista para guardar las coordenadas del sindrome y = H(m)
        coords = []

        # Calculamos cada coordenada tomando cadenas de k bits y guardandolas como enteros
        for i in range(self.n_param):
            # Para cada coordenada i = 1,...,n , nos desplazamos al bit en la posición n*k y 
            # tomamos los k bits siguientes usando la máscara definida
            coord = (X >> (i * self.k_param)) & mask
            # Después añadimos la coordenada a la lista. Resaltar que para python a un entero
            # se le pueden hacer operaciones bit a bit pero seguirá saliendo como un entero
            coords.append(coord)

        return np.array(coords, dtype=np.int64)

    def VE_impostor_KeyGen(self) -> PublicKey:

        '''
        Esta función genera unas claves para el VE_impostor
        Input: -
        Output: VE_pk (clave pública VE), VE_sk (clave privada VE)
        '''
        self.VE_pk, self.VE_sk = VE_keygen()

        self.reg_VE_secret_key(self.VE_sk)
        self.reg_VE_public_key(self.VE_pk)

        # Devolvemos solo la clave pública, la idea es que solo alguien con acceso al sistema 
        # tenga la clave privada (por representar el concepto de manera aproximada sin usar un esquema ZKP)
        return self.VE_pk

    def KeyGen(self) -> tuple[np.ndarray, np.ndarray]:

        '''
        Esta función genera un par de claves pública/privada haciendo uso del algoritmo TrapGen
        Input: -
        Output: pk (clave pública), sk (clave privada)
        '''
        # Generamos un par de claves y las registramos
        self.pk, self.sk = self.TrapGen()
        self.reg_secret_key(self.sk)
        self.reg_public_key(self.pk)

        return self.pk, self.sk
    
    def Sig(self, m):
        '''
        Esta función firma una lista de mensajes haciendo uso del algortimo SamplePre
        Input: [m] (lista de mensajes en texto plano)
        Output: sign (firma del mensaje)
        '''
        # Detectamos si la entrada es una lista de mensajes o un único mensaje
        es_lista = isinstance(m, (list, tuple))

        if es_lista:
            mensajes = list(m)
        else:
            mensajes = [m]

        L = len(mensajes)

        # Si la lista está vacía, devolvemos una lista vacía
        if L == 0:
            return []

        # Calculamos todos los síndromes H(m_i)
        # Y tendrá dimensión n x L
        Y = np.column_stack([self.H(mensaje) for mensaje in mensajes])

        # Sampleamos todos los testigos a la vez
        # witness_real tendrá dimensión m_param x L
        witness_real = self.rng.multivariate_normal(
            mean=np.zeros(self.m_param),
            cov=(self.r_param**2) * self.SIGMA,
            size=L
        ).T

        # Redondeamos los testigos
        witness_e = np.rint(witness_real).astype(np.int64)

        # Calculamos todas las afirmaciones públicas
        # statement_e tendrá dimensión n x L
        statement_e = (self.pk @ witness_e) % self.q_param

        # Calculamos todas las firmas como preimágenes
        # U tendrá dimensión m_param x L
        U = self.SamplePre((Y + statement_e) % self.q_param)

        print(np.array_equal((self.pk @ U) % self.q_param, (Y + statement_e) % self.q_param ))

        # Calculamos ahora la lista de firmas
        signs = []

        # vamos añadiendo cada firma, con su afirmación pública y su testigo cifrado
        for j in range(L):
            sign = (
                U[:, j],
                statement_e[:, j],
                VE_impostor_encrypt(witness_e[:, j], self.VE_pk)
            )

            self.reg_sign(sign)
            signs.append(sign)

        # Si la entrada original era un único mensaje, devolvemos una única firma
        if not es_lista:
            return signs[0]

        # Si era una lista, devolvemos la lista completa
        return signs
    
    
    def VE_impostor_ver(self,vepk: PublicKey,pk: np.ndarray,sign: tuple[np.ndarray,np.ndarray,bytes]) -> bool:
        
        '''
        Esta función es una sustituta de la función de verificación de la afirmación pública para 
        el esquema VE. Sirve como recreación de lo que pasaría en una implementación real, pero en
        un entorno controlado.
        Input: vepk (clave pública que usaría un usuario para verificar si se cumple la relación), 
               pk (clave pública), sign (firma que contiene el testigo cifrado y la afirmación pública)
        Output: True/False
        '''

        # Usamos la clave pública aportada por el usuario y vemos si coincide con la que tiene el sistema
        # asociado a la clave privada de esta época. Comparamos vs la clave pública registrada para la 
        # época actual. La clave privada de esa misma época es la que usamos para descifrar
        if vepk != self.VE_public_keys[self.epoc_param][1]:
            return False
        
        # El sistema descifra internamente el testigo cifrado y comprueba si se cumple la relación
        witness_e = VE_impostor_decrypt(sign[2], self.VE_secret_keys[self.epoc_param][1])
        if not np.array_equal(sign[1] % self.q_param,(pk @ witness_e) % self.q_param):
            return False
        
        # Comprobamos también que el testigo cumple la condición de la cota
        if np.linalg.norm(witness_e) > self.beta_params[self.epoc_param]:
            return False 
        
        # Si todo se cumple devolvemos True. En caso contrario, devolvemos False
        return True

    def Ver(self,vepk: PublicKey,pk: np.ndarray,m: str,sign: tuple[np.ndarray,np.ndarray,bytes]) -> bool:

        '''
        Esta función verifica que una firma es válida en la época actual
        Input: vepk (calve pública esquema VE), pk (clave pública), m (mensaje en texto plano), 
               sign (firma del mensaje m)
        Output: True/False
        '''
        
        # Calculamos la norma de la firma
        norma = np.linalg.norm(sign[0])

        # Comprobamos que la firma cumple la cota
        if limit := (norma <= self.beta_params[self.epoc_param]):
            print(f'El vector tiene norma {norma} que es menor que la cota {self.beta_params[self.epoc_param]}')
            pass

        # Comprobamos que la afirmación pública es cierta
        if VE_condition := self.VE_impostor_ver(vepk,pk,sign):
            print(f'La afirmación pública t_e es cierta para la relación definida')
            pass
        
        # Calculamos el sindrome del mensaje
        y = (self.H(m) + sign[1]) % self.q_param

        # Comprobamos si el sindrome calculado del mensaje m y la firma aportada verifican la ecuación modular
        if congruence := np.all((pk @ sign[0]) % self.q_param == y):
            print(f'La congruencia es verdadera')
            pass

        # Si todo es cierto devolvemos True. En caso contrario, devolvemos False
        return limit and congruence and VE_condition

        
    def Next(self,pk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        '''
        Esta función pasa el sistema a la siguiente época. Para ello, genera unas nuevas claves, ajusta los parámetros 
        de dispersión y actualiza el contador de época.
        Input: pk (clave pública época e)
        Output: token (token de actualización para pasar de e -> e+1)
        '''

        # Nos aseguramos que se ha introducido la clave pública de la época actual
        if not np.array_equal(pk % self.q_param, self.pk % self.q_param):
            raise ValueError('La pk introducida no coincide con la clave pública actual')
        
        # Nos aseguramos de que se puede actualizar. Sumamos 1 porque permitimos N épocas
        # desde el 0
        if self.epoc_param + 1 >= self.N_param:
                raise ValueError('No puedes actualizar más')
          
        # Aumentamos la época        
        self.epoc_param += 1

        # Generamos unas nuevas claves y las guardamos en las variables del sistema.
        # Recordemos que internamente se registran las claves con la época, por eso
        # hay que actualizar la época antes.
        self.pk,self.sk = self.KeyGen()

        # Generamos unas nuevas claves para el esquema VE. La clave secreta se actualiza
        # internamente en VE_impostor_KeyGen()
        self.VE_pk = self.VE_impostor_KeyGen()

        # Calculamos el token de actualización usando el algoritmo SamplePre
        token = self.SamplePre(pk, SIGMA_override=True)
        
        # Si el token verifica la ecuación modular, cambiamos de época
        if np.array_equal((self.pk @ token) % self.q_param,pk % self.q_param):

            # Registramos el token
            self.reg_token(token)

            # Calculamos la nueva matriz de covarianzas para el parámetro gaussiano
            # de la siguiente época (s_{e+1})
            self.SIGMA = (
                self.s_params[self.epoc_param]
            )**2 * np.eye(self.m_param, dtype=float)

            # # Mostramos los parámetros del sistema actualizados
            # self.show_system_params()

            # Devolvemos las claves y el token
            return self.pk, self.sk, token
        else:
            raise ValueError('El token no verifica la condición algebraica')
        

    def Update(self, token: np.ndarray, sign: np.ndarray) -> tuple[np.ndarray, np.ndarray, bytes]:

        
        # Nos aseguramos de que exista una época anterior
        if self.epoc_param == 0:
            raise ValueError('No podemos actualizar en la época inicial')
        
        # Recuperamos la afirmación pública con la clave privada del esquema VE de la época anterior
        witness_e_old = VE_impostor_decrypt(sign[2], self.VE_secret_keys[self.epoc_param - 1][1])

        # Verificamos si cumple la cota definida en la relación
        norm_witness_e = np.linalg.norm(witness_e_old)
        if norm_witness_e > self.beta_params[self.epoc_param - 1]:
            raise ValueError('El testigo no cumple la cota de la norma')
        
        # Tomamos la clave pública de la época anterior
        _,last_pk = self.public_keys[self.epoc_param - 1]

        # Verificamos que el token introducido es el token de transición entre la etapa anterior y la actual
        if not np.array_equal((self.pk @ token) % self.q_param,last_pk % self.q_param):
            raise ValueError('El token no actualiza correctamente de A_e a A_{e+1}')
        
        # Verificamos si el testigo cumple la ecuación definida en la relación 
        if not np.array_equal((last_pk @ witness_e_old) % self.q_param,sign[1] % self.q_param):
            raise ValueError('El testigo no cumple la ecuación matricial')
        
        # Calculamos la dispersión del ruido para corregir el efecto del token de actualización sobre las firmas. Lo calculamos
        # con el token en reales por seguridad
        token_float = token.astype(float)
        self.SIGMA_upd = (self.s_params[self.epoc_param]**2 * self.I_m) - (self.s_params[self.epoc_param - 1]**2 *(token_float @ token_float.T))
        self.SIGMA_upd = (self.SIGMA_upd + self.SIGMA_upd.T) / 2

        # Comprobamos que la matriz sea semidefinida positiva. Para ello tomamos el parámetro gaussiano de esta etapa y la anterior
        s_new = self.s_params[self.epoc_param]
        s_old = self.s_params[self.epoc_param - 1]

        # Calculamos el máximo autovalor del producto del token por su traspuesta
        token_norm = np.linalg.norm(token.astype(float), 2)

        # Comprobamos que sea semidefinida positiva
        if (s_new - s_old * token_norm <= 0):
            raise ValueError('La matriz de covarianza de la actualización no es semidefinida positiva')

        # Si lo es, entonces calculamos el ruido gaussiano
        r_noise_real = self.rng.multivariate_normal(
            mean=np.zeros(self.m_param,),
            cov=(self.r_param**2) * self.SIGMA_upd
        )

        r_noise = np.rint(r_noise_real).astype(np.int64)

        # Tomamos la firma antigua
        u_e_old = sign[0]

        # Calculamos la nueva firma actualizada
        u_e = (token @ u_e_old) + r_noise

        # Calculamos el nuevo testigo
        witness_e = (token @ witness_e_old) + r_noise


        # Comprobamos que la nueva firma sigue estando acotada
        if np.linalg.norm(u_e) > self.beta_params[self.epoc_param]:
            raise ValueError('La firma actualizada supera la cota')

        # Comprobamos que el nuevo testigo también sigue acotado
        if np.linalg.norm(witness_e) > self.beta_params[self.epoc_param]:
            raise ValueError('El nuevo testigo supera la cota')

        # Calculamos la nueva afirmación pública asociada al nuevo testigo
        statement_e = (self.pk @ witness_e) % self.q_param

        # Ciframos el testigo
        upd_sign = (u_e, statement_e, VE_impostor_encrypt(witness_e,self.VE_pk))

        # Tomamos la clave pública anterior
        _,pk_old = self.public_keys[self.epoc_param - 1]

        # Comprobamos si se cumple la ecuación con el mismo sindrome tras la actualización de firma
        if not np.array_equal(((self.pk @ u_e) - statement_e) % self.q_param,
                    ((pk_old @ u_e_old) - sign[1]) % self.q_param):
            raise ValueError('La actualización no preserva la relación entre las claves y la firma')

        # Registramos la firma actualizada
        self.reg_sign(upd_sign)

        return upd_sign
    


def medir_rendimiento(func, *args, intervalo=0.01, **kwargs):
    # Identificamos el proceso de la fución
    process = psutil.Process(os.getpid())

    # CPU inicial del proceso
    cpu_inicio = process.cpu_times()
    cpu_inicio_total = cpu_inicio.user + cpu_inicio.system

    # Memoria inicial del proceso
    memoria_inicio = process.memory_info().rss / 1024 / 1024  # MB

    # Creamos la lista para guardar las mediciones de memoria
    memorias = []
    ejecutando = True

    # Esta función realizará mediciones en segundo plano
    def monitor_memoria():
        while ejecutando:
            memoria_actual = process.memory_info().rss / 1024 / 1024
            memorias.append(memoria_actual)
            time.sleep(intervalo)

    # Iniciamos la función anterior
    hilo = threading.Thread(target=monitor_memoria)
    hilo.start()

    # Con esto guardamos el inicio de la ejecución de la función
    t0 = time.perf_counter()

    # La ejecutamos y calculamos el tiempo de ejecución
    try:
        resultado = func(*args, **kwargs)
    finally:
        tiempo_real = time.perf_counter() - t0

        # Paramos el proceso en segundo plano
        ejecutando = False
        hilo.join()

    # CPU final del proceso
    cpu_final = process.cpu_times()
    cpu_final_total = cpu_final.user + cpu_final.system

    # Tiempo total de CPU usado por la función
    tiempo_cpu_usado = cpu_final_total - cpu_inicio_total

    # Porcentaje medio de CPU usado por la función
    cpu_porcentaje = (tiempo_cpu_usado / tiempo_real) * 100 if tiempo_real > 0 else 0

    # Memoria
    memoria_final = process.memory_info().rss / 1024 / 1024
    memoria_pico = max(memorias) if memorias else memoria_final
    memoria_usada = memoria_pico - memoria_inicio

    print('=== Medición de la función ===')
    print(f'Tiempo real: {tiempo_real:.6f} s')
    print(f'CPU usado: {cpu_porcentaje:.2f} %')
    print(f'Memoria usada: {memoria_usada:.2f} MB')
    print(f'Memoria pico del proceso: {memoria_pico:.2f} MB')

    return resultado


def generar_palabras(n=1000, longitud=10):

    palabras = []

    rng = random.Random(2026)

    for _ in range(n):
        palabra = ''.join(rng.choices(string.ascii_lowercase, k=longitud))
        palabras.append(palabra)

    return palabras


def simulacion():
    sample_size = 100

    # Ahora generamos la lista de palbras que vamos a firmar
    lista_palabras = generar_palabras(n=sample_size,longitud=10)

    inicio_total = datetime.now()

    # Creamos una instancia del sistema
    print('='*50)
    print('Inicialización del sistema')
    system = usgpv(32, 5, 22900,seed=2026)


    def firmar(sample_size, lista_palabras):

        # Definimos los parámetros y generamos las claves iniciales
        system.setup()
        pk0, sk0 = system.KeyGen()
        vepk0 = system.VE_impostor_KeyGen()


        # Firmamos todas las palabras y las guardamos en una lista
        print('='*50)
        print(f'Firmando los {sample_size} mensajes')
        firmas = []
        fallos = []
        for i,palabra in enumerate(lista_palabras):
            # print(f'Firmando mensaje: {i}/{sample_size}')
            sign = system.Sig(palabra)
            if system.Ver(vepk0, pk0, palabra, sign):
                firmas.append((palabra,sign))
            else:
                fallos.append((palabra,sign))
                print(f'Error al firmar la palabra {palabra}')
        print('='*50)
        return firmas,fallos,pk0
    
    print('='*50)
    print('Estadisticas etapa 0')
    firmas,fallos,pk0 = medir_rendimiento(firmar,sample_size, lista_palabras)
    print('='*50)
    fin_epoca_0 = datetime.now()

    inicio_epoca_1 = datetime.now()
    
    def actualizacion_1(sample_size,firmas,pk0):
        # Cambiamos una vez de época de e=0 -> e=1
        print('='*50)
        print('Cambio de época y actualización')
        pk1, sk1, token = system.Next(pk0)

        # Actualizamos todas las firmas
        firmas_1 = []
        fallos_1 = []
        for i,firma in enumerate(firmas):
            # print(f'Actualizando mensaje: {i}/{sample_size}')
            sign1 = system.Update(token, firma[1])
            if system.Ver(system.VE_pk, pk1, firma[0], sign1):
                firmas_1.append((firma[0],sign1))
            else:
                fallos_1.append((firma[0],sign1))
                print(f'Error al actualizar la palabra {firma[0]}')

        return firmas_1,fallos_1,pk1,token

    print('='*50)
    print('Estadisticas etapa 1')
    firmas_1,fallos_1,pk1,token = medir_rendimiento(actualizacion_1,sample_size,firmas,pk0)
    print('='*50)
    fin_epoca_1 = datetime.now()

    inicio_epoca_2 = datetime.now()

    def actualizacion_2(sample_size,firmas_1,pk1, lista_palabras):
        # Cambiamos una vez de época de e=1 -> e=2
        print('='*50)
        print('Cambio de época y actualización')
        pk2, sk2, token_2 = system.Next(pk1)

        # Actualizamos todas las firmas
        firmas_2 = []
        fallos_2 = []
        for i,firma in enumerate(firmas_1):
            # print(f'Actualizando mensaje: {i}/{sample_size}')
            sign2 = system.Update(token_2, firma[1])
            if system.Ver(system.VE_pk, pk2, firma[0], sign2):
                firmas_2.append((firma[0],sign2))
            else:
                fallos_2.append((firma[0],sign2))
                print(f'Error al actualizar la palabra {firma[0]}')

        for i,palabra in enumerate(lista_palabras):
            # print(f'Firmando mensaje: {i}/{sample_size}')
            sign = system.Sig(palabra)
            if system.Ver(system.VE_pk, pk2, palabra, sign):
                firmas_2.append((palabra,sign))
            else:
                fallos_2.append((palabra,sign))
                print(f'Error al firmar la palabra {palabra}')
        print('='*50)
        return firmas_2,fallos_2,pk2,token_2
    
    print('='*50)
    print('Estadisticas etapa 2')
    firmas_2,fallos_2,pk2,token_2 = medir_rendimiento(actualizacion_2,sample_size,firmas_1,pk1,lista_palabras)
    print('='*50)

    fin_total = datetime.now()

    print(f'Tiempo de ejecución total: {fin_total-inicio_total}')
    print(f'Tiempo de ejecución epoca 0: {fin_epoca_0-inicio_total}')
    print(f'Tiempo de ejecución epoca 1: {fin_epoca_1-inicio_epoca_1}')
    print(f'Tiempo de ejecución epoca 2: {fin_total-inicio_epoca_2}')


def analisis_estadistico():
    # Tamaño de la muestra
    sample_size = 100000

    # Ahora generamos la lista de palbras que vamos a firmar
    print('='*50)
    print(f'Generando las {sample_size} palabras')
    lista_palabras = generar_palabras(n=sample_size,longitud=10)
    print('='*50)

    # Creamos una instancia del sistema
    print('='*50)
    print('Inicialización del sistema')
    system_usgpv = usgpv(32, 5, 22900,seed=2026)


    # Definimos los parámetros y generamos las claves iniciales
    system_usgpv.setup()
    pk0, sk0 = system_usgpv.KeyGen()
    vepk0 = system_usgpv.VE_impostor_KeyGen()

    # Firmamos todas las palabras y las guardamos en una lista
    print('='*50)
    print(f'Firmando los {sample_size} mensajes')
    muestra_usgpv = system_usgpv.Sig(lista_palabras)
    print('='*50)


    # Guardamos la muestra y trasponemos para que cada columna sea una muestra
    X_usgpv = np.asarray([sample[0] for sample in muestra_usgpv]).T

    # Esta es la matriz usada por el sistema para muestrear
    Sigma_esperada = (system_usgpv.r_param**2)*np.asarray(system_usgpv.SIGMA, dtype=float)

    # Aquí estimamos la covarianza de la muestra usando el estimador insesgado
    Sigma_empirica = np.cov(X_usgpv, ddof=1)

    # Calculamos la media de la distribución, que tiene que estar cerca del 0
    media_empirica = np.mean(X_usgpv,axis=1)
    norma_media = np.linalg.norm(media_empirica)

    print('='*50)
    print(f'Norma euclidea de la media muestral: {norma_media}')

    # Calculamos la norma de Frobenius de la diferencia
    error_fro = np.linalg.norm(Sigma_empirica - Sigma_esperada, ord="fro")
    # Calculamos la norma de Frobenius esperada
    norma_esperada_fro = np.linalg.norm(Sigma_esperada, ord="fro")

    # Calculamos el porcentaje de error relativo
    error_relativo_fro = error_fro / norma_esperada_fro

    print('='*50)
    print(f'Error Frobenius absoluto: {np.round(error_fro,2)}')
    print(f'Norma Frobenius esperada: {np.round(norma_esperada_fro,2)}')
    print(f'Error Frobenius relativo: {np.round(100 * error_relativo_fro,2)} %')

    # Calculamos la norma espectral de la diferencia
    error_esp = np.linalg.norm(Sigma_empirica - Sigma_esperada, ord=2)
    # Calculamos la norma espectral esperada
    norma_esperada_esp = np.linalg.norm(Sigma_esperada, ord=2)

    # Calculamos el porcentaje de error relativo
    error_relativo_esp = np.round(error_esp / norma_esperada_esp,2)

    print('='*50)
    print(f'Error Espectral absoluto: {np.round(error_esp,2)}')
    print(f'Norma Espectral esperada: {np.round(norma_esperada_esp,2)}')
    print(f'Error Espectral relativo: {np.round(100 * error_relativo_esp,2)} %')

    # Calculamos las trazas de ambas matrices
    tr_emp = np.trace(Sigma_empirica)
    tr_esp = np.trace(Sigma_esperada)

    print('='*50)
    print(f'Traza empírica: {np.round(tr_emp,2)}')
    print(f'Traza esperada: {np.round(tr_esp,2)}')
    print(f'Error relativo traza: {np.round(100 * (tr_emp - tr_esp) / tr_esp,2)} %')

    # Calculamos los autovalores
    autovalores_esp = np.linalg.eigvals(Sigma_esperada)
    autovalores_emp = np.linalg.eigvals(Sigma_empirica)

    # Calculamos el error del máximo y del mínimo autovalor
    error_auto_max_rel = 100*(autovalores_emp.max() - autovalores_esp.max())/autovalores_esp.max()
    error_auto_min_rel = 100*(autovalores_emp.min() - autovalores_esp.min())/autovalores_esp.min()

    print('='*50)
    print(f'Error relativo autovalor máximo: {np.round(error_auto_max_rel,2)} %')
    print(f'Error relativo autovalor mínimo: {np.round(error_auto_min_rel,2)} %')


def analisis_estadistico_pca():
    sample_size = 5000

    # Ahora generamos la lista de palabras que vamos a firmar
    print('='*50)
    print(f'Generando las {sample_size} palabras')
    lista_palabras = generar_palabras(n=sample_size,longitud=10)
    print('='*50)

    # Creamos una instancia del sistema
    print('='*50)
    print('Inicialización del sistema')
    system_usgpv = usgpv(32, 5, 22900,seed=2026)


    # Definimos los parámetros y generamos las claves iniciales
    system_usgpv.setup()
    pk0, sk0 = system_usgpv.KeyGen()
    vepk0 = system_usgpv.VE_impostor_KeyGen()

    # Firmamos todas las palabras y las guardamos en una lista
    print('='*50)
    print(f'Firmando los {sample_size} mensajes')
    muestra_usgpv = system_usgpv.Sig(lista_palabras)
    print('='*50)


    print('Primera actualizacion USGPV')
    # Ahora vamos a actualizar la muestra para ver la acción del token
    pk1, sk1, token = system_usgpv.Next(pk0)
    muestra_actualizada_usgpv = [system_usgpv.Update(token, sign) for sign in muestra_usgpv]

    print('Segunda actualizacion USGPV')
    pk2, sk2, token2 = system_usgpv.Next(pk1)
    muestra_actualizada_usgpv = [system_usgpv.Update(token2, sign) for sign in muestra_actualizada_usgpv]

    print('Tercera actualizacion USGPV')
    pk3, sk3, token3 = system_usgpv.Next(pk2)
    muestra_actualizada_usgpv = [system_usgpv.Update(token3, sign) for sign in muestra_actualizada_usgpv]

    # print('Cuarta actualizacion USGPV')
    # pk4, sk4, token4 = system_usgpv.Next(pk3)
    # muestra_actualizada_usgpv = [system_usgpv.Update(token4, sign) for sign in muestra_actualizada_usgpv]

    X_usgpv = np.asarray([sample[0] for sample in muestra_actualizada_usgpv])

    # Ahora generamos una muestra con CRSST21
    # Creamos una instancia del sistema
    print('='*50)
    print('Inicialización del sistema')
    system_crsst21 = crsst21(32, 5, 22900,seed=2026)


    # Definimos los parámetros y generamos las claves iniciales
    system_crsst21.setup()
    pk0, sk0 = system_crsst21.KeyGen()

    # Firmamos todas las palabras y las guardamos en una lista
    print('='*50)
    print(f'Firmando los {sample_size} mensajes')
    muestra_crsst21 = system_crsst21.Sig(lista_palabras)
    print('='*50)

    print('Primera actualizacion CRSST21')
    # Ahora vamos a actualizar la muestra para ver la acción del token
    pk1, sk1, token = system_crsst21.Next(pk0)
    muestra_actualizada_crsst21 = [system_crsst21.Update(token, sign) for sign in muestra_crsst21]
    
    print('Segunda actualizacion CRSST21')
    pk2, sk2, token2 = system_crsst21.Next(pk1)
    muestra_actualizada_crsst21 = [system_crsst21.Update(token2, sign) for sign in muestra_actualizada_crsst21]

    print('Tercera actualizacion CRSST21')
    pk3, sk3, token3 = system_crsst21.Next(pk2)
    muestra_actualizada_crsst21 = [system_crsst21.Update(token3, sign) for sign in muestra_actualizada_crsst21]

    # print('Cuarta actualizacion CRSST21')
    # pk4, sk4, token4 = system_crsst21.Next(pk3)
    # muestra_actualizada_crsst21 = [system_crsst21.Update(token4, sign) for sign in muestra_actualizada_crsst21]

    X_crsst21 = np.asarray(muestra_actualizada_crsst21)

    def pca_muestra(X, n_componentes=None):
        '''
        Realiza un análisis PCA sobre una muestra dada.
        Input: X (muestra), n_componentes (número de componentes principales a devolver)
        Output: componentes (número de componentes), varianza_explicada (tasa de varianza explicada por componentes), 
                scores (proyecciones sobre las componentes principales)
        '''

        # Nos aseguramos de que la muestra es del tipo np.ndarray
        X = np.asarray(X, dtype=float)

        # En primer lugar, centramos la muestra. Aunque vimos que la media era prácticamente nula, 
        # no lo es exactamente así que es conveniente centrarla igualmente.
        media = np.mean(X, axis=0)
        X_centrada = X - media


        # Recalculamos la matriz de covarianza empírica
        Sigma_empirica = np.cov(X_centrada,rowvar=False, ddof=1)

        # Para comparar las muestras, las normalizamos segun su dispersión total
        # que es la raiz de la traza de la matriz de covarianza (es decir, la raiz 
        # de la suma de las varianzas de cada variable)
        escala = np.sqrt(np.trace(Sigma_empirica))

        # Calculamos los autovalores y autovectores
        autovalores, autovectores = np.linalg.eigh(Sigma_empirica)

        # Ordenamos de mayor a menor varianza. Esta función nos da los indices de los autovalores
        # ordenados de mayor a menor
        idx = np.argsort(autovalores)[::-1]

        # Obtenemos los autovalores en ese orden
        autovalores = autovalores[idx]
        # Obtenemos los autovectores en ese orden
        autovectores = autovectores[:, idx]

        # Calculamos el porcentaje de varianza explicada por cada uno
        varianza_explicada = autovalores / np.sum(autovalores)
        varianza_acumulada = np.cumsum(varianza_explicada)

        # nos quedamos con las primeras componentes si se pide
        if n_componentes is not None:
            autovalores = autovalores[:n_componentes]
            autovectores = autovectores[:, :n_componentes]
            varianza_explicada = varianza_explicada[:n_componentes]
            varianza_acumulada = varianza_acumulada[:n_componentes]

        # Normalizamos la muestra
        X_norm = X_centrada / escala

        # Proyectamos la muestra sobre las componentes principales
        scores = X_norm @ autovectores

        # Calculamos la lista de componentes
        componentes = np.arange(1, len(varianza_explicada) + 1)


        return componentes, varianza_explicada, scores
    
    # Calculamos las componentes, la tasa de varianza explicada y las proyecciones sobre las componentes para CRSST21
    print('='*50)
    print('Calculando PCA para CRSST21')
    componentes_crsst21, varianza_explicada_crsst21, scores_crsst21 = pca_muestra(X_crsst21,n_componentes=50)
    # Calculamos las componentes, la tasa de varianza explicada y las proyecciones sobre las componentes para USGPV
    print('='*50)
    print('Calculando PCA para USGPV')
    componentes_usgpv, varianza_explicada_usgpv, scores_uspgv = pca_muestra(X_usgpv,n_componentes=50)

    # Pasamos de decimal a porcentaje en la varianza explicada
    var_crsst21_pca = 100 * np.asarray(varianza_explicada_crsst21)
    var_usgpv_pca = 100 * np.asarray(varianza_explicada_usgpv)

    def grafico_barras_pca(componentes, var_pca_1, var_pca_2):     

        ancho = 0.35
        x = np.arange(len(componentes))

        plt.figure(figsize=(10, 5))

        plt.bar(
            x - ancho/2,
            var_pca_1,
            width=ancho,
            label="CRSST21"
        )

        plt.bar(
            x + ancho/2,
            var_pca_2,
            width=ancho,
            label="USGPV"
        )

        plt.xlabel("Componente principal")
        plt.ylabel("Varianza explicada (%)")
        plt.xticks(x, componentes)
        plt.legend()
        plt.tight_layout()
        plt.show()
    


    def plot_pca_3x3(scores,color='C0'):

        nombres = ['PC1', 'PC2', 'PC3']

        fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=False, sharey=False)

        for i in range(3):
            for j in range(3):
                ax = axes[i, j]

                ax.scatter(
                    scores[:, j],
                    scores[:, i],
                    s=8,
                    alpha=0.35,
                    c= color
                )

                ax.set_xlabel(nombres[j])
                ax.set_ylabel(nombres[i])

                ax.set_xlim(-0.6, 0.6)
                ax.set_ylim(-0.6, 0.6)

                ax.grid(True, alpha=0.25)


        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')

        plt.tight_layout()
        plt.show()

    # Graficamos el gráfico de barras con ambas varianzas (las componentes coinciden).
    grafico_barras_pca(componentes_crsst21,var_crsst21_pca,var_usgpv_pca)

    
    plot_pca_3x3(scores_crsst21)
    plot_pca_3x3(scores_uspgv, color='C1')

    
    


if __name__ == '__main__':
    raise SystemExit(
        'Ejecuta este modulo mediante uno de los lanzadores de scripts/ '
        'o mediante: python ejecutar.py --help'
    )
