'''Sustituto experimental del componente de cifrado verificable.

El TFM necesita encapsular el testigo de una firma para recuperarlo durante la
actualizacion. Este modulo reproduce solo ese flujo de cifrado y descifrado con
``SealedBox`` de PyNaCl. No implementa las pruebas ni las garantias de seguridad
de un esquema de cifrado verificable real.
'''

import io
import numpy as np
from nacl.public import PrivateKey, SealedBox


def VE_keygen():
    '''
    Genera par de claves asimétricas.
    '''
    sk = PrivateKey.generate()
    pk = sk.public_key
    return pk, sk


def vector_to_bytes(v: np.ndarray) -> bytes:
    '''
    Serializa vector/matriz NumPy sin pickle.
    '''
    buffer = io.BytesIO()
    np.save(buffer, v, allow_pickle=False)
    return buffer.getvalue()


def bytes_to_vector(data: bytes) -> np.ndarray:
    '''
    Reconstruye vector/matriz NumPy.
    '''
    buffer = io.BytesIO(data)
    return np.load(buffer, allow_pickle=False)


def VE_impostor_encrypt(v: np.ndarray, pk) -> bytes:
    '''
    Cifra un vector/matriz usando clave pública.
    '''
    plaintext = vector_to_bytes(v)
    box = SealedBox(pk)
    ciphertext = box.encrypt(plaintext)
    return ciphertext


def VE_impostor_decrypt(ciphertext: bytes, sk) -> np.ndarray:
    '''
    Descifra usando clave privada.
    '''
    box = SealedBox(sk)
    plaintext = box.decrypt(ciphertext)
    return bytes_to_vector(plaintext)


if __name__ == '__main__':

    pk_E, sk_E = VE_keygen()

    x = np.array([12, -5, 33, 100, 8934, 873981, 847,32,21831], dtype=np.int64) % 128

    c = VE_impostor_encrypt(x, pk_E)

    x_rec = VE_impostor_decrypt(c, sk_E)

    print(x_rec)
    print(np.array_equal(x, x_rec))
