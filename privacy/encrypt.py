from cryptography.fernet import Fernet
import numpy as np
import pickle

KEY = Fernet.generate_key()
cipher = Fernet(KEY)


def encrypt(weights):    

    serialized = pickle.dumps(weights)

    ciphertext = cipher.encrypt(serialized)

    print("AES-128 encrypted, sending...")

    return ciphertext


def decrypt(ciphertext):

    decrypted = cipher.decrypt(ciphertext)

    weights = pickle.loads(decrypted)

    return weights