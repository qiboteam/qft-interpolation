#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 9 2021

@author: Sergi Ramos-Calderer

QJPEG

"""

import numpy as np
from qibo.models import Circuit, QFT
from qibo import gates


class qjpeg_compression:
    """Class that compresses an image in a JPEG-inspired way"""
    def __init__(self, image, subspace=3, m=1):
        self.nx = int(np.ceil(np.log2(image.shape[0])))
        self.ny = int(np.ceil(np.log2(image.shape[1])))
        self.n = self.nx+self.ny+4
        self.subspace = subspace
        self.m = m
        k = subspace
        self.reg_nx = [i for i in range(self.nx-k)]+[self.nx-k+m+i for i in range(k)]
        self.reg_ny = [self.nx+2+i for i in range(self.ny-k)]+[self.nx+2+self.ny-k+m+i for i in range(k)]
        self.reg_n = [i for i in range(self.n)]
        self.img = image

    def img2state(self):
        """Insert the gray values of the image in the relevant amplitudes of the quantum system.
        Returns:
            state_vector (np.array): amplitudes of all quantum states.

        """
        state_vector = np.zeros(2**(self.n))
        for x in range(self.img.shape[0]):
            bin_x = format(x, f'0{self.nx}b')
            for y in range(self.img.shape[1]):
                bin_y = format(y, f'0{self.ny}b')
                num = 0
                for j in range(self.nx):
                    num += int(bin_x[j])*(2**(self.n-self.reg_nx[j]-1))
                for j in range(self.ny):
                    num += int(bin_y[j])*(2**(self.n-self.reg_ny[j]-1))
                state_vector[num] = np.sqrt(self.img[x, y])
        state_vector /= np.sqrt(np.sum(state_vector**2))
        return state_vector

    def qjpeg_circuit(self):
        k = self.subspace
        nx = self.nx
        ny = self.ny
        m = self.m
        n = self.n

        c = Circuit(n)

        c.add(gates.H(self.reg_n[nx-k]))
        c.add(gates.H(self.reg_n[nx+2+ny-k]))

        c.add([gates.CNOT(self.reg_n[nx-k], i) for i in self.reg_nx[-k:]])
        c.add([gates.CNOT(self.reg_n[nx+2+ny-k], i) for i in self.reg_ny[-k:]])

        c.add(gates.X(self.reg_n[nx+2-1]))
        c.add(gates.X(self.reg_n[-1]))

        c.add(QFT(k+2).on_qubits(*([self.reg_n[nx-k]]+self.reg_nx[-k:]+[self.reg_n[nx+2-1]])))
        c.add(QFT(k+2).on_qubits(*([self.reg_n[nx+2+ny-k]]+self.reg_ny[-k:]+[self.reg_n[-1]])))

        c.add([gates.CNOT(self.reg_nx[-k:][m], self.reg_nx[-k:][i]) for i in range(m)])
        c.add([gates.CNOT(self.reg_ny[-k:][m], self.reg_ny[-k:][i]) for i in range(m)])

        c.add(QFT(k+2-m).invert().on_qubits(*([self.reg_n[nx-k]]+self.reg_nx[-k:][m:]+[self.reg_n[nx+2-1]])))
        c.add(QFT(k+2-m).invert().on_qubits(*([self.reg_n[nx+2+ny-k]]+self.reg_ny[-k:][m:]+[self.reg_n[-1]])))

        c.add(gates.X(self.reg_n[nx+2-1]))
        c.add(gates.X(self.reg_n[-1]))

        c.add([gates.CNOT(self.reg_n[nx-k], i) for i in self.reg_nx[-k+m:]])
        c.add([gates.CNOT(self.reg_n[nx+2+ny-k], i) for i in self.reg_ny[-k+m:]])

        c.add(gates.H(self.reg_n[nx-k]))
        c.add(gates.H(self.reg_n[nx+2+ny-k]))

        return c

    def extract_image(self, state):
        k = self.subspace
        m = self.m
        nx = self.nx
        ny = self.ny
        state = state.reshape(2**(self.n//2), 2**(self.n//2))
        state = np.abs(state)**2
        state = state[::2, ::2]
        img = np.zeros((2**(nx-m), 2**(ny-m)))
        xx = 0
        yy = 0
        for i in range(2**(nx-k)):
            for j in range(2**(ny-k)):
                for x in range(2**(k-m)):
                    for y in range(2**(k-m)):
                        img[xx+x,yy+y] = state[2**(k+1)*i+x, 2**(k+1)*j+y]
                yy += 2**(k-m)
            yy = 0
            xx += 2**(k-m)
        return img

    def treat_image(self, image):
        image = image*self.img[100,100]/image[50,50]
        image = image*self.img[200,200]/image[100,100]
        image = image*self.img[100,200]/image[50,100]
        image = image*self.img[200,100]/image[100,50]
        return image

    def execute(self):
        state = self.img2state()
        c = self.qjpeg_circuit()
        print(c.summary())
        state = c(state).state()
        image = self.extract_image(state)
        #image = self.treat_image(image)
        mean_img = np.sum(self.img)/self.img.size
        image = image*mean_img*image.size/np.sum(image)
        image = image.astype(int)
        return image

    def __call__(self):
        return self.execute()


class qjpeg_interpolation:
    """Class that interpolates an image in a JPEG-inspired way"""
    def __init__(self, image, subspace=3, m=1):
        self.nx = int(np.ceil(np.log2(image.shape[0])))
        self.ny = int(np.ceil(np.log2(image.shape[1])))
        
        self.m = m
        self.subspace = subspace
        k = subspace

        if len(image.shape)>2:
            self.layers = image.shape[2]
        else:
            self.layers = 1
        self.l = int(np.ceil(np.log2(self.layers)))

        self.n = self.nx+self.ny+4+2*self.m+self.l

        self.reg_nx = [i for i in range(self.nx-k)]+[self.nx-k+1]+[self.nx-k+2+m+i for i in range(k-1)]
        self.reg_ny = [self.nx+2+m+i for i in range(self.ny-k)]+[self.nx+2+m+self.ny-k+1]+[self.nx+2+m+self.ny-k+2+m+i for i in range(k-1)]
        self.reg_mx = [self.nx-k+2+i for i in range(m)]
        self.reg_my = [self.nx+2+m+self.ny-k+2+i for i in range(m)]
        self.reg_n = [i for i in range(self.n)]
        self.reg_l = [i+self.nx+self.ny+4+2*self.m for i in range(self.l)]
        self.img = image
        self.image = image

    def img2state_layers(self):
        """Insert the values of the multi-layer image in the relevant amplitudes of the quantum system.
        Returns:
            state_vector (np.array): amplitudes of all quantum states.

        """
        state_vector = np.zeros(2**(self.n))
        
        for i in range(self.layers):
            for x in range(self.image.shape[0]):
                bin_x = format(x, f'0{self.nx}b')
                for y in range(self.image.shape[1]):
                    bin_y = format(y, f'0{self.ny}b')
                    num = 0
                    for j in range(self.nx):
                        num += int(bin_x[j])*(2**(self.n-self.reg_nx[j]-1))
                    for j in range(self.ny):
                        num += int(bin_y[j])*(2**(self.n-self.reg_ny[j]-1))
                    state_vector[num+i] = np.sqrt(self.image[x, y, i])
        state_vector /= np.sqrt(np.sum(state_vector**2))
        return state_vector

    def img2state_bw(self):
        """Insert the gray values of the image in the relevant amplitudes of the quantum system.
        Returns:
            state_vector (np.array): amplitudes of all quantum states.

        """
        state_vector = np.zeros(2**(self.n))
        
        for x in range(self.image.shape[0]):
            bin_x = format(x, f'0{self.nx}b')
            for y in range(self.image.shape[1]):
                bin_y = format(y, f'0{self.ny}b')
                num = 0
                for j in range(self.nx):
                    num += int(bin_x[j])*(2**(self.n-self.reg_nx[j]-1))
                for j in range(self.ny):
                    num += int(bin_y[j])*(2**(self.n-self.reg_ny[j]-1))
                state_vector[num] = np.sqrt(self.image[x, y])
        state_vector /= np.sqrt(np.sum(state_vector**2))
        return state_vector

    def img2state(self):
        """Insert the gray values of the image in the relevant amplitudes of the quantum system.
        Returns:
            state_vector (np.array): amplitudes of all quantum states.

        """
        nx = self.nx
        ny = self.ny
        n = self.n
        state_vector = np.zeros(2**(n))
        for x in range(self.img.shape[0]):
            bin_x = format(x, f'0{nx}b')
            for y in range(self.img.shape[1]):
                bin_y = format(y, f'0{ny}b')
                num = 0
                for j in range(nx):
                    num += int(bin_x[j])*(2**(n-self.reg_nx[j]-1))
                for j in range(ny):
                    num += int(bin_y[j])*(2**(n-self.reg_ny[j]-1))
                state_vector[num] = np.sqrt(self.img[x, y])
        state_vector /= np.sqrt(np.sum(state_vector**2))
        return state_vector

    def qjpeg_circuit(self):
        k = self.subspace
        nx = self.nx
        ny = self.ny
        m = self.m
        n = self.n

        c = Circuit(n)

        c.add(gates.H(self.reg_n[nx-k]))
        c.add(gates.H(self.reg_n[nx+2+m+ny-k]))

        c.add([gates.CNOT(self.reg_n[nx-k], i) for i in self.reg_nx[-k:]])
        c.add([gates.CNOT(self.reg_n[nx+2+m+ny-k], i) for i in self.reg_ny[-k:]])

        c.add(gates.X(self.reg_n[nx+2+m-1]))
        c.add(gates.X(self.reg_n[-1]))

        c.add(QFT(k+2).on_qubits(*([self.reg_n[nx-k]]+self.reg_nx[-k:]+[self.reg_n[nx+2+m-1]])))
        c.add(QFT(k+2).on_qubits(*([self.reg_n[nx+2+m+ny-k]]+self.reg_ny[-k:]+[self.reg_n[-1]])))

        c.add([gates.CNOT(self.reg_nx[-k], i) for i in self.reg_mx])
        c.add([gates.CNOT(self.reg_ny[-k], i) for i in self.reg_my])

        c.add(QFT(k+2+m).invert().on_qubits(*([self.reg_n[nx-k]]+[self.reg_nx[-k]]+self.reg_mx+self.reg_nx[-k+1:]+[self.reg_n[nx+2+m-1]])))
        c.add(QFT(k+2+m).invert().on_qubits(*([self.reg_n[nx+2+m+ny-k]]+[self.reg_ny[-k]]+self.reg_my+self.reg_ny[-k+1:]+[self.reg_n[-1]])))

        c.add(gates.X(self.reg_n[nx+2+m-1]))
        c.add(gates.X(self.reg_n[-1]))

        c.add([gates.CNOT(self.reg_n[nx-k], i) for i in [self.reg_nx[-k]]+self.reg_mx+self.reg_nx[-k+1:]])
        c.add([gates.CNOT(self.reg_n[nx+2+m+ny-k], i) for i in [self.reg_ny[-k]]+self.reg_my+self.reg_ny[-k+1:]])

        c.add(gates.H(self.reg_n[nx-k]))
        c.add(gates.H(self.reg_n[nx+2+m+ny-k]))

        return c

    def extract_image(self, state):
        k = self.subspace
        m = self.m
        nx = self.nx
        ny = self.ny
        #state = state.reshape(2**(self.n//2), 2**(self.n//2))
        if self.l == 0:
            state = state.reshape(2**(nx+2+m), 2**(ny+2+m))
            state = np.abs(state)**2
            state = state[::2, ::2]
            img = np.zeros((2**(nx+m), 2**(ny+m)))
            xx = 0
            yy = 0
            for i in range(2**(nx-k)):
                for j in range(2**(ny-k)):
                    for x in range(2**(k+m)):
                        for y in range(2**(k+m)):
                            img[xx+x,yy+y] = state[2**(k+m+1)*i+x, 2**(k+m+1)*j+y]
                    yy += 2**(k+m)
                yy = 0
                xx += 2**(k+m)
        else:
            state = state.reshape(2**(self.nx+2+self.m), 2**(self.ny+2+self.m), 2**self.l)
            state = np.abs(state)**2
            state = state[::2, ::2, :]
            img = np.zeros((2**(nx+m), 2**(ny+m), self.layers))
            xx = 0
            yy = 0
            for i in range(2**(nx-k)):
                for j in range(2**(ny-k)):
                    for x in range(2**(k+m)):
                        for y in range(2**(k+m)):
                            for l in range(self.layers):
                                img[xx+x,yy+y,l] = state[2**(k+m+1)*i+x, 2**(k+m+1)*j+y,l]
                    yy += 2**(k+m)
                yy = 0
                xx += 2**(k+m)

        #state = np.abs(state)**2
        #state = state[::2, ::2, :]
        #img = np.zeros((2**(nx+m), 2**(ny+m)))
        #xx = 0
        #yy = 0
        #for i in range(2**(nx-k)):
        #    for j in range(2**(ny-k)):
        #        for x in range(2**(k+m)):
        #            for y in range(2**(k+m)):
        #                img[xx+x,yy+y] = state[2**(k+m+1)*i+x, 2**(k+m+1)*j+y]
        #        yy += 2**(k+m)
        #    yy = 0
        #    xx += 2**(k+m)
        return img

    def execute(self):
        c = self.qjpeg_circuit()
        if self.l == 0:
            state = self.img2state_bw()
        else:
            state = self.img2state_layers()
        #print(c.summary())
        state = c(state).state()
        image = self.extract_image(state)
        print(image.shape)
        if self.l == 0:
            mean_img = np.sum(self.img)/self.img.size

            image = image*mean_img*image.size/np.sum(image)
            image = image.astype(int)
        else:
            for i in range(self.img.shape[2]):
                mean_img = np.sum(self.img[:,:,i])/self.img[:,:,i].size
                image[:,:,i] = image[:,:,i]*mean_img*image[:,:,i].size/np.sum(image[:,:,i])
            image = image.astype(int)

        #mean_img = np.sum(self.img)/self.img.size
        #image = image*mean_img*image.size/np.sum(image)
        #image = image.astype(int)
        return np.minimum(image, 255)

    def __call__(self):
        return self.execute()


# 1D versions of the JPEG inspired algorithm


class qjpeg_compression_1d:
    """Class that compresses an image in a JPEG-inspired way"""
    def __init__(self, probabilities, subspace=3, m=1, state=False):
        self.nx = int(np.ceil(np.log2(probabilities.size)))
        self.n = self.nx+2
        self.subspace = subspace
        self.m = m
        k = subspace
        self.reg_nx = [i for i in range(self.nx-k)]+[self.nx-k+m+i for i in range(k)]
        self.reg_n = [i for i in range(self.n)]
        self.prob = probabilities
        self.state = state

    def prob2state(self):
        """Insert the values of the probability distribution in the relevant amplitudes of the quantum system.
        Returns:
            state_vector (np.array): amplitudes of all quantum states.

        """
        state_vector = np.zeros(2**(self.n))
        for i in range(self.prob.size):
            bin_i = format(i, f'0{self.nx}b')
            num = 0
            for j in range(self.nx):
                num += int(bin_i[j])*2**(self.n-self.reg_nx[j]-1)
            if self.state:
                state_vector[num] = self.prob[i]
            else:
                state_vector[num] = np.sqrt(self.prob[i])
        state_vector /= np.sqrt(np.sum(state_vector**2))
        return state_vector

    def qjpeg_circuit(self):
        k = self.subspace
        nx = self.nx
        ny = self.ny
        m = self.m
        n = self.n

        c = Circuit(n)

        c.add(gates.H(self.reg_n[nx-k]))

        c.add([gates.CNOT(self.reg_n[nx-k], i) for i in self.reg_nx[-k:]])

        c.add(gates.X(self.reg_n[-1]))

        c.add(QFT(k+2).on_qubits(*([self.reg_n[nx-k]]+self.reg_nx[-k:]+[self.reg_n[-1]])))

        c.add([gates.CNOT(self.reg_nx[-k:][m], self.reg_nx[-k:][i]) for i in range(m)])

        c.add(QFT(k+2-m).invert().on_qubits(*([self.reg_n[nx-k]]+self.reg_nx[-k:][m:]+[self.reg_n[-1]])))

        c.add(gates.X(self.reg_n[-1]))

        c.add([gates.CNOT(self.reg_n[nx-k], i) for i in self.reg_nx[-k+m:]])

        c.add(gates.H(self.reg_n[nx-k]))

        return c

    def execute(self):
        state = self.prob2state()
        c = self.qjpeg_circuit()
        print(c.summary())
        state = c(state).state()
        #image = self.treat_image(image)
        return state

    def __call__(self):
        return self.execute()


class qjpeg_interpolation_1d:
    """Class that interpolates an image in a JPEG-inspired way"""
    def __init__(self, probabilities, subspace=3, m=1, state=False):
        self.nx = int(np.ceil(np.log2(probabilities.size)))
        self.n = self.nx+2+m
        self.m = m
        self.subspace = subspace
        k = subspace

        self.reg_nx = [i for i in range(self.nx-k)]+[self.nx-k+1]+[self.nx-k+2+m+i for i in range(k-1)]
        self.reg_mx = [self.nx-k+2+i for i in range(m)]
        self.reg_n = [i for i in range(self.n)]
        self.prob = probabilities
        self.state = state

    def prob2state(self):
        """Insert the values of the probability distribution in the relevant amplitudes of the quantum system.
        Returns:
            state_vector (np.array): amplitudes of all quantum states.

        """
        state_vector = np.zeros(2**(self.n), dtype=complex)
        for i in range(self.prob.size):
            bin_i = format(i, f'0{self.nx}b')
            num = 0
            for j in range(self.nx):
                num += int(bin_i[j])*2**(self.n-self.reg_nx[j]-1)
            if self.state:
                state_vector[num] = self.prob[i]
            else:
                state_vector[num] = np.sqrt(self.prob[i])
        state_vector /= np.sqrt(np.sum(state_vector**2))
        return state_vector

    def qjpeg_circuit(self):
        k = self.subspace
        nx = self.nx
        m = self.m
        n = self.n

        c = Circuit(n)

        c.add(gates.H(self.reg_n[nx-k]))

        c.add([gates.CNOT(self.reg_n[nx-k], i) for i in self.reg_nx[-k:]])

        c.add(gates.X(self.reg_n[-1]))

        c.add(QFT(k+2).on_qubits(*([self.reg_n[nx-k]]+self.reg_nx[-k:]+[self.reg_n[-1]])))

        c.add([gates.CNOT(self.reg_nx[-k], i) for i in self.reg_mx])

        c.add(QFT(k+2+m).invert().on_qubits(*([self.reg_n[nx-k]]+[self.reg_nx[-k]]+self.reg_mx+self.reg_nx[-k+1:]+[self.reg_n[-1]])))

        c.add(gates.X(self.reg_n[-1]))

        c.add([gates.CNOT(self.reg_n[nx-k], i) for i in [self.reg_nx[-k]]+self.reg_mx+self.reg_nx[-k+1:]])

        c.add(gates.H(self.reg_n[nx-k]))

        return c

    def extract_state(self, state):
        k = self.subspace
        m = self.m
        nx = self.nx
        state = state[::2]
        state_small = np.zeros(2**(nx+m), dtype=complex)
        xx = 0
        for i in range(2**(nx-k)):
            for x in range(2**(k+m)):
                state_small[xx+x] = state[2**(k+m+1)*i+x]
            xx += 2**(k+m)
        return state_small

    def execute(self):
        state = self.prob2state()
        c = self.qjpeg_circuit()
        #print(c.summary())
        state = c(state).state()
        state = self.extract_state(state)
        return state

    def __call__(self):
        return self.execute()
