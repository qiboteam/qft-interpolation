#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 9 2021

@author: Sergi Ramos-Calderer

"""
import numpy as np
from qibo.models import Circuit, QFT
from qibo import gates


class qft_interpolation_2d:
    """Class that uses qft interpolation to upscale an image."""

    def __init__(self, image, upscale_factor):
        """Set up the important parameters for the implementation.
        Args:
            image (np.array): original image to upscale.
            upscale_factor (int): image will be upscaled by 2**upscale_factor in both directions.

        """
        self.nx = int(np.ceil(np.log2(image.shape[0])))
        self.ny = int(np.ceil(np.log2(image.shape[1])))
        self.m = upscale_factor
        if len(image.shape)>2:
            self.layers = image.shape[2]
        else:
            self.layers = 1
        self.l = int(np.ceil(np.log2(self.layers)))
        self.img = image

        # Padding the image to be a power of 2 in order to fit in a quantum state.
        if self.l==0:
            self.image = np.pad(image, (((2**self.nx - image.shape[0])//2, (2**self.nx - image.shape[0])//2), ((
                2**self.ny - image.shape[1])//2, (2**self.ny - image.shape[1])//2)))
        else:
            self.image = np.pad(image, (((2**self.nx - image.shape[0])//2, (2**self.nx - image.shape[0])//2), ((
                2**self.ny - image.shape[1])//2, (2**self.ny - image.shape[1])//2), (0,0)))

        self.q_registers()
        self.nqubits = self.nx+self.ny+2*self.m+self.l

    def q_registers(self):
        """Auxiliary function to set up the quantum registers needed."""
        self.reg_x = [i for i in range(self.nx+self.m)]
        self.reg_y = [i+self.nx+self.m for i in range(self.ny+self.m)]
        self.reg_nx = self.reg_x[:1]+self.reg_x[self.m+1:]
        self.reg_mx = self.reg_x[1:self.m+1]
        self.reg_ny = self.reg_y[:1]+self.reg_y[self.m+1:]
        self.reg_my = self.reg_y[1:self.m+1]
        self.reg_l = [i+self.nx+self.ny+2*self.m for i in range(self.l)]
    
    def img2state_layers(self):
        """Insert the values of the multi-layer image in the relevant amplitudes of the quantum system.
        Returns:
            state_vector (np.array): amplitudes of all quantum states.

        """
        state_vector = np.zeros(2**(self.nqubits))
        
        for i in range(self.layers):
            for x in range(self.image.shape[0]):
                bin_x = format(x, f'0{self.nx}b')
                for y in range(self.image.shape[1]):
                    bin_y = format(y, f'0{self.ny}b')
                    num = 0
                    for j in range(self.nx):
                        num += int(bin_x[j])*(2**(self.nqubits-self.reg_nx[j]-1))
                    for j in range(self.ny):
                        num += int(bin_y[j])*(2**(self.nqubits-self.reg_ny[j]-1))
                    state_vector[num+i] = np.sqrt(self.image[x, y, i])
        state_vector /= np.sqrt(np.sum(state_vector**2))
        return state_vector
    
    def img2state_bw(self):
        """Insert the gray values of the image in the relevant amplitudes of the quantum system.
        Returns:
            state_vector (np.array): amplitudes of all quantum states.

        """
        state_vector = np.zeros(2**(self.nqubits))
        
        for x in range(self.image.shape[0]):
            bin_x = format(x, f'0{self.nx}b')
            for y in range(self.image.shape[1]):
                bin_y = format(y, f'0{self.ny}b')
                num = 0
                for j in range(self.nx):
                    num += int(bin_x[j])*(2**(self.nqubits-self.reg_nx[j]-1))
                for j in range(self.ny):
                    num += int(bin_y[j])*(2**(self.nqubits-self.reg_ny[j]-1))
                state_vector[num] = np.sqrt(self.image[x, y])
        state_vector /= np.sqrt(np.sum(state_vector**2))
        return state_vector

    def cnot_layer(self, nqubits):
        """Changes the sign of all qubits controlled on 0 by the first one.
        Args:
            nqubits (int): size of the qubit register.

        Returns:
            c (qibo.models.Circuit): circuit with CNOT gates controlled by the first qubit.

        """
        c = Circuit(nqubits)
        for i in range(1, nqubits):
            c.add(gates.CNOT(0, i))
        return c

    def qft_int_circuit(self):
        """Quantum circuit that applies interpolation in Fourier space to resample an image.
        Returns:
            c (qibo.models.Circuit): list of gates for the qft interpolation in two dimensions.

        """
        c = Circuit(self.nqubits)

        # 2D quantum Fourier Transform
        c.add(QFT(self.nx, with_swaps=True).on_qubits(*(self.reg_nx)))
        c.add(QFT(self.ny, with_swaps=True).on_qubits(*(self.reg_ny)))

        # Invert the added qubits if most significant qubit is 0.
        c.add(self.cnot_layer(self.m+1).on_qubits(*
              (self.reg_nx[:1]+self.reg_mx)))
        c.add(self.cnot_layer(self.m+1).on_qubits(*
              (self.reg_ny[:1]+self.reg_my)))

        # Undo the QFT to return to image space
        c.add(QFT(self.ny+self.m, with_swaps=True).invert().on_qubits(*(self.reg_y)))
        c.add(QFT(self.nx+self.m, with_swaps=True).invert().on_qubits(*(self.reg_x)))

        return c

    def execute(self):
        """Run the quantum simulation starting from the quantum state that encodes the image to get the upscaled picture.
        Returns:
            self.upscaled_img (np.array): Upscaled image by a factor of 2**upscale_factor.

        """

        # Create the quantum circuit and quantum state.
        c = self.qft_int_circuit()
        if self.l == 0:
            state = self.img2state_bw()
        else:
            state = self.img2state_layers()
        img = c(state).state()

        # Reshape the state vector into a 2D image.
        if self.l == 0:
            img = img.reshape(2**(self.nx+self.m), 2**(self.ny+self.m))
        else:
            img = img.reshape(2**(self.nx+self.m), 2**(self.ny+self.m), 2**self.l)

        img = (np.abs((img))**2)
        
        if self.l == 0:
            mean_img = np.sum(self.img)/self.img.size

            img = img*mean_img*img.size/np.sum(img)
            img = img.astype(int)
        else:
            for i in range(self.img.shape[2]):
                mean_img = np.sum(self.img[:,:,i])/self.img[:,:,i].size
                img[:,:,i] = img[:,:,i]*mean_img*img[:,:,i].size/np.sum(img[:,:,i])
            img = img.astype(int)
        
        # Return only the image undoing the initial padding.
        if self.l == 0:
            self.upscaled_img = img[(2**(self.nx+self.m)-self.img.shape[0]*2**self.m)//2:
                                    (2**(self.nx+self.m) +
                                     self.img.shape[0]*2**self.m)//2,
                                    (2**(self.ny+self.m)-self.img.shape[1]*2**self.m)//2:
                                    (2**(self.ny+self.m)+self.img.shape[1]*2**self.m)//2]
        else:
            self.upscaled_img = img[(2**(self.nx+self.m)-self.img.shape[0]*2**self.m)//2:
                                    (2**(self.nx+self.m) +
                                     self.img.shape[0]*2**self.m)//2,
                                    (2**(self.ny+self.m)-self.img.shape[1]*2**self.m)//2:
                                    (2**(self.ny+self.m)+self.img.shape[1]*2**self.m)//2, :self.img.shape[2]]

        return np.minimum(self.upscaled_img, 255)

    def __call__(self):
        """Equivalent to `qft_interpolation.qft_interpolation_2d.execute`."""
        return self.execute()


class qft_interpolation_1d:
    """Class that uses qft interpolation to upscale a probability distribution."""

    def __init__(self, probabilities, upscale_factor=None, unary=False):
        """Set up the important parameters for the implementation.
        Args:
            probabilities (np.array): original probability distribution to upscale.
            upscale_factor (int): probability will be upscaled by 2**upscale_factor.
            unary (Bool): use unary uploading.

        """
        self.unary = unary
        self.n = int(np.ceil(np.log2(probabilities.size)))
        if unary:
            assert np.log2(probabilities.size).is_integer()
            self.m = (probabilities.size-self.n)
            self.params = self.rw_parameters(probabilities.size, probabilities)
            self.q_registers_un()
        else:
            self.m = upscale_factor
            self.prob = probabilities
            self.q_registers()
            
        self.nqubits = self.n+self.m

    def q_registers(self):
        """Auxiliary function to set up the quantum registers needed."""
        self.reg = [i for i in range(self.n+self.m)]
        self.reg_n = self.reg[:1]+self.reg[self.m+1:]
        self.reg_m = self.reg[1:self.m+1]
        
    def prepare_reg_un(self, reg_bin, reg_extra):
        r = []
        c = 0
        n = len(reg_bin)
        for i in reversed(range(n)):
            r.append(reg_bin[n-1-i])
            for _ in range(0,(2**i)-1):
                r.append(reg_extra[c])
                c += 1
        r.append(reg_extra[c])
        return r
    
    def q_registers_un(self):
        self.reg = [i for i in range(self.n+self.m)]
        self.reg_n = self.reg[:1]+self.reg[self.m+1:]
        self.reg_m = self.reg[1:self.m+1]
        self.reg_un = self.prepare_reg_un(self.reg_n, self.reg_m)

    def prob2state(self):
        """Insert the values of the probability distribution in the relevant amplitudes of the quantum system.
        Returns:
            state_vector (np.array): amplitudes of all quantum states.

        """
        state_vector = np.concatenate((np.sqrt(self.prob[:self.prob.size//2]),
                                       np.zeros((2**self.nqubits-2**self.n)//2),
                                       np.sqrt(self.prob[self.prob.size//2:]),
                                       np.zeros((2**self.nqubits-2**self.n)//2)))
        state_vector /= np.sqrt(np.sum(state_vector**2))
        return state_vector

    def cnot_layer(self, nqubits):
        """Changes the sign of all qubits controlled on 0 by the first one.
        Args:
            nqubits (int): size of the qubit register.

        Returns:
            c (qibo.models.Circuit): circuit with CNOT gates controlled by the first qubit.

        """
        c = Circuit(nqubits)
        for i in range(1, nqubits):
            c.add(gates.CNOT(0, i))
        return c
    
    def rw_parameters(self, qubits, pdf):
        """Parameters that encode a target probability distribution into the unary basis
        Args:
            qubits (int): number of qubits used for the unary basis.
            pdf (list): known probability distribution function that wants to be reproduced.

        Returns:
            paramters (list): values to be introduces into the fSim gates for amplitude distribution.
        """
        if qubits%2==0:
            mid = qubits // 2
        else:
            mid = (qubits-1)//2 #Important to keep track of the centre
        last = 1
        parameters = []
        for i in range(mid-1):
            angle = 2 * np.arctan(np.sqrt(pdf[i]/(pdf[i+1] * last)))
            parameters.append(angle)
            last = (np.cos(angle/2))**2 #The last solution is needed to solve the next one
        angle = 2 * np.arcsin(np.sqrt(pdf[mid-1]/last))
        parameters.append(angle)
        last = (np.cos(angle/2))**2
        for i in range(mid, qubits-1):
            angle = 2 * np.arccos(np.sqrt(pdf[i]/last))
            parameters.append(angle)
            last *= (np.sin(angle/2))**2
        return parameters
    
    def un2bin(self, nqubits):
        """Circuit that transforms the unary representation into the binary one.
        Args:
            nqubits (int): number of qubits of the unary representation.

        Returns:
            c (qibo.models.Circuit): quantum circuit with the gates needed to perform the transformation.
        """
        n = int(np.log2(nqubits))
        c = Circuit(nqubits)
        q = 0
        for m in (range(n)):
            qq = 2**(n-m-1)
            c.add(gates.CNOT(q, q+qq))
            for i in range(1, qq):
                c.add(gates.CNOT(q+i, q))
            for i in range(1, qq):
                c.add(gates.SWAP(q+i, q+i+qq).controlled_by(q))
            q += qq
        c.add(gates.X(nqubits-1))
        return c
    
    def rw_circuit(self, qubits, parameters):
        """Circuit that implements the amplitude distributor part of the option pricing algorithm.
        Args:
            qubits (int): number of qubits used for the unary basis.
            paramters (list): values to be introduces into the fSim gates for amplitude distribution.

        Returns:
            c (qibo.models.Circuit) : circuit with the gates needed for the amplitude distributor circuit
        """
        c = Circuit(qubits)
        if qubits%2==0:
            mid1 = int(qubits/2)
            mid0 = int(mid1-1)
            c.add(gates.X(mid1))
            c.add(gates.GeneralizedfSim(mid1, mid0, gates.RY(0, -2*parameters[mid0]/2).matrix, 0))
            for i in range(mid0):
                c.add(gates.GeneralizedfSim(mid0-i, mid0-i-1, gates.RY(0, -2*parameters[mid0-i-1]/2).matrix, 0))
                c.add(gates.GeneralizedfSim(mid1+i, mid1+i+1, gates.RY(0, -2*parameters[mid1+i]/2).matrix, 0))
        else:
            mid = int((qubits-1)/2)
            c.add(gates.X(mid))
            for i in range(mid):
                c.add(gates.GeneralizedfSim(mid-i, mid-i-1, gates.RY(0, -2*parameters[mid-i-1]/2).matrix, 0))
                c.add(gates.GeneralizedfSim(mid+i, mid+i+1, gates.RY(0, -2*parameters[mid+i]/2).matrix, 0))
        return c

    def qft_int_circuit(self):
        """Quantum circuit that applies interpolation in Fourier space to upscale an image.
        Returns:
            c (qibo.models.Circuit): list of gates for the qft interpolation in two dimensions.

        """
        c = Circuit(self.nqubits)

        # 1D quantum Fourier Transform
        c.add(QFT(self.n, with_swaps=True).on_qubits(*(self.reg_n)))

        # Invert the added qubits if most significant qubit is 0.
        c.add(self.cnot_layer(self.m+1).on_qubits(*
              (self.reg_n[:1]+self.reg_m)))

        # Undo the QFT to return to image space
        c.add(QFT(self.nqubits, with_swaps=True).invert().on_qubits(*(self.reg)))

        return c

    def execute(self):
        """Run the quantum simulation starting from the quantum state that encodes the image to get the upscaled picture.
        Returns:
            self.upscaled_img (np.array): Upscaled image by a factor of 2**upscale_factor.

        """

        # Create the quantum circuit and quantum state.
        c = Circuit(self.nqubits)
        if self.unary:
            c.add(self.rw_circuit(self.nqubits, self.params).on_qubits(*reversed(self.reg_un)))
            c.add(self.un2bin(self.nqubits).on_qubits(*self.reg_un))
            c += self.qft_int_circuit()
            prb = c().state()
        else:
            c += self.qft_int_circuit()
            state = self.prob2state()
            prb = c(state).state()

        self.upscaled_prb = (np.abs((prb))**2)

        return self.upscaled_prb

    def __call__(self):
        """Equivalent to `qft_interpolation.qft_interpolation_2d.execute`."""
        return self.execute()