# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
import numpy
import math

class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        new_matrix = numpy.zeros((15,15),dtype=numpy.complex_)

        for v in range(15):
            for u in range(15):
                for y in range(15):
                    for x in range(15):
                        #n = matrix[y][x]*math.exp((-1j*2*math.pi*(u*x+v*y))/15)
                        n = matrix[y][x] * ((math.cos(2*math.pi*(v*y+u*x)/15))-(1j*math.sin(2*math.pi*(v*y+u*x)/15)))
                        new_matrix[u][v] = n
        
        return new_matrix

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""

        new_matrix = numpy.zeros((15,15),dtype=numpy.complex_)

        for v in range(15):
            for u in range(15):
                for y in range(15):
                    for x in range(15):
                        #n = matrix[y][x]*math.exp((-1j*2*math.pi*(u*x+v*y))/15)
                        n = matrix[y][x] * ((math.cos(2*math.pi*(v*y+u*x)/15))+(1j*math.sin(2*math.pi*(v*y+u*x)/15)))
                        new_matrix[u][v] = n
        
        return new_matrix

    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""

        new_matrix = numpy.zeros((15,15))

        for v in range(15):
            for u in range(15):
                for y in range(15):
                    for x in range(15):
                        #n = matrix[y][x]*math.exp((-1j*2*math.pi*(u*x+v*y))/15)
                        n = matrix[y][x] * ((math.cos(2*math.pi*(v*y+u*x)/15)))
                        new_matrix[u][v] = n
        
        return new_matrix

    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""

        new_matrix = numpy.absolute(self.forward_transform(matrix))

        return matrix