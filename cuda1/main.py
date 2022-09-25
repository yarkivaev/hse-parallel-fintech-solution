import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.driver import Event


module = SourceModule("""

    __global__ void MatrixVectorMul(int height, int width, float* matrix, float* vector, float* result) {
        // YOUR CODE HERE

    }

    __global__ void GetGrade(float* clientSums, int A, int B, int length, int* creditGrade) {
        
        // YOUR CODE HERE
    }
""")


if __name__ == '__main__':
    # Enter your code here
    # Create arrays, kernels and check your code
    pass
