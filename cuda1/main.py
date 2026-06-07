import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

BLOCK_SIZE = 256
SHARED_BYTES = BLOCK_SIZE * 4

module = SourceModule("""

    __global__ void MatrixVectorMul(int height, int width, float* matrix, float* vector, float* result) {
        extern __shared__ float partial[];
        int row = blockIdx.x;
        int tid = threadIdx.x;
        if (row >= height) {
            return;
        }
        float local = 0.0f;
        for (int col = tid; col < width; col += blockDim.x) {
            local += matrix[row * width + col] * vector[col];
        }
        partial[tid] = local;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            result[row] = partial[0];
        }
    }

    __global__ void GetGrade(float* clientSums, int A, int B, int length, int* creditGrade) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            float sum = clientSums[idx];
            if (sum < A) {
                creditGrade[idx] = 0;
            } else if (sum < B) {
                creditGrade[idx] = 1;
            } else {
                creditGrade[idx] = 2;
            }
        }
    }
""")

matrix_vector_mul = module.get_function("MatrixVectorMul")
get_grade = module.get_function("GetGrade")


def reference_ratings(matrix, weights, a, b):
    matrix = np.asarray(matrix, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    sums = matrix @ weights
    grades = np.where(sums < a, 0, np.where(sums < b, 1, 2)).astype(np.int32)
    return sums, grades


def compute_ratings(matrix, weights, a, b):
    matrix = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float32))
    height, width = matrix.shape
    length = height
    d_matrix = cuda.mem_alloc(matrix.nbytes)
    d_vector = cuda.mem_alloc(weights.nbytes)
    d_sums = cuda.mem_alloc(height * np.dtype(np.float32).itemsize)
    d_grades = cuda.mem_alloc(height * np.dtype(np.int32).itemsize)
    try:
        cuda.memcpy_htod(d_matrix, matrix)
        cuda.memcpy_htod(d_vector, weights)
        matrix_vector_mul(
            np.int32(height),
            np.int32(width),
            d_matrix,
            d_vector,
            d_sums,
            grid=(height, 1),
            block=(BLOCK_SIZE, 1, 1),
            shared=SHARED_BYTES,
        )
        get_grade(
            d_sums,
            np.int32(a),
            np.int32(b),
            np.int32(length),
            d_grades,
            grid=((length + BLOCK_SIZE - 1) // BLOCK_SIZE, 1),
            block=(BLOCK_SIZE, 1, 1),
        )
        client_sums = np.empty(height, dtype=np.float32)
        credit_grades = np.empty(height, dtype=np.int32)
        cuda.memcpy_dtoh(client_sums, d_sums)
        cuda.memcpy_dtoh(credit_grades, d_grades)
        return client_sums, credit_grades
    finally:
        d_matrix.free()
        d_vector.free()
        d_sums.free()
        d_grades.free()


def validate(matrix, weights, a, b, rtol=1e-4, atol=1e-3):
    gpu_sums, gpu_grades = compute_ratings(matrix, weights, a, b)
    ref_sums, ref_grades = reference_ratings(matrix, weights, a, b)
    if not np.allclose(gpu_sums, ref_sums, rtol=rtol, atol=atol):
        raise AssertionError("GPU sums differ from NumPy reference")
    if not np.array_equal(gpu_grades, ref_grades):
        raise AssertionError("GPU grades differ from NumPy reference")
    return gpu_sums, gpu_grades


def run_case(height, width, seed, a, b, label):
    rng = np.random.default_rng(seed)
    matrix = rng.integers(0, 2, size=(height, width)).astype(np.float32)
    weights = rng.integers(-100000, 100001, size=width).astype(np.float32)
    validate(matrix, weights, a, b)
    _, grades = reference_ratings(matrix, weights, a, b)
    counts = {g: int(np.sum(grades == g)) for g in (0, 1, 2)}
    print(f"{label}: OK  N={height} M={width}  grades={counts}")


if __name__ == '__main__':
    run_case(10_000, 50_000, 42, -5000, 5000, "large_M")
    run_case(10_000, 50, 7, -5000, 5000, "small_M")
    print("All checks passed")
