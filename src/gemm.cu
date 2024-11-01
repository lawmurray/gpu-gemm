/*
 * Copyright 2024 Lawrence Murray.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

/**
 * Matrix tile in global memory.
 * 
 * @tparam R Number of rows.
 * @tparam C Number of columns.
 * @tparam L Stride between columns.
 */
template<int R, int C, int L = R>
requires (R%4 == 0 && L >= R)
union global_tile {
  /**
   * Constructor.
   */
  __device__ global_tile(float* x) : x(x) {
    //
  }

  /**
   * Constructor.
   */
  template<int R1, int C1, int L1>
  __device__ global_tile(const global_tile<R1,C1,L1>& o, const int i,
      const int j) :
      x(&o.x[i + j*L1]) {
    //
  }

  float* __restrict__ x;
  float4* __restrict__ x4;
};

/**
 * Matrix tile in shared memory.
 * 
 * @tparam R Number of rows.
 * @tparam C Number of columns.
 * @tparam L Stride between columns.
 */
template<int R, int C, int L = R>
requires (R%4 == 0 && L >= R)
union shared_tile {
  /**
   * Copy into this tile from global memory using 32-bit loads.
   * 
   * @tparam T Number of threads in the group sharing the copy.
   * 
   * @param o Global memory tile.
   * @param i0 Row offset in @p o.
   * @param j0 Column offset in @p o.
   * @param t_id Id of this thread within the group sharing the copy.
   */
  template<int T, int R1, int C1, int L1>
  requires (T%R == 0)
  __device__ void copy(const global_tile<R1,C1,L1>& o, const int i0,
      const int j0, const int t_id) {
    int dst0 = __cvta_generic_to_shared(x);
    for (int s = 0; s < R*C/T; ++s) {
      int i = t_id%R;
      int j = t_id/R + s*(T/R);
      int dst = dst0 + (i + j*L)*sizeof(float);
      const float* src = &o.x[i0 + i + (j0 + j)*L1];
      asm("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"(dst), "l"(src),
          "n"(sizeof(float)));
    }
  }

  /**
   * Copy into this tile from global memory using 128-bit loads.
   * 
   * @tparam T Number of threads in the group sharing the copy.
   * 
   * @param o Global memory tile.
   * @param i0 Row offset in @p o.
   * @param j0 Column offset in @p o.
   * @param t_id Id of this thread within the group sharing the copy.
   */
  template<int T, int R1, int C1, int L1>
  requires (R%4 == 0 && L%4 == 0 && L1%4 == 0) && (T%(R/4) == 0)
  __device__ void copy4(const global_tile<R1,C1,L1>& o, const int i0,
      const int j0, const int t_id) {
    int dst0 = __cvta_generic_to_shared(x4);
    for (int s = 0; s < R*C/4/T; ++s) {
      int i = t_id%(R/4);
      int j = t_id/(R/4) + s*(T/(R/4));
      int dst = dst0 + (i + j*(L/4))*sizeof(float4);
      const float4* src = &o.x4[i0 + i + (j0 + j)*(L1/4)];
      asm("cp.async.cg.shared.global [%0], [%1], %2;" :: "r"(dst),
          "l"(src), "n"(sizeof(float4)));
    }
  }

  /**
   * Copy into this tile from global memory using 32-bit loads, with
   * transpose.
   * 
   * @tparam T Number of threads participating in the copy.
   * 
   * @param o Global memory tile.
   * @param i0 Row offset in @p o.
   * @param j0 Column offset in @p o.
   * @param t_id Thread id within the group.
   */
  template<int T, int R1, int C1, int L1>
  requires (T%C == 0)
  __device__ void copy_transpose(const global_tile<R1,C1,L1>& o, const int i0,
      const int j0, const int t_id) {
    int dst0 = __cvta_generic_to_shared(x);
    for (int s = 0; s < C*R/T; ++s) {
      int i = t_id%C;
      int j = t_id/C + s*(T/C);
      int dst = dst0 + (j + i*L)*sizeof(float);
      const float* src = &o.x[i0 + i + (j0 + j)*L1];
      asm("cp.async.ca.shared.global.L2::256B [%0], [%1], %2;" :: "r"(dst),
          "l"(src), "n"(sizeof(float)));
    }
  }

  float x[R*C];
  float4 x4[R*C/4];
};

/**
 * Vector tile in registers.
 * 
 * @tparam N Size.
 * @tparam S Stride when loading from or storing to memory.
 */
template<int N, int S = 1>
union register_vector {
  /**
   * Load from a shared memory tile.
   */
  template<int R1, int C1, int L1>
  __device__ void load(const shared_tile<R1,C1,L1>& o, const int i0,
      const int j0) {
    for (int i = 0; i < N; ++i) {
      x[i] = o.x[i0 + j0*L1 + i*S];
    }
  }

  /**
   * Load from a shared memory tile.
   */
  template<int R1, int C1, int L1>
  requires (N%4 == 0 && L1%4 == 0)
  __device__ void load4(const shared_tile<R1,C1,L1>& o, const int i0,
      const int j0) {
    for (int i = 0; i < N/4*S; i += S) {
      x4[i] = o.x4[i0 + j0*(L1/4) + i];
    }
  }

  float x[N];
  float4 x4[N/4];
};

/**
 * Matrix tile in registers.
 * 
 * @tparam R Number of rows.
 * @tparam C Number of columns.
 * @tparam RS Row stride when loading from or storing to memory.
 * @tparam CS Column stride when loading from or storing to memory.
 */
template<int R, int C, int RS = 1, int CS = 1>
union register_tile {
  /**
   * Store to a global memory tile.
   */
  template<int R1, int C1, int L1>
  __device__ void store(global_tile<R1,C1,L1>& o, const int i0,
      const int j0) {
    for (int j = 0; j < C; ++j) {
      for (int i = 0; i < R; ++i) {
        o.x[i0 + j0*L1 + i*RS + j*(CS*L1)] = x[i + j*R];
      }
    }
  }

  /**
   * Store to a global memory tile.
   */
  template<int R1, int C1, int L1>
  requires (R%4 == 0 && L1%4 == 0)
  __device__ void store4(global_tile<R1,C1,L1>& o, const int i0,
      const int j0) {
    for (int j = 0; j < C/4; ++j) {
      for (int i = 0; i < R/4; ++i) {
        /* when storing, write through so as not to evict useful data from
         * inputs from the L2 cache */
        for (int b = 0; b < 4; ++b) {
          __stwt(&o.x4[i0 + j0*L1 + i*RS + j*(CS*L1) + b*(L1/4)], x4[i + j*R + b*(R/4)]);
        }
      }
    }
  }

  /**
   * Outer product of two vectors and add.
   * 
   * @param a First argument.
   * @param b Second argument.
   * 
   * Computes $ab^\top$ and adds to this tile.
   */
  template<int S1, int S2>
  __device__ void add_outer(const register_vector<R,S1>& a,
      const register_vector<C,S2>& b) {
    for (int i = 0; i < R; ++i) {
      for (int j = 0; j < C; ++j) {
        x[i + j*R] += a.x[i]*b.x[j];
      }
    }
  }

  float x[R*C]{0};
  float4 x4[R*C/4];
};

/**
 * Two-dimensional point.
 */
struct point2 { int i, j; };

/**
 * Coordinates for a given serial index along a two-dimensional Hilbert curve.
 * 
 * @tparam M Number of rows.
 * @tparam N Number of columns.
 * 
 * @param s Serial index.
 * 
 * @return Coordinates.
 */
template<int M, int N>
requires (M == N || M == N/2)
__device__ point2 hilbert2(const int s) {
  int i = 0, j = 0;
  int t = s;
  for (int k = 1; k < max(M, N); k *= 2) {
    int bi = 1 & (t/2);  // local gray code, u shape top left to bottom left
    int bj = 1 & (t ^ bi);
    if (bj == 0) {
      if (bi == 1) {
        i = k - 1 - i;  // flip up-down
        j = k - 1 - j;  // flip left-right
      }
      int tmp = i;  // transpose
      i = j;
      j = tmp;
    }
    i += k*bi;
    j += k*bj;
    t /= 4;
  }
  return {i, j};
}

/**
 * Difference between two matrices.
 * 
 * @tparam M Number of rows.
 * @tparam N Number of columns.
 * 
 * @param C Matrix.
 * @param D Matrix.
 * 
 * @return Maximum absolute element-wise difference.
 */
template<int M, int N>
float diff(const float* C, const float* D) {
  float mx = 0.0;
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      mx = std::max(mx, std::abs(C[i + j*M] - D[i + j*M]));
    }
  }
  return mx;
}

/**
 * Matrix-matrix multiplication kernel.
 * 
 * @tparam M Number of rows of $A$ and $C$.
 * @tparam N Number of columns of $B$ and $C$.
 * @tparam K Number of columns of $A$ and rows of $B$.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * @param C Matrix $C$.
 * 
 * Computes $C = AB$.
 */
template<int M, int N, int K>
__global__ void gemm_kernel(float* __restrict__ A, float* __restrict__ B,
    float* __restrict__ C) {
  /* config */
  constexpr int nthreads = 256;      // number of threads per block
  constexpr int wsize = 32;          // number of threads per warp
  constexpr int nstages = 4;  // number of asynchronous pipeline stages

  /* level 0 tile size (original matrices in global memory) */
  constexpr int M0 = M;
  constexpr int N0 = N;
  constexpr int K0 = K;

  /* level 1 tile size (global memory, thread block level) */
  constexpr int M1 = 256;
  constexpr int N1 = 128;
  constexpr int K1 = K0;  // must be K0

  /* level 2 tile size (shared memory, thread block level) */
  constexpr int M2 = 256;
  constexpr int N2 = 128;
  constexpr int K2 = 8;

  /* level 3 tile size (shared memory, warp level) */
  constexpr int M3 = 64;
  constexpr int N3 = 64;
  constexpr int K3 = K2;  // must be K2

  /* level 3 warp grid size */
  constexpr int M3_warps = M2/M3;
  constexpr int N3_warps = N2/N3;

  /* level 4 tile size (registers, thread level) */
  constexpr int M4 = 8;
  constexpr int N4 = 16;
  constexpr int K4 = 1;

  /* level 4 thread grid size */
  constexpr int M4_threads = 8;
  constexpr int N4_threads = 4;

  /* thread and warp ids */
  const int b_id = blockIdx.x;       // id of this block within the grid
  const int w_id = threadIdx.x/32;   // id of this warp within its block
  const int t_id = threadIdx.x%32;   // id of this thread within its warp
  const int row_id = w_id%M3_warps;  // id of row handled warp at level 3
  const int col_id = w_id/M3_warps;  // id of column handled warp at level 3

  /* barrier ids associated with the row and column handled by the warp */
  const int row_barrier = row_id;
  const int col_barrier = M3_warps + col_id;

  /* level 0 tiles (original matrices) */
  global_tile<M0,K0> A0(A);
  global_tile<K0,N0> B0(B);
  global_tile<M0,N0> C0(C);

  /* level 1 tiles */
  auto [b_i, b_j] = hilbert2<M0/M1,N0/N1>(b_id);
  global_tile<M1,K1,M0> A1(A0, b_i*M1, 0);
  global_tile<K1,N1,K0> B1(B0, 0, b_j*N1);

  /* level 3 buffers (B is transposed) */
  __shared__ shared_tile<M3,K3> A3[M3_warps][nstages];
  __shared__ shared_tile<N3,K3> BT3[N3_warps][nstages];

  /* start pipeline */
  const int r_id = t_id + row_id*wsize;
  const int c_id = t_id + col_id*wsize;
  for (int stage = 0; stage < nstages - 1; ++stage) {
    BT3[col_id][stage].copy_transpose<nthreads/N3_warps>(B1, stage*K2, col_id*N3, r_id);
    A3[row_id][stage].copy4<nthreads/M3_warps>(A1, row_id*(M3/4), stage*K2, c_id);
    asm("cp.async.commit_group;");
  }

  /* level 4 tiles */
  register_vector<M4,M4_threads> a4;
  register_vector<N4,N4_threads> b4;
  register_tile<M4,N4,M4_threads,N4_threads> C4;

  /* level 4 offsets to first elements */
  const int i4 = t_id%M4_threads;
  const int j4 = t_id/M4_threads;

  /* multiply */
  for (int k2 = 0; k2 < K1/K2; ++k2) {
    asm("cp.async.wait_group %0;" :: "n"(nstages - 2));
    asm("barrier.sync.aligned %0, %1;" :: "r"(col_barrier), "n"(nthreads/N3_warps));
    asm("barrier.sync.aligned %0, %1;" :: "r"(row_barrier), "n"(nthreads/M3_warps));

    int stage = k2%nstages;
    for (int k4 = 0; k4 < K3/K4; ++k4) {
      a4.load4(A3[row_id][stage], i4, k4*K4);
      b4.load4(BT3[col_id][stage], j4, k4*K4);
      C4.add_outer(a4, b4);
    }

    int next_stage = (k2 + (nstages - 1))%nstages;
    int next_k = (k2 + (nstages - 1))%(K1/K2);
    BT3[col_id][next_stage].copy_transpose<nthreads/N3_warps>(B1, next_k*K2, col_id*N3, r_id);
    A3[row_id][next_stage].copy4<nthreads/M3_warps>(A1, row_id*(M3/4), next_k*K2, c_id);
    asm("cp.async.commit_group;");
  }

  /* write final result */
  C4.store4(C0, b_i*(M1/4) + row_id*(M3/4) + i4, b_j*(N1/4) + col_id*(N3/4) + j4);
}

/**
 * Matrix-matrix multiplication.
 * 
 * @tparam M Number of rows of $A$ and $C$.
 * @tparam N Number of columns of $B$ and $C$.
 * @tparam K Number of columns of $A$ and rows of $B$.
 * 
 * @param A Matrix $A$.
 * @param B Matrix $B$.
 * @param C Matrix $C$.
 * 
 * Computes $C = AB$.
 */
template<int M, int N, int K>
void gemm(float* __restrict__ A, float* __restrict__ B,
    float* __restrict__ C) {
  dim3 block(256);
  dim3 grid((M/256)*(N/128));
  gemm_kernel<M,N,K><<<grid,block>>>(A, B, C);
}

/**
 * Cache flush kernel.
 * 
 * @tparam F Flush size.
 * 
 * @param f Vector.
 */
template<int F>
__global__ void flush_kernel(float* f) {
  f[threadIdx.x + blockIdx.x*blockDim.x] += 1.0f;
}

/**
 * Cache flush.
 * 
 * @tparam F Flush size.
 * 
 * @param f Vector.
 */
template<int F>
void flush(float* f) {
  dim3 block(256);
  dim3 grid(F/256);
  flush_kernel<F><<<grid,block>>>(f);
}

int main(int argc, char** argv) {
  /* initialize cublas */
  cublasHandle_t handle;
  cublasCreate(&handle);

  /* initialize curand */
  constexpr int seed = 1;
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);

  auto run_test = [&]<int M, int N, int K, int ntrials, int nwarmup>() {
    /* number of floating point operations for TFLOPS numbers; each output
     * has M*N elements, each computed from K multiplications and K - 1
     * additions */
    constexpr long flop = M*N*(2l*K - 1l);

    /* cache flush size */
    constexpr int F = 32*1024*1024;

    /* initialize matrices; the output matrices, C and D, are allocated with
     * managed memory to support problem sizes somewhat beyond the available
     * device memory, while ensuring that the input matrices, A and B, are
     * always on device, which is more important for performance */
    float *A, *B, *C, *D, *f;
    cudaMalloc((void**)&A, M*K*sizeof(float));
    cudaMalloc((void**)&B, K*N*sizeof(float));
    cudaMallocManaged((void**)&C, M*N*sizeof(float));
    cudaMallocManaged((void**)&D, M*N*sizeof(float));
    cudaMalloc((void**)&f, F*sizeof(float));
    curandGenerateUniform(gen, A, M*K);
    curandGenerateUniform(gen, B, K*N);
    curandGenerateUniform(gen, f, F);

    /* initialize events */
    cudaEvent_t start1[ntrials], stop1[ntrials];
    cudaEvent_t start2[ntrials], stop2[ntrials];
    for (int trial = 0; trial < ntrials; ++trial) {
      cudaEventCreate(&start1[trial]);
      cudaEventCreate(&stop1[trial]);
      cudaEventCreate(&start2[trial]);
      cudaEventCreate(&stop2[trial]);
    }

    /* initialize scalars */
    float scalar0 = 0.0f, scalar1 = 1.0f;

    /* warm up */
    for (int trial = 0; trial < nwarmup; ++trial) {
      cudaMemPrefetchAsync(C, M*N*sizeof(float), 0);
      curandGenerateUniform(gen, C, M*N);  // clear output
      flush<F>(f);  // flush L2 cache
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &scalar1, A, M,
          B, K, &scalar0, C, M);

      cudaMemPrefetchAsync(D, M*N*sizeof(float), 0);
      curandGenerateUniform(gen, D, M*N);  // clear output
      flush<F>(f);  // flush L2 cache
      gemm<M,N,K>(A, B, D);
    }

    /* benchmark */
    for (int trial = 0; trial < ntrials; ++trial) {
      cudaMemPrefetchAsync(C, M*N*sizeof(float), 0);
      curandGenerateUniform(gen, C, M*N);  // clear output
      flush<F>(f);  // flush L2 cache
      cudaEventRecord(start1[trial]);
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &scalar1, A, M,
          B, K, &scalar0, C, M);
      cudaEventRecord(stop1[trial]);

      cudaMemPrefetchAsync(D, M*N*sizeof(float), 0);
      curandGenerateUniform(gen, D, M*N);  // clear output
      flush<F>(f);  // flush L2 cache
      cudaEventRecord(start2[trial]);
      gemm<M,N,K>(A, B, D);
      cudaEventRecord(stop2[trial]);
    }

    /* results */
    float ms, ms1 = 0.0f, ms2 = 0.0f;
    for (int trial = 0; trial < ntrials; ++trial) {
      cudaEventSynchronize(stop1[trial]);
      cudaEventElapsedTime(&ms, start1[trial], stop1[trial]);
      ms1 += ms;

      cudaEventSynchronize(stop2[trial]);
      cudaEventElapsedTime(&ms, start2[trial], stop2[trial]);
      ms2 += ms;
    }
    ms1 /= ntrials;
    ms2 /= ntrials;
    float tflops1 = flop/ms1/1.0e9f;
    float tflops2 = flop/ms2/1.0e9f;
    float error = diff<M,M>(C, D);    

    /* report results */
    std::printf("| %6d | %6d | %6d | %11.3f | %11.3f | %15.3f | %15.3f | %6d | %9.3f |\n",
        M, N, K, ms1, ms2, tflops1, tflops2, ntrials, error);

    /* destroy events */
    for (int trial = 0; trial < ntrials; ++trial) {
      cudaEventDestroy(start1[trial]);
      cudaEventDestroy(stop1[trial]);
      cudaEventDestroy(start2[trial]);
      cudaEventDestroy(stop2[trial]);
    }

    /* free memory */
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);
  };

  /* table header */
  std::printf("|      M |      N |      K | cublas (ms) | custom (ms) | cublas (tflops) | custom (tflops) | trials |       err |\n");
  std::printf("| -----: | -----: | -----: | ----------: | ----------: | --------------: | --------------: | -----: | :-------: |\n");

  /* run tests and report */
  run_test.template operator()<2048,2048,2048,10000,100>();
  run_test.template operator()<4096,4096,4096,10000,100>();
  run_test.template operator()<8192,8192,8192,1000,100>();
  run_test.template operator()<16384,16384,16384,1000,100>();
  run_test.template operator()<32768,32768,32768,10,1>();

  return 0;
}
