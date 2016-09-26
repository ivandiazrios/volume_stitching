#ifdef USE_CUDA

#ifndef REGISTRATION_CUH
#define REGISTRATION_CUH

#include "vec4.hpp"
#include "mat4x4.hpp"
#include <limits>

#define MAX_THREADS 1024

extern cudaTextureObject_t tex;

unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

template <typename T, unsigned int block_size>
__global__ void getdiff(cudaTextureObject_t tex, float *device_sum_diff, float *device_sum_count, T *data1, int depth, int height, int width, int n_max, mat4x4<float> m) {
    
    extern __shared__ float s_data[];
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    
    unsigned int tid = threadIdx.z + blockDim.z * (threadIdx.y + blockDim.y * threadIdx.x) ;
    unsigned int block_id = blockIdx.z + gridDim.z * (blockIdx.y + gridDim.y * blockIdx.x);
    unsigned int threads_per_block = blockDim.x * blockDim.y * blockDim.z; 

    unsigned int index = z + width * (y + height * x);
    x = x ^ z; z = z ^ x; x = x ^ z; //swap x and z due to row major order of data
    
    float arg1;

	if (z < depth && y < height && x < width && (arg1=data1[index])) {
		
		vec4<float> v = m * vec4<float>(x,y,z,1.0f);
		float arg2 = (tex3D<float>(tex, v.a1+0.5f, v.a2+0.5f, v.a3+0.5f) * n_max); 
		
		s_data[tid] = arg2 ? (arg1 - arg2) * (arg1 - arg2) : 0.0f;
		s_data[tid + threads_per_block] = arg2 ? 1.0f : 0.0f;

	} else {

		s_data[tid] = 0;
		s_data[tid + threads_per_block] = 0;

	}
    
    __syncthreads();
    
    for (unsigned int s=block_size/2; s>0; s>>=1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
			s_data[tid + threads_per_block] += s_data[tid + threads_per_block + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        device_sum_diff[block_id] = s_data[0];
		device_sum_count[block_id] = s_data[threads_per_block];
    }
}

template <unsigned int block_size>
__global__ void getsum(unsigned int n, float *device_sum) {
    
    volatile extern __shared__ float shared_data[];
    
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(block_size*2) + tid;
    
    float sum = (i < n) ? device_sum[i] : 0;
    
    if (i + blockDim.x < n)
        sum += device_sum[i+blockDim.x];
    
    shared_data[tid] = sum;
    
    __syncthreads();
    
    if(block_size >=1024) { if(tid < 512) { shared_data[tid] = sum = sum + shared_data[tid + 512]; } __syncthreads(); }
    if(block_size >= 512) { if(tid < 256) { shared_data[tid] = sum = sum + shared_data[tid + 256]; } __syncthreads(); }
    if(block_size >= 256) { if(tid < 128) { shared_data[tid] = sum = sum + shared_data[tid + 128]; } __syncthreads(); }
    if(block_size >= 128) { if(tid <  64) { shared_data[tid] = sum = sum + shared_data[tid +  64]; } __syncthreads(); }
    
    if (tid < 32) {
        if (block_size >= 64) shared_data[tid] = sum = sum + shared_data[tid + 32];
        if (block_size >= 32) shared_data[tid] = sum = sum + shared_data[tid + 16];
        if (block_size >= 16) shared_data[tid] = sum = sum + shared_data[tid +  8];
        if (block_size >=  8) shared_data[tid] = sum = sum + shared_data[tid +  4];
        if (block_size >=  4) shared_data[tid] = sum = sum + shared_data[tid +  2];
        if (block_size >=  2) shared_data[tid] = sum = sum + shared_data[tid +  1];
    }
    
    if (tid == 0) {
        device_sum[blockIdx.x] = shared_data[0];
    }
}

void get_num_blocks_threads(unsigned int n, unsigned int &num_blocks, unsigned int &num_threads) {
    num_threads = (n < MAX_THREADS*2) ? nextPow2((n + 1)/ 2) : MAX_THREADS;
    num_blocks  = (n + (num_threads * 2 - 1)) / (num_threads * 2);
}

float reduce_sum(unsigned int n, float *device_sum) {

    unsigned int num_blocks, num_threads;
    
    get_num_blocks_threads(n, num_blocks, num_threads);
	
	while (n > 1000) {
        
        unsigned int shared_mem_size = (num_threads <= 32) ?
        2 * num_threads * sizeof(float) :
        num_threads * sizeof(float);
        
        switch (num_threads) {
            case 1024:
                getsum<1024> <<<num_blocks, num_threads, shared_mem_size>>>(n, device_sum);
                break;
            case 512:
                getsum<512> <<<num_blocks, num_threads, shared_mem_size>>>(n, device_sum);
                break;
            case 256:
                getsum<256> <<<num_blocks, num_threads, shared_mem_size>>>(n, device_sum);
                break;
            case 128:
                getsum<128> <<<num_blocks, num_threads, shared_mem_size>>>(n, device_sum);
                break;
            case 64:
                getsum<64> <<<num_blocks, num_threads, shared_mem_size>>>(n, device_sum);
                break;
            case 32:
                getsum<32> <<<num_blocks, num_threads, shared_mem_size>>>(n, device_sum);
                break;
            case 16:
                getsum<16> <<<num_blocks, num_threads, shared_mem_size>>>(n, device_sum);
                break;
            case 8:
                getsum<8> <<<num_blocks, num_threads, shared_mem_size>>>(n, device_sum);
                break;
            case 4:
                getsum<4> <<<num_blocks, num_threads, shared_mem_size>>>(n, device_sum);
                break;
            case 2:
                getsum<2> <<<num_blocks, num_threads, shared_mem_size>>>(n, device_sum);
                break;
            case 1:
                getsum<1> <<<num_blocks, num_threads, shared_mem_size>>>(n, device_sum);
                break;
        }
        
        // Calculate blocks, threads per block and number of elements left to sum
        get_num_blocks_threads(n, num_blocks, num_threads);
        n = (n + (num_threads*2-1)) / (num_threads*2);
    }
    
    float *host_sum = new float[n];
    float sum = 0;
    
    cudaMemcpy(host_sum, device_sum, n*sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; i++) {
        sum += host_sum[i];	
    }

    delete [] host_sum;
   
    return sum;
    
}

template <typename T>
float red(mat4x4<float> start_mat, unsigned int depth, unsigned int height, unsigned int width, T *d_x, float *device_sum_diff, float *device_sum_count) {
    
    // x,y,z threads per block
    const int tpb_x = 8;
    const int tpb_y = 8;
    const int tpb_z = 4;

    // grid x,y,z dimensions
    int X = (depth+tpb_x-1)/tpb_x;
    int Y = (height+tpb_y-1)/tpb_y;
    int Z = (width+tpb_z-1)/tpb_z;
    
    unsigned int n = X * Y * Z;

    dim3 Dg(X, Y, Z);
    dim3 Db(tpb_x, tpb_y, tpb_z);
    
    const unsigned int threads_per_block = tpb_x * tpb_y * tpb_z;
    unsigned int shared_mem_size = 2*threads_per_block*sizeof(float);

    getdiff<T, threads_per_block><<<Dg, Db, shared_mem_size>>>(tex, device_sum_diff, device_sum_count, d_x, depth, height, width, std::numeric_limits<T>::max(), start_mat);

    float result = reduce_sum(n, device_sum_diff);
    float count = reduce_sum(n, device_sum_count);

    if (count == 0) {
        return std::numeric_limits<float>::max();
    } else {
        return result/count;
    }
}
    
template <typename T>
float stitcher<T>::reduce(mat4x4<float> transform) {
    
    transform = world_to_source * transform * target_to_world;

    return red<T>(transform, depth, height, width, device_target_data, device_sum_diff, device_sum_count);
}

#endif

#endif
