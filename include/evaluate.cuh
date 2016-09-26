#ifdef USE_CUDA

#ifndef EVALUATE_CUH
#define EVALUATE_CUH

#include "mat4x4.hpp"
#include "vec4.hpp"

template <typename T>
__global__ void getsphere(float *device_sum_diff, float *device_sum_count, T *data1, int depth, int height, int width, mat4x4<float> mref, mat4x4<float> mat) {

	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
	

	unsigned int index = z + width * (y + height * x);
	x = x ^ z; z = z ^ x; x = x ^ z;

	vec4<float> centre = vec4<float>((float) width / 2, 
									 (float) height / 2,
									 (float) depth / 2, 1.0f);
	vec4<float> voxel = vec4<float>(x,y,z,1.0f);
	

	if (voxel.get_distance(centre) < 3.0f) {

		vec4<float> mapped_voxel = mref * (mat * voxel);
		device_sum_diff[index] = voxel.get_distance(mapped_voxel);
		device_sum_count[index] = 1;
	} else {
		device_sum_diff[index] = 0;
		device_sum_count[index] = 0;
	}
}


template <typename T>
float red_sphere(mat4x4<float> Mref, mat4x4<float> mat, unsigned int depth, unsigned int height, unsigned int width, T *d_x, float *device_sum_diff, float *device_sum_count) {
    
    // x,y,z threads per block
    const int tpb_x = 8;
    const int tpb_y = 8;
    const int tpb_z = 4;
    
    // grid x,y,z dimensions
    int X = (depth+tpb_x-1)/tpb_x;
    int Y = (height+tpb_y-1)/tpb_y;
    int Z = (width+tpb_z-1)/tpb_z;
    
    dim3 Dg(X, Y, Z);
    dim3 Db(8, 8, 4);
    
    getsphere<T><<<Dg, Db>>>(device_sum_diff, device_sum_count, d_x, depth, height, width, Mref, mat);
    
    unsigned int n = depth * height * width;

	float result = reduce_sum(n, device_sum_diff);
	float count = reduce_sum(n, device_sum_count);

	return result/count;
}

template <typename T>
float stitcher<T>::reducediff(mat4x4<float> Mref, mat4x4<float> mat) {

	return red_sphere(Mref.inverse(), mat, depth, height, width, device_target_data, device_sum_diff, device_sum_count);	

}

#endif

#endif
