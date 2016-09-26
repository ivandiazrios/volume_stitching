#ifdef USE_CUDA

#ifndef FUSION_CUH
#define FUSION_CUH

#include "vec4.hpp"
#include "mat4x4.hpp"
#include <limits>

extern cudaTextureObject_t tex;

template <typename T>
class F_max {
public:
	__device__
	T operator()(T arg1, T arg2) {
    	return (arg1 > arg2 ? arg1 : arg2);
	}
};

template <typename T, typename fuse_method>
__global__ void getmax(cudaTextureObject_t tex, int n, T *data1, int depth, int height, int width, int n_max, mat4x4<float> m, fuse_method fuser)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    
    if (x < depth && y < height && z < width) {
        
        unsigned int index = z + width * (y + height * x);
        
        x = x ^ z; z = z ^ x; x = x ^ z; //swap x and z
        
        vec4<float> v = m * vec4<float>(x,y,z,1.0f);
        
        data1[index] = fuser((tex3D<float>(tex, v.a1+0.5f, v.a2+0.5f, v.a3+0.5f) * n_max), data1[index]);
    }
}

template <typename T>
void stitcher<T>::fuse(mat4x4<float> transform) {
    
    transform = world_to_source * transform * target_to_world;
    
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
    
    getmax<T, F_max<T> ><<<Dg, Db>>>(tex, num_voxels, device_target_data, depth, height, width, std::numeric_limits<T>::max(), transform, F_max<T>());
}

#endif

#endif
