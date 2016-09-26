/* GPU stitching implementation */

#ifdef USE_CUDA

#include "mat4x4.hpp"
#include "vec4.hpp"
#include <helper_cuda.h>
#include "stitching.hpp"
#include "registration.cuh"
#include "fusion.cuh"
#include "evaluate.cuh"
#include <stdint.h>

cudaTextureObject_t tex = 0;
cudaArray *d_volumeArray = 0;

template<class T>
void create_tex(T *data, int depth, int height, int width) {

    const cudaExtent volumeSize = make_cudaExtent(width, height, depth);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    
    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void *)data, volumeSize.width * sizeof(T), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));
    
    cudaResourceDesc    texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array  = d_volumeArray;
    cudaTextureDesc     texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeBorder; 
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.addressMode[2] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeNormalizedFloat;

    checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
}

template <typename T>
void stitcher<T>::set_target(void *target_data, mat4x4<float> target_to_world) {

    cudaMemcpy(device_target_data, (T*)target_data, num_voxels*sizeof(T), cudaMemcpyHostToDevice);
    this->target_data = (T*) target_data;
    this->target_to_world = target_to_world;
    
}

template <typename T>
void stitcher<T>::set_source(void *source_data, mat4x4<float> world_to_source) {
    
    create_tex((T*)source_data, depth, height, width);
    this->source_data = (T*) source_data;
    this->world_to_source = world_to_source;
    
}

template <typename T>
void stitcher<T>::finish() {
    
    // Copy fused volume back to host and unbind source volume
    cudaMemcpy(target_data, device_target_data, num_voxels*sizeof(T), cudaMemcpyDeviceToHost);
    cudaDestroyTextureObject(tex);
    
}

template <typename T>
stitcher<T>::stitcher(unsigned int depth, unsigned int height, unsigned int width)
: depth(depth), height(height), width(width)
, num_voxels(depth * height * width) {
    
    cudaMalloc(&device_target_data, num_voxels*sizeof(T));
    cudaMalloc(&device_sum_diff, num_voxels*sizeof(float));
    cudaMalloc(&device_sum_count, num_voxels*sizeof(float));
    
    // create 3D array for source data
    const cudaExtent volumeSize = make_cudaExtent(width, height, depth);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));
}

template <typename T>
stitcher<T>::~stitcher() {
	
    cudaFreeArray(d_volumeArray);

    cudaFree(device_target_data);
    cudaFree(device_sum_diff);
    cudaFree(device_sum_count);
}

template class stitcher<bool>;
template class stitcher<uint8_t>;
template class stitcher<uint16_t>;
template class stitcher<uint32_t>;
template class stitcher<int8_t>;
template class stitcher<int16_t>;
template class stitcher<int32_t>;
template class stitcher<float>;

#endif
