/* CPU stitching implementation */

#ifndef USE_CUDA

#include "stitching.hpp"
#include <stdint.h>
#include "vec4.hpp"
#include <limits>

#define ROW_INDEX(depth, height, width, i, j, k) \
k + width * (j + height * i)

static const int MIN_NEG_VALUE = 10000;

/* quick_floor and quick_ceil provide quicker floor and ceil implementations 
 than the stl functions - require you to know the minimum possible value */

template<class T>
inline int quick_floor(T x) {
    return (int)(x + (T)MIN_NEG_VALUE) - MIN_NEG_VALUE;
}

template<class T>
inline int quick_ceil(T x) {
    return MIN_NEG_VALUE - (int)((T)MIN_NEG_VALUE - x);
}

/* trilinear interpolation */
template<class T>
T interpolate(T *data, int depth, int height, int width, float a1, float a2, float a3) {
    int x0 = quick_floor(a1), x1 = quick_ceil(a1);
    int y0 = quick_floor(a2), y1 = quick_ceil(a2);
    int z0 = quick_floor(a3), z1 = quick_ceil(a3);
    if (x0 < 0|| y0 < 0 || z0 < 0 || x1 >= depth || y1 >= height || z1 >= width)
        return T();
    
    T xd = a1 - x0;
    T yd = a2 - y0;
    T zd = a3 - z0;
    
    return
    data[ROW_INDEX(depth, height, width, x0, y0, z0)]*(T(1)-xd)*(T(1)-yd)*(T(1)-zd) +
    data[ROW_INDEX(depth, height, width, x1, y0, z0)]*xd*(T(1)-yd)*(T(1)-zd) +
    data[ROW_INDEX(depth, height, width, x0, y1, z0)]*(T(1)-xd)*yd*(T(1)-zd) +
    data[ROW_INDEX(depth, height, width, x0, y0, z1)]*(T(1)-xd)*(T(1)-yd)*zd +
    data[ROW_INDEX(depth, height, width, x1, y0, z1)]*xd*(T(1)-yd)*zd +
    data[ROW_INDEX(depth, height, width, x0, y1, z1)]*(T(1)-xd)*yd*zd +
    data[ROW_INDEX(depth, height, width, x1, y1, z0)]*xd*yd*(T(1)-zd) +
    data[ROW_INDEX(depth, height, width, x1, y1, z1)]*xd*yd*zd;
}

template <typename T>
void stitcher<T>::set_target(void *target_data, mat4x4<float> target_to_world) {
    
    this->target_data = (T*) target_data;
    this->target_to_world = target_to_world;
    
}

template<typename T>
void stitcher<T>::set_source(void *source_data, mat4x4<float> world_to_source) {
    
    this->source_data = (T*) source_data;
    this->world_to_source = world_to_source;
    
}

template <typename T>
stitcher<T>::stitcher(unsigned int depth, unsigned int height, unsigned int width)
: depth(depth), height(height), width(width)
, num_voxels(depth * height * width) {}

template <typename T>
void stitcher<T>::finish() {} // Only needed for GPU version

template <typename T>
stitcher<T>::~stitcher() {}; // Only needed for GPU version
 
template <typename T>
float stitcher<T>::reduce(mat4x4<float> transform) {
    
    unsigned int index;
    vec4<float> out_vec;
    
    transform = world_to_source * transform * target_to_world;
    
    float sum = 0;
    int n = 0;
    
    for (int i = 0; i < depth; i++)
    for (int j = 0; j < height; j++)
    for (int k = 0; k < width; k++) {
                
        index = k + width * (j + height * i);
        out_vec = transform * vec4<float>(k,j,i,1.0f);
        
        float arg1 = target_data[index];
        float arg2 = interpolate(source_data, depth, height, width,
                                 out_vec.a3, out_vec.a2, out_vec.a1);
        
        if (arg1 && arg2) {
            sum += (arg1 - arg2) * (arg1 - arg2);
            n++;
        }
    }
   
    if (n == 0) {
        return std::numeric_limits<float>::max();
    } else {
        return sum/n;
    }
}

template <typename T>
void stitcher<T>::fuse(mat4x4<float> transform) {
    
    unsigned int index;
    vec4<float> out_vec;
    
    transform = world_to_source * transform * target_to_world;
    
    for (int i = 0; i < depth; i++)
    for (int j = 0; j < height; j++)
    for (int k = 0; k < width; k++) {
                
        index = k + width * (j + height * i);
        out_vec = transform * vec4<float>(k,j,i,1.0f);
        
        unsigned char arg1 = target_data[index];
        unsigned char arg2 = (unsigned char) interpolate(source_data,
                                                         depth, height, width,
                                                         out_vec.a3, out_vec.a2, out_vec.a1);
        
        target_data[index] = arg1 > arg2 ? arg1 : arg2;
    }
}

/* calculate error in mm wrt to ground truth matrices */
template <typename T>
float stitcher<T>::reducediff(mat4x4<float> mref, mat4x4<float> mat) {
    
    unsigned int index;
    vec4<float> voxel;
    mref = mref.inverse();
    vec4<float> centre = vec4<float>((float) width / 2,
                                     (float) height / 2,
                                     (float) depth / 2, 1.0f);
    float sum = 0;
    int n = 0;
    
    for (int i = 0; i < depth; i++)
    for (int j = 0; j < height; j++)
    for (int k = 0; k < width; k++) {
                
        index = k + width * (j + height * i);
        voxel = vec4<float>(k,j,i,1.0f);
        
        if (voxel.get_distance(centre) < 3.0f) {

            vec4<float> mapped_voxel = mref * (mat * voxel);
            sum += voxel.get_distance(mapped_voxel);
            n++;
        }
    }
    
    return sum / n;
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
