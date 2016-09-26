#ifndef STITCHING_HPP
#define STITCHING_HPP

#include "base_stitcher.hpp"

template <typename T>
class stitcher: public base_stitcher {
private:
    
    T *target_data, *source_data, *device_target_data;
    float *device_sum_diff, *device_sum_count;
    unsigned int depth, height, width;
    unsigned int num_voxels;
    mat4x4<float> target_to_world, world_transform, world_to_source;
    
    // Keep track of average registration time
    template <int transform_type>
    float get_gradient(float step, mat4x4<float> &mat);
    
public:
    
    mat4x4<float> perform_registration(mat4x4<float> world_transform, int number_iterations, float t_rate, float r_rate, bool show_iters);
    
    // Compile flag specific (CUDA/CPP Implementation)
    float reducediff(mat4x4<float>, mat4x4<float>);
    void set_target(void *target_data, mat4x4<float> target_to_world);
    void set_source(void *target_data, mat4x4<float> world_to_source);
    
    void fuse(mat4x4<float> transform);
    float reduce(mat4x4<float> transform);
    void finish();
    stitcher(unsigned int depth, unsigned int height, unsigned int width);
    ~stitcher();

};


enum rigid_transform {X_T, Y_T, Z_T, X_R, Y_R, Z_R};

template <typename T>
template <int transform_type>
float stitcher<T>::get_gradient(float step, mat4x4<float> &mat) {
    
    float y1, y2;
    
    if (transform_type == X_T) {
        
        y1 = reduce(mat.translate(-step, 0, 0));
        y2 = reduce(mat.translate(step, 0, 0));
        
    } else if (transform_type == Y_T) {
        
        y1 = reduce(mat.translate(0, -step, 0));
        y2 = reduce(mat.translate(0, step, 0));
        
    } else if (transform_type == Z_T) {
        
        y1 = reduce(mat.translate(0, 0, -step));
        y2 = reduce(mat.translate(0, 0, step));
        
    } else if (transform_type == X_R) {
        
        y1 = reduce(mat.rotate_x(-step));
        y2 = reduce(mat.rotate_x(step));
        
    } else if (transform_type == Y_R) {
        
        y1 = reduce(mat.rotate_y(-step));
        y2 = reduce(mat.rotate_y(step));
        
    } else if (transform_type == Z_R){
        
        y1 = reduce(mat.rotate_z(-step));
        y2 = reduce(mat.rotate_z(step));
        
    }
    
    return (y2 - y1) / (2 * step);
}

template <typename T>
mat4x4<float> stitcher<T>::perform_registration(mat4x4<float> world_transform, int number_iterations, float t_rate, float r_rate, bool show_iters) {
    
    /* translation/rotation step =>
            gradient = (f(parameter+step) - f(parameter-step)) / (2*step) */
    float t_step = 0.01;
    float r_step = 0.0001;
    
    float t_learning_rate = t_rate;
    float r_learning_rate = r_rate;
    
    float xt_gradient, yt_gradient, zt_gradient; // x translation gradient ...
    float xt_diff,     yt_diff,     zt_diff;     // x translation diff ...
    
    float xr_gradient, yr_gradient, zr_gradient; // x rotation gradient ...
    float xr_diff,     yr_diff,     zr_diff;     // x rotation diff ...
    
    float error_value = reduce(world_transform);
    
    if (show_iters)
        printf("Iteration 00: MSE %f\n", error_value);
   	
    for (int iterations = 0; iterations < number_iterations; iterations++) {
        
        xt_gradient = get_gradient<X_T>(t_step, world_transform);
        yt_gradient = get_gradient<Y_T>(t_step, world_transform);
        zt_gradient = get_gradient<Z_T>(t_step, world_transform);
        
        xt_diff = - t_learning_rate * xt_gradient;
        yt_diff = - t_learning_rate * yt_gradient;
        zt_diff = - t_learning_rate * zt_gradient;
        
        mat4x4<float> temp_transform = world_transform.translate(xt_diff, yt_diff, zt_diff);
        float final_metric = reduce(temp_transform);
        
        if (final_metric < error_value) {
            error_value = final_metric;
            t_learning_rate *= 1.05;
            world_transform = temp_transform;
        } else {
            t_learning_rate *= 0.5;
        }
        
        xr_gradient = get_gradient<X_R>(r_step, world_transform);
        yr_gradient = get_gradient<Y_R>(r_step, world_transform);
        zr_gradient = get_gradient<Z_R>(r_step, world_transform);
        
        xr_diff = - r_learning_rate * xr_gradient;
        yr_diff = - r_learning_rate * yr_gradient;
        zr_diff = - r_learning_rate * zr_gradient;
        
        temp_transform = mat4x4<float>::bryant_to_matrix(xr_diff, yr_diff, zr_diff) * world_transform;
        final_metric = reduce(temp_transform);
        
        if (final_metric < error_value) {
            error_value = final_metric;
            world_transform = temp_transform;
            r_learning_rate *= 1.05;
        } else {
            r_learning_rate *= 0.5;
        }
        
        if (show_iters)
            printf("Iteration %02d: MSE %f\n", iterations+1, error_value);
    }

    return world_transform;
}

#endif
