#ifndef BASE_STITCHER_HPP
#define BASE_STITCHER_HPP

#include "mat4x4.hpp"

class base_stitcher {
public:
    
    // Compile flag specific - implementation in stitching.cpp or stitching.cu
    virtual void set_target(void *target_data, mat4x4<float> target_to_world) =0;
    virtual void set_source(void *target_data, mat4x4<float> world_to_source) =0;
    
    virtual void fuse(mat4x4<float> transform) =0;
    virtual void finish() =0;
    virtual ~base_stitcher() {};
    virtual float reducediff(mat4x4<float>, mat4x4<float>) =0;
    //////////////////////////////////////////////////////////////////////////
    
    virtual mat4x4<float> perform_registration(mat4x4<float> world_transform, int number_iterations, float t_rate, float r_rate, bool show_iters) =0;
    
    static base_stitcher *new_stitcher(int datatype, unsigned int depth,
                                                     unsigned int height,
                                                     unsigned int width);
    
};

#endif
