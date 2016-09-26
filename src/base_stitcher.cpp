#include "base_stitcher.hpp"
#include "stitching.hpp"
#include <nifti1_io.h>
#include <stdint.h>

base_stitcher *base_stitcher::new_stitcher(int datatype, unsigned int depth, unsigned int height, unsigned int width) {
    
    switch (datatype) {
        case DT_BINARY:            // binary (1 bit/voxel)
            return new stitcher<bool>(depth, height, width);
        case DT_UNSIGNED_CHAR:     // unsigned char (8 bits/voxel)
            return new stitcher<uint8_t>(depth, height, width);
        case DT_SIGNED_SHORT:      // signed short (16 bits/voxel)
            return new stitcher<int16_t>(depth, height, width);
        case DT_SIGNED_INT:        // signed int (32 bits/voxel)
            return new stitcher<int32_t>(depth, height, width);
        case DT_FLOAT:             // float (32 bits/voxel)
            return new stitcher<float>(depth, height, width);
        case DT_INT8:              // signed char (8 bits)
            return new stitcher<int8_t>(depth, height, width);
        case DT_UINT16:            // unsigned short (16 bits)
            return new stitcher<uint16_t>(depth, height, width);
        case DT_UINT32:            // unsigned int (32 bits)
            return new stitcher<uint32_t>(depth, height, width);
    }
    return nullptr;
}
