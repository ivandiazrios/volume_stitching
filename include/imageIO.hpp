#ifndef IMAGEIO_H
#define IMAGEIO_H

#include <nifti1_io.h>
#include "mat4x4.hpp"

class NiftiIO {
public:
    NiftiIO(const char *fname);
    void read_image(const char *fname);
    void write_image(const char *fname);
    void *get_data();
    int get_datatype();
    int *get_dimensions();
    int get_num_voxels();
    bool check_compatible(NiftiIO &source);
    bool check_supported_type();
    std::string datatype_to_string();
    mat4x4<float> get_total_matrix(const char *total_matrices);
	mat4x4<float> get_truth_matrix(const char *truth_matrices);
    mat4x4<float> get_ijk_to_xyz_matrix();
    ~NiftiIO();
    
private:
    nifti_image *nim;
    mat4x4<float> read_matrix_from_file(const char *);
};

#endif
