#include "imageIO.hpp"
#include "string.h"
#include <boost/filesystem.hpp>

NiftiIO::NiftiIO(const char *fname) {
    read_image(fname);
}

void NiftiIO::read_image(const char *fname) {
    nim = nifti_image_read(fname, 1);
}

void NiftiIO::write_image(const char *fname) {
    nifti_set_filenames(nim, fname, 0, 1);
    nifti_image_write(nim);
}

void *NiftiIO::get_data() {
    return nim->data;
}

int NiftiIO::get_datatype() {
    return nim->datatype;
}

int *NiftiIO::get_dimensions() {
    return nim->dim;
}

int NiftiIO::get_num_voxels() {
    return nim->nvox;
}

NiftiIO::~NiftiIO() {
    nifti_image_free(nim);
}

mat4x4<float> NiftiIO::read_matrix_from_file(const char *f) {
    boost::filesystem::path path(nim->fname);
    std::string filename(path.stem().string());
    
    std::ifstream file(f);
    std::string matrix_filename;
    int lines_read = 0;
    
    mat4x4<float> mat;
    
    while (std::getline(file, matrix_filename))
    {
        if ((lines_read % 5)==0 && filename.compare(matrix_filename)==0) {
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    file >> mat.m[i][j];
            
            file.close();
            return mat;
        } else {
            lines_read++;
        }
    }
    
    // report absent matrix and the use of the identity matrix instead
    std::cout << "Could not find matrix for " << filename << " in " << f << ", using identity matrix instead" << std::endl;
    
    file.close();
    return mat4x4<float>::get_identity_matrix();
}

mat4x4<float> NiftiIO::get_truth_matrix(const char *truth_matrices) {
    return read_matrix_from_file(truth_matrices);
}

mat4x4<float> NiftiIO::get_total_matrix(const char *total_matrices) {
    return read_matrix_from_file(total_matrices);
}

bool NiftiIO::check_supported_type() {
    switch (get_datatype()) {
        case DT_BINARY:            /* binary (1 bit/voxel)         */
        case DT_UNSIGNED_CHAR:     /* unsigned char (8 bits/voxel) */
        case DT_SIGNED_SHORT:      /* signed short (16 bits/voxel) */
        case DT_SIGNED_INT:        /* signed int (32 bits/voxel)   */
        case DT_FLOAT:             /* float (32 bits/voxel)        */
        case DT_INT8:              /* signed char (8 bits)         */
        case DT_UINT16:            /* unsigned short (16 bits)     */
        case DT_UINT32:            /* unsigned int (32 bits)       */
            return true;
        default:
            return false;
    }
}

std::string NiftiIO::datatype_to_string() {
    return std::string(nifti_datatype_string(get_datatype()));
}

bool NiftiIO::check_compatible(NiftiIO &source) {
    
    if (nim->dim[1] != source.nim->dim[1] ||
        nim->dim[2] != source.nim->dim[2] ||
        nim->dim[3] != source.nim->dim[3]) {
        
        std::cerr << "TARGET AND SOURCE VOLUMES MUST HAVE EQUAL DIMENSIONS\n";
        return false;
        
    } else if (this->get_datatype() != source.get_datatype()) {
        
        std::cerr << "TARGET AND SOURCE VOLUMES MUST HAVE THE SAME DATATYPE\n";
        return false;
        
    } else if (nim->dx != source.nim->dx ||
               nim->dy != source.nim->dy ||
               nim->dz != source.nim->dz) {
        
        std::cerr << "TARGET AND SOURCE VOLUME VOXEL'S MUST HAVE THE SAME SPACING\n";
        return false;
        
    }
    return true;
}

mat4x4<float> NiftiIO::get_ijk_to_xyz_matrix() {

    /* Return transformation matrix from voxel coordinates to coordinates 
     in tracker coordinate system */
    
	float origin_x = -((nim->nx-1.0)/2.0)*nim->dx;
	float origin_y = 0;
	float origin_z = -((nim->nz-1.0)/2.0)*nim->dz;
	
	float trans[4][4] =
	{{nim->dx, 0, 0, origin_x},
	{0, nim->dy, 0, origin_y},
	{0, 0, nim->dz, origin_z},
	{0, 0, 0, 1}};
	return mat4x4<float>(trans);
}

