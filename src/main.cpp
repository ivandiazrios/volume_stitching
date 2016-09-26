#include "imageIO.hpp"
#include <iostream>
#include "stitching.hpp"
#include "mat4x4.hpp"

typedef unsigned char T;

void print_simple_usage() {

    using namespace std;

    cout << "Usage: stitching -target <filename> -source <filename> -total_mats <filename> [OPTIONS]" << endl;
    cout << "       See the help for more details (-h)" << endl;
}

void print_usage() {
    using namespace std;
    int spacing = 30;
    
    cout << left;
    
    cout << "Usage: stitching -target <filename> -source <filename> -total_mats <filename> [OPTIONS]" << endl;

    cout << setw(spacing) << "\t-target <filename>" << "Filename of the target image (mandatory)" << endl;
    cout << setw(spacing) << "\t-source <filename>" << "Filename of the target image (mandatory)" << endl;
    cout << setw(spacing) << "\t-total_mats <filename>" << "Filename of the tracker to world matrices (mandatory)" << endl;
    
    cout << "*** OPTIONS ***" << endl;
    cout << setw(spacing) << "\t-ident" << "Use identity matrix as initial world transformation matrix" << endl;
    cout << setw(spacing) << "\t-o <filename>" << "Path to output image" << endl;
    cout << setw(spacing) << "\t-out_mat <filename>" << "Path to output world transformation matrix" << endl;
    cout << setw(spacing) << "\t-iter <int>" << "Number of iterations of gradient descent" << endl;
    cout << setw(spacing) << "\t-total_mats <filename>" << "Filename of the ground truth world transformation matrices" << endl;
    cout << setw(spacing) << "\t-use_truth" << "Use ground truth matrix as initial world transformation matrix" << endl;
    cout << setw(spacing) << "\t-evaluate" << "Evaluate pre and post registration error, requires ground truth matrices" << endl;
    cout << setw(spacing) << "\t-t_rate <float>" << "Set translation learning rate" << endl;
    cout << setw(spacing) << "\t-r_rate <float>" << "Set rotation learning rate" << endl;
    cout << setw(spacing) << "\t-show_iters" << "Show iterations progress" << endl;    
}

int main(int argc, char *argv[]) {
    
    // COMMAND LINE ARGS //
    
    const char *target_file = nullptr;
    
    // command line arg indices for first and last source files
    int start_source = -1, end_source = -1;
    
    //output nifti file and matrix file
    const char *output_file = nullptr;
    const char *out_mat_file = nullptr;
    
    // use identity matrix as world transform
    bool ident = false;
    
    // use ground truth matrix as world transform
    bool use_truth = false;
    
    // evaluate alignment errors
    bool evaluate = false;
    
    // show iteration progress
	bool show_iters = false;
    
	int num_iterations = 20;
    
    // path to tracker data
    const char *total_matrices = nullptr;
    
    // path to ground truth matrices
    const char *truth_matrices = nullptr;

	// translation and rotation learning rates
    float t_rate = 0.1;
	float r_rate = 0.001;
    
    // Parse command line args
    for(int i=1;i<argc;i++) {
        if (strcmp(argv[i], "-h")==0) {
            print_usage();
            return EXIT_SUCCESS;
        } else if (strcmp(argv[i], "-target")==0) {
            target_file = argv[++i];
        } else if ((strcmp(argv[i], "-source"))==0) {
            start_source = ++i;
            while (i < argc && argv[i][0] != '-') i++;
            end_source = i--;
        } else if (strcmp(argv[i], "-ident")==0) {
            ident = true;
        } else if (strcmp(argv[i], "-o")==0) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "-iter")==0) {
			num_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-out_mat")==0) {
            out_mat_file = argv[++i];
		} else if (strcmp(argv[i], "-use_truth")==0) {
			use_truth = true;
        } else if (strcmp(argv[i], "-evaluate")==0) {
            evaluate = true;
		} else if (strcmp(argv[i], "-t_rate")==0) {
			t_rate = atof(argv[++i]);
		} else if (strcmp(argv[i], "-r_rate")==0) {
			r_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "-truth_mats")==0) {
            truth_matrices = argv[++i];
        } else if (strcmp(argv[i], "-total_mats")==0) {
            total_matrices = argv[++i];
        } else if (strcmp(argv[i], "-show_iters")==0) {
			show_iters = true;
		}
    }
    
    // check necessary args supplied
    if (!target_file || start_source == 1 || end_source == -1 || !total_matrices ||
        ((use_truth || evaluate) && !truth_matrices)) {
		print_simple_usage();
		return EXIT_FAILURE;
	}
    
    NiftiIO target(target_file);
    
    int *dim = target.get_dimensions();

	// Switch dim[3] and dim[1] due to row major order data
    unsigned int depth = dim[3], height = dim[2], width = dim[1];
    
    // call factory method to get correct stitcher type
    base_stitcher *stitch = base_stitcher::new_stitcher(target.get_datatype(),
                                                        depth, height, width);
    
    // if unsupported type abort execution
    if (!stitch) {
        std::cerr << "NIFTI FILE DATATYPE NOT SUPPORTED " << target.datatype_to_string() << std::endl;
        return EXIT_FAILURE;
    }
    
    stitch->set_target(target.get_data(), target.get_ijk_to_xyz_matrix());
    
    mat4x4<float> world_transform; // World transform matrix between a pair of images
    mat4x4<float> truth_transform; // Ground truth matrix between a pair of images
    
    for (int i = start_source; i < end_source; i++) {
        
        NiftiIO source(argv[i]);
        
        if (!target.check_compatible(source)) {
            return EXIT_FAILURE;
        }
        
        stitch->set_source(source.get_data(), source.get_ijk_to_xyz_matrix().inverse());
        
        if (evaluate) {
            truth_transform = source.get_truth_matrix(truth_matrices).inverse() *
                              target.get_truth_matrix(truth_matrices);
        }
        
        if (ident) {
            world_transform = mat4x4<float>::get_identity_matrix();
        } else if (use_truth) {
            world_transform = source.get_truth_matrix(truth_matrices).inverse() *
            target.get_truth_matrix(truth_matrices);
        } else {
            world_transform = source.get_total_matrix(total_matrices).inverse() *
            target.get_total_matrix(total_matrices);
        }
       
		float no_reg_error, reg_error;

        if (evaluate) {
            // get pre registration error
            no_reg_error = stitch->reducediff(truth_transform, world_transform);
        }

        world_transform = stitch->perform_registration(world_transform, num_iterations,
                                                       t_rate, r_rate, show_iters);
        stitch->fuse(world_transform);

        stitch->finish();

        if (evaluate) {
            // get post registration error
            reg_error = stitch->reducediff(truth_transform, world_transform);

			std::cout << std::endl;
			std::cout << "Alignment error (mm) before registration: " << no_reg_error << std::endl;
			std::cout << "Alignment error (mm) after registration:  " << reg_error << std::endl;
        }
    }
    
    if (out_mat_file) {
        world_transform.write_to_file(out_mat_file);
    }
    
    if (output_file) {
        target.write_image(output_file);
    }
    
    delete stitch;
}

