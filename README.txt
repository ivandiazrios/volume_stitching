Real-time 3D Ultrasound Stitching
Imperial College London Computing BEng Final Year Project

The increasing adoption of 3D ultrasound by the medical 
community has drawn a focus on the technical improvements 
that can be made to increase its quality as a diagnostic 
imaging technique. 

This project proposes an implementation for real-time 
ultrasound image fusion and regis- tration in order to 
rectify small alignment errors in ultrasound acquisition 
systems. The required levels of parallelization for 
real-time performance are achieved through the use of the 
CUDA parallel computing platform. 

The work accomplished in this project forms part of a 
larger research initiative with King’s College London 
which hopes to increase the usefulness of medical ultrasound.

-----------------
BUILD INTRUCTIONS
-----------------

This code is built using CMake. To create an external build:

>> mkdir bin
>> cd bin
>> cmake PROJECT_ROOT_DIRECTORY -DUSE_CUDA=[ON/OFF]
>> make

To install:

>> make install

The GPU version is used if the USE_CUDA flag is set to ON. If not, and by 
default, the CPU version is used.

-----------
MATRIX DATA
-----------

Matrix data is required for the tracker info and can optionally also be used
to evaluate the alignment by providing the ground truth matrices. The data 
for multiple images is stored in one file, where the first line is the name 
of the image file and the next four the matrix values, and this is repeated 
consecutively in the file for all matrices.

The tracker matrices (matrices to go to world coordinates) are passed in 
using the total_mats flag. The matrix will then be applied to every voxel
of the image.

The evaluation matrices (ground truth matrices) are the correct alignments
between an image and the first image acquired in that session, which we can
then use to compute the optimal alignment between all pairs of images in a
given session.
