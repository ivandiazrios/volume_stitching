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
