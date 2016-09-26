import os
import os.path
import glob
from subprocess import check_output
import commands
from itertools import product
from collections import defaultdict
import csv

input_directory = "Project/sample_data/real_data/images_nii/"
input_directory = os.path.join(os.path.expanduser('~/'), input_directory)

command1 = "cd /homes/id413/Project/volume_stitching/bin ;"
command2 = "./stitching -target %s -source %s -iter 60 -t_rate 0.1 -r_rate 0.001" 

commands.getoutput(command1 + " make ")
command = command1 + command2

files = []

for input_file in glob.glob(os.path.join(input_directory, '*.nii')):
    files.append(input_file)

jump = 1

for i in range(0, len(files)-jump):
    file1 = files[i]
    file2 = files[i+jump]
    x = commands.getoutput(command % (file1, file2)) 
    print x

