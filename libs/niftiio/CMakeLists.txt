cmake_minimum_required(VERSION 2.8)

FILE(GLOB NIFTIIO_INC
    "include/*.h"
)

FILE(GLOB NIFTIIO_SRC
    "src/*.c"
)

ADD_LIBRARY(niftiio ${NIFTIIO_SRC} ${NIFTIIO_INC})

TARGET_INCLUDE_DIRECTORIES(niftiio PUBLIC "./include")
