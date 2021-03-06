##
# Second-level Makefile cross-compiling the parallel application.
#
#
# Copyright (c) 2012, Neurasmus B.V., The Netherlands,
# web: www.neurasmus.com email: info@neurasmus.com
#
# Any use or reproduction in whole or in parts is prohibited
# without the written consent of the copyright owner.
#
# All Rights Reserved.
#
# Author: Sebastian Isaza
# Created: 07-08-2012
# Modified: 08-08-2012
# 
##

# --- MACROS
# define program name
MAIN		:= Infoli

# define the c compiler to use
CC		:= gcc

# define any compile-time flags
CFLAGS		:= -O3

# define the NVIDIA compiler to use
NVCC		:= nvcc

PROJ_BASE	:= .

INCLUDES	:= -I$(PROJ_BASE)

# define any libraries to link into executable
# LDFLAGS       := -lm -lX11 -lpthread

# define CUDA flags
CUDA_LDFLAGS	:= $(LDFLAGS) -lrt -lcudart

LIBS		:= -lm
CUDA_LIBS	:= $(LIBS) -L"$(CUDA_INSTALL_PATH)/lib64"

# define CUDA source files
SRCS= infoli.cu

# define CUDA header files
HDRS= infoli.h

################################################################################
# CUDA code generation flags
#GENCODE_SM20	:=-gencode arch=compute_20,code=sm_20
GENCODE_SM30    :=-gencode arch=compute_30,code=sm_30
#GENCODE_SM32   :=-gencode arch=compute_32,code=sm_32
#GENCODE_SM35   :=-gencode arch=compute_35,code=sm_35
GENCODE_SM50    :=-gencode arch=compute_50,code=sm_50
GENCODE_SMXX    :=-gencode arch=compute_50,code=compute_50
GENCODE_FLAGS   ?=$(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30) \
                $(GENCODE_SM32) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SMXX)

NVCCFLAGS	:= $(GENCODE_FLAGS)

debug		:= 0 
ifneq ($(debug), 1)
	CFLAGS		:= -O3 -g0 
	NVCCFLAGS	+= -O3
else
	CFLAGS		:= -O0 -g3 -Wall
	NVCCFLAGS	+= -O0
endif

# --- TARGETS
all: ${MAIN} run
 
#cuda: infoli.c
#	$(CC) -o $(MAIN) infoli.c $(INCLUDES) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)

# build the cuda program
${MAIN}: ${SRCS} ${HDRS}
	@echo #
	@echo "-- BUILDING PROGRAM --"
	$(NVCC) -c ${SRCS} $(NVCCFLAGS) $(INCLUDES)

# run the cuda program
run: 
	@echo #
	@echo "-- RUNNING THE PROGRAM --"
	$(MAIN)
	@echo #

clean:
	@echo #
	@echo "-- CLEANING PROJECT FILES --"
	rm -f ${MAIN} *.o *~
	@echo #

