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
MAIN= InfOli

# define the C compiler to use
CC= gcc

# define any compile-time flags
CFLAGS= -O3

# define any libraries to link into executable
LIBS= -lm

# define C source files
SRCS= infoli.c

# define C header files
HDRS= infoli.h

# --- TARGETS
all: ${MAIN}

#Builds the program
${MAIN}: ${SRCS} ${HDRS}
	@echo #
	@echo "-- BUILDING PROGRAM --"
	${CC} ${SRCS} ${CFLAGS} ${LIBS} -o ${MAIN}

clean:
	@echo #
	@echo "-- CLEANING PROJECT FILES --"
	$(RM) *.o ${MAIN}

