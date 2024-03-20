# Makefile for compiling C code into a shared library for use with Python

# Compiler settings - Can be changed to clang if desired
CC = gcc

# Compiler flags:
# -Ofast for aggressive optimizations
# -fPIC for position-independent code (needed for shared library)
# -shared for creating a shared library
CFLAGS = -Ofast -fPIC -shared

# Target library name
TARGET_LIB_LINUX = librun.so
TARGET_LIB_WIN = run.dll

# Source files - Automatically finds all .c files
SRC = $(wildcard *.c)

# Object files
OBJ = $(SRC:.c=.o)

# OS specific part
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    TARGET_LIB = $(TARGET_LIB_LINUX)
endif
ifeq ($(UNAME_S),Darwin)
    TARGET_LIB = $(TARGET_LIB_LINUX)
endif
ifeq ($(UNAME_S),Windows_NT)
    TARGET_LIB = $(TARGET_LIB_WIN)
    CC = x86_64-w64-mingw32-gcc
endif

# Rule to make the shared library
all: $(TARGET_LIB)

$(TARGET_LIB): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Rule for cleaning up
clean:
	rm -f $(TARGET_LIB) $(OBJ)

# Rule for installing (optional, might require additional actions for Python usage)
install:
	cp $(TARGET_LIB) /path/to/your/python/project
