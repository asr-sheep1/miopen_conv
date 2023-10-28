ROCM_PATH?= $(wildcard /opt/rocm)
HIP_PATH?= $(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc
MIOPEN_INSTALL_PATH=/opt/rocm/miopen
INCLUDE_DIRS=-I$(HIP_PATH)/include -I$(ROCM_PATH)/include  -I$(MIOPEN_INSTALL_PATH)/include
LD_FLAGS=-L$(ROCM_PATH)/lib  -L$(MIOPEN_INSTALL_PATH)/lib -lMIOpen   -lrocblas
TARGET=--amdgpu-target=gfx906
LAYER_TIMING=1

#HIPCC_FLAGS=-g -Wall $(CXXFLAGS) $(TARGET) $(INCLUDE_DIRS)
HIPCC_FLAGS=-g -O3 -Wall $(CXXFLAGS) $(TARGET) $(INCLUDE_DIRS)



all: main

HEADERS= miopen.hpp naive_algo.hpp

main: my_conv.cpp $(HEADERS)
	$(HIPCC) $(HIPCC_FLAGS) my_conv.cpp $(LD_FLAGS) -o my_conv

clean:
	rm -f *.o *.out my_conv
