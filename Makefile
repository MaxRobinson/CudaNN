all: network.cu
	nvcc --std=c++11 network.cu -lcublas -lcudart -lcuda -lcurand -o network.exe