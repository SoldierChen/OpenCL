#include <stdlib.h>
#include <malloc.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "config.h"
#include "graph.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

#define AOCL_ALIGNMENT 64

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;

//Each kernel uses an independent queue
static cl_command_queue queueReadActiveVertices;
static cl_command_queue queueReadNgbInfo;
static cl_command_queue queueTraverseNgb;
static cl_command_queue queueStatusWrite;

static cl_command_queue queueInspectActiveVertices;
static cl_command_queue queueGather0; 
static cl_command_queue queueGather1; 
static cl_command_queue queueGather2; 
static cl_command_queue queueGather3; 
static cl_command_queue queueApply0;
static cl_command_queue queueApply1;
static cl_command_queue queueApply2;
static cl_command_queue queueApply3;

static cl_kernel readActiveVertices;
static cl_kernel readNgbInfo;
static cl_kernel traverseNgb;
static cl_kernel statusWrite;

static cl_kernel inspectActiveVertices;
static cl_kernel gather0;
static cl_kernel gather1;
static cl_kernel gather2;
static cl_kernel gather3;
static cl_kernel apply0;
static cl_kernel apply1;
static cl_kernel apply2;
static cl_kernel apply3;

static cl_program program;
static cl_int status;

static PROP_TYPE* CPUResult;
static PROP_TYPE* FPGAResult;

static PROP_TYPE* vertexProp;
static int* inRowPointerArray;
static int* inDeg;
static int* inEdges;
static PROP_TYPE* inEdgeProp;
static int* outRowPointerArray;
static int* outDeg;
static int* outEdges;
static int* activeVertices;
static STATUS_TYPE* activeStatus;
static int* activeVertexNum;

#define RAND_RANGE(N) ((float)rand() / ((float)RAND_MAX + 1) * (N))

static void dumpError(const char *str, cl_int status) {
    printf("%s\n", str);
    printf("Error code: %d\n", status);
}

static void freeResources(){
	// opencl environments
	if(readActiveVertices) clReleaseKernel(readActiveVertices);  
	if(readNgbInfo) clReleaseKernel(readNgbInfo);  
	if(traverseNgb) clReleaseKernel(traverseNgb);  
	if(statusWrite) clReleaseKernel(statusWrite);  

	if(inspectActiveVertices) clReleaseKernel(inspectActiveVertices);  
	if(gather0) clReleaseKernel(gather0);  
	if(gather1) clReleaseKernel(gather1);  
	if(gather2) clReleaseKernel(gather2);  
	if(gather3) clReleaseKernel(gather3);  
	if(apply0) clReleaseKernel(apply0);  
	if(apply1) clReleaseKernel(apply1);  
	if(apply2) clReleaseKernel(apply2);  
	if(apply3) clReleaseKernel(apply3);  

	if(program) clReleaseProgram(program);

	if(queueReadActiveVertices) clReleaseCommandQueue(queueReadActiveVertices);
	if(queueReadNgbInfo) clReleaseCommandQueue(queueReadNgbInfo);
	if(queueTraverseNgb) clReleaseCommandQueue(queueTraverseNgb);
	if(queueStatusWrite) clReleaseCommandQueue(queueStatusWrite);

	if(queueInspectActiveVertices) clReleaseCommandQueue(queueInspectActiveVertices);
	if(queueGather0) clReleaseCommandQueue(queueGather0);
	if(queueGather1) clReleaseCommandQueue(queueGather1);
	if(queueGather2) clReleaseCommandQueue(queueGather2);
	if(queueGather3) clReleaseCommandQueue(queueGather3);
	if(queueApply0) clReleaseCommandQueue(queueApply0);
	if(queueApply1) clReleaseCommandQueue(queueApply1);
	if(queueApply2) clReleaseCommandQueue(queueApply2);
	if(queueApply3) clReleaseCommandQueue(queueApply3);

	if(CPUResult != NULL) free(CPUResult);
	if(FPGAResult != NULL) free(FPGAResult);

	// Shared memory objects
	if(vertexProp) clSVMFreeAltera(context, vertexProp);
	if(inRowPointerArray) clSVMFreeAltera(context, inRowPointerArray);
	if(inDeg) clSVMFreeAltera(context, inDeg);
	if(inEdges) clSVMFreeAltera(context, inEdges);
	if(inEdgeProp) clSVMFreeAltera(context, inEdgeProp);
	if(outRowPointerArray) clSVMFreeAltera(context, outRowPointerArray);
	if(outDeg) clSVMFreeAltera(context, outDeg);
	if(outEdges) clSVMFreeAltera(context, outEdges);
	if(activeVertices) clSVMFreeAltera(context, activeVertices);
	if(activeStatus) clSVMFreeAltera(context, activeStatus);
	if(activeVertexNum) clSVMFreeAltera(context, activeVertexNum);

	if(context) clReleaseContext(context);
}

void cleanup(){}

cl_int varMap(const int &nodeNum, const int &edgeNum){

	cl_int localStatus;
	std::vector<cl_int> statusVec;
	// readActiveVertices kernel
	localStatus = clEnqueueSVMMap(queueReadActiveVertices, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeVertices, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueReadActiveVertices, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeVertexNum, sizeof(int) * 4, 0, NULL, NULL); 
	statusVec.push_back(localStatus);


	// read neighbor information
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ, 
			(void *)outRowPointerArray, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ, 
			(void *)outDeg, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeVertexNum, sizeof(int) * 4, 0, NULL, NULL); 
	statusVec.push_back(localStatus);


	// traverse neighbor
	localStatus = clEnqueueSVMMap(queueTraverseNgb, CL_TRUE, CL_MAP_READ, 
			(void *)outEdges, sizeof(int) * edgeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueTraverseNgb, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeVertexNum, sizeof(int) * 4, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	// write status
	localStatus = clEnqueueSVMMap(queueStatusWrite, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeStatus, sizeof(STATUS_TYPE) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	// gather apply
	localStatus = clEnqueueSVMMap(queueInspectActiveVertices, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeStatus, sizeof(STATUS_TYPE) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueInspectActiveVertices, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)vertexProp, sizeof(PROP_TYPE) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueInspectActiveVertices, CL_TRUE, CL_MAP_READ, 
			(void *)inRowPointerArray, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueInspectActiveVertices, CL_TRUE, CL_MAP_READ, 
			(void *)inDeg, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMMap(queueGather0, CL_TRUE, CL_MAP_READ, 
			(void *)inEdges, sizeof(int) * edgeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueGather0, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)vertexProp, sizeof(PROP_TYPE) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);	
	localStatus = clEnqueueSVMMap(queueGather0, CL_TRUE, CL_MAP_READ, 
			(void *)inEdgeProp, sizeof(PROP_TYPE) * edgeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMMap(queueGather1, CL_TRUE, CL_MAP_READ, 
			(void *)inEdges, sizeof(int) * edgeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueGather1, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)vertexProp, sizeof(PROP_TYPE) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);	
	localStatus = clEnqueueSVMMap(queueGather1, CL_TRUE, CL_MAP_READ, 
			(void *)inEdgeProp, sizeof(PROP_TYPE) * edgeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMMap(queueGather2, CL_TRUE, CL_MAP_READ, 
			(void *)inEdges, sizeof(int) * edgeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueGather2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)vertexProp, sizeof(PROP_TYPE) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);	
	localStatus = clEnqueueSVMMap(queueGather2, CL_TRUE, CL_MAP_READ, 
			(void *)inEdgeProp, sizeof(PROP_TYPE) * edgeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMMap(queueGather3, CL_TRUE, CL_MAP_READ, 
			(void *)inEdges, sizeof(int) * edgeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueGather3, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)vertexProp, sizeof(PROP_TYPE) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);	
	localStatus = clEnqueueSVMMap(queueGather3, CL_TRUE, CL_MAP_READ, 
			(void *)inEdgeProp, sizeof(PROP_TYPE) * edgeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMMap(queueApply0, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)vertexProp, sizeof(PROP_TYPE) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueApply0, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeVertices, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueApply0, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeVertexNum, sizeof(int) * 4, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	
	localStatus = clEnqueueSVMMap(queueApply1, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)vertexProp, sizeof(PROP_TYPE) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueApply1, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeVertices, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueApply1, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeVertexNum, sizeof(int) * 4, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMMap(queueApply2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)vertexProp, sizeof(PROP_TYPE) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueApply2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeVertices, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueApply2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeVertexNum, sizeof(int) * 4, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMMap(queueApply3, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)vertexProp, sizeof(PROP_TYPE) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueApply3, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeVertices, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueApply3, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)activeVertexNum, sizeof(int) * 4, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	for(auto s : statusVec){
		if(s != CL_SUCCESS){
			dumpError("Failed clEnqueueSVMMap", s);
			freeResources();
			return 1;	
		}
	}

	std::cout << "All the shared memory objects are mapped successfully." << std::endl;
	return CL_SUCCESS;

}

cl_int varUnmap(){

	cl_int localStatus;
	std::vector<cl_int> statusVec;
	localStatus = clEnqueueSVMUnmap(queueReadActiveVertices, (void *)activeVertices, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueReadActiveVertices, (void *)activeVertexNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)outRowPointerArray, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)outDeg, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)activeVertexNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	
	localStatus = clEnqueueSVMUnmap(queueTraverseNgb, (void *)outEdges, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueTraverseNgb, (void *)activeVertexNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMUnmap(queueStatusWrite, (void *)activeStatus, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMUnmap(queueInspectActiveVertices, (void *)activeStatus, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueInspectActiveVertices, (void *)vertexProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueInspectActiveVertices, (void *)inRowPointerArray, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueInspectActiveVertices, (void *)inDeg, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMUnmap(queueGather0, (void *)inEdges, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueGather0, (void *)vertexProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueGather0, (void *)inEdgeProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMUnmap(queueGather1, (void *)inEdges, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueGather1, (void *)vertexProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueGather1, (void *)inEdgeProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMUnmap(queueGather2, (void *)inEdges, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueGather2, (void *)vertexProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueGather2, (void *)inEdgeProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMUnmap(queueGather3, (void *)inEdges, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueGather3, (void *)vertexProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueGather3, (void *)inEdgeProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	localStatus = clEnqueueSVMUnmap(queueApply0, (void *)vertexProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueApply0, (void *)activeVertices, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueApply0, (void *)activeVertexNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	
	localStatus = clEnqueueSVMUnmap(queueApply1, (void *)vertexProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueApply1, (void *)activeVertices, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueApply1, (void *)activeVertexNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	
	localStatus = clEnqueueSVMUnmap(queueApply2, (void *)vertexProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueApply2, (void *)activeVertices, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueApply2, (void *)activeVertexNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	
	localStatus = clEnqueueSVMUnmap(queueApply3, (void *)vertexProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueApply3, (void *)activeVertices, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueApply3, (void *)activeVertexNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	
	for(auto s : statusVec){
		if(s != CL_SUCCESS) {
			dumpError("Failed clEnqueueSVMMap", s);
			freeResources();
			return 1;
		}
	}

	std::cout << "All the shared memory objects are unmapped successfully." << std::endl;
	return CL_SUCCESS;

}

cl_int setHardwareEnv(
		cl_uint &numPlatforms,
		cl_uint &numDevices
		){

	status = clGetPlatformIDs(1, &platform, &numPlatforms);
	if(status != CL_SUCCESS) {
		dumpError("Failed clGetPlatformIDs.", status);
		freeResources();
		return 1;
	}

	if(numPlatforms != 1) {
		printf("Found %d platforms!\n", numPlatforms);
		freeResources();
		return 1;
	}

	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &numDevices);
	if(status != CL_SUCCESS) {
		dumpError("Failed clGetDeviceIDs.", status);
		freeResources();
		return 1;
	}

	if(numDevices != 1) {
		printf("Found %d devices!\n", numDevices);
		freeResources();
		return 1;
	}

	context = clCreateContext(0, 1, &device, NULL, NULL, &status);
	if(status != CL_SUCCESS) {
		dumpError("Failed clCreateContext.", status);
		freeResources();
		return 1;
	}

	return CL_SUCCESS;
}

unsigned int setKernelEnv(){
	cl_int localStatus;
	queueReadActiveVertices = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateCommandQueue.", localStatus);
		freeResources();
		return 1;
	}

	queueReadNgbInfo = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateCommandQueue.", localStatus);
		freeResources();
		return 1;
	}

	queueTraverseNgb = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateCommandQueue.", localStatus);
		freeResources();
		return 1;
	}

	queueStatusWrite = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateCommandQueue.", localStatus);
		freeResources();
		return 1;
	}

	queueInspectActiveVertices = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateCommandQueue.", localStatus);
		freeResources();
		return 1;
	}

	queueGather0 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateCommandQueue.", localStatus);
		freeResources();
		return 1;
	}

	queueGather1 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateCommandQueue.", localStatus);
		freeResources();
		return 1;
	}

	queueGather2 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateCommandQueue.", localStatus);
		freeResources();
		return 1;
	}

	queueGather3 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateCommandQueue.", localStatus);
		freeResources();
		return 1;
	}

	queueApply0 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateCommandQueue.", localStatus);
		freeResources();
		return 1;
	}
	queueApply1 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateCommandQueue.", localStatus);
		freeResources();
		return 1;
	}
	queueApply2 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateCommandQueue.", localStatus);
		freeResources();
		return 1;
	}
	queueApply3 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateCommandQueue.", localStatus);
		freeResources();
		return 1;
	}

	size_t binSize = 0;
	unsigned char* binaryFile = loadBinaryFile("./graph_fpga.aocx", &binSize);
	if(!binaryFile) {
		dumpError("Failed loadBinaryFile.", localStatus);
		freeResources();
		return 1;
	}

	cl_int kernelStatus;
	program = clCreateProgramWithBinary(
			context, 1, &device, &binSize, 
			(const unsigned char**)&binaryFile, 
			&kernelStatus, &localStatus);

	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clCreateProgramWithBinary.", localStatus);
		freeResources();
		return 1;
	}

	delete [] binaryFile;

	localStatus = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	if(localStatus != CL_SUCCESS) {
		dumpError("Failed clBuildProgram.", localStatus);
		freeResources();
		return 1;
	}

	return CL_SUCCESS;

}

void processOnCPU(
		int               nodeNum,
		int               edgeNum
		)
{
	int itNum = 0;
	while(true){
		itNum++;

		// Scatter
		for(int i = 0; i < activeVertexNum[0]; i++){
			int vidx = activeVertices[i];
			int start = outRowPointerArray[vidx];
			int num = outDeg[vidx];
			for(int j = 0; j < num; j++){
				int ngbIdx = outEdges[start + j];
				activeStatus[ngbIdx] = 1;
			}
		}

		// gather and apply
		int count = 0;
		for(int i = 0; i < nodeNum; i++){
			if(activeStatus[i] == 1){
				activeStatus[i] = 0;
				int start = inRowPointerArray[i];
				PROP_TYPE vProp = vertexProp[i];
				int num = inDeg[i];

				// Gather
				PROP_TYPE tmpProp = vProp;
				for(int j = 0; j < num; j++){
					int ngbIdx = inEdges[start + j];
					PROP_TYPE eProp = inEdgeProp[start + j];
					PROP_TYPE uProp = vertexProp[ngbIdx];
					tmpProp = compute(uProp, eProp, tmpProp);
				}

				// Apply
				if(tmpProp != vProp){
					vertexProp[i] = tmpProp;
					activeVertices[count++] = i;
				}
			}
		}

		activeVertexNum[0] = count;
		std::cout << "iteration: " << itNum << " frontier = " << count << std::endl;

		if(count == 0){
			break;
		}
	}
}


// Read graph from edge based files
Graph* createGraph(const std::string &gName, const std::string &mode){
	Graph* gptr;
	std::string dir;
	if(mode == "harp") dir = "../../graph-data/";
	else if(mode == "sim") dir = "/data/DATA/liucheng/graph-data/";
	else {
		std::cout << "unknown execution environment." << std::endl;
		exit(0);
	}

	if(gName == "dblp"){
		gptr = new Graph(dir + "dblp.ungraph.txt");
	}
	else if(gName == "youtube"){
		gptr = new Graph(dir + "youtube.ungraph.txt");
	}
	else if(gName == "lj"){
		gptr = new Graph(dir + "lj.ungraph.txt");
	}
	else if(gName == "pokec"){
		gptr = new Graph(dir + "pokec-relationships.txt");
	}
	else if(gName == "wiki-talk"){
		gptr = new Graph(dir + "wiki-Talk.txt");
	}
	else if(gName == "lj1"){
		gptr = new Graph(dir + "LiveJournal1.txt");
	}
	else if(gName == "orkut"){
		gptr = new Graph(dir + "orkut.ungraph.txt");
	}
	else if(gName == "rmat-21-32"){
		gptr = new Graph(dir + "rmat-21-32.txt");
	}
	else if(gName == "rmat-19-32"){
		gptr = new Graph(dir + "rmat-19-32.txt");
	}
	else if(gName == "rmat-21-128"){
		gptr = new Graph(dir + "rmat-21-128.txt");
	}
	else if(gName == "twitter"){
		gptr = new Graph(dir + "twitter_rv.txt");
	}
	else if(gName == "friendster"){
		gptr = new Graph(dir + "friendster.ungraph.txt");
	}
	else if(gName == "example"){
		gptr = new Graph(dir + "rmat-1k-10k.txt");
	}
	else{
		std::cout << "Unknown graph name." << std::endl;
		exit(EXIT_FAILURE);
	}

	return gptr;
}

int align(int num, int dataWidth, int alignedWidth){
	if(dataWidth > alignedWidth){
		std::cout << "Aligning to smaller data width is not supported." << std::endl;
		return -1;
	}
	else{
		int wordNum = alignedWidth / dataWidth;
		int alignedNum = ((num - 1)/wordNum + 1) * wordNum;
		return alignedNum;
	}
}


// Initialize the memory objects with the graph data
void processInit(CSR* csr, const int &nodeNum, 
		const int &edgeNum, const int &source){
	for(int i = 0; i < nodeNum; i++){
		if(i < csr->vertexNum){	
			inRowPointerArray[i] = csr->rpai[i];
			inDeg[i] = csr->rpai[i + 1] - csr->rpai[i];
			outRowPointerArray[i] = csr->rpao[i];
			outDeg[i] = csr->rpao[i + 1] - csr->rpao[i];
		}
		else{
			inRowPointerArray[i] = 0;
			inDeg[i] = 0;
			outRowPointerArray[i] = 0;
			outDeg[i] = 0;
		}

		vertexProp[i] = MAX_PROP;
		activeStatus[i] = 0;	

	}

	for(int i = 0; i < edgeNum; i++){
		inEdges[i] = csr->ciai[i];
		inEdgeProp[i] = 1; // when it is BFS
		outEdges[i] = csr->ciao[i];
	}

	vertexProp[source] = 0;
	activeVertexNum[0] = 1;
	activeVertices[0] = source;
}

// Reset part of the input data for repeated execution.
void processReset(const int &nodeNum, const int &source){
	for(int i = 0; i < nodeNum; i++){
		vertexProp[i] = MAX_PROP;
		activeStatus[i] = 0;
	}
	vertexProp[source] = 0;
	activeVertexNum[0] = 1;
	activeVertexNum[1] = 0;
	activeVertexNum[2] = 0;
	activeVertexNum[3] = 0;
	activeVertices[0] = source;
}

int main(int argc, char ** argv){

	cl_uint numPlatforms;
	cl_uint numDevices;

	status = setHardwareEnv(numPlatforms, numDevices);
	std::cout << "Creating device memory objects." << std::endl;

	int source = 365723;
	std::string mode = "harp";
	std::string graphName = "youtube";
	if(graphName == "youtube")    source = 320872;
	if(graphName == "lj1")        source = 3928512;
	if(graphName == "pokec")      source = 182045;
	if(graphName == "rmat-19-32") source = 104802;
	if(graphName == "rmat-21-32") source = 365723;

	Graph* gptr = createGraph(graphName, mode);
	CSR* csr = new CSR(*gptr);
	free(gptr);

	std::cout << "Graph is loaded." << std::endl;
	int nodeNum = align(csr->vertexNum, 8, 8*1024); 
	int edgeNum = csr->ciao.size();
	std::cout << "node num: " << csr->vertexNum << " is aligned to " << nodeNum << std::endl;
	std::cout << "edge num: " << edgeNum << std::endl;
	int offset1 = nodeNum/4;
	int offset2 = nodeNum/2;
	int offset3 = nodeNum*3/4;

	// Result of CPU and FPGA
	PROP_TYPE* CPUResult = (PROP_TYPE*) malloc(sizeof(PROP_TYPE) * nodeNum);
	PROP_TYPE* FPGAResult = (PROP_TYPE*) malloc(sizeof(PROP_TYPE) * nodeNum);

	// Declare shared memory objects
	vertexProp         = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * nodeNum, 1024); 
	inRowPointerArray  = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
	inDeg              = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
	inEdges            = (int*) clSVMAllocAltera(context, 0, sizeof(int) * edgeNum, 1024);
	inEdgeProp         = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * edgeNum, 1024);
	outRowPointerArray = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
	outDeg             = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
	outEdges           = (int*) clSVMAllocAltera(context, 0, sizeof(int) * edgeNum, 1024);
	activeVertices     = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024); 

	// 0-->vertex is inactive, 1--> vertex is active
	activeStatus       = (STATUS_TYPE*) clSVMAllocAltera(context, 0, sizeof(STATUS_TYPE) * nodeNum, 1024); 
	activeVertexNum    = (int*) clSVMAllocAltera(context, 0, sizeof(int)*4, 1024);

	if(!inRowPointerArray || !outDeg || !outRowPointerArray || !inDeg || !activeVertices
	   || !activeStatus || !vertexProp || !inEdges || !inEdgeProp || !outEdges || !activeVertexNum) 
	{
		dumpError("Failed to allocate buffers.", status);
		freeResources();
		return 1;	
	}
	std::cout << "Memory allocation is done." << std::endl;

	// Graph processing on CPU
	processInit(csr, nodeNum, edgeNum, source);	
	processOnCPU(nodeNum, edgeNum);

	for(int i = 0; i < nodeNum; i++){
		CPUResult[i] = vertexProp[i];
	}

	processReset(nodeNum, source);
	std::cout << "Graph processing on CPU is done." << std::endl;

	// Graph processing on FPGA
	status = setKernelEnv();
	std::cout << "Creating graph processing kernels." << std::endl;
	{
		readActiveVertices = clCreateKernel(program, "readActiveVertices", &status);
		if(status != CL_SUCCESS) {
			dumpError("Failed clCreateKernel read active vertices.", status);
			freeResources();
			return 1;
		} 

		readNgbInfo = clCreateKernel(program, "readNgbInfo", &status);
		if(status != CL_SUCCESS) {
			dumpError("Failed clCreateKernel status readNgbInfo.", status);
			freeResources();
			return 1;
		}

		traverseNgb = clCreateKernel(program, "traverseNgb", &status);
		if(status != CL_SUCCESS) {
			dumpError("Failed clCreateKernel traverse Ngb.", status);
			freeResources();
			return 1;
		}

		statusWrite = clCreateKernel(program, "statusWrite", &status);
		if(status != CL_SUCCESS) {
			dumpError("Failed clCreateKernel status write.", status);
			freeResources();
			return 1;
		}

		inspectActiveVertices = clCreateKernel(program, "inspectActiveVertices", &status);
		if(status != CL_SUCCESS){
			dumpError("Failed clCreateKernel inspect active vertices.", status);
			freeResources();
			return 1;
		}

		gather0 = clCreateKernel(program, "gather0", &status);
		if(status != CL_SUCCESS){
			dumpError("Failed clCreateKernel gather0.", status);
			freeResources();
			return 1;
		}
		gather1 = clCreateKernel(program, "gather1", &status);
		if(status != CL_SUCCESS){
			dumpError("Failed clCreateKernel gather1.", status);
			freeResources();
			return 1;
		}
		gather2 = clCreateKernel(program, "gather2", &status);
		if(status != CL_SUCCESS){
			dumpError("Failed clCreateKernel gather2.", status);
			freeResources();
			return 1;
		}
		gather3 = clCreateKernel(program, "gather3", &status);
		if(status != CL_SUCCESS){
			dumpError("Failed clCreateKernel gather3.", status);
			freeResources();
			return 1;
		}

		apply0 = clCreateKernel(program, "apply0", &status);
		if(status != CL_SUCCESS){
			dumpError("Failed clCreateKernel apply0.", status);
			freeResources();
			return 1;
		}
		apply1 = clCreateKernel(program, "apply1", &status);
		if(status != CL_SUCCESS){
			dumpError("Failed clCreateKernel apply1.", status);
			freeResources();
			return 1;
		}
		apply2 = clCreateKernel(program, "apply2", &status);
		if(status != CL_SUCCESS){
			dumpError("Failed clCreateKernel apply2.", status);
			freeResources();
			return 1;
		}
		apply3 = clCreateKernel(program, "apply3", &status);
		if(status != CL_SUCCESS){
			dumpError("Failed clCreateKernel apply3.", status);
			freeResources();
			return 1;
		}

		// set kernel arguments
		std::cout << "Set kernel arguments." << std::endl;
		clSetKernelArgSVMPointerAltera(readActiveVertices, 0, (void*)activeVertices);
		clSetKernelArgSVMPointerAltera(readActiveVertices, 1, (void*)activeVertexNum);
		clSetKernelArg(readActiveVertices, 2, sizeof(int), (void*)&offset1);
		clSetKernelArg(readActiveVertices, 3, sizeof(int), (void*)&offset2);
		clSetKernelArg(readActiveVertices, 4, sizeof(int), (void*)&offset3);

		clSetKernelArgSVMPointerAltera(readNgbInfo, 0, (void*)outRowPointerArray);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 1, (void*)outDeg);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 2, (void*)activeVertexNum);

		clSetKernelArgSVMPointerAltera(traverseNgb, 0, (void*)outEdges);
		clSetKernelArgSVMPointerAltera(traverseNgb, 1, (void*)activeVertexNum);

		clSetKernelArgSVMPointerAltera(statusWrite, 0, (void*)activeStatus);

		clSetKernelArgSVMPointerAltera(inspectActiveVertices, 0, (void*)activeStatus);
		clSetKernelArgSVMPointerAltera(inspectActiveVertices, 1, (void*)vertexProp);
		clSetKernelArgSVMPointerAltera(inspectActiveVertices, 2, (void*)inRowPointerArray);
		clSetKernelArgSVMPointerAltera(inspectActiveVertices, 3, (void*)inDeg);
		clSetKernelArg(inspectActiveVertices, 4, sizeof(int), (void*)&nodeNum);

		clSetKernelArgSVMPointerAltera(gather0, 0, (void*)inEdges);
		clSetKernelArgSVMPointerAltera(gather0, 1, (void*)vertexProp);
		clSetKernelArgSVMPointerAltera(gather0, 2, (void*)inEdgeProp);

		clSetKernelArgSVMPointerAltera(gather1, 0, (void*)inEdges);
		clSetKernelArgSVMPointerAltera(gather1, 1, (void*)vertexProp);
		clSetKernelArgSVMPointerAltera(gather1, 2, (void*)inEdgeProp);

		clSetKernelArgSVMPointerAltera(gather2, 0, (void*)inEdges);
		clSetKernelArgSVMPointerAltera(gather2, 1, (void*)vertexProp);
		clSetKernelArgSVMPointerAltera(gather2, 2, (void*)inEdgeProp);

		clSetKernelArgSVMPointerAltera(gather3, 0, (void*)inEdges);
		clSetKernelArgSVMPointerAltera(gather3, 1, (void*)vertexProp);
		clSetKernelArgSVMPointerAltera(gather3, 2, (void*)inEdgeProp);

		clSetKernelArgSVMPointerAltera(apply0, 0, (void*)vertexProp);
		clSetKernelArgSVMPointerAltera(apply0, 1, (void*)activeVertices);
		clSetKernelArgSVMPointerAltera(apply0, 2, (void*)activeVertexNum);

		clSetKernelArgSVMPointerAltera(apply1, 0, (void*)vertexProp);
		clSetKernelArgSVMPointerAltera(apply1, 1, (void*)activeVertices);
		clSetKernelArgSVMPointerAltera(apply1, 2, (void*)activeVertexNum);
		clSetKernelArg(apply1, 3, sizeof(int), (void*)&offset1);

		clSetKernelArgSVMPointerAltera(apply2, 0, (void*)vertexProp);
		clSetKernelArgSVMPointerAltera(apply2, 1, (void*)activeVertices);
		clSetKernelArgSVMPointerAltera(apply2, 2, (void*)activeVertexNum);
		clSetKernelArg(apply2, 3, sizeof(int), (void*)&offset2);

		clSetKernelArgSVMPointerAltera(apply3, 0, (void*)vertexProp);
		clSetKernelArgSVMPointerAltera(apply3, 1, (void*)activeVertices);
		clSetKernelArgSVMPointerAltera(apply3, 2, (void*)activeVertexNum);
		clSetKernelArg(apply3, 3, sizeof(int), (void*)&offset3);

		varMap(nodeNum, edgeNum);
		std::cout << "End of kernel argument setup." << std::endl;
	}

	// Run BFS iteration
	std::cout << "start the processing iterations." << std::endl;
	cl_event eventReadActiveVertices;
	cl_event eventReadNgbInfo;
	cl_event eventTraverseNgb;
	cl_event eventStatusWrite;
	cl_event eventInspectActiveVertices;
	cl_event eventGather0;
	cl_event eventGather1;
	cl_event eventGather2;
	cl_event eventGather3;
	cl_event eventApply0;
	cl_event eventApply1;
	cl_event eventApply2;
	cl_event eventApply3;

	auto startTime = std::chrono::high_resolution_clock::now();
	double k0Time = 0; 
	double k1Time = 0;
	int itNum = 0;
	while(activeVertexNum[0] > 0){
		itNum++;
		auto c0 = std::chrono::high_resolution_clock::now();
		status = clEnqueueTask(queueReadActiveVertices, readActiveVertices, 0, NULL, &eventReadActiveVertices);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch scatter.", status);
			freeResources();
			return 1;
		}	

		status = clEnqueueTask(queueReadNgbInfo, readNgbInfo, 0, NULL, &eventReadNgbInfo);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch scatter.", status);
			freeResources();
			return 1;
		}

		status = clEnqueueTask(queueTraverseNgb, traverseNgb, 0, NULL, &eventTraverseNgb);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch scatter.", status);
			freeResources();
			return 1;
		}	

		status = clEnqueueTask(queueStatusWrite, statusWrite, 0, NULL, &eventStatusWrite);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch status write.", status);
			freeResources();
			return 1;
		}

		clFinish(queueReadActiveVertices);
		clFinish(queueReadNgbInfo);
		clFinish(queueTraverseNgb);
		clFinish(queueStatusWrite);

		auto c1 = std::chrono::high_resolution_clock::now();
		status = clEnqueueTask(queueInspectActiveVertices, 
				inspectActiveVertices, 0, NULL, &eventInspectActiveVertices);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch gatherApply.", status);
			freeResources();
			return 1;
		}
		status = clEnqueueTask(queueGather0, gather0, 0, NULL, &eventGather0);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch gather.", status);
			freeResources();
			return 1;
		}
		status = clEnqueueTask(queueGather1, gather1, 0, NULL, &eventGather1);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch gather.", status);
			freeResources();
			return 1;
		}
		status = clEnqueueTask(queueGather2, gather2, 0, NULL, &eventGather2);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch gather.", status);
			freeResources();
			return 1;
		}
		status = clEnqueueTask(queueGather3, gather3, 0, NULL, &eventGather3);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch gather.", status);
			freeResources();
			return 1;
		}

		status = clEnqueueTask(queueApply0, apply0, 0, NULL, &eventApply0);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch apply0.", status);
			freeResources();
			return 1;
		}
		status = clEnqueueTask(queueApply1, apply1, 0, NULL, &eventApply1);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch apply1.", status);
			freeResources();
			return 1;
		}
		status = clEnqueueTask(queueApply2, apply2, 0, NULL, &eventApply2);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch apply2.", status);
			freeResources();
			return 1;
		}
		status = clEnqueueTask(queueApply3, apply3, 0, NULL, &eventApply3);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch apply3.", status);
			freeResources();
			return 1;
		}

		clFinish(queueInspectActiveVertices);
		clFinish(queueGather0);
		clFinish(queueGather1);
		clFinish(queueGather2);
		clFinish(queueGather3);
		clFinish(queueApply0);
		clFinish(queueApply1);
		clFinish(queueApply2);
		clFinish(queueApply3);

		auto c2 = std::chrono::high_resolution_clock::now();
		k0Time += std::chrono::duration_cast<std::chrono::milliseconds>(c1 - c0).count();
		k1Time += std::chrono::duration_cast<std::chrono::milliseconds>(c2 - c1).count();
		int totalActiveVertexNum = activeVertexNum[0] + activeVertexNum[1] + activeVertexNum[2] + activeVertexNum[3];
		std::cout << "BFS level = " << itNum << " frontier = " << totalActiveVertexNum << std::endl;
	}
	auto endTime = std::chrono::high_resolution_clock::now();
	auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << "BFS runtime: " << totalTime << std::endl;
	std::cout << "scatter runtime: " << k0Time << std::endl;
	std::cout << "gatherApply runtime: " << k1Time << std::endl;

	// Copy FPGA result
	for(int i = 0; i < nodeNum; i++){
		FPGAResult[i] = vertexProp[i];
	}

	// Verification step	
	bool passed = true; 
	for(int i = 0; i < nodeNum; i++){
		if(CPUResult[i] != FPGAResult[i]){
			passed =  false;
			break;
		}
	}
	std::cout << "Verification: " << (passed? "Passed": "Failed") << std::endl;

	varUnmap();
	freeResources();
	return 0;
}
