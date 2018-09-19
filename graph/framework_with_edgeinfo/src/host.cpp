#include <stdlib.h>
#include <malloc.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <cstdio>
#include <math.h>
#include <ctime>
#include "config.h"
#include "graph.h"
#include "safequeue.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <tr1/unordered_map>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <pthread.h>
#include<float.h>
#include <sys/time.h>

using namespace std::tr1;
using namespace aocl_utils;
using namespace std;

static cl_command_queue queueReadActiveVertices;
static cl_command_queue queueReadNgbInfo;
static cl_command_queue queueProcessEdge;
static cl_command_queue queueVertexApply;
static cl_command_queue queueActiveVertexOutput;


static cl_kernel readActiveVertices;
static cl_kernel readNgbInfo;
static cl_kernel processEdge;
static cl_kernel vertexApply; 
static cl_kernel activeVertexOutput; 



static cl_program program;

static cl_int status;
static PROP_TYPE* vertexProp;
static PROP_TYPE* tmpVertexProp;
static int* rpa;
static int* blkRpa;
static int* blkRpaNum;
static int* outDeg;
static int* blkRpaLast;
static int* cia;
static int* cia_padding;
static int* vertexScore; // global - like outDeg
static PROP_TYPE* edgeProp;
static int* blkCia;
static PROP_TYPE* blkEdgeProp;
static int* activeVertices;
static int* activeVertexNum;
static int* blkActiveVertices;
static int* blkActiveVertexNum;
static int* itNum;
static int* fpgaIterNum;
static int* blkEdgeNum;
static int* blkVertexNum;
static int* eop; // end of processing
static int* srcRange;
static int* sinkRange;
static int* error;
int vertexNum;
int edgeNum; 
int blkNum;
int base_score;
CSR* csr;
std::vector<CSR_BLOCK*> blkVec;
int processing_edges = 0;
int edge_replic_factor = 2;
//Notify the FPGA thread
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;

static int fpga_partition_x = 0;
static int fpga_partition_y = 0;
static int max_partition_degree = 0; 
typedef int ScoreT;
double fpga_runtime =0;
#define AOCL_ALIGNMENT 64
#define THREAD_NUM 1
#define MAX_ITER 1
#define BFS
#define DEBUG
#ifdef PR 
#define PROP_TYPE int
#define kDamp 0.85f
#define epsilon  0.001f
#endif
#ifdef CC 
#define PROP_TYPE int
#endif
#define INT2FLOAT (pow(2,23))
int float2int(float a){
	return (int)(a * INT2FLOAT);
}

float int2float(int a){
	return ((float)a / INT2FLOAT);
}

#define RAND_RANGE(N) ((float)rand() / ((float)RAND_MAX + 1) * (N))

static void freeResources(){
	if(readActiveVertices) clReleaseKernel(readActiveVertices);  
	if(readNgbInfo)        clReleaseKernel(readNgbInfo);  
	if(processEdge)        clReleaseKernel(processEdge);  
	if(program)            clReleaseProgram(program);

	if(queueReadActiveVertices) clReleaseCommandQueue(queueReadActiveVertices);
	if(queueReadNgbInfo)        clReleaseCommandQueue(queueReadNgbInfo);
	if(queueProcessEdge)        clReleaseCommandQueue(queueProcessEdge);

	// We set all the objects to be shared by CPU and FPGA, though
	// some of them are only used by CPU process.
	if(vertexProp)         clSVMFreeAltera(context, vertexProp);
	if(tmpVertexProp)      clSVMFreeAltera(context, tmpVertexProp);
	if(rpa)                clSVMFreeAltera(context, rpa);
	if(blkRpa)             clSVMFreeAltera(context, blkRpa);
	if(outDeg)             clSVMFreeAltera(context, outDeg);
	if(cia)                clSVMFreeAltera(context, cia);
	if(edgeProp)           clSVMFreeAltera(context, edgeProp);
	if(blkCia)             clSVMFreeAltera(context, blkCia);
	if(blkEdgeProp)        clSVMFreeAltera(context, blkEdgeProp);
	if(activeVertices)     clSVMFreeAltera(context, activeVertices);
	if(blkActiveVertices)  clSVMFreeAltera(context, blkActiveVertices);
	if(activeVertexNum)    clSVMFreeAltera(context, activeVertexNum);
	if(blkActiveVertexNum) clSVMFreeAltera(context, blkActiveVertexNum);
	if(itNum)              clSVMFreeAltera(context, itNum);
	if(blkVertexNum)       clSVMFreeAltera(context, blkVertexNum);
	if(blkEdgeNum)         clSVMFreeAltera(context, blkEdgeNum);
	if(eop)                clSVMFreeAltera(context, eop);

	if(context)            clReleaseContext(context);
}

void cleanup(){}

void dumpError(const char *str) {
	printf("Error: %s\n", str);
	freeResources();
}

void checkStatus(const char *str) {
	if(status != 0 || status != CL_SUCCESS){
		dumpError(str);
		printf("Error code: %d\n", status);
	}
}


void setKernelEnv(){
	queueReadActiveVertices = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkStatus("Failed clCreateCommandQueue of queueReadActiveVertices.");
	queueReadNgbInfo = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkStatus("Failed clCreateCommandQueue of queueReadNgbInfo.");
	queueProcessEdge = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkStatus("Failed clCreateCommandQueue of queueProcessEdge.");
	queueVertexApply = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkStatus("Failed clCreateCommandQueue of queueProcessEdge.");
	queueActiveVertexOutput = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkStatus("Failed clCreateCommandQueue of queueProcessEdge.");

	size_t binSize = 0;
	unsigned char* binaryFile = loadBinaryFile("./graph_fpga.aocx", &binSize);
	if(!binaryFile) dumpError("Failed loadBinaryFile.");

	program = clCreateProgramWithBinary(
		context, 1, &device, &binSize, (const unsigned char**)&binaryFile, 
		&status, &status);
	if(status != CL_SUCCESS) delete [] binaryFile;
	checkStatus("Failed clCreateProgramWithBinary of program.");

	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkStatus("Failed clBuildProgram.");

	std::cout << "set kernel env." << std::endl;
}
// access FPGA using the main thread 
void setHardwareEnv(){
	cl_uint numPlatforms;
	cl_uint numDevices;
	status = clGetPlatformIDs(1, &platform, &numPlatforms);
	checkStatus("Failed clGetPlatformIDs.");
	printf("Found %d platforms!\n", numPlatforms);

	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &numDevices);
	checkStatus("Failed clGetDeviceIDs.");
	printf("Found %d devices!\n", numDevices);

	context = clCreateContext(0, 1, &device, NULL, NULL, &status);
	checkStatus("Failed clCreateContext.");
}

Graph* createGraph(const std::string &gName, const std::string &mode){
	Graph* gptr;
	std::string dir;
	if(mode == "harp") dir = "./";
	else if(mode == "sim") dir = "/data/DATA/liucheng/graph-data/";
	else if(mode == "rmat") dir = "/data/DATA/xinyu/work/krongen/";
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
	else if(gName == "rmat-12-4"){
		gptr = new Graph(dir + "rmat-12-4.txt");
	}
	else if(gName == "rmat-23-4"){
		gptr = new Graph(dir + "rmat-23-4.txt");
	}
	else if(gName == "rmat-23-16"){
		gptr = new Graph(dir + "rmat-23-16.txt");
	}
	else if(gName == "wiki-Talk"){
		gptr = new Graph(dir + "soc-wiki-Talk-dir.mtx");
	}
	else if(gName == "orkut"){
		gptr = new Graph(dir + "soc-orkut-dir.edges");
	}
	else if(gName == "twitter-higgs"){
		gptr = new Graph(dir + "soc-twitter-higgs.edges");
	}
	else if(gName == "mouse-gene"){
		gptr = new Graph(dir + "bio-mouse-gene.edges");
	}
	else if(gName == "flixster"){
		gptr = new Graph(dir + "soc-flixster.mtx");
	}
	else{
		std::cout << "Unknown graph name." << std::endl;
		exit(EXIT_FAILURE);
	}

	return gptr;
}


void globalVarInit(
	CSR* csr, 
	const int &vertexNum, 
	const int &edgeNum
	)
{
	printf("blkNum %d \n", blkNum);

	vertexProp         = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * vertexNum, 1024); 
	tmpVertexProp      = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * vertexNum, 1024);
	activeVertices     = (int*) clSVMAllocAltera(context, 0, sizeof(int) * vertexNum, 1024);

	outDeg             = (int*) clSVMAllocAltera(context, 0, sizeof(int) * vertexNum, 1024);
	vertexScore        = (int*) clSVMAllocAltera(context, 0, sizeof(int) * vertexNum, 1024);

	blkRpa             = (int*) clSVMAllocAltera(context, 0, sizeof(int) * (blkNum * BLK_SIZE * blkNum), 1024); // because it is from 2 dimension to 1 dimension, so * 2
	//outdeg_padding     = (int*) clSVMAllocAltera(context, 0, sizeof(int) * (ceil(vertexNum / BLK_SIZE) * BLK_SIZE * blkNum), 1024);
	blkActiveVertices  = (int*) clSVMAllocAltera(context, 0, sizeof(int) * (blkNum * BLK_SIZE * blkNum), 1024);
	
	//blkRpaNum          = (int*) clSVMAllocAltera(context, 0, sizeof(int) * (vertexNum), 1024);
	blkRpaLast    	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * blkNum * blkNum, 1024);
	cia_padding        = (int*) clSVMAllocAltera(context, 0, sizeof(int) * edgeNum * edge_replic_factor * 2, 1024);	
	
	blkEdgeProp        = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * 1, 1024); ///////////////////////////////// problem here
	blkActiveVertexNum = (int*) clSVMAllocAltera(context, 0, sizeof(int) * blkNum * blkNum, 1024); // The MAX partitions FPGA need to process
	blkEdgeNum     	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * blkNum * blkNum, 1024); 
	blkVertexNum 	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * blkNum * blkNum, 1024); 
	srcRange 	   	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * blkNum * blkNum * 2, 1024);  
	sinkRange 	   	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * blkNum * blkNum * 2, 1024);  
	itNum     		   = (int*) clSVMAllocAltera(context, 0, sizeof(int), 1024);
	fpgaIterNum        = (int*) clSVMAllocAltera(context, 0, sizeof(int), 1024);  
	eop  		       = (int*) clSVMAllocAltera(context, 0, sizeof(int), 1024); 
	error 	       	   = (int*) clSVMAllocAltera(context, 0, sizeof(int), 1024); 

	
	rpa                = (int*)malloc( sizeof(int) * (vertexNum + 1)); //= (int*) clSVMAllocAltera(context, 0, sizeof(int) * (vertexNum + 1), 1024);
	cia           	   = (int*)malloc( sizeof(int) * edgeNum * 2); //= (int*) clSVMAllocAltera(context, 0, sizeof(int) * edgeNum * 2, 1024); // undirection graph
	edgeProp           = (int*)malloc( sizeof(int) * edgeNum );//= (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * edgeNum, 1024);
	activeVertexNum    = (int*)malloc( sizeof(int) * 1); //= (int*) clSVMAllocAltera(context, 0, sizeof(int), 1024);
	
	

	if(!vertexProp || !tmpVertexProp || !rpa || !blkRpa 
		|| !outDeg || !cia || !edgeProp 
		|| !activeVertices|| !activeVertexNum 
		|| !blkActiveVertices || !blkActiveVertexNum 
		|| !itNum || !blkEdgeNum || !blkVertexNum || !eop 
		|| !srcRange || !sinkRange
		){
		dumpError("Failed to allocate buffers.");
	}
	else{
		printf("SVMAllocAltera Done! \n");
	}

	for(int i = 0; i < vertexNum; i++){
		if(i < csr->vertexNum){ // 'vertexNum' may be aligned.	
			rpa[i] = csr->rpao[i];
			outDeg[i] = csr->rpao[i + 1] - csr->rpao[i];
	}
	else{
		rpa[i] = 0;
		outDeg[i] = 0;
		}
	}
	rpa[vertexNum] = csr->rpao[vertexNum]; 
	for(int i = 0; i < edgeNum; i++){
		cia[i] = csr->ciao[i];
		edgeProp[i] = rand()%100;
	}
}
void kernelVarMap(
	const int &vertexNum, 
	const int &edgeNum
	)
{
	status = clEnqueueSVMMap(queueReadActiveVertices, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)blkActiveVertices, sizeof(int) * blkNum * blkNum, 0, NULL, NULL); 
	checkStatus("enqueue activeVertices.");
	status = clEnqueueSVMMap(queueReadActiveVertices, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)blkActiveVertexNum, sizeof(int) * blkNum * blkNum, 0, NULL, NULL); 
	checkStatus("enqueue activeVertexNum.");
		status = clEnqueueSVMMap(queueReadActiveVertices, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)fpgaIterNum, sizeof(int), 0, NULL, NULL); 
	checkStatus("enqueue activeVertexNum.");

	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ, 
		(void *)blkRpa, sizeof(int) * (blkNum * BLK_SIZE * blkNum), 0, NULL, NULL); 
	checkStatus("enqueue blkRpa.");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ, 
		(void *)blkRpaLast, sizeof(int) * blkNum * blkNum, 0, NULL, NULL); 
	checkStatus("enqueue blkCia");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)blkEdgeProp, sizeof(PROP_TYPE) * edgeNum, 0, NULL, NULL); 
	checkStatus("enqueue blkEdgeProp");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)vertexProp, sizeof(int) * vertexNum, 0, NULL, NULL); 
	checkStatus("enqueue outDeg");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)blkActiveVertexNum, sizeof(int) * blkNum * blkNum, 0, NULL, NULL); 
	checkStatus("enqueue blkActiveVertexNum");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)blkVertexNum, sizeof(int) * blkNum * blkNum, 0, NULL, NULL); 
	checkStatus("enqueue blkVertexNum");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)blkEdgeNum, sizeof(int) * blkNum * blkNum, 0, NULL, NULL); 
	checkStatus("enqueue blkEdgeNum.");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)srcRange, sizeof(int) * blkNum * blkNum * 2, 0, NULL, NULL); 
	checkStatus("enqueue srcRange.");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)fpgaIterNum, sizeof(int), 0, NULL, NULL); 
	checkStatus("enqueue itNum.");

	// traverse neighbor
	status = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ, 
		(void *)vertexProp, sizeof(int) * vertexNum, 0, NULL, NULL); 
	checkStatus("enqueue vertexProp.");
	status = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ, 
		(void *)tmpVertexProp, sizeof(int) * vertexNum, 0, NULL, NULL); 
	checkStatus("enqueue tmpVertexProp.");
	status = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)eop, sizeof(int), 0, NULL, NULL); 
	checkStatus("enqueue activeVertexNum.");
	status = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)fpgaIterNum, sizeof(int), 0, NULL, NULL); 
	checkStatus("enqueue semaphore.");
	status = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)srcRange, sizeof(int) * blkNum * blkNum * 2, 0, NULL, NULL); 
	checkStatus("enqueue srcRange");
	status = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)sinkRange, sizeof(int) * blkNum * blkNum * 2, 0, NULL, NULL); 
	checkStatus("enqueue sinkRange.");
	std::cout << "All the shared memory objects are mapped successfully." << std::endl;
}
void createKernels(
	const int &vertexNum, 
	const int &edgeNum
	)
{
	std::cout << "Creating graph processing kernels." << std::endl;
	readActiveVertices = clCreateKernel(program, "readActiveVertices", &status);
	checkStatus("Failed clCreateKernel read active vertices.");
	readNgbInfo = clCreateKernel(program, "readNgbInfo", &status);
	checkStatus("Failed clCreateKernel status readNgbInfo.");
	processEdge = clCreateKernel(program, "processEdge", &status);
	checkStatus("Failed clCreateKernel processEdge.");
	vertexApply = clCreateKernel(program, "vertexApply", &status);
	checkStatus("Failed clCreateKernel processEdge.");
	activeVertexOutput = clCreateKernel(program, "activeVertexOutput", &status);
	checkStatus("Failed clCreateKernel processEdge.");

	clSetKernelArgSVMPointerAltera(readActiveVertices, 0, (void*)blkActiveVertices);	
	clSetKernelArgSVMPointerAltera(readActiveVertices, 1, (void*)blkActiveVertexNum);
	clSetKernelArgSVMPointerAltera(readActiveVertices, 2, (void*)fpgaIterNum);


	clSetKernelArgSVMPointerAltera(readNgbInfo, 0, (void*)blkRpa);
	clSetKernelArgSVMPointerAltera(readNgbInfo, 1, (void*)blkRpaLast);
	clSetKernelArgSVMPointerAltera(readNgbInfo, 2, (void*)cia_padding);
	clSetKernelArgSVMPointerAltera(readNgbInfo, 3, (void*)blkEdgeProp);
	clSetKernelArgSVMPointerAltera(readNgbInfo, 4, (void*)vertexProp);	// vertexScore - PR
	clSetKernelArgSVMPointerAltera(readNgbInfo, 5, (void*)blkActiveVertexNum);
	clSetKernelArgSVMPointerAltera(readNgbInfo, 6, (void*)blkVertexNum);
	clSetKernelArgSVMPointerAltera(readNgbInfo, 7, (void*)blkEdgeNum);
	clSetKernelArgSVMPointerAltera(readNgbInfo, 8, (void*)srcRange);
	clSetKernelArgSVMPointerAltera(readNgbInfo, 9, (void*)fpgaIterNum);

	clSetKernelArgSVMPointerAltera(processEdge, 0, (void*)vertexProp);
	clSetKernelArgSVMPointerAltera(processEdge, 1, (void*)tmpVertexProp);
	clSetKernelArgSVMPointerAltera(processEdge, 2, (void*)eop);
	clSetKernelArgSVMPointerAltera(processEdge, 3, (void*)fpgaIterNum);
	clSetKernelArgSVMPointerAltera(processEdge, 4, (void*)srcRange);
	clSetKernelArgSVMPointerAltera(processEdge, 5, (void*)sinkRange);

	clSetKernelArgSVMPointerAltera(vertexApply, 0, (void*)vertexProp);
	clSetKernelArgSVMPointerAltera(vertexApply, 1, (void*)tmpVertexProp);
	clSetKernelArgSVMPointerAltera(vertexApply, 2, (void*)activeVertices);
	clSetKernelArgSVMPointerAltera(vertexApply, 3, (void*)outDeg);
	clSetKernelArgSVMPointerAltera(vertexApply, 4, (void*)vertexScore);
	clSetKernelArgSVMPointerAltera(vertexApply, 5, (void*)error);
	clSetKernelArg(vertexApply, 6, sizeof(int), (void*)&vertexNum);
	clSetKernelArg(vertexApply, 7, sizeof(int), (void*)&base_score);

	clSetKernelArgSVMPointerAltera(activeVertexOutput, 0, (void*)activeVertices);
	clSetKernelArgSVMPointerAltera(activeVertexOutput, 1, (void*)error);
}

#ifdef BFS
void singleThreadSWProcessing(
		CSR* csr,
		std::vector<CSR_BLOCK*> &blkVec, 
		PROP_TYPE* swProp, 
		const int &blkNum,
		const int &vertexNum,
		const int &source
		)
{
	itNum[0] = 0;
	while(activeVertexNum[0] > 0){
		#ifdef DEBUG
		std::cout << "software processing, iteration: " << itNum[0];
		std::cout << " active vertex number: " << activeVertexNum[0] << std::endl;
		#endif
		for(int j = 0; j < activeVertexNum[0]; j++){
			int v = activeVertices[j];
			int start = csr->rpao[v];
			int end = csr->rpao[v+1];
			PROP_TYPE uProp = vertexProp[v];
			for(int i = start; i < end; i++){
				int ngbVidx = csr->ciao[i];
				//PROP_TYPE eProp = csr->eProps[i];
				tmpVertexProp[ngbVidx] = compute(uProp, 1, tmpVertexProp[ngbVidx]);
			}
		}

		// Decide the frontier
		int idx = 0;
		for(int i = 0; i < csr->vertexNum; i++){
			PROP_TYPE vProp = vertexProp[i];
			PROP_TYPE tProp = tmpVertexProp[i];
			bool update = updateCondition(vProp, tProp);
			if(update){
				vertexProp[i] = tProp;
				activeVertices[idx++] = i;

				if(itNum[0] == 0) printf("cpu vertex %d \t", i);
			}
		}

		activeVertexNum[0] = idx;
		itNum[0]++;
	}
	// Collect the result 
	for(int i = 0; i < csr->vertexNum; i++){
		swProp[i] = vertexProp[i];
	}
}
#endif
#ifdef PR
void singleThreadSWProcessing(
	CSR* csr,
	std::vector<CSR_BLOCK*> &blkVec, 
	PROP_TYPE* ptProp, 
	const int &blkNum,
	const int &vertexNum,
	const int &source
	)
{	
	base_score = float2int((1.0f - kDamp) /vertexNum);
	printf("base_score original %.*f \n", 10,(1.0f- kDamp) /vertexNum);
	printf("base_score int %d \n", base_score);
	printf("base_score after int %.*f\n", 10,int2float(base_score));
	itNum[0] = 0;
	while(itNum[0] < MAX_ITER){
		//std::cout << "Processing with partition, iteration: " << itNum[0] << std::endl;
		//#pragma omp parallel for
		for (int u=0; u < vertexNum; u++) {
			int start = rpa[u];
			int num = rpa[u+1] - rpa[u];
			for(int j = 0; j < num; j++){
					tmpVertexProp[cia[start + j]] += vertexProp[u] / (csr->rpao[u+1] - csr->rpao[u]);
			}	
		}
		int error = 0;
		//#pragma omp parallel for reduction(+:error)
		for(int i = 0; i < vertexNum; i++){
			PROP_TYPE tProp = tmpVertexProp[i];
			PROP_TYPE old_score = vertexProp[i];
			vertexProp[i] = base_score + kDamp * tProp;
			error += fabs(vertexProp[i] - old_score);
			tmpVertexProp[i] = 0;
			if(outDeg[i] > 0) vertexScore[i] = vertexProp[i]/outDeg[i];
		}
		printf(" %2d    %lf\n", itNum[0], int2float(error));
		activeVertexNum[0] = vertexNum;
		itNum[0]++;
	}
}
#endif
#ifdef PR
void ptProcessing(
	CSR* csr,
	std::vector<CSR_BLOCK*> &blkVec, 
	PROP_TYPE* ptProp, 
	const int &blkNum,
	const int &vertexNum,
	const int &source
	)
{
	itNum[0] = 0;
	while(itNum[0] < MAX_ITER){

		double begin = getCurrentTimestamp();
		for(int row = 0; row < blkNum * blkNum; row++){
			//if(row == fpga_partition_x) continue;
			CSR_BLOCK* blkPtr = blkVec[row];
			int srcStart = blkPtr->srcStart;
			int srcEnd = blkPtr->srcEnd;
			for(int v = srcStart; v < srcEnd; v++){
				int ciaIdxStart = blkPtr->rpa[v - srcStart];
				int ciaIdxEnd = blkPtr->rpa[v + 1 - srcStart];
				for(int i = ciaIdxStart; i < ciaIdxEnd; i++){
					int ngbIdx = blkPtr->cia[i]; 
					tmpVertexProp[ngbIdx] += vertexScore[v];
				}
			}
		} 
		double end = getCurrentTimestamp();
		printf("[INFO]CPU compute phrase costs %lf ms\n", (end - begin) * 1000);

		int error = 0;
		begin =  getCurrentTimestamp();
		for(int i = 0; i < vertexNum; i++){
			PROP_TYPE tProp = tmpVertexProp[i];
			PROP_TYPE old_score = vertexProp[i];
			vertexProp[i] = base_score + kDamp * tProp;
			error += fabs(vertexProp[i] - old_score);
			tmpVertexProp[i] = 0;
			if(outDeg[i] > 0) vertexScore[i] = vertexProp[i]/outDeg[i];
		}
		end = getCurrentTimestamp();
		printf("[INFO]CPU apply phrase costs %lf ms\n", (end - begin) * 1000);

		printf(" %2d    %lf\n", itNum[0],int2float(error));
		activeVertexNum[0] = vertexNum;
		itNum[0]++;
	}
}
#endif

void paddingBlk(
				const std::vector<CSR_BLOCK*> &blkVec
				){
	int vertex_size_block = BLK_SIZE;
	int edgesIdx = 0;
	int outEdgesOld = 0;
	for(int row = 0; row < blkNum*blkNum; row ++){
		CSR_BLOCK* blkPtr = blkVec[row];
		blkVertexNum[row]     = blkPtr->vertexNum; 
		blkEdgeNum  [row]     = blkPtr->edgeNum;
		srcRange [2 * row]    = blkPtr->srcStart;
		srcRange [2 * row + 1]= blkPtr->srcEnd;
		sinkRange[2 * row]    = blkPtr->sinkStart;
		sinkRange[2 * row + 1]= blkPtr->sinkEnd;

		int pad = -1;
		int padding_num = 32;

		//printf("blocksize %d blkNum %d padding num %d ", BLK_SIZE, blkNum, padding_num);
		//printf("row %d vertexNum %d edgeNum %d\n", row, blkPtr->vertexNum, blkPtr->edgeNum);

		for(int i = 0; i < blkPtr->vertexNum; i++){
			int start = blkPtr->rpa[i];
			int num = blkPtr->rpa[i + 1] - blkPtr->rpa[i];
			std::vector<std::vector<int>> bankVec(padding_num);
			for(int j = 0; j < num; j++){
				int ngbIdx = blkPtr->cia[start + j];
				bankVec[ngbIdx % padding_num].push_back(ngbIdx);
			}

			outEdgesOld = edgesIdx;
			blkRpa[row * vertex_size_block + i] = edgesIdx;
	
			bool empty_flag = false;
			for(int i = 0; i < padding_num; i ++) empty_flag |= (!bankVec[i].empty());
	
			while (empty_flag){
				for(int i = 0; i < padding_num; i ++){
					if(!bankVec[i].empty()){
						cia_padding[edgesIdx ++] = bankVec[i].back();
						bankVec[i].pop_back();
					}
					else
						cia_padding[edgesIdx ++] = pad;
				}
				empty_flag = false;
				for(int i = 0; i < padding_num; i ++) empty_flag |= (!bankVec[i].empty());
			}
			//outdeg_padding[row * vertex_size_block + i] = edgesIdx - outEdgesOld;
		}
		blkRpaLast[row] = edgesIdx;
	}
	printf("Edge replication factor %d \n", edgesIdx / edgeNum);
}

void launchFPGA(CSR * csr, std::vector<CSR_BLOCK*> &blkVec)
{
		cl_event eventReadActiveVertices;
		cl_event eventReadNgbInfo;
		cl_event eventProcessEdge;			
		blkActiveVertexNum[0] = BLK_SIZE;			
	
		for(int row = 0; row < blkNum * blkNum; row++){	
			//double begin = getCurrentTimestamp();
			CSR_BLOCK* blkPtr = blkVec[row];
			fpgaIterNum[0] = row;

			double begin = getCurrentTimestamp();
			int i = 0;
			for(int k = 0; k < activeVertexNum[0]; k++){
				int v = activeVertices[k];
				if(v >= blkPtr->srcStart && v < blkPtr->srcEnd){
					blkActiveVertices[row * BLK_SIZE + (i ++)] =  v;
				}
			}
			blkActiveVertexNum[row] = i;
			double end = getCurrentTimestamp();
      printf("time to ensure the next active vertex num %f \n", (end - begin)* 1000);
			if(!i) continue;
			#if 0
			//double begin = getCurrentTimestamp();
			int srcStart = srcRange [2 * row];
			int srcEnd = srcRange [2 * row + 1];
			//printf("blkActiveVertexNum[%d] %d\n", row, blkActiveVertexNum[row]);
			for(int i = 0; i < blkActiveVertexNum[row]; i++){
				int v = blkActiveVertices[row * BLK_SIZE + i];
				int begin = blkRpa[row * BLK_SIZE + v - srcStart];
				int end = ((v - srcStart)!= BLK_SIZE)? blkRpa[row * BLK_SIZE + v - srcStart + 1] : blkRpaLast[row];
			if(v == 130551) printf("num %d, vertexIdx %d \n",end - begin, v);
			if(v == 247144) printf("num %d, vertexIdx %d \n",end - begin, v);
			if(v == 268646) printf("num %d, vertexIdx %d \n",end - begin, v);
  		        for(int k = begin; k < end; k++){
  		            int ngbIdx = cia_padding[k];
  		            if(ngbIdx != -1){
  		              		tmpVertexProp[ngbIdx]  = (tmpVertexProp[ngbIdx] > vertexProp[v])? vertexProp[v] + 1 :tmpVertexProp[ngbIdx]; //compute(uProp, 1, tmpVertexProp[     ngbIdx]);
  		          	}
  		       	}
  		  	}
	      	double end = getCurrentTimestamp();
			fpga_runtime = (end - begin);
			//printf("[INFO] row %d task CPU run time is %lf\n", row, fpga_runtime * 1000);
			#else
			status = clEnqueueTask(queueReadActiveVertices, readActiveVertices, 0, NULL, &eventReadActiveVertices);
			checkStatus("Failed to launch readActiveVertices.");
			status = clEnqueueTask(queueReadNgbInfo, readNgbInfo, 0, NULL, &eventReadNgbInfo);
			checkStatus("Failed to launch readNgbInfo.");
			status = clEnqueueTask(queueProcessEdge, processEdge, 0, NULL, &eventProcessEdge);
			checkStatus("Failed to launch processEdge.");
			
		//	sleep(3);
			// clWaitForEvents(1, &eventReadActiveVertices);
			// clWaitForEvents(1, &eventReadNgbInfo);
			// clWaitForEvents(1, &eventProcessEdge);

			clFinish(queueReadActiveVertices);
			clFinish(queueReadNgbInfo);
			clFinish(queueProcessEdge);
	
			//double end = getCurrentTimestamp();
			//fpga_runtime += (end - begin);
			//printf("[INFO] row %d FPGA run time is %lf ms\n", row, (end - begin) * 1000);
			#endif
		}
		printf("[INFO] FPGA total run time is %lf ms\n", fpga_runtime * 1000);
}
void fpgaApplyPhrase(){

	status = clEnqueueTask(queueVertexApply, vertexApply, 0, NULL, NULL);
	checkStatus("Failed to launch vertexApply.");
	status = clEnqueueTask(queueActiveVertexOutput, activeVertexOutput, 0, NULL, NULL);
	checkStatus("Failed to launch activeVertexOutput.");

	clFinish(queueVertexApply);
	clFinish(queueActiveVertexOutput);
}
#ifdef BFS
void ptProcessing(
		CSR* csr,
		std::vector<CSR_BLOCK*> &blkVec, 
		PROP_TYPE* ptProp, 
		const int &blkNum,
		const int &vertexNum,
		const int &source
		)
{
	itNum[0] = 0;
	while(activeVertexNum[0] > 0){
		#ifdef DEBUG
		std::cout << "software processing, iteration: " << itNum[0];
		std::cout << " active vertex number: " << activeVertexNum[0] << std::endl;
		#endif
		for(int i = 0; i < blkNum; i++){	
			for(int j = 0; j < blkNum; j++){
				int blkIdx = j * blkNum + i;
				CSR_BLOCK* blkPtr = blkVec[blkIdx];
				int srcStart = blkPtr->srcStart;
				int srcEnd = blkPtr->srcEnd;
				std::vector<int> frontier;
				for(int k = 0; k < activeVertexNum[0]; k++){
					int v = activeVertices[k];
					if(v >= srcStart && v < srcEnd){
						frontier.push_back(v);
					}
				}
				if(frontier.size() == 0) continue;

				for(auto v : frontier){
					int ciaIdxStart = blkPtr->rpa[v - srcStart];
					int ciaIdxEnd = blkPtr->rpa[v + 1 - srcStart];

					for(int k = ciaIdxStart; k < ciaIdxEnd; k++){
						int ngbIdx = blkPtr->cia[k];
						PROP_TYPE uProp = vertexProp[v];
						//PROP_TYPE eProp = blkVec[blkIdx]->eProps[k];
						tmpVertexProp[ngbIdx] = compute(uProp, 1, tmpVertexProp[ngbIdx]);
					}
				}
			}
		}

		// Decide active vertices and apply
		int idx = 0;
		for(int i = 0; i < vertexNum; i++){
			PROP_TYPE vProp = vertexProp[i];
			PROP_TYPE tProp = tmpVertexProp[i];
			bool update = updateCondition(vProp, tProp);
			if(update){
				vertexProp[i] = tProp;
				activeVertices[idx++] = i;
			}
		}

		activeVertexNum[0] = idx;
		itNum[0]++;
	}

	// Collect the result
	for(int i = 0; i < vertexNum; i++){
		ptProp[i] = vertexProp[i];
	}
}
#endif
double fpgaProcessing(
	CSR* csr,
	std::vector<CSR_BLOCK*> &blkVec, 
	PROP_TYPE* hybridProp, 
	int &blkNum,
	const int &vertexNum,
	const int &edgeNum,
	const int mode // 1 is single thread, 2 is multi-thread
		)
{	
	double runtime = 0;
	itNum[0] = 0;
	while(activeVertexNum[0] > 0){
		#ifdef DEBUG
		std::cout << "software processing, iteration: " << itNum[0];
		std::cout << " active vertex number: " << activeVertexNum[0] << std::endl;
		#endif
		#if 0
		for(int i = 0; i < blkNum * blkNum; i++){	
				int blkIdx = i;//j * blkNum + i;
				CSR_BLOCK* blkPtr = blkVec[blkIdx];
				int srcStart = blkPtr->srcStart;
				int srcEnd = blkPtr->srcEnd;
				std::vector<int> frontier;
				for(int k = 0; k < activeVertexNum[0]; k++){
					int v = activeVertices[k];
					if(v >= srcStart && v < srcEnd){
						frontier.push_back(v);
					}
				}
				if(frontier.size() == 0) continue;

				for(auto v : frontier){
					int ciaIdxStart = blkPtr->rpa[v - srcStart];
					int ciaIdxEnd = blkPtr->rpa[v + 1 - srcStart];

			if(v == 130551) printf("num %d, vertexIdx %d \n",ciaIdxEnd - ciaIdxStart, v);
			if(v == 247144) printf("num %d, vertexIdx %d \n",ciaIdxEnd - ciaIdxStart, v);
			if(v == 268646) printf("num %d, vertexIdx %d \n",ciaIdxEnd - ciaIdxStart, v);

					for(int k = ciaIdxStart; k < ciaIdxEnd; k++){
						int ngbIdx = blkPtr->cia[k];
						PROP_TYPE uProp = vertexProp[v];
						//PROP_TYPE eProp = blkVec[blkIdx]->eProps[k];
						tmpVertexProp[ngbIdx] = compute(uProp, 1, tmpVertexProp[ngbIdx]);
					}
				}
		}

		#else
			launchFPGA( csr , blkVec);
		#endif

    	const double begin = getCurrentTimestamp();
		#if 0
		int idx = 0;
		for(int i = 0; i < vertexNum; i++){
			PROP_TYPE vProp = vertexProp[i];
			PROP_TYPE tProp = tmpVertexProp[i];
			bool update = updateCondition(vProp, tProp);				
			if(update){
				vertexProp[i] = tProp;
				activeVertices[idx++] = i;
				if(itNum[0] == 0) printf("active vertex %d \t", i);
			}
		}
		activeVertexNum[0] = idx;
		printf("next iteration active vertexNum %d \n", idx);
		#else
			fpgaApplyPhrase();
			activeVertexNum[0] = error[0];
		#endif
		const double end = getCurrentTimestamp();
		printf("[INFO] FPGA apply phrase runtime is %f ms \n", (end - begin) * 1000);
		//---------------------//
		runtime += (end - begin);
		itNum[0]++;
	}
	return (runtime * 1000);//*1.0/CLOCKS_PER_SEC;
}

// Init the variables for a new processing.
void processInit(
	const int &vertexNum,
	const int &edgeNum,
	const int &source
	)
{
	eop[0] = 0;

	#ifdef PR
	float init_score_float = 1.0f / vertexNum;
	int init_score_int = float2int(init_score_float);
	for(int i = 0; i < vertexNum; i++){
		vertexProp[i] = init_score_int;
		tmpVertexProp[i] = 0;
		activeVertices[i] = i;
		if(outDeg[i] > 0) vertexScore[i] = vertexProp[i]/outDeg[i];
	}
	printf("init_score original %f \n",init_score_float);
	printf("init_score original to int %d \n",init_score_int);
	printf("init_score after int %f\n", int2float(init_score_int));
	activeVertexNum[0] = vertexNum;
	//activeVertexNum[0] = 0;
	#endif

	#ifdef BFS
	for(int i = 0; i < vertexNum; i++){
		vertexProp[i]    = MAX_PROP;
		tmpVertexProp[i] = MAX_PROP;
	}
	vertexProp[source] = 0;
	tmpVertexProp[source] = 0;
	activeVertexNum[0] = 1;
	activeVertices[0] = source;
	#endif

}

void csrPartition(
	CSR* csr,
	std::vector<CSR_BLOCK*> &blkVec,
	const int &blkNum
	)
{
	std::cout << "The graph is divided into " << blkNum * blkNum << " partitions\n";
	for(int cordx = 0; cordx < blkNum; cordx++){
		for(int cordy = 0; cordy < blkNum; cordy++){
			CSR_BLOCK* csrBlkPtr = new CSR_BLOCK(cordx, cordy, csr);
			blkVec.push_back(csrBlkPtr);
			// find the partition with max degree
			if(csrBlkPtr->edgeNum / BLK_SIZE > 16) printf("cordx %d, cordy %d \n", cordx, cordy);
			if(csrBlkPtr->edgeNum > max_partition_degree){
				max_partition_degree = csrBlkPtr->edgeNum;
				fpga_partition_x = cordx;
				fpga_partition_y = cordy;
				printf("fpga_partition_x %d, fpga_partition_y %d, max_partition_degree %d \n",fpga_partition_x, fpga_partition_y,max_partition_degree);
			}
		}
	}
}


// CPU thread related to main function 
int main(int argc, char **argv) {
	double begin;
	double end;
	double elapsedTime;
	int startVertexIdx;
	std::string gName = "pokec";
	std::string mode = "harp"; // or 3arp
	edge_replic_factor = 3;
	if(gName == "youtube")    startVertexIdx = 320872;
	if(gName == "lj1")        startVertexIdx = 3928512;
	if(gName == "pokec")      startVertexIdx = 182045;
	if(gName == "rmat-19-32") startVertexIdx = 104802;
	if(gName == "rmat-21-32") startVertexIdx = 365723;
	Graph* gptr = createGraph(gName, mode);
	csr = new CSR(*gptr);
	vertexNum = csr->vertexNum;
	edgeNum   = csr->edgeNum;
	free(gptr);

	PROP_TYPE *swProp      = (PROP_TYPE*)malloc(vertexNum * sizeof(PROP_TYPE));
	PROP_TYPE *ptProp      = (PROP_TYPE*)malloc(vertexNum * sizeof(PROP_TYPE));
	PROP_TYPE *hybridProp  = (PROP_TYPE*)malloc(vertexNum * sizeof(PROP_TYPE));

	blkNum = (vertexNum + BLK_SIZE - 1)/BLK_SIZE;
	printf("blkNum %d, ceil num %d \n", blkNum, (int)ceil(static_cast<double>(vertexNum) / BLK_SIZE));

	// init fpga
	setHardwareEnv();
	globalVarInit(csr, vertexNum, edgeNum);
	setKernelEnv();
	createKernels(vertexNum, edgeNum);

	//kernelVarMap(vertexNum, edgeNum);

	//software processing on CPU
	std::cout << "software BFS starts." << std::endl;
	processInit(vertexNum, edgeNum, startVertexIdx);
	begin = getCurrentTimestamp();
	singleThreadSWProcessing(csr, blkVec, ptProp, blkNum, vertexNum, startVertexIdx);
	end = getCurrentTimestamp();
	elapsedTime = (end - begin) * 1000;
	std::cout << "[INFO] singleThreadSWProcessing BFS takes " << elapsedTime << " ms." << std::endl;
	printf("\n");

	// software processing with only 2 partitions
	csrPartition(csr, blkVec, blkNum);

	std::cout << "soft partitioned BFS starts." << std::endl;
	processInit(vertexNum, edgeNum, startVertexIdx);
	begin = getCurrentTimestamp();
	ptProcessing(csr, blkVec, ptProp, blkNum, vertexNum, startVertexIdx);
	end = getCurrentTimestamp();
	elapsedTime = (end - begin) * 1000;
	std::cout << "[INFO] ptProcessing BFS takes " << elapsedTime << " ms." << std::endl;
	printf("\n");
	
	std::cout << "fpga processing." << std::endl;	
	processInit(vertexNum, edgeNum, startVertexIdx);	
	paddingBlk(blkVec);
	printf("Padding finish \n");
	begin = getCurrentTimestamp();
	elapsedTime = fpgaProcessing(csr, blkVec, hybridProp, blkNum, vertexNum, edgeNum, 2);
	end = getCurrentTimestamp();
  	elapsedTime = (end -begin) * 1000;
	std::cout << "[INFO] fpga processing takes " << elapsedTime << " ms." << std::endl;
	
	freeResources();

	return 0;
}
