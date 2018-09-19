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

#include <sys/time.h>

using namespace std::tr1;
using namespace aocl_utils;
using namespace std;

static cl_command_queue queueReadActiveVertices;
static cl_command_queue queueReadNgbInfo;
static cl_command_queue queueProcessEdge;

static cl_kernel readActiveVertices;
static cl_kernel readNgbInfo;
static cl_kernel processEdge;

static cl_program program;

static cl_int status;
static PROP_TYPE* vertexProp;
static PROP_TYPE* tmpVertexProp;
static int* rpa;
static int* blkRpa;
static int* blkRpaNum;
static int* outDeg;
static int* outdeg_padding;
static int* cia;
static int* cia_padding;
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
int vertexNum;
int edgeNum; 
int blkNum;
volatile bool globalVarInitDone = false;
volatile bool fpgaTaskDone = false;
volatile bool isComplete = false;
CSR* csr;
std::vector<CSR_BLOCK*> blkVec;
int processing_edges = 0;

//Notify the FPGA thread
std::condition_variable cond;
std::mutex condMutex;

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;

static int fpga_partition_x = 0;
static int fpga_partition_y = 0;
static int max_partition_degree = 0; 
static CSR_BLOCK* fpga_partition;

double fpga_runtime =0;
#define AOCL_ALIGNMENT 64
#define THREAD_NUM 1
#define MAX_ITER 1
//#define DEBUG 
#define MODE "sim"
#define PR
#ifdef PR 
#define PROP_TYPE float
#define kDamp 0.85f
#define epsilon  0.001f
#endif
#ifdef CC 
#define PROP_TYPE int
#endif

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

void kernelVarMap(
	const int &vertexNum, 
	const int &edgeNum
	)
{
	status = clEnqueueSVMMap(queueReadActiveVertices, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)blkActiveVertices, sizeof(int) * vertexNum, 0, NULL, NULL); 
	checkStatus("enqueue activeVertices.");
	status = clEnqueueSVMMap(queueReadActiveVertices, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)blkActiveVertexNum, sizeof(int), 0, NULL, NULL); 
	checkStatus("enqueue activeVertexNum.");

	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ, 
		(void *)blkRpa, sizeof(int) * (vertexNum + 1), 0, NULL, NULL); 
	checkStatus("enqueue blkRpa.");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ, 
		(void *)cia_padding, sizeof(int) * edgeNum * 20, 0, NULL, NULL); 
	checkStatus("enqueue blkCia");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)blkEdgeProp, sizeof(PROP_TYPE) * edgeNum, 0, NULL, NULL); 
	checkStatus("enqueue blkEdgeProp");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)outDeg, sizeof(int) * vertexNum, 0, NULL, NULL); 
	checkStatus("enqueue outDeg");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)blkActiveVertexNum, sizeof(int), 0, NULL, NULL); 
	checkStatus("enqueue blkActiveVertexNum");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)blkVertexNum, sizeof(int), 0, NULL, NULL); 
	checkStatus("enqueue blkVertexNum");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)blkEdgeNum, sizeof(int), 0, NULL, NULL); 
	checkStatus("enqueue blkEdgeNum.");
	status = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)srcRange, sizeof(int) * 2, 0, NULL, NULL); 
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
		(void *)srcRange, sizeof(int) * 2, 0, NULL, NULL); 
	checkStatus("enqueue srcRange");
	status = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)sinkRange, sizeof(int) * 2, 0, NULL, NULL); 
	checkStatus("enqueue sinkRange.");
	std::cout << "All the shared memory objects are mapped successfully." << std::endl;
}
void kernelVarUnmap(){
	status = clEnqueueSVMUnmap(queueReadActiveVertices, 
		(void *)blkActiveVertices, 0, NULL, NULL); 
	checkStatus("unmap activeVertices.");
	status = clEnqueueSVMUnmap(queueReadActiveVertices, 
		(void *)blkActiveVertexNum, 0, NULL, NULL); 
	checkStatus("Unmap activeVertexNum.");

	status = clEnqueueSVMUnmap(queueReadNgbInfo, 
		(void *)blkRpa, 0, NULL, NULL); 
	checkStatus("Unmap blkRpa.");
	status = clEnqueueSVMUnmap(queueReadNgbInfo, 
		(void *)blkCia, 0, NULL, NULL); 
	checkStatus("Unmap blkCia.");
	status = clEnqueueSVMUnmap(queueReadNgbInfo, 
		(void *)blkEdgeProp, 0, NULL, NULL); 
	checkStatus("Unmap activeVertexNum.");
	status = clEnqueueSVMUnmap(queueReadNgbInfo, 
		(void *)outDeg, 0, NULL, NULL); 
	checkStatus("Unmap edgeProp.");
	status = clEnqueueSVMUnmap(queueReadNgbInfo, 
		(void *)blkActiveVertexNum, 0, NULL, NULL); 
	checkStatus("Unmap cia.");
	status = clEnqueueSVMUnmap(queueReadNgbInfo, 
		(void *)blkVertexNum, 0, NULL, NULL); 
	checkStatus("Unmap blkVertexNum.");
	status = clEnqueueSVMUnmap(queueReadNgbInfo, 
		(void *)blkEdgeNum, 0, NULL, NULL); 
	checkStatus("Unmap blkEdgeNum.");
	status = clEnqueueSVMUnmap(queueReadNgbInfo, 
		(void *)srcRange, 0, NULL, NULL); 
	checkStatus("Unmap srcRange.");
	status = clEnqueueSVMUnmap(queueReadNgbInfo, 
		(void *)fpgaIterNum, 0, NULL, NULL); 
	checkStatus("Unmap itNum.");

	status = clEnqueueSVMUnmap(queueProcessEdge, 
		(void *)vertexProp, 0, NULL, NULL); 
	checkStatus("Unmap vertexProp.");
	status = clEnqueueSVMUnmap(queueProcessEdge, 
		(void *)tmpVertexProp, 0, NULL, NULL); 
	checkStatus("Unmap tmpVertexProp.");
	status = clEnqueueSVMUnmap(queueProcessEdge, 
		(void *)eop, 0, NULL, NULL); 
	checkStatus("Unmap eop.");
	status = clEnqueueSVMUnmap(queueProcessEdge, 
		(void *)fpgaIterNum, 0, NULL, NULL); 
	checkStatus("Unmap itNum.");
	status = clEnqueueSVMUnmap(queueProcessEdge, 
		(void *)srcRange, 0, NULL, NULL); 
	checkStatus("Unmap srcRange.");
	status = clEnqueueSVMUnmap(queueProcessEdge, 
		(void *)sinkRange, 0, NULL, NULL); 
	checkStatus("Unmap sinkRange.");
}

void setKernelEnv(){
	queueReadActiveVertices = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkStatus("Failed clCreateCommandQueue of queueReadActiveVertices.");
	queueReadNgbInfo = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkStatus("Failed clCreateCommandQueue of queueReadNgbInfo.");
	queueProcessEdge = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
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
	else if(gName == "rmat-12-4"){
		gptr = new Graph(dir + "rmat-12-4.txt");
	}
	else if(gName == "rmat-23-4"){
		gptr = new Graph(dir + "rmat-23-4.txt");
	}
	else if(gName == "rmat-23-16"){
		gptr = new Graph(dir + "rmat-23-16.txt");
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
	vertexProp         = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * vertexNum, 1024); 
	tmpVertexProp      = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * vertexNum, 1024);
	rpa                = (int*) clSVMAllocAltera(context, 0, sizeof(int) * (vertexNum + 1), 1024);
	blkRpa             = (int*) clSVMAllocAltera(context, 0, sizeof(int) * (vertexNum + 1), 1024);
	blkRpaNum          = (int*) clSVMAllocAltera(context, 0, sizeof(int) * (vertexNum), 1024);
	outDeg             = (int*) clSVMAllocAltera(context, 0, sizeof(int) * vertexNum, 1024);
	cia           	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * edgeNum * 2, 1024); // undirection graph
	cia_padding    	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * edgeNum * 6, 1024);
	edgeProp           = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * edgeNum, 1024);
	blkCia             = (int*) clSVMAllocAltera(context, 0, sizeof(int) * 2, 1024);	
	outdeg_padding     = (int*) clSVMAllocAltera(context, 0, sizeof(int) * vertexNum, 1024);
	blkEdgeProp        = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * edgeNum, 1024);
	activeVertices     = (int*) clSVMAllocAltera(context, 0, sizeof(int) * vertexNum, 1024);
	activeVertexNum    = (int*) clSVMAllocAltera(context, 0, sizeof(int), 1024);
	blkActiveVertices  = (int*) clSVMAllocAltera(context, 0, sizeof(int) * vertexNum, 1024);
	blkActiveVertexNum = (int*) clSVMAllocAltera(context, 0, sizeof(int) * 256, 1024); // The MAX partitions FPGA need to process
	itNum     		   = (int*) clSVMAllocAltera(context, 0, sizeof(int), 1024);
	fpgaIterNum        = (int*) clSVMAllocAltera(context, 0, sizeof(int), 1024);  
	blkEdgeNum     	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * 100, 1024); 
	blkVertexNum 	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * 100, 1024); 
	eop  		       = (int*) clSVMAllocAltera(context, 0, sizeof(int), 1024); 
	srcRange 	   	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * 200, 1024);  
	sinkRange 	   	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * 200, 1024);  

	if(!vertexProp || !tmpVertexProp || !rpa || !blkRpa 
		|| !outDeg || !cia || !edgeProp || !blkCia 
		|| !blkEdgeProp || !activeVertices|| !activeVertexNum 
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

	clSetKernelArgSVMPointerAltera(readActiveVertices, 0, (void*)blkActiveVertices);	
	clSetKernelArgSVMPointerAltera(readActiveVertices, 1, (void*)blkActiveVertexNum);
	clSetKernelArgSVMPointerAltera(readActiveVertices, 2, (void*)fpgaIterNum);


	clSetKernelArgSVMPointerAltera(readNgbInfo, 0, (void*)blkRpa);
	clSetKernelArgSVMPointerAltera(readNgbInfo, 1, (void*)outdeg_padding);
	clSetKernelArgSVMPointerAltera(readNgbInfo, 2, (void*)cia_padding);
	clSetKernelArgSVMPointerAltera(readNgbInfo, 3, (void*)blkEdgeProp);
	clSetKernelArgSVMPointerAltera(readNgbInfo, 4, (void*)outDeg);	
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
}
typedef float ScoreT;
void singleThreadSWProcessing(
	CSR* csr,
	std::vector<CSR_BLOCK*> &blkVec, 
	PROP_TYPE* ptProp, 
	const int &blkNum,
	const int &vertexNum,
	const int &source
	)
{
	const ScoreT base_score = (1.0f - kDamp) / vertexNum;
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
		double error = 0;
		//#pragma omp parallel for reduction(+:error)
		for(int i = 0; i < vertexNum; i++){
			PROP_TYPE tProp = tmpVertexProp[i];
			PROP_TYPE old_score = vertexProp[i];
			vertexProp[i] = base_score + kDamp * tProp;
			error += fabs(vertexProp[i] - old_score);
			tmpVertexProp[i] = 0;
		}
		printf(" %2d    %lf\n", itNum[0], error);
		activeVertexNum[0] = vertexNum;
		itNum[0]++;
	}
}
void multiThreadSWProcessing(
	CSR* csr,
	std::vector<CSR_BLOCK*> &blkVec, 
	PROP_TYPE* ptProp, 
	const int &blkNum,
	const int &vertexNum,
	const int &source
	)
{
	const ScoreT base_score = (1.0f - kDamp) / vertexNum;
	itNum[0] = 0;
	while(itNum[0] < MAX_ITER){
		//std::cout << "Processing with partition, iteration: " << itNum[0] << std::endl;
		#pragma omp parallel for
		for (int u=0; u < vertexNum; u++) {
			int start = rpa[u];
			int num = rpa[u+1] - rpa[u];
			for(int j = 0; j < num; j++){
					tmpVertexProp[cia[start + j]] += vertexProp[u] / (csr->rpao[u+1] - csr->rpao[u]);
			}	
		}
		double error = 0;
		#pragma omp parallel for reduction(+:error)
		for(int i = 0; i < vertexNum; i++){
			PROP_TYPE tProp = tmpVertexProp[i];
			PROP_TYPE old_score = vertexProp[i];
			vertexProp[i] = base_score + kDamp * tProp;
			error += fabs(vertexProp[i] - old_score);
			tmpVertexProp[i] = 0;
		}
		printf(" %2d    %lf\n", itNum[0], error);
		activeVertexNum[0] = vertexNum;
		itNum[0]++;
	}
}

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
	const ScoreT base_score = (1.0f - kDamp) / vertexNum;
	itNum[0] = 0;
	while(itNum[0] < MAX_ITER){
		//std::cout << "Processing with partition, iteration: " << itNum[0] << std::endl;
		double begin = getCurrentTimestamp();
		struct timeval t1, t2;
		gettimeofday(&t1, NULL);
		for (int u=0; u < vertexNum; u++) {
			int start = rpa[u];
			int num = rpa[u+1] - rpa[u];
			for(int j = 0; j < num; j++){
				if((csr->rpao[u+1] - csr->rpao[u]) != 0)
					tmpVertexProp[cia[start + j]] += vertexProp[u] / (csr->rpao[u+1] - csr->rpao[u]);
			}	
		}
		gettimeofday(&t2, NULL);
		double end = getCurrentTimestamp();
		printf("[INFO]sw pt big CSR takes mainthread %lf ms\n", ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0));

		
		// int row = fpga_partition_x;
		// CSR_BLOCK* blkPtr = blkVec[row];
		// int srcStart = blkPtr->srcStart;
		// int srcEnd = blkPtr->srcEnd;
		// begin = getCurrentTimestamp();
		// for(int v = srcStart; v < srcEnd; v++){
		// 	int ciaIdxStart = blkPtr->rpa[v - srcStart];
		// 	int ciaIdxEnd = blkPtr->rpa[v + 1 - srcStart];
		// 	for(int i = ciaIdxStart; i < ciaIdxEnd; i++){
		// 		int ngbIdx = blkPtr->cia[i]; 
		// 		if((csr->rpao[v+1] - csr->rpao[v]) != 0)
		// 			tmpVertexProp[ngbIdx] += vertexProp[v] / (csr->rpao[v+1] - csr->rpao[v]);
		// 	}
		// }
		// end = getCurrentTimestamp();
		// double fpga_runtime = (end - begin);
		// printf("[INFO]fpga task on CPU main thread run time %lf ms\n", fpga_runtime * 1000);
		
		for(int row = 0; row < blkNum; row++){
			//if(row == fpga_partition_x) continue;
			CSR_BLOCK* blkPtr = blkVec[row];
			int srcStart = blkPtr->srcStart;
			int srcEnd = blkPtr->srcEnd;
			for(int v = srcStart; v < srcEnd; v++){
				int ciaIdxStart = blkPtr->rpa[v - srcStart];
				int ciaIdxEnd = blkPtr->rpa[v + 1 - srcStart];
				for(int i = ciaIdxStart; i < ciaIdxEnd; i++){
					int ngbIdx = blkPtr->cia[i]; 
					if((csr->rpao[v+1] - csr->rpao[v]) != 0)
						tmpVertexProp[ngbIdx] += vertexProp[v] / (csr->rpao[v+1] - csr->rpao[v]);
				}
			}
		} 
		end = getCurrentTimestamp();
		double small_partition_runtime = (end - begin);
		printf("[INFO]The small partition on CPU main thread run time %lf ms\n", small_partition_runtime * 1000);

		double error = 0;

		for(int i = 0; i < vertexNum; i++){
			PROP_TYPE tProp = tmpVertexProp[i];
			PROP_TYPE old_score = vertexProp[i];
			vertexProp[i] = base_score + kDamp * tProp;
			error += fabs(vertexProp[i] - old_score);
			tmpVertexProp[i] = 0;
		}
		printf(" %2d    %lf\n", itNum[0], error);
		activeVertexNum[0] = vertexNum;
		itNum[0]++;
	}
}
#endif


#ifdef PR
void ptProcessingMultiThread(
	CSR* csr,
	std::vector<CSR_BLOCK*> &blkVec, 
	PROP_TYPE* ptProp, 
	const int &blkNum,
	const int &vertexNum,
	const int &source
	)
{
	const ScoreT base_score = (1.0f - kDamp) / vertexNum;
	itNum[0] = 0;
	while(itNum[0] < MAX_ITER){
		//std::cout << "Processing with partition, iteration: " << itNum[0] << std::endl;
		double begin = getCurrentTimestamp();
		struct timeval t1, t2;
		gettimeofday(&t1, NULL);
		#pragma omp parallel for
		for (int u=0; u < vertexNum; u++) {
			int start = rpa[u];
			int num = rpa[u+1] - rpa[u];
			for(int j = 0; j < num; j++){
				if((csr->rpao[u+1] - csr->rpao[u]) != 0)
					tmpVertexProp[cia[start + j]] += vertexProp[u] / (csr->rpao[u+1] - csr->rpao[u]);
			}	
		}
		gettimeofday(&t2, NULL);
		double end = getCurrentTimestamp();
		printf("[INFO]sw pt big CSR takes mainthread %lf ms\n", ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0));

		
		// int row = fpga_partition_x;
		// CSR_BLOCK* blkPtr = blkVec[row];
		// int srcStart = blkPtr->srcStart;
		// int srcEnd = blkPtr->srcEnd;
		// begin = getCurrentTimestamp();
		// for(int v = srcStart; v < srcEnd; v++){
		// 	int ciaIdxStart = blkPtr->rpa[v - srcStart];
		// 	int ciaIdxEnd = blkPtr->rpa[v + 1 - srcStart];
		// 	for(int i = ciaIdxStart; i < ciaIdxEnd; i++){
		// 		int ngbIdx = blkPtr->cia[i]; 
		// 		if((csr->rpao[v+1] - csr->rpao[v]) != 0)
		// 			tmpVertexProp[ngbIdx] += vertexProp[v] / (csr->rpao[v+1] - csr->rpao[v]);
		// 	}
		// }
		// end = getCurrentTimestamp();
		// double fpga_runtime = (end - begin);
		// printf("[INFO]fpga task on CPU main thread run time %lf ms\n", fpga_runtime * 1000);
		begin = getCurrentTimestamp();
		#pragma omp parallel for
		for(int row = 0; row < blkNum; row++){
			//if(row == fpga_partition_x) continue;
			CSR_BLOCK* blkPtr = blkVec[row];
			int srcStart = blkPtr->srcStart;
			int srcEnd = blkPtr->srcEnd;
			for(int v = srcStart; v < srcEnd; v++){
				int ciaIdxStart = blkPtr->rpa[v - srcStart];
				int ciaIdxEnd = blkPtr->rpa[v + 1 - srcStart];
				for(int i = ciaIdxStart; i < ciaIdxEnd; i++){
					int ngbIdx = blkPtr->cia[i]; 
					if((csr->rpao[v+1] - csr->rpao[v]) != 0)
						tmpVertexProp[ngbIdx] += vertexProp[v] / (csr->rpao[v+1] - csr->rpao[v]);
				}
			}
		} 
		end = getCurrentTimestamp();
		double small_partition_runtime = (end - begin);
		printf("[INFO]The small partition on CPU main thread run time %lf ms\n", small_partition_runtime * 1000);

		double error = 0;
		#pragma omp parallel for reduction(+:error)
		for(int i = 0; i < vertexNum; i++){
			PROP_TYPE tProp = tmpVertexProp[i];
			PROP_TYPE old_score = vertexProp[i];
			vertexProp[i] = base_score + kDamp * tProp;
			error += fabs(vertexProp[i] - old_score);
			tmpVertexProp[i] = 0;
		}
		printf(" %2d    %lf\n", itNum[0], error);
		activeVertexNum[0] = vertexNum;
		itNum[0]++;
	}
}
#endif

void paddingBlk(const std::vector<CSR_BLOCK*> &blkVec){

	//int edge_size_block = max_partition_degree;
	int vertex_size_block = BLK_SIZE;
	int edgesIdx = 0;
	int outEdgesOld = 0;
	for(int row = 0; row < blkNum; row ++){
		CSR_BLOCK* blkPtr = blkVec[row];
		blkVertexNum[row]     = blkPtr->vertexNum; 
		blkEdgeNum  [row]     = blkPtr->edgeNum;
		srcRange [2 * row]    = blkPtr->srcStart;
		srcRange [2 * row + 1]= blkPtr->srcEnd;
		sinkRange[2 * row]    = blkPtr->sinkStart;
		sinkRange[2 * row + 1]= blkPtr->sinkEnd;

		//printf("row %d \n", row);
		//std::cout << "fpga edgeNum: " << blkEdgeNum[0] << std::endl;
		//std::cout << "vetexnum  " << blkVertexNum[0] <<std::endl;
		//int idx = 0;
/*		for(int i = 0; i < activeVertexNum[0]; i++){
			int v = activeVertices[i];
			if(v >= blkPtr->srcStart && v < blkPtr->srcEnd){
				blkActiveVertices[row * vertex_size_block + idx++] = v;
			}
		}
		*/
//		blkActiveVertexNum[row] = idx;
		// for(int i = 0; i <= blkPtr->vertexNum; i++){
		// 	blkRpa[row * vertex_size_block + i] = blkPtr->rpa[i];
		// }

		// for(int i = 0; i < blkEdgeNum[0]; i++){
		// 	blkCia[row * edge_size_block + i]   = blkPtr->cia[i];
		// }
		#if 1 // csr block partition padding
		int pad = -1;
		int padding_num = 64;
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
			outdeg_padding[row * vertex_size_block + i] = edgesIdx - outEdgesOld;
		}
		#endif
	}
}
volatile bool cpuThreadUp = false;

void  cpuThreadFunc(CSR * csr){	
	struct timeval t1, t2;	
	gettimeofday(&t1, NULL);
	{
		#pragma omp parallel for// ordered schedule(static)
		for (int u=0; u < csr->vertexNum; u++) {
			int start = rpa[u];
			int num = rpa[u+1] - rpa[u];
			for(int j = start; j < start + num; j++){
					tmpVertexProp[cia[j]] += vertexProp[u] / outDeg[u]; 
			}	
		}
	}
	gettimeofday(&t2, NULL);
	printf("[INFO] hybridProcessing pt CSR takes %lf ms\n", ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0));
}
void  cpuSingleThreadFunc(CSR * csr){	
	struct timeval t1, t2;	
	gettimeofday(&t1, NULL);
	{
		//#pragma omp parallel for// ordered schedule(static)
		for (int u=0; u < csr->vertexNum; u++) {
			int start = rpa[u];
			int num = rpa[u+1] - rpa[u];
			for(int j = start; j < start + num; j++){
					tmpVertexProp[cia[j]] += vertexProp[u] / outDeg[u]; 
			}	
		}
	}
	gettimeofday(&t2, NULL);
	printf("[INFO] hybridProcessing pt CSR takes %lf ms\n", ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0));
}
void launchFPGA(CSR * csr, std::vector<CSR_BLOCK*> &blkVec)
{
		//FPGA thread CPU version
	#if 0
		double begin = getCurrentTimestamp();
		
		for(int row = 0; row < blkNum; row++){
			CSR_BLOCK* blkPtr = blkVec[row];
			int srcStart = blkPtr->srcStart;
			int srcEnd = blkPtr->srcEnd;
			for(int v = srcStart; v < srcEnd; v++){
				int ciaIdxStart = blkPtr->rpa[v - srcStart];
				int ciaIdxEnd = blkPtr->rpa[v + 1 - srcStart];
				for(int i = ciaIdxStart; i < ciaIdxEnd; i++){
					int ngbIdx = blkPtr->cia[i]; 
					if((csr->rpao[v+1] - csr->rpao[v]) != 0)
						tmpVertexProp[ngbIdx] += vertexProp[v] / (csr->rpao[v+1] - csr->rpao[v]);
				}
			}
		}
		
		double end = getCurrentTimestamp();
		printf("[INFO] FPGA samll partition newly created thread runtime is %f ms \n", (end - begin) * 1000);
 
	#else
			cl_event eventReadActiveVertices;
			cl_event eventReadNgbInfo;
			cl_event eventProcessEdge;			
		
			blkActiveVertexNum[0] = BLK_SIZE;
		
		
		for(int row = 0; row <  fpga_partition_x + 6; row++){
			//int  row = fpga_partition_x;
			double begin = getCurrentTimestamp();
			CSR_BLOCK* blkPtr = blkVec[row];
			fpgaIterNum[0] = row;
			blkActiveVertexNum[row] = BLK_SIZE;
			for(int i = 0; i < blkActiveVertexNum[row]; i ++){
				blkActiveVertices[row * BLK_SIZE + i] =  blkPtr->srcStart + i;
			}
			#if 0
			//double begin = getCurrentTimestamp();
			int srcStart = srcRange [2 * row];
			int srcEnd = srcRange [2 * row + 1];
			for(int i = 0; i < blkActiveVertexNum[row]; i++){
				int v = blkActiveVertices[row * BLK_SIZE + i];
  		          for(int k = blkRpa[row * BLK_SIZE + v - srcStart]; k < blkRpa[row * BLK_SIZE + v - srcStart + 1]; k++){
  		            int ngbIdx = cia_padding[k];
  		            	if(ngbIdx != -1){
  		              			tmpVertexProp[ngbIdx] += vertexProp[v] / outDeg[v]; //compute(uProp, 1, tmpVertexProp[     ngbIdx]);
  		          		}
  		       	}
  		  	}
	      	double end = getCurrentTimestamp();
			fpga_runtime = (end - begin);
			printf("[INFO] row %d task CPU run time is %lf\n", row, fpga_runtime * 1000);
			#else
			status = clEnqueueTask(queueReadActiveVertices, readActiveVertices, 0, NULL, &eventReadActiveVertices);
			checkStatus("Failed to launch readActiveVertices.");
			status = clEnqueueTask(queueReadNgbInfo, readNgbInfo, 0, NULL, &eventReadNgbInfo);
			checkStatus("Failed to launch readNgbInfo.");
			status = clEnqueueTask(queueProcessEdge, processEdge, 0, NULL, &eventProcessEdge);
			checkStatus("Failed to launch processEdge.");
		
			clFinish(queueReadActiveVertices);
			clFinish(queueReadNgbInfo);
			clFinish(queueProcessEdge);
			double end = getCurrentTimestamp();
			fpga_runtime += (end - begin);
			printf("[INFO] row %d FPGA run time is %lf ms\n", row, (end - begin) * 1000);
			#endif
		}
			printf("[INFO] FPGA total run time is %lf ms\n", fpga_runtime * 1000);

		//Process the last one on CPU since it is not the BLOCK SIZE
		{
			double begin = getCurrentTimestamp();
			for(int row = fpga_partition_x + 6; row < blkNum; row++){
				//if ((row - fpga_partition_x ) < 5) continue;
				CSR_BLOCK* blkPtr = blkVec[row];
				int srcStart = blkPtr->srcStart;
				int srcEnd = blkPtr->srcEnd;

				for(int v = srcStart; v < srcEnd; v++){
					int ciaIdxStart = blkPtr->rpa[v - srcStart];
					int ciaIdxEnd = blkPtr->rpa[v + 1 - srcStart];
					for(int i = ciaIdxStart; i < ciaIdxEnd; i++){
						int ngbIdx = blkPtr->cia[i]; 
							tmpVertexProp[ngbIdx] += vertexProp[v] / outDeg[v];
					}
				}
			} 
			double end = getCurrentTimestamp();
			printf("[INFO] Offload part of CPU runtime is %lf ms\n", (end - begin) * 1000);
		}
	#endif
}
double hybridProcessing(
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
	paddingBlk(blkVec);
	const ScoreT base_score = (1.0f - kDamp) / vertexNum;
	isComplete = false;
	
	while(itNum[0] < MAX_ITER){	
		fpgaTaskDone = false;
		const double begin = getCurrentTimestamp();
		
		cond.notify_all();

		if(mode == 1)
			cpuSingleThreadFunc(csr);
		else
			cpuThreadFunc(csr);

		while(!fpgaTaskDone);

		const double apply_begin = getCurrentTimestamp();
		double error = 0;
		#pragma omp parallel for reduction(+:error)
		for(int i = 0; i < vertexNum; i++){
			PROP_TYPE tProp = tmpVertexProp[i];
			PROP_TYPE old_score = vertexProp[i];
			vertexProp[i] = base_score + kDamp * tProp;
			error += fabs(vertexProp[i] - old_score);
			tmpVertexProp[i] = 0;
		}
		printf(" %2d    %lf\n", itNum[0], error);
		const double end = getCurrentTimestamp();
		printf("[INFO] Apply phrase runtime is %f ms \n", (end - apply_begin) * 1000);
		
		runtime += (end - begin);
		itNum[0]++;
	}
	
	isComplete = true;
	cond.notify_all();
	
	return (runtime * 1000);//*1.0/CLOCKS_PER_SEC;
}
void * fpgaThreadRun(){
	setHardwareEnv();
	globalVarInit(csr, vertexNum, edgeNum);
	setKernelEnv();
	createKernels(vertexNum, edgeNum);
	kernelVarMap(vertexNum, edgeNum);
	globalVarInitDone = true;
	while(true){
		std::unique_lock<std::mutex> condLock(condMutex);
		cond.wait(condLock);
		if(isComplete) break;
		launchFPGA(csr, blkVec);
		fpgaTaskDone = true;
	}
	return 0;
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
	for(int i = 0; i < vertexNum; i++){
		vertexProp[i] = 1.0f / vertexNum;
		tmpVertexProp[i] = 0;
		activeVertices[i] = i;
	}
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

void csrReorganize(
	CSR* csr,
	std::vector<CSR_BLOCK*> &blkVec,
	const int &blkNum){
	// remove the FPGA partition from CSR
	CSR_BLOCK* blkPtr = blkVec[0];
	int k = 0;
	rpa[0] = 0;
	for(int i = 0; i < csr->vertexNum; i++){
		for(int j = csr->rpao[i]; j < csr->rpao[i + 1]; j ++ ){
			if((csr->ciao[j] < blkPtr -> sinkStart) ||  (csr->ciao[j] > blkPtr -> sinkEnd))
				cia[k ++] = csr->ciao[j];
		}
		rpa[i + 1] = k;
	}
	printf("edge number after reorganizeed %d \n", k);
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
			// find the partition with max degree
			if(csrBlkPtr->edgeNum > max_partition_degree){
				max_partition_degree = csrBlkPtr->edgeNum;
				fpga_partition_x = cordx;
				fpga_partition_y = cordy;
				fpga_partition = csrBlkPtr;
				printf("fpga_partition_x %d, fpga_partition_y %d, max_partition_degree %d \n",fpga_partition_x, fpga_partition_y,max_partition_degree);
			}
			//printf("cordx %d, cordy %d, degree %d \n",cordx, cordy,csrBlkPtr->edgeNum);
		}
	}

	for(int cordx = 0; cordx < blkNum; cordx++){
		CSR_BLOCK* csrBlkPtr = new CSR_BLOCK(cordx, fpga_partition_y, csr);
		blkVec.push_back(csrBlkPtr);
	}
}


// CPU thread related to main function 
int main(int argc, char **argv) {
	double begin;
	double end;
	double elapsedTime;
	int startVertexIdx;
	std::string gName = "rmat-23-16";
	std::string mode = "harp"; // or harp

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

	// Partition the CSR
	blkNum = (vertexNum + BLK_SIZE - 1)/BLK_SIZE;
	
	std::thread fpgaThread;
	fpgaThread = std::thread(fpgaThreadRun);
	while(!globalVarInitDone);
	//software processing on CPU
	//software processing on CPU
	std::cout << "software PageRank starts." << std::endl;
	processInit(vertexNum, edgeNum, startVertexIdx);
	begin = getCurrentTimestamp();
	singleThreadSWProcessing(csr, blkVec, ptProp, blkNum, vertexNum, startVertexIdx);
	end = getCurrentTimestamp();
	elapsedTime = (end - begin) * 1000;
	std::cout << "[INFO] singleThreadSWProcessing PR takes " << elapsedTime << " ms." << std::endl;

	printf("\n");

	std::cout << "software multi-thread PageRank starts." << std::endl;
	processInit(vertexNum, edgeNum, startVertexIdx);
	begin = getCurrentTimestamp();
	multiThreadSWProcessing(csr, blkVec, ptProp, blkNum, vertexNum, startVertexIdx);
	end = getCurrentTimestamp();
	elapsedTime = (end - begin) * 1000;
	std::cout << "[INFO] multiThreadSWProcessing PR takes " << elapsedTime << " ms." << std::endl;
	printf("\n");
	// software processing with only 2 partitions
	csrPartition(csr, blkVec, blkNum);
	csrReorganize(csr, blkVec, blkNum);

	std::cout << "soft partitioned PageRank starts." << std::endl;
	processInit(vertexNum, edgeNum, startVertexIdx);
	begin = getCurrentTimestamp();
	ptProcessing(csr, blkVec, ptProp, blkNum, vertexNum, startVertexIdx);
	end = getCurrentTimestamp();
	elapsedTime = (end - begin) * 1000;
	std::cout << "[INFO] ptProcessing PR takes " << elapsedTime << " ms." << std::endl;
	printf("\n");

	std::cout << "soft partitioned multi-thread PageRank starts." << std::endl;
	processInit(vertexNum, edgeNum, startVertexIdx);
	begin = getCurrentTimestamp();
	ptProcessingMultiThread (csr, blkVec, ptProp, blkNum, vertexNum, startVertexIdx);
	end = getCurrentTimestamp();
	elapsedTime = (end - begin) * 1000;
	std::cout << "[INFO] ptProcessingMultiThread PR takes " << elapsedTime << " ms." << std::endl;
	printf("\n");
	
	std::cout << "Hybrid cpu single thread + fpga processing." << std::endl;
	processInit(vertexNum, edgeNum, startVertexIdx);
	elapsedTime = hybridProcessing(csr, blkVec, hybridProp, blkNum, vertexNum, edgeNum, 2);
	std::cout << "[INFO] Hybrid cpu multi-thread + fpga processing takes " << elapsedTime << " ms." << std::endl;
	
	fpgaThread.join();
	freeResources();

	return 0;
}
