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
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include "config.h"
#include "graph.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <tr1/unordered_map>
using namespace std::tr1;
using namespace aocl_utils;

#define AOCL_ALIGNMENT 64

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;

//Each kernel uses an independent queue
	static cl_command_queue queueReadActiveVertices;
	static cl_command_queue queueReadNgbInfo;
	static cl_command_queue queueProcessEdge;
	static cl_command_queue queueUpdateNextFrontier;

	static cl_kernel readActiveVertices;
	static cl_kernel readNgbInfo;
	static cl_kernel processEdge;
	static cl_kernel updateNextFrontier;

	static cl_program program;
	static cl_int status;

	static PROP_TYPE* CPUResult;
	static PROP_TYPE* FPGAResult;

	static PROP_TYPE* vertexProp;
	static PROP_TYPE* tmpVertexProp;
	static int* inRowPointerArray;
	static int* inDeg;
	static int* inEdges;
	static PROP_TYPE* inEdgeProp;
	static int* outRowPointerArray;
	static int * outDeg;
	static int* outdeg_padding;
	static int* outEdges;
	static int* cia;
	static int* activeVertices;
	static STATUS_TYPE* activeStatus;
	static int* activeVertexNum;
	static int* nextActiveVertices;
	static int* itNum;
	static PROP_TYPE* semaphore;

//#define PR
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
static void dumpError(const char *str, cl_int status) {
	printf("%s\n", str);
	printf("Error code: %d\n", status);
}

static void freeResources(){
	// opencl environments
	if(readActiveVertices) clReleaseKernel(readActiveVertices);  
	if(readNgbInfo) clReleaseKernel(readNgbInfo);  
	if(processEdge) clReleaseKernel(processEdge);  
	if(updateNextFrontier) clReleaseKernel(updateNextFrontier);  

	if(program) clReleaseProgram(program);

	if(queueReadActiveVertices) clReleaseCommandQueue(queueReadActiveVertices);
	if(queueReadNgbInfo) clReleaseCommandQueue(queueReadNgbInfo);
	if(queueProcessEdge) clReleaseCommandQueue(queueProcessEdge);
	if(queueUpdateNextFrontier) clReleaseCommandQueue(queueUpdateNextFrontier);

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
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ, 
		(void *)outdeg_padding, sizeof(int) *nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)activeVertexNum, sizeof(int) * 4, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)inEdgeProp, sizeof(int) * 4, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)cia, sizeof(int) * edgeNum , 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)outEdges, sizeof(int) * 16 * edgeNum , 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	
	
	// traverse neighbor
	localStatus = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ, 
		(void *)vertexProp, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ, 
		(void *)tmpVertexProp, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)activeVertexNum, sizeof(int) * 4, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)semaphore, sizeof(PROP_TYPE) * 4, 0, NULL, NULL); 
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
	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)outdeg_padding, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)activeVertexNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)inEdgeProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)cia, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	
	localStatus = clEnqueueSVMUnmap(queueProcessEdge, (void *)vertexProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueProcessEdge, (void *)tmpVertexProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueProcessEdge, (void *)activeVertexNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueProcessEdge, (void *)semaphore, 0, NULL, NULL); 
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

cl_int setHardwareEnv(cl_uint &numPlatforms, cl_uint &numDevices){

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

	queueProcessEdge = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
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
#ifdef BFS
void processOnCPU(int nodeNum, int edgeNum){
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
#endif
#ifdef PR
typedef float ScoreT;
void processOnCPU(int nodeNum, int edgeNum){
	int itNum = 0;
	int max_iter = 11;
	const ScoreT init_score = 1.0f / nodeNum;
	const ScoreT base_score = (1.0f - kDamp) / nodeNum;
	std::vector<float> scores(nodeNum, init_score);
	std::vector<float> outgoing_contrib(nodeNum);
	for (int iter=0; iter < max_iter; iter++){
		double error = 0;
		for (int n=0; n < nodeNum; n++)
			{	if(outDeg[n]!=0)
					outgoing_contrib[n] = scores[n] / outDeg[n];
				else 
					outgoing_contrib[n] = 0;
			}
			for (int u=0; u < nodeNum; u++) {
				ScoreT incoming_total = 0;
				int start = inRowPointerArray[u];
				int num = inDeg[u];
				for(int j = 0; j < num; j++){
					incoming_total += outgoing_contrib[inEdges[start + j]];
				}
				ScoreT old_score = scores[u];
				scores[u] = base_score + kDamp * incoming_total;
				error += fabs(scores[u] - old_score);
			}
			printf(" %2d    %lf\n", iter, error);
			//if (error < epsilon)
			//	break;
		}
	}
#endif
#ifdef CC
#define NodeID int
	void processOnCPU(int nodeNum, int edgeNum){
		int itNum = 0;
		int max_iter = 19;
		int count = nodeNum;
		bool change = true;
		std::vector<int> comp(nodeNum);
		for (int n=0; n < nodeNum; n++) comp[n] = n;
			for (int u=0; u < nodeNum; u++) {
				int start = outRowPointerArray[u];
				int num = outDeg[u];
				for(int j = 0; j < num; j++){
					int v = cia[start + j];
					int comp_u = comp[u];
					int comp_v = comp[v];
					if (comp_u == comp_v) continue;
					NodeID high_comp = comp_u > comp_v ? comp_u : comp_v;
					NodeID low_comp = comp_u + (comp_v - high_comp);
					if (high_comp == comp[high_comp]) {
						comp[high_comp] = low_comp;
						count --;
					}
				}
			}
			for (NodeID n=0; n < nodeNum; n++) {
				while (comp[n] != comp[comp[n]]) {
					comp[n] = comp[comp[n]];
				}
			}
			unordered_map<NodeID, NodeID> count_hashmap;
			for (NodeID comp_i : comp)
				count_hashmap[comp_i] += 1;
			printf("the cluster num is %d \n", count_hashmap.size());
			printf("the cluster num is %d \n", count);
		}
#endif
// Read graph from edge based files
		Graph* createGraph(const std::string &gName, const std::string &mode){
			Graph* gptr;
			std::string dir;
			if(mode == "harp") dir = "./";
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
			else if(gName == "rmat-12-4"){
				gptr = new Graph(dir + "rmat-12-4.txt");
			}
			else if(gName == "rmat-12-16"){
				gptr = new Graph(dir + "rmat-12-16.txt");
			}
			else if(gName == "rmat-14-32"){
				gptr = new Graph(dir + "rmat-14-32.txt");
			}
			else if(gName == "rmat-17-32"){
				gptr = new Graph(dir + "rmat-17-32.txt");
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
// define the padding number to int4 int8 int 16


void processInit(CSR* csr, const int &nodeNum, const int &edgeNum, const int &source){
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
		cia[i] = csr->ciao[i];
	}
#ifdef PR
	for(int i = 0; i < nodeNum; i++){
		vertexProp[i] = 1.0f / nodeNum;
		tmpVertexProp[i] = 1.0f / nodeNum;
		activeVertices[i] = i;
	}
	activeVertexNum[0] = nodeNum;
#endif
}


void padding (CSR* csr, const int &nodeNum, const int &edgeNum, int pading_num){
// pre-processing to parallel : padding
	int pad = -1;
	int edgesIdx = 0;
	int threshold = (nodeNum % 4)? (nodeNum/4 + 1) : (nodeNum / 4); 
	int outEdgesOld = 0;
	for(int i = 0; i < nodeNum; i++){
		int start = outRowPointerArray[i];
		int num = outDeg[i];

		std::vector<std::vector<int>> bankVec(pading_num);

		for(int j = 0; j < num; j++){
			int ngbIdx = csr->ciao[start + j];
			bankVec[ngbIdx % pading_num].push_back(ngbIdx);
		}
		outEdgesOld = edgesIdx;
		outRowPointerArray[i] = edgesIdx;

		bool empty_flag = false;
		for(int i = 0; i < pading_num; i ++) empty_flag |= (!bankVec[i].empty());

		while (empty_flag){
			for(int i = 0; i < pading_num; i ++){
				if(!bankVec[i].empty()){
					outEdges[edgesIdx ++] = bankVec[i].back();
					bankVec[i].pop_back();
				}
				else
					outEdges[edgesIdx ++] = pad;
			}

			empty_flag = false;
			for(int i = 0; i < pading_num; i ++) empty_flag |= (!bankVec[i].empty());
		}
		outdeg_padding[i] = edgesIdx - outEdgesOld;
	}

}

// Reset part of the input data for repeated execution.
void processReset(const int &nodeNum, const int &source){
	for(int i = 0; i < nodeNum; i++){
		vertexProp[i] = 1.0f / nodeNum;
		tmpVertexProp[i] = 1.0f / nodeNum;
		//vertexProp[i] = MAX_PROP;
		activeStatus[i] = 0;
		activeVertices[i] = i;
	}
	//vertexProp[source] = 0;
	activeVertexNum[0] = nodeNum;
}

int main(int argc, char ** argv){

	cl_uint numPlatforms;
	cl_uint numDevices;

	status = setHardwareEnv(numPlatforms, numDevices);
	std::cout << "Creating device memory objects." << std::endl;
	
	int source = 365723;
	std::string mode = "sim";
	std::string graphName = "lj1";
	if(graphName == "youtube")    source = 320872;
	if(graphName == "lj1")        source = 3928512;
	if(graphName == "pokec")      source = 182045;
	if(graphName == "rmat-19-32") source = 104802;
	if(graphName == "rmat-21-32") source = 365723;
	if(graphName == "rmat-12-4")  source = 100;
	if(graphName == "rmat-12-16")  source = 1100;
	
	Graph* gptr = createGraph(graphName, mode);
	CSR* csr = new CSR(*gptr);
	free(gptr);

	std::cout << "Graph is loaded." << std::endl;
	int nodeNum = align(csr->vertexNum, 8, 8*1024); 
	int edgeNum = csr->ciao.size();
	std::cout << "node num: " << csr->vertexNum << " is aligned to " << nodeNum << std::endl;
	std::cout << "edge num: " << edgeNum << std::endl;

	// Result of CPU and FPGA
	PROP_TYPE* CPUResult = (PROP_TYPE*) malloc(sizeof(PROP_TYPE) * nodeNum);
	PROP_TYPE* FPGAResult = (PROP_TYPE*) malloc(sizeof(PROP_TYPE) * nodeNum);

	// Declare shared memory objects
	vertexProp         = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * nodeNum, 1024); 
	tmpVertexProp      = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * nodeNum, 1024); 
	inRowPointerArray  = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
	inDeg              = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
	inEdges            = (int*) clSVMAllocAltera(context, 0, sizeof(int) * edgeNum, 1024);
	inEdgeProp         = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * edgeNum, 1024);
	outRowPointerArray = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
	outDeg             = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
	outdeg_padding     = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
	outEdges           = (int*) clSVMAllocAltera(context, 0, sizeof(int) * edgeNum * 3, 1024);
	cia           	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * edgeNum, 1024);
	activeVertices     = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
	nextActiveVertices = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);  
	itNum     		   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * 2, 1024); 
	semaphore  		   = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * 2, 1024); 
	memset(semaphore,sizeof(PROP_TYPE) * 2, 0);
	// 0-->vertex is inactive, 1--> vertex is active
	activeStatus       = (STATUS_TYPE*) clSVMAllocAltera(context, 0, sizeof(STATUS_TYPE) * nodeNum, 1024); 
	activeVertexNum    = (int*) clSVMAllocAltera(context, 0, sizeof(int)*4, 1024);

	if(!inRowPointerArray || !outDeg || !outRowPointerArray || !inDeg || !activeVertices
		|| !activeStatus || !vertexProp || !inEdges || !inEdgeProp || !outEdges ||!outdeg_padding ||!activeVertexNum || !cia || !semaphore || !itNum) 
	{
		dumpError("Failed to allocate buffers.", status);
		freeResources();
		return 1;	
	}
	std::cout << "Memory allocation is done." << std::endl;
	// Graph processing on CPU
	processInit(csr, nodeNum, edgeNum, source);	
	//padding 
	padding(csr, nodeNum, edgeNum, 16);	

	std::cout << "Initialization is done." << std::endl;

	 const double cpu_start = getCurrentTimestamp();
	 processOnCPU(nodeNum, edgeNum);
	 const double cpu_end = getCurrentTimestamp();
	 printf("CPU runtime is: %f\n", cpu_end - cpu_start);
	// std::cout << "Graph processing on CPU is done." << std::endl;
	processReset(nodeNum, source);
	//processInit(csr, nodeNum, edgeNum, source);	
	int padding_num = 0;
	for(int i = 0; i < nodeNum; i++){
		int start = outRowPointerArray[i];
		int num = outdeg_padding[i];
		//printf("\n vertex %d num %d start %d\t",i,num,start);
		for(int j = start; j < start + num; j ++){
			if(outEdges[i] == -1) padding_num ++;
			//printf("%d \t", outEdges[j]);
		}
	}
	printf("original edgeNum %d padding num %d total edges %d \n",edgeNum, padding_num, (edgeNum + padding_num));
	
#if 0
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

		processEdge = clCreateKernel(program, "processEdge", &status);
		if(status != CL_SUCCESS) {
			dumpError("Failed clCreateKernel processEdge.", status);
			freeResources();
			return 1;
		}

		// set kernel arguments
		std::cout << "Set kernel arguments." << std::endl;
		printf ("nodeNum is %d, edgeNum is %d, itNum is %d \n",nodeNum, edgeNum, itNum[0]);
		printf ("activeVertexNum is %d\n",activeVertexNum[0]);

		clSetKernelArgSVMPointerAltera(readActiveVertices, 0, (void*)activeVertices);	
		clSetKernelArgSVMPointerAltera(readActiveVertices, 1, (void*)activeVertexNum);

		clSetKernelArgSVMPointerAltera(readNgbInfo, 0, (void*)outRowPointerArray);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 1, (void*)outdeg_padding);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 2, (void*)outDeg);	
		clSetKernelArgSVMPointerAltera(readNgbInfo, 3, (void*)outEdges);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 4, (void*)inEdgeProp);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 5, (void*)activeVertexNum);
		clSetKernelArg(readNgbInfo, 6, sizeof(int), (void*)&nodeNum);
		clSetKernelArg(readNgbInfo, 7, sizeof(int), (void*)&edgeNum);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 8, (void*)itNum);

		clSetKernelArgSVMPointerAltera(processEdge, 0, (void*)vertexProp);
		clSetKernelArgSVMPointerAltera(processEdge, 1, (void*)tmpVertexProp);
		clSetKernelArgSVMPointerAltera(processEdge, 2, (void*)semaphore);
		clSetKernelArg(processEdge, 3, sizeof(int), (void*)&nodeNum);
		clSetKernelArgSVMPointerAltera(processEdge, 4, (void*)itNum);
	}

	// Run BFS iteration
	std::cout << "start the processing iterations." << std::endl;
	cl_event eventReadActiveVertices;
	cl_event eventReadNgbInfo;
	cl_event eventProcessEdge;

	const double start_time = getCurrentTimestamp();
	auto startTime = std::chrono::high_resolution_clock::now();
	itNum[1] = nodeNum;
for(int i = 0; i < 1; i ++){
	itNum[0] = 0;
	activeVertexNum[0] = nodeNum; //128 * 4 * i;
	bool loop_conditon = true;
	double k0Time = 0; 
	// exit conditon 
	auto c0 = std::chrono::high_resolution_clock::now();
	while(loop_conditon){

		//auto c0 = std::chrono::high_resolution_clock::now();

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

		status = clEnqueueTask(queueProcessEdge, processEdge, 0, NULL, &eventProcessEdge);
		if(status != CL_SUCCESS){
			dumpError("Failed to launch scatter.", status);
			freeResources();
			return 1;
		}	

		clFinish(queueReadActiveVertices);
		clFinish(queueReadNgbInfo);
		clFinish(queueProcessEdge);

		// auto c1 = std::chrono::high_resolution_clock::now();

		// k0Time +=  std::chrono::duration_cast<std::chrono::milliseconds>(c1 - c0).count();

		#ifdef PR
		PROP_TYPE error = 0;
		for(int i = 0; i < nodeNum; i++){
			tmpVertexProp[i] = (1.0 - kDamp) / nodeNum + kDamp * tmpVertexProp[i];
			error += fabs(tmpVertexProp[i] - vertexProp[i]);
			vertexProp[i] = tmpVertexProp[i];
		}
		printf(" %2d    %lf\n",  itNum[0], error);
		if (itNum[0] == 10)
			break;
		loop_conditon = true;
		itNum[0]++;
		#endif

	}

	auto c1 = std::chrono::high_resolution_clock::now();

	k0Time =  std::chrono::duration_cast<std::chrono::milliseconds>(c1 - c0).count();
	//std::cout << " FPGA traverse runtime: " << k0Time << std::endl;
	//compute the edges number
	int total_edges = 0;
	for(int i = 0; i < activeVertexNum[0]; i++){
		int num = outdeg_padding[i];
		total_edges += num;//printf("%d \t", outEdges[j]);
	}
	//std::cout << activeVertexNum[0] <<"activeVertices," << " edge number: " << total_edges << std::endl;
	std::cout << activeVertexNum[0] <<" " << k0Time << " " << total_edges << std::endl;
}
	const double end_time = getCurrentTimestamp();
	auto endTime = std::chrono::high_resolution_clock::now();
	auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout << " runtime: " << totalTime << std::endl;
	double time = (end_time - start_time);
	printf(" FPGA runtime is: %f\n", time);
  // Copy FPGA result
	// for(int i = 0; i < nodeNum; i++){
	// 	FPGAResult[i] = vertexProp[i];
	// }
	// // Verification step	
	// bool passed = true; 
	// for(int i = 0; i < nodeNum; i++){
	// 	if(CPUResult[i] != FPGAResult[i]){
	// 		passed =  false;
	// 		break;
	// 	}
	// }
	// std::cout << "Verification: " << (passed? "Passed": "Failed") << std::endl;
	varUnmap();
	freeResources();
#endif
	return 0;
}
