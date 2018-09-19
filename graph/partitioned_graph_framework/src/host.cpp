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
#include <cstdio>
#include <ctime>
#include "config.h"
#include "graph.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <tr1/unordered_map>
using namespace std::tr1;
using namespace aocl_utils;
//Each kernel uses an independent queue
{
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
	static int* vPropRange;
	static int* sinkRange;
	static int* inRowPointerArray;
	static int* inDeg;
	static int* inEdges;
	static PROP_TYPE* inEdgeProp;
	static int* outRowPointerArray;
	static int* outDeg;
	static int* rpaNum;
	static int* outEdges;
	static int* cia;
	static int* activeVertices;
	static STATUS_TYPE* activeStatus;
	static int* activeVertexNum;
	static int* nextActiveVertices;
	static int* itNum;
	static int* edgeNum_g;
	static int* vertexNum_g;
	static PROP_TYPE* semaphore;

	static cl_platform_id platform;
	static cl_device_id device;
	static cl_context context;
}

#define AOCL_ALIGNMENT 64

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
static const char *error_message =
"Error: Result mismatch:\n"
"i = %d CPU result = %d Device result = %d\n";
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
		(void *)rpaNum, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ, 
		(void *)outDeg, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)activeVertexNum, sizeof(int) * 4, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)inEdgeProp, sizeof(int) * 4, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)cia, sizeof(int) * edgeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)edgeNum_g, sizeof(int) * 2, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueReadNgbInfo, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)vertexNum_g, sizeof(int) * 2, 0, NULL, NULL); 
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
	localStatus = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)vertexNum_g, sizeof(int) * 2, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)vPropRange, sizeof(int) * 2, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueProcessEdge, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)sinkRange, sizeof(int) * 2, 0, NULL, NULL); 
	statusVec.push_back(localStatus);

	// write status
	localStatus = clEnqueueSVMMap(queueUpdateNextFrontier, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
		(void *)activeVertices, sizeof(int) * nodeNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMMap(queueUpdateNextFrontier, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
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
	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)rpaNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)outDeg, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)activeVertexNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)inEdgeProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)cia, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)vertexNum_g, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueReadNgbInfo, (void *)edgeNum_g, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	
	localStatus = clEnqueueSVMUnmap(queueProcessEdge, (void *)vertexProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueProcessEdge, (void *)tmpVertexProp, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueProcessEdge, (void *)activeVertexNum, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueProcessEdge, (void *)semaphore, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueProcessEdge, (void *)vertexNum_g, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueProcessEdge, (void *)vPropRange, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueProcessEdge, (void *)sinkRange, 0, NULL, NULL); 
	statusVec.push_back(localStatus);


	localStatus = clEnqueueSVMUnmap(queueUpdateNextFrontier, (void *)activeVertices, 0, NULL, NULL); 
	statusVec.push_back(localStatus);
	localStatus = clEnqueueSVMUnmap(queueUpdateNextFrontier, (void *)activeVertexNum, 0, NULL, NULL); 
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

	queueUpdateNextFrontier = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &localStatus);
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
	else{
		std::cout << "Unknown graph name." << std::endl;
		exit(EXIT_FAILURE);
	}

	return gptr;
}
void propInit(
	int vertexNum, 
	PROP_TYPE* props
	)
{
	for(int i = 0; i < vertexNum; i++){
		props[i]  = MAX_PROP;
	}
}

void propInit(
	int vertexNum, 
	PROP_TYPE* props,
	const int &startVertexIdx
	)
{
	for(int i = 0; i < vertexNum; i++){
		props[i]  = MAX_PROP;
	}

	props[startVertexIdx] = 0;
}

void propInit(
	int vertexNum, 
	PROP_TYPE* props,
	std::vector<int> &activeVertices
	)
{
	activeVertices.clear();
	for(int i = 0; i < vertexNum; i++){
		activeVertices.push_back(i);
	}

	for(int i = 0; i < vertexNum; i++){
		props[i]  = MAX_PROP;
	}
}

void propInit(
	int vertexNum, 
	PROP_TYPE* props, 
	std::vector<int> &activeVertices,
	const int &startVertexIdx
	)
{
	for(int i = 0; i < vertexNum; i++){
		props[i]  = MAX_PROP;
	}

	props[startVertexIdx] = 0;
	activeVertices.clear();
	activeVertices.push_back(startVertexIdx);
}
#ifdef PR
typedef float ScoreT;
void swProcessing(int nodeNum, int edgeNum){
	int itNum = 0;
	int max_iter = 19;
	const ScoreT init_score = 1.0f / nodeNum;
	const ScoreT base_score = (1.0f - kDamp) / nodeNum;
	std::vector<float> scores(nodeNum, init_score);
	std::vector<float> outgoing_contrib(nodeNum);

	for (int iter=0; iter < max_iter; iter++){
		double error = 0;
		for (int n=0; n < nodeNum; n++)
			{	if(outDeg[n]!=0)
				outgoing_contrib[n] = vertexProp[n] / outDeg[n];
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
				ScoreT old_score = vertexProp[u];
				vertexProp[u] = base_score + kDamp * incoming_total;
				error += fabs(vertexProp[u] - old_score);
			}
			printf(" %2d    %lf\n", iter, error);
			if (error < epsilon)
				break;
		}
	}
#endif
#ifdef CC
#define NodeID int
	void swProcessing(int nodeNum, int edgeNum){
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
#ifdef BFS
		void swProcessing(
			CSR* csr, 
			PROP_TYPE* swProp,
			PROP_TYPE* tmpProp,
			std::vector<int> &activeVertices
			)
		{
			int itNum = 0;
			while(activeVertices.size() > 0){
				std::cout << "software processing, iteration: " << itNum << std::endl;
		// Traverse active vertices and process each outgoing edges
				for(auto v : activeVertices){
					int start = csr->rpao[v];
					int end = csr->rpao[v+1];
					PROP_TYPE uProp = swProp[v];
					for(int i = start; i < end; i++){
						int ngbVidx = csr->ciao[i];
				//PROP_TYPE eProp = csr->eProps[i];
						tmpProp[ngbVidx] = compute(uProp, 1, tmpProp[ngbVidx]);
					}
				}

		// Decide the frontier
				activeVertices.clear();
				for(int i = 0; i < csr->vertexNum; i++){
					PROP_TYPE vProp = swProp[i];
					PROP_TYPE tProp = tmpProp[i];
					bool update = updateCondition(vProp, tProp);
					if(update){
						swProp[i] = tProp;
						activeVertices.push_back(i);
					}
				}

				itNum++;
			}
		}
#endif
// Iterate the partitioned CSR for BFS
#ifdef BFS
		void ptProcessing(
			std::vector<CSR_BLOCK*> &blkVec, 
			PROP_TYPE* ptProp, 
			PROP_TYPE* tmpProp,
			std::vector<int> &activeVertices,
			const int &blkNum,
			const int &vertexNum
			)
		{
			int itNum = 0;
			while(activeVertices.size() > 0){
				std::cout << "Processing with partition, iteration: " << itNum << std::endl;
		// process blocks in a column major
        //#pragma omp parallel for
				for(int i = 0; i < blkNum; i++){
					int srcStart = blkVec[i*blkNum]->srcStart;
					int srcEnd = blkVec[i*blkNum]->srcEnd;

					std::vector<int> frontier;
					for(auto v : activeVertices){
						if(v >= srcStart && v < srcEnd){
							frontier.push_back(v);
						}
					}
					if(!frontier.size()) continue;
			//if(itNum < 4)
					printf("%d vertices added to activeVertices[]\n",frontier.size());
					for(int j = 0; j < blkNum; j++){
						int blkIdx = i * blkNum + j;
						CSR_BLOCK* blkPtr = blkVec[blkIdx];	
						int sinkStart = blkPtr->sinkStart;
						int sinkEnd = blkPtr->sinkEnd;
				// Traverse active frontiers that are in the block
						for(auto v : frontier){
							int ciaIdxStart = blkPtr->rpa[v - srcStart];
							int ciaIdxEnd = blkPtr->rpa[v + 1 - srcStart];
							{
								if(itNum< 0) 
									printf("vidx %d, Buffer idx %d ngbNUM %d, srcstart %d, ciaostart %d \n", v,v - srcStart, ciaIdxEnd - ciaIdxStart, srcStart, ciaIdxStart);		
								for(int k = ciaIdxStart; k < ciaIdxEnd; k++){
									int ngbIdx = blkPtr->cia[k];
							//PROP_TYPE eProp = blkVec[blkIdx]->eProps[k];
									tmpProp[ngbIdx] = compute(ptProp[v], 1, tmpProp[ngbIdx]);
								}
						//if(itNum <= 1) printf("details:: vidx %d ciaIdxNUM %d ciaStart %d\n", v, ciaIdxEnd - ciaIdxStart, ciaIdxStart);
							}
						}
					}
				}
		// Decide active vertices and apply
				activeVertices.clear();
				for(int i = 0; i < vertexNum; i++){
					PROP_TYPE vProp = ptProp[i];
					PROP_TYPE tProp = tmpProp[i];
					bool update = updateCondition(vProp, tProp);
					if(update){
						ptProp[i] = tProp;
						activeVertices.push_back(i);
					}
				}

				itNum++;
			}
		}
#endif
#ifdef PR
		void ptProcessing(
			std::vector<CSR_BLOCK*> &blkVec, 
			PROP_TYPE* ptProp, 
			PROP_TYPE* tmpProp,
			std::vector<int> &activeVertices,
			const int &blkNum,
			const int &vertexNum
			)
		{
			int itNum = 0;
			int loop_conditon = 1;
			while(loop_conditon){
				std::cout << "Processing with partition, iteration: " << itNum << std::endl;
		// process blocks in a column major
        //#pragma omp parallel for
				for(int i = 0; i < blkNum; i++){
					int srcStart = blkVec[i*blkNum]->srcStart;
					int srcEnd = blkVec[i*blkNum]->srcEnd;

					std::vector<int> frontier;
					for(auto v : activeVertices){
						if(v >= srcStart && v < srcEnd){
							frontier.push_back(v);
						}
					}
					if(!frontier.size()) continue;
			//if(itNum < 4)
			//printf("%d vertices added to activeVertices[]\n",frontier.size());
					for(int j = 0; j < blkNum; j++){
						int blkIdx = i * blkNum + j;
						CSR_BLOCK* blkPtr = blkVec[blkIdx];	
						int sinkStart = blkPtr->sinkStart;
						int sinkEnd = blkPtr->sinkEnd;
				// Traverse active frontiers that are in the block
						for(auto v : frontier){
							int ciaIdxStart = blkPtr->rpa[v - srcStart];
							int ciaIdxEnd = blkPtr->rpa[v + 1 - srcStart];
					//int outDeg = ciaIdxEnd - ciaIdxStart;
							{
								for(int k = ciaIdxStart; k < ciaIdxEnd; k++){
									int ngbIdx = blkPtr->cia[k];
							//PROP_TYPE eProp = blkVec[blkIdx]->eProps[k];
									if(outDeg)
										tmpProp[ngbIdx] += ptProp[v] / outDeg[v]; 
								}
						//if(itNum <= 1) printf("details:: vidx %d ciaIdxNUM %d ciaStart %d\n", v, ciaIdxEnd - ciaIdxStart, ciaIdxStart);
							}
						}
					}
				}
		// Decide active vertices and apply

				PROP_TYPE error = 0;
			//activeVertices_v.clear();
				for(int i = 0; i < vertexNum; i++){
					PROP_TYPE incoming_score = tmpProp[i] - ptProp[i];
					tmpProp[i] = (1.0 - kDamp) / vertexNum + kDamp * incoming_score;
				//if(i < 100) printf("tmpVertexProp %f , vertexProp %f",tmpProp[i], ptProp[i]);
				//if(i < 100) printf("incmingscore %f \n",incoming_score);
					error += fabs(tmpProp[i] - ptProp[i]);
				//activeVertices_v.push_back(i);
				}
				for(int i = 0; i < vertexNum; i++){
					ptProp[i] = tmpProp[i];
				}
				printf("error is %lf iteration %2d %d vertices added to activeVertices[]\n",error, itNum, activeVertices.size());
				loop_conditon = error < epsilon? 0 : 1;
				itNum++;
			}
		}
#endif
		int processInit(CSR* csr, const int nodeNum, 
			const int edgeNum, const int source, 	
			PROP_TYPE* props, 
			PROP_TYPE* tmpProps, 
			std::vector<int> &activeVertices_v,
			const int &startVertexIdx){
			
			cl_uint numPlatforms;
			cl_uint numDevices;
			std::cout << "Creating device memory objects." << std::endl;
			status = setHardwareEnv(numPlatforms, numDevices);
	// malloc shared memorys 
			vertexProp         = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * nodeNum, 1024); 
			tmpVertexProp      = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * nodeNum, 1024);
			inRowPointerArray  = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
			inDeg              = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
			inEdges            = (int*) clSVMAllocAltera(context, 0, sizeof(int) * edgeNum, 1024);
			inEdgeProp         = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * edgeNum, 1024);
			outRowPointerArray = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
			outDeg             = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
			rpaNum             = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
			outEdges           = (int*) clSVMAllocAltera(context, 0, sizeof(int) * edgeNum, 1024);
			cia           	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * edgeNum, 1024);
			activeVertices     = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);
			nextActiveVertices = (int*) clSVMAllocAltera(context, 0, sizeof(int) * nodeNum, 1024);  
			itNum     		   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * 2, 1024); 
			edgeNum_g     	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * 2, 1024); 
			vertexNum_g 	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * 2, 1024);
			vPropRange 	   	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * 2, 1024);  
			sinkRange 	   	   = (int*) clSVMAllocAltera(context, 0, sizeof(int) * 2, 1024);  
			semaphore  		   = (PROP_TYPE*) clSVMAllocAltera(context, 0, sizeof(PROP_TYPE) * 2, 1024); 
			memset(semaphore,sizeof(PROP_TYPE) * 2, 0);
			activeStatus       = (STATUS_TYPE*) clSVMAllocAltera(context, 0, sizeof(STATUS_TYPE) * nodeNum, 1024); 
			activeVertexNum    = (int*) clSVMAllocAltera(context, 0, sizeof(int)*4, 1024);
			if(!inRowPointerArray || !outDeg || !outRowPointerArray || !inDeg || !activeVertices
				|| !activeStatus || !vertexProp || !inEdges || !inEdgeProp || !outEdges || !activeVertexNum || !cia 
				|| !semaphore || !itNum || !edgeNum_g || !vertexNum_g || !tmpVertexProp || !sinkRange || !rpaNum) 
			{
				dumpError("Failed to allocate buffers 2.", status);
				freeResources();
				return 1;	
			}
			std::cout << "Memory allocation is done." << std::endl;
			
	// init buffers 
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
				tmpVertexProp[i] = MAX_PROP;
				activeStatus[i] = 0;	
			}
			for(int i = 0; i < edgeNum; i++){
				inEdges[i] = csr->ciai[i];
		inEdgeProp[i] = 1; // when it is BFS
		outEdges[i] = csr->ciao[i];
		cia[i] = csr->ciao[i];
	}
#ifdef PR
	activeVertices_v.clear();
	for(int i = 0; i < nodeNum; i++){
		vertexProp[i] = 1.0f / nodeNum;
		tmpVertexProp[i] = 1.0f / nodeNum;
		activeVertices[i] = i;
		activeVertices_v.push_back(i);
	}
	activeVertexNum[0] = nodeNum;
#endif
#ifdef CC
	activeVertices_v.clear();
	for(int i = 0; i < nodeNum; i++){
		vertexProp[i] = i;
		activeVertices[i] = i;
		activeVertices_v.push_back(i);
	}
	activeVertexNum[0] = nodeNum;
#endif 
#ifdef BFS
	activeVertices_v.clear();
	vertexProp[source] = 0;
	tmpVertexProp[source] = 0;
	activeVertexNum[0] = 1;
	activeVertices[0] = source;
	activeVertices_v.push_back(source);
#endif

	for(int i = 0; i < nodeNum; i ++){
		props[i] = vertexProp[i];
		tmpProps[i] = vertexProp[i];
	}
	//printf ("nodeNum is %d, edgeNum is %d \n",nodeNum, edgeNum);
	//printf ("activeVertexNum is %d\n",activeVertexNum[0]);
	return 1;
}
int ptProcessingOnFpga (
	std::vector<CSR_BLOCK*> &blkVec, 
	PROP_TYPE* ptProp, 
	PROP_TYPE* tmpProp,
	std::vector<int> &activeVertices_v,
	const int &blkNum,
	const int &vertexNum,
	const int edgeNum
	)
{
	// the num for one partition block 
	
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

		updateNextFrontier = clCreateKernel(program, "updateNextFrontier", &status);
		if(status != CL_SUCCESS) {
			dumpError("Failed clCreateKernel updateNextFrontier.", status);
			freeResources();
			return 1;
		}
		// set kernel arguments
		std::cout << "Set kernel arguments." << std::endl;

		clSetKernelArgSVMPointerAltera(readActiveVertices, 0, (void*)activeVertices);	
		clSetKernelArgSVMPointerAltera(readActiveVertices, 1, (void*)activeVertexNum);

		clSetKernelArgSVMPointerAltera(readNgbInfo, 0, (void*)outRowPointerArray);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 1, (void*)rpaNum);	
		clSetKernelArgSVMPointerAltera(readNgbInfo, 2, (void*)outDeg);	
		clSetKernelArgSVMPointerAltera(readNgbInfo, 3, (void*)cia);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 4, (void*)inEdgeProp);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 5, (void*)activeVertexNum);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 6, (void*)vertexNum_g);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 7, (void*)edgeNum_g);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 8, (void*)itNum);
		clSetKernelArgSVMPointerAltera(readNgbInfo, 9, (void*)vPropRange);

		clSetKernelArgSVMPointerAltera(processEdge, 0, (void*)vertexProp);
		clSetKernelArgSVMPointerAltera(processEdge, 1, (void*)tmpVertexProp);
		clSetKernelArgSVMPointerAltera(processEdge, 2, (void*)activeVertexNum);
		clSetKernelArgSVMPointerAltera(processEdge, 3, (void*)semaphore);
		clSetKernelArgSVMPointerAltera(processEdge, 4, (void*)vertexNum_g);
		clSetKernelArgSVMPointerAltera(processEdge, 5, (void*)itNum);
		clSetKernelArgSVMPointerAltera(processEdge, 6, (void*)vPropRange);
		clSetKernelArgSVMPointerAltera(processEdge, 7, (void*)sinkRange);

		clSetKernelArgSVMPointerAltera(updateNextFrontier, 0, (void*)nextActiveVertices);
		clSetKernelArgSVMPointerAltera(updateNextFrontier, 1, (void*)activeVertexNum);
		clSetKernelArg(updateNextFrontier, 2, sizeof(int), (void*)&vertexNum);
		varMap(vertexNum, edgeNum);
		std::cout << "End of kernel argument setup." << std::endl;
	}
	// Run BFS iteration
	std::cout << "start the processing iterations." << std::endl;
	cl_event eventReadActiveVertices;
	cl_event eventReadNgbInfo;
	cl_event eventProcessEdge;
	cl_event eventUpdateNextFrontier;

	const double start_time = getCurrentTimestamp();
	auto startTime = std::chrono::high_resolution_clock::now();
	double k0Time = 0; 
	itNum[0] = 0;
	itNum[1] = vertexNum;
	bool loop_conditon = true;
	while(loop_conditon){
		std::cout << "Processing with partition, iteration: " << itNum[0] << std::endl;
		// process blocks in a column major
        //#pragma omp parallel for
		for(int i = 0; i < blkNum; i++){
			int srcStart = blkVec[i*blkNum]->srcStart;
			int srcEnd = blkVec[i*blkNum]->srcEnd;

			std::vector<int> frontier;
			int k = 0;
			for(auto v : activeVertices_v){
				if(v >= srcStart && v < srcEnd){
					frontier.push_back(v);
					activeVertices[k++] = v; 
				}
			}

			if(!k) continue;
			for(int j = 0; j < blkNum; j++){
				int blkIdx = i * blkNum + j;
				CSR_BLOCK* blkPtr = blkVec[blkIdx];	
				int sinkStart = blkPtr->sinkStart;
				int sinkEnd = blkPtr->sinkEnd;
				{	//printf("col %d sinkStart %d , sinkEnd %d \n", blkNum, sinkStart,sinkEnd);
					//assign args for every iteration
				activeVertexNum[0] = frontier.size();
					vertexNum_g[0] = srcEnd-srcStart; //blkPtr->vertexNum;
					edgeNum_g[0] = blkPtr->edgeNum;//sinkEnd- sinkStart;//blkPtr->edgeNum;
					//printf("fpga edgeNum : dst %d edgeNum %d",edgeNum_g[0],blkPtr->edgeNum);
					//printf("fpga vertexNum : src %d vertexNum %d \n",vertexNum_g[0], blkPtr->vertexNum);
					vPropRange[0] = srcStart;
					vPropRange[1] = srcEnd;
					sinkRange[0] = sinkStart;
					sinkRange[1] = sinkEnd;
					for(int i = srcStart; i < srcEnd; i ++){	
						outRowPointerArray[i - srcStart] = blkPtr->rpa[i - srcStart];
					}
					for(int i = 0; i < activeVertexNum[0]; i ++){
						rpaNum[activeVertices[i]-srcStart] = blkPtr->rpa[activeVertices[i] + 1 -srcStart] - blkPtr->rpa[activeVertices[i] - srcStart];
					}
					for(int i = 0; i < blkPtr->edgeNum; i ++){	
						inEdgeProp[i] = blkPtr->eProps[i];
						cia[i] = blkPtr->cia[i];
					}
					// fpga start
					auto c0 = std::chrono::high_resolution_clock::now();
					status = clEnqueueTask(queueReadActiveVertices, readActiveVertices, 0, NULL, &eventReadActiveVertices);
					if(status != CL_SUCCESS){
						dumpError("Failed to launch readActiveVertices.", status);
						freeResources();
						return 1;
					}	
					status = clEnqueueTask(queueReadNgbInfo, readNgbInfo, 0, NULL, &eventReadNgbInfo);
					if(status != CL_SUCCESS){
						dumpError("Failed to launch readNgbInfo.", status);
						freeResources();
						return 1;
					}
					status = clEnqueueTask(queueProcessEdge, processEdge, 0, NULL, &eventProcessEdge);
					if(status != CL_SUCCESS){
						dumpError("Failed to launch processEdge.", status);
						freeResources();
						return 1;
					}	
					clFinish(queueReadActiveVertices);
					clFinish(queueReadNgbInfo);
					clFinish(queueProcessEdge);
					clFinish(queueUpdateNextFrontier);
					auto c1 = std::chrono::high_resolution_clock::now();
					k0Time += std::chrono::duration_cast<std::chrono::milliseconds>(c1 - c0).count();
				}
			}
		}
		#ifdef PR
		PROP_TYPE error = 0;
			//activeVertices_v.clear();
		for(int i = 0; i < vertexNum; i++){
			PROP_TYPE incoming_score = tmpVertexProp[i] - vertexProp[i];
			tmpVertexProp[i] = (1.0 - kDamp) / vertexNum + kDamp * incoming_score;
				//if(i < 100) printf("tmpVertexProp %f , vertexProp %f",tmpVertexProp[i], vertexProp[i]);
				//if(i < 100) printf("incmingscore %f \n",incoming_score);
			error += fabs(tmpVertexProp[i] - vertexProp[i]);
				//activeVertices_v.push_back(i);
		}
		for(int i = 0; i < vertexNum; i++){
			vertexProp[i] = tmpVertexProp[i];
		}
		printf("%d vertices added to activeVertices[]\n",activeVertices_v.size());
		#else
		// Decide active vertices and apply
		printf("general update condition \n");
		activeVertices_v.clear();
		for(int i = 0; i < vertexNum; i++){
			PROP_TYPE vProp = vertexProp[i];
			PROP_TYPE tProp = tmpVertexProp[i];
			bool update = updateCondition(vProp, tProp);
			if(update){
				vertexProp[i] = tProp;
				ptProp[i] = tProp;
				activeVertices_v.push_back(i);
			}
		}
		#endif
		// define the loop condition
		#ifdef BFS
		std::cout << "BFS level = " << itNum[0] << " BFS next frontier = " << activeVertices_v.size() << std::endl;
		itNum[0]++;
		loop_conditon = (activeVertices_v.size() > 0)? 1 : 0;
		#endif
		#ifdef SSSP
		std::cout << "SSSP level = " << itNum[0] << " next frontier = " << activeVertices_v.size() << std::endl;
		itNum[0]++;
		loop_conditon = (activeVertices_v.size() > 0)? 1 : 0;
		#endif
		#ifdef PR
		itNum[0]++;
		loop_conditon = (error > epsilon)? 1 : 0;
		std::cout << "PageRank iteration = " << itNum[0] << " next frontier = " << activeVertices_v.size() << std::endl;
		printf ("%lf\n",error);

		#endif
		#ifdef CC
		printf("semaphore[1] = %d\n", semaphore[1]);
		loop_conditon = false;
		#endif
	}

	return 1;
}

int verify(PROP_TYPE* swProp, PROP_TYPE* ptProp, const int &num){
	bool match = true;
	for (int i = 0; i < num; i++) {
		if (swProp[i] != ptProp[i]) {
			printf(error_message, i, swProp[i], ptProp[i]);	
			match = false;
			break;
		} 
	}
	if (match) {
		printf("TEST PASSED.\n");
		return EXIT_SUCCESS;
	} else {
		printf("TEST FAILED.\n");
		return EXIT_FAILURE;
	}
}

int main(int argc, char **argv) {
	std::clock_t begin;
	std::clock_t end;
	double elapsedTime;

	int startVertexIdx;
	std::string gName = "youtube";
	std::string mode = "sim"; // or harp

	if(gName == "youtube")    startVertexIdx = 320872;
	if(gName == "lj1")        startVertexIdx = 3928512;
	if(gName == "pokec")      startVertexIdx = 182045;
	if(gName == "rmat-19-32") startVertexIdx = 104802;
	if(gName == "rmat-21-32") startVertexIdx = 365723;
	Graph* gptr = createGraph(gName, mode);
	CSR* csr = new CSR(*gptr);
	free(gptr);

	std::cout << "Graph is loaded." << std::endl;
	int vertexNum = csr->vertexNum;
	int edgeNum = csr->ciao.size();
	std::cout << "node num: " << csr->vertexNum << std::endl;
	std::cout << "edge num: " << edgeNum << std::endl;

	PROP_TYPE *swProp   = (PROP_TYPE*)malloc(vertexNum * sizeof(PROP_TYPE));
	PROP_TYPE *ptProp   = (PROP_TYPE*)malloc(vertexNum * sizeof(PROP_TYPE));
	PROP_TYPE *tmpProp  = (PROP_TYPE*)malloc(vertexNum * sizeof(PROP_TYPE));
	std::vector<int> activeVertices_v; 
	// Partition the CSR
	int blkNum = (vertexNum + BLK_SIZE - 1)/BLK_SIZE;
	std::cout << "The graph is divided into " << blkNum * blkNum << " partitions\n";
	std::vector<CSR_BLOCK*> blkVec;
	std::cout << "The amount of edges in the partitions is " << std::endl;
	for(int cordx = 0; cordx < blkNum; cordx++){
		for(int cordy = 0; cordy < blkNum; cordy++){
			CSR_BLOCK* csrBlkPtr = new CSR_BLOCK(cordx, cordy, csr);
			blkVec.push_back(csrBlkPtr);
			//std::cout << csrBlkPtr->edgeNum << " ";
		}
	}
	std::cout << std::endl;
	// Partition CSR finished
	std::cout << "soft bfs starts." << std::endl;
	//propInit(vertexNum, swProp, activeVertices_v, startVertexIdx);
	//propInit(vertexNum, tmpProp, startVertexIdx);
	begin = clock();
	//swProcessing(csr, swProp, tmpProp, activeVertices_v);
	processInit(csr, vertexNum, edgeNum, startVertexIdx, ptProp, tmpProp ,activeVertices_v, startVertexIdx);
	swProcessing(vertexNum, edgeNum);
	end = clock();
	elapsedTime = (end - begin)*1.0/CLOCKS_PER_SEC;
	std::cout << "Software bfs takes " << elapsedTime << " seconds." << std::endl;

	std::cout << "soft bfs with partition starts." << std::endl;
	propInit(vertexNum, ptProp, activeVertices_v, startVertexIdx);
	propInit(vertexNum, tmpProp, startVertexIdx);
	processInit(csr, vertexNum, edgeNum, startVertexIdx, ptProp, tmpProp ,activeVertices_v, startVertexIdx);
	begin = clock();
	ptProcessing(blkVec, ptProp, tmpProp, activeVertices_v, blkNum, vertexNum);
	end = clock();
	elapsedTime = (end - begin)*1.0/CLOCKS_PER_SEC;
	std::cout << "Software bfs with partition takes " << elapsedTime << " seconds." << std::endl;
	std::cout << "Verify BFS with partition: " << std::endl;									
	verify(swProp, ptProp, vertexNum);

	std::cout << "Hardare bfs with partition starts." << std::endl;
	processInit(csr, vertexNum, edgeNum, startVertexIdx, ptProp, tmpProp ,activeVertices_v, startVertexIdx);
	begin = clock();
	ptProcessingOnFpga (blkVec, ptProp, tmpProp, activeVertices_v, blkNum, vertexNum, edgeNum);
	end = clock();
	elapsedTime = (end - begin)*1.0/CLOCKS_PER_SEC;
	std::cout << "Hardware bfs with partition takes " << elapsedTime << " seconds." << std::endl;
	std::cout << "Verify Hardware BFS with partition: " << std::endl;									
	verify(swProp, vertexProp, vertexNum);

	freeResources();
	return 0;
}
