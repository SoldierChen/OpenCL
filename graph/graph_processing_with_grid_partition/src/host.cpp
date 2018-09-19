/**********
Author: Cheng Liu
Email: st.liucheng@gmail.com
Software: SDx 2016.4
Date: July 6th 2017
**********/

#include "xcl.h"
#include "graph.h"

#include <cstdio>
#include <vector>
#include <ctime>

static const char *error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

#define OCL_CHECK(call)                                                        \
	do {                                                                       \
		cl_int err = call;                                                     \
		if (err != CL_SUCCESS) {                                               \
			printf("Error calling " #call ", error: %s\n", oclErrorCode(err)); \
			exit(EXIT_FAILURE);                                                \
		}                                                                      \
	} while (0);

Graph* createGraph(const std::string &gName){
	Graph* gptr;
	if(gName == "dblp"){
		gptr = new Graph("/home/liucheng/gitrepo/graph-data/dblp.ungraph.txt");
	}
	else if(gName == "youtube"){
		gptr = new Graph("/home/liucheng/gitrepo/graph-data/youtube.ungraph.txt");
	}
	else if(gName == "lj"){
		gptr = new Graph("/home/liucheng/gitrepo/graph-data/lj.ungraph.txt");
	}
	else if(gName == "pokec"){
		gptr = new Graph("/home/liucheng/gitrepo/graph-data/pokec-relationships.txt");
	}
	else if(gName == "wiki-talk"){
		gptr = new Graph("/home/liucheng/gitrepo/graph-data/wiki-Talk.txt");
	}
	else if(gName == "lj1"){
		gptr = new Graph("/home/liucheng/gitrepo/graph-data/LiveJournal1.txt");
	}
	else if(gName == "orkut"){
		gptr = new Graph("/home/liucheng/gitrepo/graph-data/orkut.ungraph.txt");
	}
	else if(gName == "rmat-21-32"){
		gptr = new Graph("/home/liucheng/gitrepo/graph-data/rmat-21-32.txt");
	}
	else if(gName == "rmat-19-32"){
		gptr = new Graph("/home/liucheng/gitrepo/graph-data/rmat-19-32.txt");
	}
	else if(gName == "rmat-21-128"){
		gptr = new Graph("/home/liucheng/gitrepo/graph-data/rmat-21-128.txt");
	}
	else if(gName == "twitter"){
		gptr = new Graph("/home/liucheng/gitrepo/graph-data/twitter_rv.txt");
	}
	else if(gName == "friendster"){
		gptr = new Graph("/home/liucheng/gitrepo/graph-data/friendster.ungraph.txt");
	}
	else if(gName == "example"){
		gptr = new Graph("/home/liucheng/gitrepo/graph-data/rmat-1k-10k.txt");
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


// Iterate the partitioned CSR for BFS
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
			for(int j = 0; j < blkNum; j++){
				int blkIdx = i * blkNum + j;
				CSR_BLOCK* blkPtr = blkVec[blkIdx];	

				// Traverse active frontiers that are in the block
				for(auto v : frontier){
					int ciaIdxStart = blkPtr->rpa[v - srcStart];
					int ciaIdxEnd = blkPtr->rpa[v + 1 - srcStart];
					{
						for(int k = ciaIdxStart; k < ciaIdxEnd; k++){
							int ngbIdx = blkPtr->cia[k];
							//PROP_TYPE eProp = blkVec[blkIdx]->eProps[k];
							tmpProp[ngbIdx] = compute(ptProp[v], 1, tmpProp[ngbIdx]);
						}
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

	if(gName == "youtube")    startVertexIdx = 320872;
	if(gName == "lj1")        startVertexIdx = 3928512;
	if(gName == "pokec")      startVertexIdx = 182045;
	if(gName == "rmat-19-32") startVertexIdx = 104802;
	if(gName == "rmat-21-32") startVertexIdx = 365723;
	Graph* gptr = createGraph(gName);
	CSR* csr = new CSR(*gptr);
	free(gptr);

	std::cout << "Graph is loaded." << std::endl;
	int vertexNum = csr->vertexNum;
	PROP_TYPE *swProp   = (PROP_TYPE*)malloc(vertexNum * sizeof(PROP_TYPE));
	PROP_TYPE *ptProp   = (PROP_TYPE*)malloc(vertexNum * sizeof(PROP_TYPE));
	PROP_TYPE *tmpProp  = (PROP_TYPE*)malloc(vertexNum * sizeof(PROP_TYPE));
	std::vector<int> activeVertices; 

	// Partition the CSR
	int blkNum = (vertexNum + BLK_SIZE - 1)/BLK_SIZE;
	std::cout << "The graph is divided into " << blkNum * blkNum << " partitions\n";
	std::vector<CSR_BLOCK*> blkVec;
	std::cout << "The amount of edges in the partitions is " << std::endl;
	for(int cordx = 0; cordx < blkNum; cordx++){
		for(int cordy = 0; cordy < blkNum; cordy++){
			CSR_BLOCK* csrBlkPtr = new CSR_BLOCK(cordx, cordy, csr);
			blkVec.push_back(csrBlkPtr);
			std::cout << csrBlkPtr->edgeNum << " ";
		}
	}
	std::cout << std::endl;

	std::cout << "soft bfs starts." << std::endl;
	propInit(vertexNum, swProp, activeVertices, startVertexIdx);
	propInit(vertexNum, tmpProp, startVertexIdx);
	begin = clock();
	swProcessing(csr, swProp, tmpProp, activeVertices);
	end = clock();
	elapsedTime = (end - begin)*1.0/CLOCKS_PER_SEC;
	std::cout << "Software bfs takes " << elapsedTime << " seconds." << std::endl;

	std::cout << "soft bfs with partition starts." << std::endl;
	propInit(vertexNum, ptProp, activeVertices, startVertexIdx);
	propInit(vertexNum, tmpProp, startVertexIdx);
	begin = clock();
	ptProcessing(blkVec, ptProp, tmpProp, activeVertices, blkNum, vertexNum);
	end = clock();
	elapsedTime = (end - begin)*1.0/CLOCKS_PER_SEC;
	std::cout << "Software bfs with partition takes " << elapsedTime << " seconds." << std::endl;

	std::cout << "Verify BFS with partition: " << std::endl;
	verify(swProp, ptProp, vertexNum);

	return 0;

	/*
	hwBfsInit(vertexNum, hwDepth, startVertexIdx);

	xcl_world world = xcl_world_single();
    cl_program program = xcl_import_binary(world, "bfs");
	cl_kernel krnl_bfs = xcl_get_kernel(program, "bfs");

    // Transfer data from system memory over PCIe to the FPGA on-board DDR memory.
    cl_mem devMemRpao = xcl_malloc(world, CL_MEM_READ_ONLY, rpaoSize * sizeof(int));
	cl_mem devMemCiao = xcl_malloc(world, CL_MEM_READ_ONLY, ciaoSize * sizeof(int));
    cl_mem devMemDepth = xcl_malloc(world, CL_MEM_READ_WRITE, vertexNum * sizeof(char));
    cl_mem devMemFrontierSize = xcl_malloc(world, CL_MEM_WRITE_ONLY, sizeof(int));
    xcl_memcpy_to_device(world, devMemRpao, csr->rpao.data(), csr->rpao.size() * sizeof(int));
    xcl_memcpy_to_device(world, devMemCiao, csr->ciao.data(), csr->ciao.size() * sizeof(int));
	xcl_memcpy_to_device(world, devMemDepth, hwDepth, vertexNum * sizeof(char));

	int nargs = 0;
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemDepth);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemDepth);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemDepth);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemDepth);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemDepth);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemDepth);

	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemRpao);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemRpao);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemRpao);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemRpao);

	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemRpao);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemRpao);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemRpao);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemRpao);

	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemCiao);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemCiao);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemCiao);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemCiao);

	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(cl_mem), &devMemFrontierSize);
	xcl_set_kernel_arg(krnl_bfs, nargs++, sizeof(int), &vertexNum);

	char level = 0;
	begin = clock();
	while(frontierSize != 0){
		xcl_set_kernel_arg(krnl_bfs, nargs, sizeof(char), &level);
		xcl_run_kernel3d(world, krnl_bfs, 1, 1, 1);
		xcl_memcpy_from_device(world, &frontierSize, devMemFrontierSize, sizeof(int));
		level++;
	}
	clFinish(world.command_queue);
	xcl_memcpy_from_device(world, hwDepth, devMemDepth, vertexNum * sizeof(char));
	end = clock();
	elapsedTime = (end - begin)*1.0/CLOCKS_PER_SEC;
	std::cout << "hardware bfs takes " << elapsedTime << " seconds." << std::endl;
	std::cout << "level = " << (int)level << std::endl;
	
	clReleaseMemObject(devMemRpao);
	clReleaseMemObject(devMemCiao);
	clReleaseMemObject(devMemDepth);
	clReleaseMemObject(devMemFrontierSize);
	clReleaseKernel(krnl_bfs);
	clReleaseProgram(program);
	xcl_release_world(world);

	verify(swDepth, hwDepth, csr->vertexNum);
	free(csr);
	free(swDepth);
	free(hwDepth);

	*/

	return 0;
}
