//#include "config.h"
//type define
#define SW 0
#if SW == 1
	#define VERTEX_MAX  256*1024 //(128*1024)
	#define EDGE_MAX  16384 // (1024*1024)
#else
	#define VERTEX_MAX  (128*1024)
	#define EDGE_MAX  (64*1024)
#endif
#define PROP_TYPE int
#define PR
#define kDamp 0.85f
#define BRAM_BANK 4
#define LOG2_BRAM_BANK 2
typedef struct EdgeInfo{
	int vertexIdx;
	int4 ngbVidx;
	//PROP_TYPE eProp;
	int outDeg;
} edge_info_t;

channel int activeVertexCh __attribute__((depth(2048)));
channel edge_info_t edgeInfoCh __attribute__((depth(2048)));
channel int edgeInfoChEof __attribute__((depth(4)));
channel int nextFrontierCh __attribute__((depth(2048)));
channel int nextFrontierChEof __attribute__((depth(4)));

 // BFS // SSSP // PR // CC
__attribute__((always_inline)) void compute(int srcVidx, int dstVidx, PROP_TYPE eProp, int outDeg,
	PROP_TYPE* tmpVPropBuffer, PROP_TYPE* vPropBuffer, int * count){

		//PROP_TYPE srcVprop = vPropBuffer[srcVidx];
		//PROP_TYPE dstVprop = tmpVPropBuffer[(int)(dstVidx / 4)];

	#ifdef PR 
			
			if(outDeg) {
				int idx = dstVidx >> LOG2_BRAM_BANK;
				PROP_TYPE operand0 = vPropBuffer[srcVidx];
				PROP_TYPE operand1 = tmpVPropBuffer[idx];
				//printf(" %d \t",dstVidx/4);
			    tmpVPropBuffer[idx] = ( operand0 / outDeg) + operand1;
			}
	#endif
	#ifdef BFS
			tmpVPropBuffer[dstVidx] = (dstVprop > srcVprop + 1)? (srcVprop + 1) : dstVprop;
	#endif
	#ifdef SSSP
			tmpVPropBuffer[dstVidx] = (dstVprop > srcVprop + eProp)? (srcVprop + eProp) : dstVprop;
	#endif
}

__kernel void __attribute__((task)) readActiveVertices(
		__global const int* restrict activeVertices, 
		__global const int* restrict activeVertexNum
		)
{
	int vertexIdx;
	for(int i = 0; i < activeVertexNum[0]; i++){
		int vertexIdx = activeVertices[i];
		write_channel_altera(activeVertexCh, vertexIdx);
	}
	//printf("kernel 1 finished \n");
}

__kernel void __attribute__((task)) readNgbInfo(
		__global const int* restrict rpaStart,
		__global const int* restrict rpaNum,
		__global const int* restrict outDeg,
		__global const int4 * restrict cia,
		__global const PROP_TYPE* restrict edgeProp,
		__global const int* restrict activeVertexNum,
		const int vertexNum,
		const int edgeNum,
		__global const int* restrict itNum
		)
{	
	#if SW == 1
	    int * rpaStartBuffer = malloc(sizeof(int) * VERTEX_MAX);
	    int * rpaNumBuffer = malloc(sizeof(int) * VERTEX_MAX);
	    int * outDegBuffer = malloc(sizeof(int) * VERTEX_MAX);
	#else
		int rpaStartBuffer[VERTEX_MAX];
		int rpaNumBuffer[VERTEX_MAX];
		PROP_TYPE outDegBuffer[VERTEX_MAX];
	#endif
//	int4 ciaBuffer[EDGE_MAX/4];
//	int edgePropBuffer[EDGE_MAX];
	// Load vertex and edge information
	if(SW|(itNum[0] == 0)){
		for(int i = 0; i < vertexNum; i++){
			rpaStartBuffer[i] = rpaStart[i];
			rpaNumBuffer[i] = rpaNum[i];
			outDegBuffer[i] = outDeg[i];
		}
		//for(int i = 0; i < edgeNum/4; i++){
		//	ciaBuffer[i] = cia[i];
		//  edgePropBuffer[i] = edgeProp[i];
		//}
	}
	for(int i = 0; i < activeVertexNum[0]; i++){
		int vertexIdx = read_channel_altera(activeVertexCh);
		int start = rpaStartBuffer[vertexIdx];
		int num = rpaNumBuffer[vertexIdx];
		int vertexOutDeg = outDegBuffer[vertexIdx];
		//printf("\n vertex %d num %d start %d \t ",vertexIdx, num, start);
		for(int j = start/4; j < (start + num)/4; j++){
			int4 ngbVidx = cia[j];
			//printf("%d %d %d %d \n",ngbVidx.s0, ngbVidx.s1, ngbVidx.s2, ngbVidx.s3 );
			//int4 eProp = edgePropBuffer[start + j];
			edge_info_t edge_info;
			edge_info.vertexIdx = vertexIdx;
			edge_info.ngbVidx = ngbVidx;
			//edge_info.eProp = eProp;
			edge_info.outDeg = vertexOutDeg;
			write_channel_altera(edgeInfoCh, edge_info);
		}
	}
	write_channel_altera(edgeInfoChEof, 0xffff);
}

__kernel void __attribute__((task)) processEdge(
		__global PROP_TYPE* restrict vertexProp,
		__global int4 * restrict tmpVertexProp,
		__global PROP_TYPE* restrict semaphore,
		const int vertexNum,
		__global const int* restrict itNum
		)
{
	#if SW == 1
		PROP_TYPE * vPropBuffer 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX);
		PROP_TYPE * tmpVPropBuffer0 = malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1 = malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer2 = malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer3 = malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
	#else 
		PROP_TYPE vPropBuffer[VERTEX_MAX];
		PROP_TYPE tmpVPropBuffer0[VERTEX_MAX/BRAM_BANK];
		PROP_TYPE tmpVPropBuffer1[VERTEX_MAX/BRAM_BANK];
		PROP_TYPE tmpVPropBuffer2[VERTEX_MAX/BRAM_BANK];
		PROP_TYPE tmpVPropBuffer3[VERTEX_MAX/BRAM_BANK];
	#endif
	edge_info_t edge_info; 
	PROP_TYPE eProp = 0;
	int end_flag = 0;
	bool valid_data = 0;
	bool valid_flag = 0;
	int count[1] = {vertexNum};

	for(int i = 0; i < vertexNum; i++){
		vPropBuffer[i] = vertexProp[i];
	}
	for(int i = 0; i < vertexNum/BRAM_BANK; i++){
		tmpVPropBuffer0[i] = 0;
		tmpVPropBuffer1[i] = 0;
		tmpVPropBuffer2[i] = 0;
		tmpVPropBuffer3[i] = 0;
	}
#pragma ivdep
	while(true){	
		edge_info = read_channel_nb_altera(edgeInfoCh, &valid_data);
		if(valid_data){
			int srcVidx = edge_info.vertexIdx;
			int4 dstVidx = edge_info.ngbVidx;
			int outDeg = edge_info.outDeg;

			if(dstVidx.s0 != -1)
			#pragma ivdep
				compute(srcVidx, dstVidx.s0, eProp, outDeg, tmpVPropBuffer0, vPropBuffer, count);
			if(dstVidx.s1 != -1)
			#pragma ivdep
				compute(srcVidx, dstVidx.s1, eProp, outDeg, tmpVPropBuffer1, vPropBuffer, count);
			if(dstVidx.s2 != -1)
			#pragma ivdep
				compute(srcVidx, dstVidx.s2, eProp, outDeg, tmpVPropBuffer2, vPropBuffer, count);
			if(dstVidx.s3 != -1)
			#pragma ivdep
				compute(srcVidx, dstVidx.s3, eProp, outDeg, tmpVPropBuffer3, vPropBuffer, count);		
		}
		int end_flag_tmp = read_channel_nb_altera(edgeInfoChEof, &valid_flag);
		if(valid_flag) end_flag = end_flag_tmp;		
		if(end_flag == 0xffff && !valid_data && !valid_flag) break;
	}
	//semaphore[1] = apply(vertexNum, tmpVPropBuffer, vPropBuffer, count);
	//printf("semaphore[1] = %f\n", semaphore[1]);

	for(int i = 0; i < vertexNum / 4; i++){
		int4 ddr_data;
		ddr_data.s0 = tmpVPropBuffer0[i];
		ddr_data.s1 = tmpVPropBuffer1[i];
		ddr_data.s2 = tmpVPropBuffer2[i];
		ddr_data.s3 = tmpVPropBuffer3[i];
		tmpVertexProp[i] = ddr_data;
	}
}
