//#include "config.h"
//type define
#define SW 1
#if SW == 1
	#define VERTEX_MAX  262144//262144//40960//40960//(128*1024)
	#define EDGE_MAX  1747834//5610680////163840 // (1024*1024)
#else
	#define VERTEX_MAX  (16*1024)
	#define EDGE_MAX  (64*1024)
#endif
#define PROP_TYPE float
#define PR
#define kDamp 0.85
typedef struct EdgeInfo{
	int vertexIdx;
	int ngbVidx;
	PROP_TYPE eProp;
	int outDeg;
} edge_info_t;

channel int activeVertexCh __attribute__((depth(2048)));
channel edge_info_t edgeInfoCh __attribute__((depth(2048)));
channel int edgeInfoChEof __attribute__((depth(4)));
channel int nextFrontierCh __attribute__((depth(2048)));
channel int nextFrontierChEof __attribute__((depth(4)));

 // BFS // SSSP // PR
__attribute__((always_inline)) void compute(int srcStart, int srcVidx, int dstStart, int dstVidx, PROP_TYPE eProp, int outDeg,
	PROP_TYPE* tmpVPropBuffer, PROP_TYPE* vPropBuffer, int * count){

		int arrySrcIdx = srcVidx - srcStart;
		int arryDstIdx = dstVidx - dstStart;
		PROP_TYPE srcVprop = vPropBuffer[srcVidx - srcStart];
		PROP_TYPE dstVprop = tmpVPropBuffer[dstVidx - dstStart];
	
	#ifdef CC
			PROP_TYPE comp_u = srcVprop;
        	PROP_TYPE comp_v = dstVprop;
			if (comp_u != comp_v){
        			PROP_TYPE high_comp = comp_u > comp_v ? comp_u : comp_v;
        			PROP_TYPE low_comp = comp_u + (comp_v - high_comp);
        			if (high_comp == tmpVPropBuffer[high_comp - dstStart]) {
          				tmpVPropBuffer[high_comp] = low_comp;
          				count[0] --;
        			}
        	}
	#endif
	#ifdef PR 
			if(outDeg)
			    tmpVPropBuffer[arryDstIdx] += (vPropBuffer[arrySrcIdx] / outDeg);
	#endif
	#ifdef BFS
			tmpVPropBuffer[arryDstIdx] = (dstVprop > srcVprop + 1)? (srcVprop + 1) : dstVprop;
	#endif
	#ifdef SSSP
			tmpVPropBuffer[arryDstIdx] = (dstVprop > srcVprop + eProp)? (srcVprop + eProp) : dstVprop;
	#endif
}
__attribute__((always_inline)) PROP_TYPE apply(int vertexNum, PROP_TYPE * tmpVPropBuffer, PROP_TYPE * vPropBuffer, int * count){

	#ifdef CC
    		return *count;
	#endif
    #ifdef PR 
    	PROP_TYPE error = 0;
    	for(int i = 0; i < vertexNum; i++){
    		PROP_TYPE incoming_score = tmpVPropBuffer[i] - vPropBuffer[i];
			tmpVPropBuffer[i] = (1.0 - kDamp) / vertexNum + kDamp * incoming_score;
			error += fabs(tmpVPropBuffer[i] - vPropBuffer[i]);
		}
		return error;
	#endif
	#ifdef BFS
			return 0;
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
}

__kernel void __attribute__((task)) readNgbInfo(
		__global int* restrict rpaStart,
		__global int* restrict rpaNum,
		__global int* restrict outDegGlobal,		
		__global int* restrict cia,
		__global PROP_TYPE* restrict edgeProp,
		__global int* restrict activeVertexNum,
		__global int* restrict vertexNum,
		__global int* restrict edgeNum,
		__global int* restrict itNum,
		__global int* restrict vPropRange

		)
{	
	#if 1
	int * rpaStartBuffer = (int*)malloc(sizeof(int) * VERTEX_MAX * 2);
	int * rpaNumBuffer = (int*)malloc(sizeof(int) * VERTEX_MAX * 2);
	int * ciaBuffer = (int*)malloc(sizeof(int) * EDGE_MAX * 2);
	int * edgePropBuffer = (int*)malloc(sizeof(int) * EDGE_MAX * 2);
	#else
	int rpaStartBuffer[VERTEX_MAX];
	int rpaNumBuffer[VERTEX_MAX];
	int ciaBuffer[EDGE_MAX];
	int edgePropBuffer[EDGE_MAX];
	#endif
	int srcStart = vPropRange[0];
	int srcEnd = vPropRange[1];
	int srcNum = srcEnd - srcStart;
	// Load vertex and edge information
	if(SW|(itNum[0] == 0)){
		for(int i = 0; i < srcNum; i++){
			rpaStartBuffer[i] = rpaStart[i];
		}
		for(int i = 0; i < srcNum; i++){
			rpaNumBuffer[i] = rpaNum[i];
		}
		for(int i = 0; i < edgeNum[0]; i++){
			ciaBuffer[i] = cia[i];
			edgePropBuffer[i] = edgeProp[i];
		}
	}
	//printf("srcStart %d\n", srcStart);
	for(int i = 0; i < activeVertexNum[0]; i++){
		int vertexIdx = read_channel_altera(activeVertexCh);
		int start = rpaStartBuffer[vertexIdx - srcStart];
		int num = rpaNumBuffer[vertexIdx - srcStart];
		//printf("vidx %d, Buffer idx %d ngbNUM %d, srcstart %d, ciaostart %d \n", vertexIdx,vertexIdx - srcStart, num, srcStart, start);		
		for(int j = 0; j < num; j++){
			int ngbVidx = ciaBuffer[start + j];
			int eProp = edgePropBuffer[start + j];
			edge_info_t edge_info;
			edge_info.vertexIdx = vertexIdx;
			edge_info.ngbVidx = ngbVidx;
			edge_info.eProp = eProp;
			edge_info.outDeg = outDegGlobal[vertexIdx];
			write_channel_altera(edgeInfoCh, edge_info);
		}
	}
	write_channel_altera(edgeInfoChEof, 0xffff);
}

__kernel void __attribute__((task)) processEdge(
		__global PROP_TYPE* restrict vertexProp,
		__global PROP_TYPE* restrict tmpVertexProp,
		__global int* restrict activeVertexNum,
		__global PROP_TYPE* restrict semaphore,
		__global const int* restrict vertexNum,
		__global const int* restrict itNum,
		__global const int* restrict srcRange,
		__global const int* restrict sinkRange
		)
{	

	#if 1
	PROP_TYPE * vPropBuffer = (PROP_TYPE*)malloc(sizeof(PROP_TYPE) * VERTEX_MAX * 2);
	PROP_TYPE * tmpVPropBuffer = (PROP_TYPE*)malloc(sizeof(PROP_TYPE) * VERTEX_MAX * 2);
	#else
	PROP_TYPE vPropBuffer[VERTEX_MAX];
	PROP_TYPE tmpVPropBuffer[VERTEX_MAX];
	#endif
	edge_info_t edge_info; 
	int dstVidx = 0;
	int srcVidx = 0;
	PROP_TYPE eProp = 0;
	int outDeg = 0;
	int end_flag = 0;
	bool valid_data = 0;
	bool valid_flag = 0;
	int count[1] = {vertexNum[0]};
	int srcStart = srcRange[0];
	int srcEnd = srcRange[1];
	int srcNum = srcEnd - srcStart;
	int dstStart = sinkRange[0];
	int dstEnd = sinkRange[1];
	int dstNum = dstEnd - dstStart;
	if(SW|(itNum[0] == 0)){
		for(int i = 0; i < srcNum; i++){
			vPropBuffer[i] = vertexProp[i + srcStart];
		}
	}
	if(SW|(itNum[0] == 0)){
		for(int i = 0; i < dstNum; i++){
			tmpVPropBuffer[i] = tmpVertexProp[i + dstStart];
		}
	}		
	
	while(true){	
		edge_info = read_channel_nb_altera(edgeInfoCh, &valid_data);
		if(valid_data){
			srcVidx = edge_info.vertexIdx;
			dstVidx = edge_info.ngbVidx;
			eProp = edge_info.eProp;
			outDeg = edge_info.outDeg;
			// add user defined compute at inline compute function.
			compute(srcStart, srcVidx, dstStart, dstVidx, eProp, outDeg, tmpVPropBuffer, vPropBuffer, count);
			//if(srcVidx == 320872) printf("details:: vidx %d ciaIdxStart %d outDeg %d\n",srcVidx, dstVidx, outDeg);
		}
		int end_flag_tmp = read_channel_nb_altera(edgeInfoChEof, &valid_flag);
		if(valid_flag) end_flag = end_flag_tmp;		
		if(end_flag == 0xffff && !valid_data && !valid_flag) break;
	}
	//semaphore[1] = apply(vertexNum[0], tmpVPropBuffer, vPropBuffer, count);
	//printf("semaphore[1] = %f\n", semaphore[1]);
/*	
	for(int i = 0; i < vertexNum[0]; i++){
		if(tmpVPropBuffer[i] != vPropBuffer[i]){
			vPropBuffer[i] = tmpVPropBuffer[i];
			write_channel_altera(nextFrontierCh, i);
		}
	}
*/
	#if SW == 1
	if(SW|(semaphore[0] == 1)){
		for(int i = 0; i < dstNum; i++){
			tmpVertexProp[i + dstStart] = tmpVPropBuffer[i];
		}
	}
	#endif
	
	//write_channel_altera(nextFrontierChEof, 0xffff);
}
__kernel void __attribute__((task)) updateNextFrontier(
		__global int* restrict nextFrontier,
		__global int* restrict nextFrontierSize,
		const int vertexNum
		)
{
	int idx = 0;
	int end_flag = 0;
	bool valid_eof = 0;
	bool valid_data = 0;
	int cnt = 0;
	int w_buffer [64];
	while(true){
		int vidx = read_channel_nb_altera(nextFrontierCh, &valid_data);
		if(valid_data){
			w_buffer[cnt++] = vidx;
		}
		if(cnt == 64) {
			for (int i = 0; i < 64; i ++)
				nextFrontier[idx ++] = w_buffer[i];
			cnt = 0;
		}
		int end_flag_tmp = read_channel_nb_altera(nextFrontierChEof, &valid_eof);
		if(valid_eof) end_flag = end_flag_tmp;

		if((end_flag == 0xffff)&&(!valid_data)&&(!valid_eof)){
			for (int i = 0; i < cnt; i ++) 
				nextFrontier[idx++] = w_buffer[i];
			  nextFrontierSize[0] = idx; 
			break;
		}
	}
}


