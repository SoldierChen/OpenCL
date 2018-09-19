//#include "config.h"
//type define

#define SW 0
#if SW == 1
	#define VERTEX_MAX  4096 //(128*1024)
	#define EDGE_MAX  16384 // (1024*1024)
#else
	#define VERTEX_MAX  (128*1024)
	#define EDGE_MAX  (64*1024)
#endif
#define PROP_TYPE int
#define CC
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
__attribute__((always_inline)) void compute(int srcVidx, int dstVidx, PROP_TYPE eProp, int outDeg,
	PROP_TYPE* tmpVPropBuffer, PROP_TYPE* vPropBuffer, int * count){

		PROP_TYPE srcVprop = tmpVPropBuffer[srcVidx];
		PROP_TYPE dstVprop = tmpVPropBuffer[dstVidx];
	
	#ifdef CC
			PROP_TYPE comp_u = srcVprop;
        	PROP_TYPE comp_v = dstVprop;
			if (comp_u != comp_v){
        			PROP_TYPE high_comp = comp_u > comp_v ? comp_u : comp_v;
        			PROP_TYPE low_comp = comp_u + (comp_v - high_comp);
        			if (high_comp == tmpVPropBuffer[high_comp]) {
          				tmpVPropBuffer[high_comp] = low_comp;
          				count[0] --;
        			}
        	}
	#endif
	#ifdef PR 
			if(outDeg)
			    tmpVPropBuffer[dstVidx] += (vPropBuffer[srcVidx] / outDeg);
	#endif
	#ifdef BFS
			tmpVPropBuffer[dstVidx] = (dstVprop > srcVprop + 1)? (srcVprop + 1) : dstVprop;
	#endif
	#ifdef SSSP
			tmpVPropBuffer[dstVidx] = (dstVprop > srcVprop + eProp)? (srcVprop + eProp) : dstVprop;
	#endif
}
__attribute__((always_inline)) PROP_TYPE apply(int vertexNum, PROP_TYPE * tmpVPropBuffer, 
						PROP_TYPE * vPropBuffer, int * count){

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
		__global const int* restrict rpaStart,
		__global const int* restrict rpaNum,
		__global const int* restrict cia,
		__global const PROP_TYPE* restrict edgeProp,
		__global const int* restrict activeVertexNum,
		const int vertexNum,
		const int edgeNum,
		__global const int* restrict itNum
		)
{	
	int rpaStartBuffer[VERTEX_MAX];
	int rpaNumBuffer[VERTEX_MAX];
	// Load vertex and edge information
	if(SW|(itNum[0] == 0)){
		for(int i = 0; i < vertexNum; i++){
			rpaStartBuffer[i] = rpaStart[i];
			rpaNumBuffer[i] = rpaNum[i];
		}
	}
	for(int i = 0; i < activeVertexNum[0]; i++){
		int vertexIdx = read_channel_altera(activeVertexCh);
		int start = rpaStartBuffer[vertexIdx];
		int num = rpaNumBuffer[vertexIdx];
		for(int j = 0; j < num; j++){
			int ngbVidx = cia[start + j];
			int eProp = edgeProp[start + j];
			edge_info_t edge_info;
			edge_info.vertexIdx = vertexIdx;
			edge_info.ngbVidx = ngbVidx;
			edge_info.eProp = eProp;
			edge_info.outDeg = num;
			write_channel_altera(edgeInfoCh, edge_info);
		}
	}
	write_channel_altera(edgeInfoChEof, 0xffff);
}

__kernel void __attribute__((task)) processEdge(
		__global PROP_TYPE* restrict vertexProp,
		__global int* restrict activeVertexNum,
		__global PROP_TYPE* restrict semaphore,
		const int vertexNum,
		__global const int* restrict itNum
		//__global const int* restrict count_g
		)
{
	PROP_TYPE vPropBuffer[VERTEX_MAX];
	PROP_TYPE tmpVPropBuffer[VERTEX_MAX];
	edge_info_t edge_info; 
	int dstVidx = 0;
	int srcVidx = 0;
	PROP_TYPE eProp = 0;
	int outDeg = 0;
	int end_flag = 0;
	bool valid_data = 0;
	bool valid_flag = 0;
	int count[1] = {vertexNum};
	if(SW|(itNum[0] == 0)){
		for(int i = 0; i < vertexNum; i++){
			vPropBuffer[i] = vertexProp[i];
			tmpVPropBuffer[i] = vertexProp[i];
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
			compute(srcVidx, dstVidx, eProp, outDeg, tmpVPropBuffer, vPropBuffer, count);
		}

		int end_flag_tmp = read_channel_nb_altera(edgeInfoChEof, &valid_flag);
		if(valid_flag) end_flag = end_flag_tmp;		
		if(end_flag == 0xffff && !valid_data && !valid_flag) break;
	}

	semaphore[1] = apply(vertexNum, tmpVPropBuffer, vPropBuffer, count);
	//printf("semaphore[1] = %f\n", semaphore[1]);
	
	for(int i = 0; i < vertexNum; i++){
		if(tmpVPropBuffer[i] != vPropBuffer[i]){
			vPropBuffer[i] = tmpVPropBuffer[i];
			write_channel_altera(nextFrontierCh, i);
		}
	}


	#if SW == 1
	if(SW|(semaphore[0] == 1)){
		for(int i = 0; i < vertexNum; i++){
			vertexProp[i] = vPropBuffer[i];
		}
	}
	#endif
	
	write_channel_altera(nextFrontierChEof, 0xffff);
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
			nextFrontierSize[1] = idx; 
			break;
		}
	}
}


