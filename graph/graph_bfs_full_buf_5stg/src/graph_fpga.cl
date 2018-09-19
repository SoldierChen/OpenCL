//#include "config.h"
#define PROP_TYPE int
typedef struct EdgeInfo{
	int vertexIdx;
	int ngbVidx;
	int eProp;
} edge_info_t;
typedef struct NgbInfo{
	int vertexIdx;
	int start;
	int num;
} ngb_info_t;

channel int activeVertexCh __attribute__((depth(2048)));
channel edge_info_t edgeInfoCh __attribute__((depth(2048)));
channel ngb_info_t ngbInfoCh __attribute__((depth(512)));
channel int edgeInfoChEof __attribute__((depth(4)));
channel int nextFrontierCh __attribute__((depth(2048)));
channel int nextFrontierChEof __attribute__((depth(4)));

#define SW 0

#if SW == 1
	#define VERTEX_MAX  4096 //(128*1024)
	#define EDGE_MAX  16384 // (1024*1024)
#else
	#define VERTEX_MAX  (64*1024)
	#define EDGE_MAX  (512*1024)
#endif

__attribute__((always_inline)) int compute(int srcVprop,int dstProp, int eProp){
	return (dstProp > srcVprop + 1)? (srcVprop + 1) : dstProp;
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
		__global const int* restrict activeVertexNum,
		const int vertexNum,
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
		ngb_info_t ngb_info;
		ngb_info.vertexIdx = vertexIdx;
		ngb_info.start = rpaStartBuffer[vertexIdx];
		ngb_info.num = rpaNumBuffer[vertexIdx];
		write_channel_altera(ngbInfoCh, ngb_info);
	}
}

__kernel void __attribute__((task)) readNgbs(
		__global const int* restrict cia,
		__global const int* restrict edgeProp,
		__global const int* restrict activeVertexNum,
		const int edgeNum,
		__global const int* restrict itNum
		)
{	

	int ciaBuffer[EDGE_MAX];
	int edgePropBuffer[EDGE_MAX];
	// Load vertex and edge information
	if(SW|(itNum[0] == 0)){
		for(int i = 0; i < edgeNum; i++){
			ciaBuffer[i] = cia[i];
			edgePropBuffer[i] = edgeProp[i];
		}
	}

	for(int i = 0; i < activeVertexNum[0]; i++){
		ngb_info_t ngb_info;
		ngb_info = read_channel_altera(ngbInfoCh);
		int vertexIdx = ngb_info.vertexIdx;
		int start = ngb_info.start;
		int num = ngb_info.num;
		for(int j = 0; j < num; j++){
			int ngbVidx = ciaBuffer[start + j];
			int eProp = edgeProp[start + j];
			edge_info_t edge_info;
			edge_info.vertexIdx = vertexIdx;
			edge_info.ngbVidx = ngbVidx;
			edge_info.eProp = eProp;
			write_channel_altera(edgeInfoCh, edge_info);
		}
	}
	write_channel_altera(edgeInfoChEof, 0xffff);
}

__kernel void __attribute__((task)) processEdge(
		__global int* restrict vertexProp,
		__global int* restrict activeVertexNum,
		const int vertexNum,
		__global const int* restrict itNum
		)
{

	int vPropBuffer[VERTEX_MAX];
	int tmpVPropBuffer[VERTEX_MAX];
	edge_info_t edge_info; 
	int dstVidx = 0;
	int srcVidx = 0;
	int eProp = 0;
	int end_flag = 0;
	bool valid_data = 0;
	bool valid_flag = 0;
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
			int srcVprop = vPropBuffer[srcVidx];
			int dstVprop = vPropBuffer[dstVidx];
			// add user defined compute at inline compute function.
			int vtmp = compute(srcVprop, dstVprop, eProp);
			// update the tmp property
			// need ***** explore whether update immediately and update conditions. 			
			tmpVPropBuffer[dstVidx] = vtmp;
		}

		int end_flag_tmp = read_channel_nb_altera(edgeInfoChEof, &valid_flag);
		if(valid_flag) end_flag = end_flag_tmp;
		
		if(end_flag == 0xffff && !valid_data && !valid_flag) break;
	}

	// Compare and decide the next frontier
	for(int i = 0; i < vertexNum; i++){
		if(tmpVPropBuffer[i] != vPropBuffer[i]){
			vPropBuffer[i] = tmpVPropBuffer[i];
			write_channel_altera(nextFrontierCh, i);
		}
	}

	#if SW == 1
			if(SW|(itNum[0] == 0)){
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
	while(true){
		int vidx = read_channel_nb_altera(nextFrontierCh, &valid_data);
		if(valid_data) nextFrontier[idx++] = vidx;

		int end_flag_tmp = read_channel_nb_altera(nextFrontierChEof, &valid_eof);
		if(valid_eof) end_flag = end_flag_tmp;
		// exit condition
		if((end_flag == 0xffff)&&(!valid_data)&&(!valid_eof)){
			nextFrontierSize[0] = idx; 
			break;
		}
	}
}


