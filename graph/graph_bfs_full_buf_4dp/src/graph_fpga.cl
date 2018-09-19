//#include "config.h"
//type define
typedef struct EdgeInfo{
	int vertexIdx;
	int ngbVidx;
	int eProp;
} edge_info_t;

channel int activeVertexCh __attribute__((depth(2048)));
channel edge_info_t edgeInfoCh __attribute__((depth(2048)));
channel int nextFrontierCh __attribute__((depth(2048)));
channel int nextFrontierChEof __attribute__((depth(4)));


#define VERTEX_MAX 0xffff
#define EDGE_MAX 0xffff

__attribute__((always_inline)) int compute(int srcVprop,int dstProp, int eProp){

		return srcVprop	+ 1;

}
__kernel void __attribute__((task)) readActiveVertices(
		__global const int* restrict activeVertices, 
		const int totalActiveVertexNum
		)
{
	int vertexIdx;
	for(int i = 0; i < totalActiveVertexNum; i++){
		int vertexIdx = activeVertices[i];
		write_channel_altera(activeVertexCh, vertexIdx);
	}
}

__kernel void __attribute__((task)) readNgbInfo(
		__global const int* restrict outRowPointerArray,
		__global const int* restrict rpaStart,
		__global const int* restrict rpaNum,
		__global const int* restrict cia,
		__global const int* restrict edgeProp,
		//__global const int* restrict outDeg,

		const int activeVertexNum,
		const int vertexNum,
		const int edgeNum,
		const int itNum
		)
{
	int rpaStartBuffer[VERTEX_MAX];
	int rpaNumBuffer[VERTEX_MAX];
	int ciaBuffer[EDGE_MAX];
	int edgePropBuffer[EDGE_MAX];

	// Load vertex and edge information
	if(itNum == 0){
		for(int i = 0; i < vertexNum; i++){
			rpaStartBuffer[i] = rpaStart[i];
			rpaNumBuffer[i] = rpaNum[i];
		}

		for(int i = 0; i < edgeNum; i++){
			ciaBuffer[i] = cia[i];
			edgePropBuffer[i] = edgeProp[i];
		}

	}

	for(int i = 0; i < activeVertexNum; i++){
		int vertexIdx = read_channel_altera(activeVertexCh);
		int start = rpaStartBuffer[vertexIdx];
		int num = rpaNumBuffer[vertexIdx];

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
}


__kernel void __attribute__((task)) processEdge(
		__global const int* restrict vertexProp,
		__global int* restrict activeVertexNum,
		const int vertexNum,
		const int itNum
		)
{
	int vPropBuffer[VERTEX_MAX];
	int tmpVPropBuffer[VERTEX_MAX];
	edge_info_t edge_info; 
	int dstVidx = 0;
	int srcVidx = 0;
	int eProp = 0;
	if(itNum == 0){
		for(int i = 0; i < vertexNum; i++){
			vPropBuffer[i] = vertexProp[i];
		}
	}	

	while(true){
		
		edge_info = read_channel_altera(edgeInfoCh);
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

	// Compare and decide the next frontier
	for(int i = 0; i < vertexNum; i++){
		if(tmpVPropBuffer[i] != vPropBuffer[i]){
			vPropBuffer[dstVidx] = tmpVPropBuffer[i];
			write_channel_altera(nextFrontierCh, dstVidx);
		}
	}
	write_channel_altera(nextFrontierChEof, 0xffff);
}


__kernel void __attribute__((task)) updateNextFrontier(
		__global int* restrict nextFrontier,
		__global int* restrict nextFrontierSize,
		const int vertexNum
		)
{
	bool vStatus[VERTEX_MAX];

	// Reset vStatus
	for(int i = 0; i < vertexNum; i++){
		vStatus[i] = 0;
	}

	int idx = 0;
	bool valid_eof = 0;
	bool valid_data = 0;
	while(true){
		int vidx = read_channel_nb_altera(nextFrontierCh, &valid_data);
		int end_flag = 	read_channel_nb_altera(nextFrontierChEof, &valid_eof);
		if(valid_data && (end_flag != 0xffff)){
				if(vStatus[vidx] == 0){
					vStatus[vidx] = 1;
					nextFrontier[idx++] = vidx;
				}
		}

		if((end_flag == 0xffff)&&(!valid_data)) break;
	}
}


