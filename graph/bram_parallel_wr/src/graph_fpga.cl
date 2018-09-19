#define SW 0
#if SW == 1
	#define VERTEX_MAX  4096 //(128*1024)
	#define EDGE_MAX  16384 // (1024*1024)
#else
	#define VERTEX_MAX  (256*1024)
	#define EDGE_MAX  (64*1024)
#endif
#define PROP_TYPE int

channel int activeVertexCh __attribute__((depth(2048)));

__kernel void __attribute__((task)) readActiveVertices(
		__global const int* restrict activeVertices, 
		__global const int* restrict activeVertexNum,
		__global  int* restrict itNum
		)
{
	int activeVerticesBuf[VERTEX_MAX];
	if(SW|(itNum[0] == 0)){
		for(int i = 0; i < VERTEX_MAX; i++){
			activeVerticesBuf[i] = activeVertices[i];
		}
	}

	int vertexIdx;
	for(int i = 0; i < activeVertexNum[0]; i++){
		int vertexIdx = activeVerticesBuf[i];
		write_channel_altera(activeVertexCh, vertexIdx);
	}
}

__kernel void __attribute__((task)) readNgbInfo(
		__global  int* restrict dstDDR,
		__global const int* restrict rpaNum,
		__global const int* restrict cia,
		__global const PROP_TYPE* restrict edgeProp,
		__global const int* restrict activeVertexNum,
		const int vertexNum,
		const int edgeNum,
		__global  int* restrict itNum
		)
{	
	int dstBuffer[VERTEX_MAX];
	if(SW|(itNum[0] == 0)){
		for(int i = 0; i < VERTEX_MAX; i++){
			dstBuffer[i] = i;
		}
	}

	for(int i = 0; i < activeVertexNum[0]; i++){
		int vertexIdx = read_channel_altera(activeVertexCh);
		dstBuffer[vertexIdx] += vertexIdx;
	}

	if(SW|(itNum[0] == 100)){
		for(int i = 0; i < VERTEX_MAX; i++){
			dstDDR[i] = dstBuffer[i];
		}
	}
}
