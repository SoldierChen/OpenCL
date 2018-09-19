#define SW 0
#if SW == 1
	#define VERTEX_MAX  4096 //(128*1024)
	#define EDGE_MAX  16384 // (1024*1024)
#else
	#define VERTEX_MAX  (128*1024)
	#define EDGE_MAX  (64*1024)
#endif
#define PROP_TYPE int

channel int4 activeVertexCh __attribute__((depth(2048)));

__kernel void __attribute__((task)) readActiveVertices(
		__global const int4 * restrict activeVertices, 
		__global const int * restrict activeVertexNum,
		__global  int * restrict itNum
		)
{
	int4 activeVerticesBuf[VERTEX_MAX/4];
	if(SW|(itNum[0] == 0)){
		for(int i = 0; i < VERTEX_MAX/4; i++){
			activeVerticesBuf[i] = activeVertices[i];
		}
	}

	int4 vertexIdx;
	for(int i = 0; i < activeVertexNum[0]/4; i++){
		int4 vertexIdx = activeVerticesBuf[i];
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
	int dstBuffer0[VERTEX_MAX];
	int dstBuffer1[VERTEX_MAX];
	int dstBuffer2[VERTEX_MAX];
	int dstBuffer3[VERTEX_MAX];
	if(SW|(itNum[0] == 0)){
		for(int i = 0; i < VERTEX_MAX; i++){
			dstBuffer0[i] = i;
			dstBuffer1[i] = i;
			dstBuffer2[i] = i;
			dstBuffer3[i] = i;
		}
	}

	for(int i = 0; i < activeVertexNum[0]/4; i++){
		int4 vertexIdx = read_channel_altera(activeVertexCh);
		dstBuffer0[vertexIdx.s0] += vertexIdx.s0;
		dstBuffer1[vertexIdx.s1] += vertexIdx.s0;
		dstBuffer2[vertexIdx.s2] += vertexIdx.s0;
		dstBuffer3[vertexIdx.s3] += vertexIdx.s0;
	}

	if(SW|(itNum[0] == 100)){
		for(int i = 0; i < VERTEX_MAX; i++){
			dstDDR[i] = dstBuffer0[i] | dstBuffer1[i] | dstBuffer2[i] | dstBuffer3[i];
		}
	}
}
