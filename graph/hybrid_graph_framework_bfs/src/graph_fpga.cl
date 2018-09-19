//#define SW 1
#define BFS
#define EOF_FLAG 0xffff
#define PROP_TYPE int

#define VERTEX_MAX  (128*1024)//262144//40960//40960//(128*1024)
#define EDGE_MAX    (2*1024*1024)//5610680////163840 // (1024*1024)

typedef struct EdgeInfo{
	int vertexIdx;
	int ngbVidx;
	//PROP_TYPE eProp;
	int outDeg;
} EDGE_INFO;

channel int activeVertexCh    __attribute__((depth(1024)));
channel EDGE_INFO edgeInfoCh  __attribute__((depth(1024)));
channel int edgeInfoChEof     __attribute__((depth(4)));

__kernel void __attribute__((task)) readActiveVertices(
		__global const int* restrict blkActiveVertices, 
		__global const int* restrict blkActiveVertexNum
		)
{
	for(int i = 0; i < blkActiveVertexNum[0]; i++){
		int vertexIdx = blkActiveVertices[i];
		write_channel_altera(activeVertexCh, vertexIdx);
	}
}

__kernel void __attribute__((task)) readNgbInfo(
		__global int*  restrict blkRpa,
		__global int*  restrict blkCia,
		__global PROP_TYPE* restrict blkEdgeProp,
		__global int*  restrict outDeg,
		__global int*  restrict blkActiveVertexNum,
		__global int*  restrict blkVertexNum,
		__global int*  restrict blkEdgeNum,
		__global int*  restrict srcRange,
		__global int*  restrict itNum
		)
{	
	#ifdef SW
	int* rpaStartBuffer = (int*)malloc(sizeof(int) * VERTEX_MAX);
	int* rpaNumBuffer   = (int*)malloc(sizeof(int) * VERTEX_MAX);
	int* outDegBuffer   = (int*)malloc(sizeof(int) * VERTEX_MAX);
	int* ciaBuffer      = (int*)malloc(sizeof(int) * EDGE_MAX);
	PROP_TYPE* edgePropBuffer = (int*)malloc(sizeof(int) * EDGE_MAX);
	#else
	int rpaStartBuffer[VERTEX_MAX];
	int rpaNumBuffer[VERTEX_MAX];
	int outDegBuffer[VERTEX_MAX];
	//int ciaBuffer[EDGE_MAX];
	//PROP_TYPE edgePropBuffer[EDGE_MAX];
	#endif

	int srcStart = srcRange[0];
    int rpao_old = blkRpa[0];
		for(int i = 0; i < blkVertexNum[0]; i++){
			rpaStartBuffer[i] = rpao_old;
			rpaNumBuffer[i]   = blkRpa[i + 1] - rpao_old;
			rpao_old = blkRpa[i + 1];
		}
		for(int i = 0; i < blkVertexNum[0]; i++){
			outDegBuffer[i]   = outDeg[srcStart + i];
		}

/*		for(int i = 0; i < blkEdgeNum[0]; i++){
			ciaBuffer[i]      = blkCia[i];
			edgePropBuffer[i] = blkEdgeProp[i];
		}
*/

	for(int i = 0; i < blkActiveVertexNum[0]; i++){
		int vertexIdx = read_channel_altera(activeVertexCh);
		int bufIdx = vertexIdx - srcStart;
		int start = rpaStartBuffer[bufIdx];
		int num = rpaNumBuffer[bufIdx];
		int deg = outDegBuffer[bufIdx];

		for(int j = 0; j < num; j++){
			int ngbVidx = blkCia[start + j];
			//int eProp = edgePropBuffer[start + j];
			EDGE_INFO edgeInfo;
			edgeInfo.vertexIdx = vertexIdx;
			edgeInfo.ngbVidx = ngbVidx;
			//edgeInfo.eProp = eProp;
			edgeInfo.outDeg = deg;
			write_channel_altera(edgeInfoCh, edgeInfo);
		}
	}
	write_channel_altera(edgeInfoChEof, EOF_FLAG);
}

__kernel void __attribute__((task)) processEdge(
		__global PROP_TYPE* restrict vertexProp,
		__global PROP_TYPE* restrict tmpVertexProp,
		__global const int* restrict eop,
		__global const int* restrict itNum,
		__global const int* restrict srcRange,
		__global const int* restrict sinkRange
		)
{	

	#ifdef SW
	PROP_TYPE * vertexPropBuffer    = (PROP_TYPE*)malloc(sizeof(PROP_TYPE) * VERTEX_MAX);
	PROP_TYPE * tmpVertexPropBuffer = (PROP_TYPE*)malloc(sizeof(PROP_TYPE) * VERTEX_MAX);
	#else
	PROP_TYPE vertexPropBuffer[VERTEX_MAX];
	PROP_TYPE tmpVertexPropBuffer[VERTEX_MAX];
	#endif

	int endFlag = 0;
	bool validData = 0;
	bool validFlag = 0;

	int srcStart = srcRange[0];
	int srcEnd = srcRange[1];
	int srcNum = srcEnd - srcStart;
	int dstStart = sinkRange[0];
	int dstEnd = sinkRange[1];
	int dstNum = dstEnd - dstStart;

		for(int i = 0; i < srcNum; i++){
			vertexPropBuffer[i] = vertexProp[i + srcStart];
		}
		for(int i = 0; i < dstNum; i++){
			tmpVertexPropBuffer[i] = tmpVertexProp[i + dstStart];
		}

	while(true){
		EDGE_INFO edgeInfo = read_channel_nb_altera(edgeInfoCh, &validData);
		if(validData){
			int srcVidx      = edgeInfo.vertexIdx;
			int dstVidx      = edgeInfo.ngbVidx;
			//PROP_TYPE eProp  = edgeInfo.eProp;
			int outDeg       = edgeInfo.outDeg;

			int srcBufIdx = srcVidx - srcStart;
			int dstBufIdx = dstVidx - dstStart;
			PROP_TYPE srcProp = vertexPropBuffer[srcVidx - srcStart];
			PROP_TYPE dstProp = tmpVertexPropBuffer[dstVidx - dstStart];

            #ifdef PR 
			if(outDeg != 0)
				tmpVertexPropBuffer[dstBufIdx] += (vertexPropBuffer[srcBufIdx] / outDeg);
            #endif

            #ifdef BFS
			//tmpVertexPropBuffer[dstBufIdx] = (dstProp > srcProp + 1)? (srcProp + 1) : dstProp;
			if((vertexPropBuffer[srcVidx - srcStart] + 1) < tmpVertexPropBuffer[dstVidx - dstStart]){
				tmpVertexPropBuffer[dstVidx - dstStart] = vertexPropBuffer[srcVidx - srcStart] + 1;
			}
            #endif

            #ifdef SSSP
			tmpVertexPropBuffer[dstBufIdx] = (dstProp > srcProp + eProp)? (srcProp + eProp) : dstProp;
            #endif
		}
		int tmpEndFlag = read_channel_nb_altera(edgeInfoChEof, &validFlag);
		if(validFlag) endFlag = tmpEndFlag;	
		if(endFlag == EOF_FLAG && !validData && !validFlag) break;
	}
		for(int i = 0; i < dstNum; i++){
			tmpVertexProp[i + dstStart] = tmpVertexPropBuffer[i];
		}
}


