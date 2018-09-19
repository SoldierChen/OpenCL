//#define SW 1
#define PR
#define EOF_FLAG 0xffff
#define PROP_TYPE float

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
		__global int16*  restrict blkRpa,
		__global int*  restrict blkCia,
		__global PROP_TYPE* restrict blkEdgeProp,
		__global int16*  restrict outDeg,
		__global int*  restrict blkActiveVertexNum,
		__global int*  restrict blkVertexNum,
		__global int*  restrict blkEdgeNum,
		__global int*  restrict srcRange,
		__global int*  restrict itNum
		)
{	
	#ifdef SW
	//int* rpaStartBuffer = (int*)malloc(sizeof(int) * VERTEX_MAX);
	//int* rpaNumBuffer   = (int*)malloc(sizeof(int) * VERTEX_MAX);
	//int* outDegBuffer   = (int*)malloc(sizeof(int) * VERTEX_MAX);
	/*int* ciaBuffer      = (int*)malloc(sizeof(int) * EDGE_MAX);

	int** rpaStartBuffer=(int **)malloc( VERTEX_MAX  *sizeof(int *));  
     for(int i=0; i< VERTEX_MAX ;i++)  
         rpaStartBuffer[i]=(int *)malloc(16*sizeof(int));  
    int** rpaNumBuffer=(int **)malloc( VERTEX_MAX  *sizeof(int *));  
     for(int i=0; i< VERTEX_MAX ;i++)  
         rpaNumBuffer[i]=(int *)malloc(16*sizeof(int));  
    int** outDegBuffer=(int **)malloc( VERTEX_MAX  *sizeof(int *));  
     for(int i=0; i< VERTEX_MAX ;i++)  
         outDegBuffer[i]=(int *)malloc(16*sizeof(int));  
	*/
    //int outDegBuffer[VERTEX_MAX/16][16];
    //int* outDegBuffer   = (int*)malloc(sizeof(int) * VERTEX_MAX);

    int (*rpaStartBuffer)[16]=(int(*)[16])malloc(sizeof(int)* VERTEX_MAX /16 *16);
    int (*rpaNumBuffer)[16]=(int(*)[16])malloc(sizeof(int)* VERTEX_MAX /16 *16);   
    int (*outDegBuffer)[16]=(int(*)[16])malloc(sizeof(int)* VERTEX_MAX /16 * 16);        
	//PROP_TYPE* edgePropBuffer = (int*)malloc(sizeof(int) * EDGE_MAX);
	#else
	int rpaStartBuffer[VERTEX_MAX >> 4][16];
	//int rpaNumBuffer[VERTEX_MAX >> 4][16];
	int outDegBuffer[VERTEX_MAX >> 4][16];

	#endif

	int srcStart = srcRange[0];
   // int rpao_old = blkRpa[0];
		for(int i = 0; i < (blkVertexNum[0] >> 4); i++){
			int16 rpa_uint16 = blkRpa[i];
			rpaStartBuffer[i][0] = rpa_uint16.s0;
			rpaStartBuffer[i][1] = rpa_uint16.s1;
			rpaStartBuffer[i][2] = rpa_uint16.s2;
			rpaStartBuffer[i][3] = rpa_uint16.s3;
			rpaStartBuffer[i][4] = rpa_uint16.s4;
			rpaStartBuffer[i][5] = rpa_uint16.s5;
			rpaStartBuffer[i][6] = rpa_uint16.s6;
			rpaStartBuffer[i][7] = rpa_uint16.s7;
			rpaStartBuffer[i][8] = rpa_uint16.s8;
			rpaStartBuffer[i][9] = rpa_uint16.s9;
			rpaStartBuffer[i][10] = rpa_uint16.sa;
			rpaStartBuffer[i][11] = rpa_uint16.sb;
			rpaStartBuffer[i][12] = rpa_uint16.sc;
			rpaStartBuffer[i][13] = rpa_uint16.sd;
			rpaStartBuffer[i][14] = rpa_uint16.se;
			rpaStartBuffer[i][15] = rpa_uint16.sf;
			/*
			rpaStartBuffer[i] = blkRpa[i];
			rpaNumBuffer[i]   = blkRpa[i + 1] - blkRpa[i];
			rpao_old = blkRpa[i + 1];
			*/
		}
		for(int i = 0; i < (blkVertexNum[0] >> 4); i++){
			int16 deg_uint16 = outDeg[(srcStart >> 4) + i];
			outDegBuffer[i][0] = deg_uint16.s0;
			outDegBuffer[i][1] = deg_uint16.s1;
			outDegBuffer[i][2] = deg_uint16.s2;
			outDegBuffer[i][3] = deg_uint16.s3;
			outDegBuffer[i][4] = deg_uint16.s4;
			outDegBuffer[i][5] = deg_uint16.s5;
			outDegBuffer[i][6] = deg_uint16.s6;
			outDegBuffer[i][7] = deg_uint16.s7;
			outDegBuffer[i][8] = deg_uint16.s8;
			outDegBuffer[i][9] = deg_uint16.s9;
			outDegBuffer[i][10] = deg_uint16.sa;
			outDegBuffer[i][11] = deg_uint16.sb;
			outDegBuffer[i][12] = deg_uint16.sc;
			outDegBuffer[i][13] = deg_uint16.sd;
			outDegBuffer[i][14] = deg_uint16.se;
			outDegBuffer[i][15] = deg_uint16.sf;
    }
    /*
		for(int i = 0; i < blkVertexNum[0] - 1; i++){
			rpaNumBuffer[i >> 4][i & 0xf] = rpaStartBuffer[(i+1) >> 4][(i+1) & 0xf] - rpaStartBuffer[i >> 4][i & 0xf];
		}
			rpaNumBuffer[(blkVertexNum[0]-1) >> 4][(blkVertexNum[0]-1) & 0xf] = 0;
      */
/*
		for(int i = 0; i < blkVertexNum[0]; i++){
			outDegBuffer[i]   = outDeg[srcStart + i];
		}
*/
		//for(int i = 0; i < blkVertexNum[0] >> 4; i++){

			//rpaNumBuffer[i]   = blkRpa[i + 1] - blkRpa[i];
			//rpao_old = blkRpa[i + 1];
		//}

/*		for(int i = 0; i < blkEdgeNum[0]; i++){
			ciaBuffer[i]      = blkCia[i];
			edgePropBuffer[i] = blkEdgeProp[i];
		}
*/
	for(int i = 0; i < blkActiveVertexNum[0]; i++){
		int vertexIdx = read_channel_altera(activeVertexCh);
		int bufIdx = vertexIdx - srcStart;
		int start = rpaStartBuffer[bufIdx >> 4][bufIdx & 0xf];
		int end = rpaStartBuffer[(bufIdx + 1) >> 4][(bufIdx + 1) & 0xf];
		int num = start - end;
		//int deg = outDegBuffer[bufIdx >> 4][bufIdx & 0xe];
		int deg = outDegBuffer[bufIdx >> 4][bufIdx & 0xf];

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
		__global float16* restrict vertexProp,
		__global float16* restrict tmpVertexProp,
		__global const int* restrict eop,
		__global const int* restrict itNum,
		__global const int* restrict srcRange,
		__global const int* restrict sinkRange
		)
{	

	#ifdef SW
	//PROP_TYPE * vertexPropBuffer    = (PROP_TYPE*)malloc(sizeof(PROP_TYPE) * VERTEX_MAX);
	//PROP_TYPE * tmpVertexPropBuffer = (PROP_TYPE*)malloc(sizeof(PROP_TYPE) * VERTEX_MAX);
	PROP_TYPE (*vertexPropBuffer)[16]=(PROP_TYPE(*)[16])malloc(sizeof(PROP_TYPE)* VERTEX_MAX /16 * 16);
	PROP_TYPE (*tmpVertexPropBuffer)[16]=(PROP_TYPE(*)[16])malloc(sizeof(PROP_TYPE)* VERTEX_MAX /16 * 16);  
	#else
	PROP_TYPE vertexPropBuffer[VERTEX_MAX >> 4][16];
	PROP_TYPE tmpVertexPropBuffer[VERTEX_MAX >> 4][16];
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

		for(int i = 0; i < (srcNum >> 4); i++){
			
			float16 prop_uint16 = vertexProp[(srcStart >> 4) + i];
			vertexPropBuffer[i][0] =  prop_uint16.s0;
			vertexPropBuffer[i][1] =  prop_uint16.s1;
			vertexPropBuffer[i][2] =  prop_uint16.s2;
			vertexPropBuffer[i][3] =  prop_uint16.s3;
			vertexPropBuffer[i][4] =  prop_uint16.s4;
			vertexPropBuffer[i][5] =  prop_uint16.s5;
			vertexPropBuffer[i][6] =  prop_uint16.s6;
			vertexPropBuffer[i][7] =  prop_uint16.s7;
			vertexPropBuffer[i][8] =  prop_uint16.s8;
			vertexPropBuffer[i][9] =  prop_uint16.s9;
			vertexPropBuffer[i][10] = prop_uint16.sa;
			vertexPropBuffer[i][11] = prop_uint16.sb;
			vertexPropBuffer[i][12] = prop_uint16.sc;
			vertexPropBuffer[i][13] = prop_uint16.sd;
			vertexPropBuffer[i][14] = prop_uint16.se;
			vertexPropBuffer[i][15] = prop_uint16.sf;
		
			//vertexPropBuffer[i] = vertexProp[i + srcStart];
		}
		for(int i = 0; i < (dstNum >> 4); i++){
			float16 prop_uint16 = tmpVertexProp[(dstStart >> 4) + i];
			tmpVertexPropBuffer[i][0] =  prop_uint16.s0;
			tmpVertexPropBuffer[i][1] =  prop_uint16.s1;
			tmpVertexPropBuffer[i][2] =  prop_uint16.s2;
			tmpVertexPropBuffer[i][3] =  prop_uint16.s3;
			tmpVertexPropBuffer[i][4] =  prop_uint16.s4;
			tmpVertexPropBuffer[i][5] =  prop_uint16.s5;
			tmpVertexPropBuffer[i][6] =  prop_uint16.s6;
			tmpVertexPropBuffer[i][7] =  prop_uint16.s7;
			tmpVertexPropBuffer[i][8] =  prop_uint16.s8;
			tmpVertexPropBuffer[i][9] =  prop_uint16.s9;
			tmpVertexPropBuffer[i][10] = prop_uint16.sa;
			tmpVertexPropBuffer[i][11] = prop_uint16.sb;
			tmpVertexPropBuffer[i][12] = prop_uint16.sc;
			tmpVertexPropBuffer[i][13] = prop_uint16.sd;
			tmpVertexPropBuffer[i][14] = prop_uint16.se;
			tmpVertexPropBuffer[i][15] = prop_uint16.sf;
			
			//tmpVertexPropBuffer[i] = tmpVertexProp[i + dstStart];
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
			//PROP_TYPE srcProp = vertexPropBuffer[srcVidx - srcStart];
			//PROP_TYPE dstProp = tmpVertexPropBuffer[dstVidx - dstStart];

            #ifdef PR 
				if(outDeg != 0)
					tmpVertexPropBuffer[dstBufIdx >> 4][dstBufIdx & 0xf] += (vertexPropBuffer[srcBufIdx >> 4][srcBufIdx & 0xf] / outDeg);
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


		//for(int i = 0; i < dstNum; i++){
			//tmpVertexProp[i + dstStart] = tmpVertexPropBuffer[i];
		//}

		for(int i = 0; i < (dstNum >> 4); i++){
			
			float16 prop_uint16;
			prop_uint16.s0 = tmpVertexPropBuffer[i][0] ; 
			prop_uint16.s1 = tmpVertexPropBuffer[i][1] ; 
			prop_uint16.s2 = tmpVertexPropBuffer[i][2] ; 
			prop_uint16.s3 = tmpVertexPropBuffer[i][3] ; 
			prop_uint16.s4 = tmpVertexPropBuffer[i][4] ; 
			prop_uint16.s5 = tmpVertexPropBuffer[i][5] ; 
			prop_uint16.s6 = tmpVertexPropBuffer[i][6] ; 
			prop_uint16.s7 = tmpVertexPropBuffer[i][7] ; 
			prop_uint16.s8 = tmpVertexPropBuffer[i][8] ; 
			prop_uint16.s9 = tmpVertexPropBuffer[i][9] ; 
			prop_uint16.sa = tmpVertexPropBuffer[i][10]; 
			prop_uint16.sb = tmpVertexPropBuffer[i][11]; 
			prop_uint16.sc = tmpVertexPropBuffer[i][12]; 
			prop_uint16.sd = tmpVertexPropBuffer[i][13]; 
			prop_uint16.se = tmpVertexPropBuffer[i][14]; 
			prop_uint16.sf = tmpVertexPropBuffer[i][15]; 
			tmpVertexProp[(dstStart >> 4) + i] = prop_uint16;
			//tmpVertexProp[i + dstStart] = tmpVertexPropBuffer[i];
		}

}
