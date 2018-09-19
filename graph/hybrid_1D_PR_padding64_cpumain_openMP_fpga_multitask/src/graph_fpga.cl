//#define SW 1
#define PR
#define EOF_FLAG 0xffff
#define PROP_TYPE float

#define VERTEX_MAX  (128*1024)//262144//40960//40960//(128*1024)
#define EDGE_MAX    (2*1024*1024)//5610680////163840 // (1024*1024)
#define BRAM_BANK 64
#define LOG2_BRAM_BANK 6
#define PAD_TYPE int16
#define PAD_WITH 64

typedef struct EdgeInfo{
	int vertexIdx;
	int16 ngbVidx0;
	int16 ngbVidx1;
	int16 ngbVidx2;
	int16 ngbVidx3;
	//PROP_TYPE eProp;
	int outDeg;
} EDGE_INFO;

channel int activeVertexCh    __attribute__((depth(1024)));
channel EDGE_INFO edgeInfoCh  __attribute__((depth(1024)));
channel int edgeInfoChEof     __attribute__((depth(4)));

__attribute__((always_inline)) void compute(int srcVidx, int dstVidx, int dstStart, int outDeg,
	PROP_TYPE* tmpVPropBuffer, PROP_TYPE srcProp){
		//PROP_TYPE srcVprop = vPropBuffer[srcVidx];
		//PROP_TYPE dstVprop = tmpVPropBuffer[(int)(dstVidx / 4)];
	#ifdef PR 
			if(dstVidx != -1) {
				int idx = (dstVidx - dstStart) >> LOG2_BRAM_BANK;
				PROP_TYPE operand1 = tmpVPropBuffer[idx];
			    tmpVPropBuffer[idx] = ( srcProp / outDeg) + operand1;
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
		__global const int* restrict blkActiveVertices, 
		__global const int* restrict blkActiveVertexNum,
		__global const int* restrict iterNum
		)
{	
	int iteration = iterNum[0];
	int baseAddr = iteration << 17;
	for(int i = 0; i < blkActiveVertexNum[iteration]; i++){
		int vertexIdx = blkActiveVertices[i + baseAddr];
		write_channel_altera(activeVertexCh, vertexIdx);
	}

}

__kernel void __attribute__((task)) readNgbInfo(
		__global int16*  restrict blkRpa,
		__global int16*  restrict blkRpaNum,
		__global PAD_TYPE*  restrict blkCia,
		__global PAD_TYPE* restrict blkEdgeProp,
		__global int16*  restrict outDeg,
		__global int*  restrict blkActiveVertexNum,
		__global int*  restrict blkVertexNum,
		__global int*  restrict blkEdgeNum,
		__global int*  restrict srcRange,
		__global int*  restrict iterNum
		)
{	
	#if SW == 1
	int (*rpaStartBuffer)[16]=(int(*)[16])malloc(sizeof(int)* VERTEX_MAX /16 *16);
	int (*rpaNumBuffer)[16]=(int(*)[16])malloc(sizeof(int)* VERTEX_MAX /16 *16);
	int (*outDegBuffer)[16]=(int(*)[16])malloc(sizeof(int)* VERTEX_MAX /16 * 16);
	#else
	int rpaStartBuffer[VERTEX_MAX >> 4][16];
	int rpaNumBuffer[VERTEX_MAX >> 4][16];
	int outDegBuffer[VERTEX_MAX >> 4][16];
	#endif
	int iteration = iterNum[0];
	int baseAddr = iteration << 13; // <<17 >>4
	int srcStart = srcRange[iteration << 1];
    //int rpao_old = blkRpa[0];
	for(int i = 0; i < (VERTEX_MAX >> 4); i++){
			int16 rpa_uint16 = blkRpa[i + baseAddr];
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
	}
	for(int i = 0; i < (VERTEX_MAX >> 4); i++){
			int16 rpa_uint16 = blkRpaNum[i + baseAddr];
			rpaNumBuffer[i][0] = rpa_uint16.s0;
			rpaNumBuffer[i][1] = rpa_uint16.s1;
			rpaNumBuffer[i][2] = rpa_uint16.s2;
			rpaNumBuffer[i][3] = rpa_uint16.s3;
			rpaNumBuffer[i][4] = rpa_uint16.s4;
			rpaNumBuffer[i][5] = rpa_uint16.s5;
			rpaNumBuffer[i][6] = rpa_uint16.s6;
			rpaNumBuffer[i][7] = rpa_uint16.s7;
			rpaNumBuffer[i][8] = rpa_uint16.s8;
			rpaNumBuffer[i][9] = rpa_uint16.s9;
			rpaNumBuffer[i][10] = rpa_uint16.sa;
			rpaNumBuffer[i][11] = rpa_uint16.sb;
			rpaNumBuffer[i][12] = rpa_uint16.sc;
			rpaNumBuffer[i][13] = rpa_uint16.sd;
			rpaNumBuffer[i][14] = rpa_uint16.se;
			rpaNumBuffer[i][15] = rpa_uint16.sf;
	}
	for(int i = 0; i < (VERTEX_MAX >> 4); i++){
			int16 rpa_uint16 = outDeg[i + (srcStart >> 4)];
			outDegBuffer[i][0] = rpa_uint16.s0;
			outDegBuffer[i][1] = rpa_uint16.s1;
			outDegBuffer[i][2] = rpa_uint16.s2;
			outDegBuffer[i][3] = rpa_uint16.s3;
			outDegBuffer[i][4] = rpa_uint16.s4;
			outDegBuffer[i][5] = rpa_uint16.s5;
			outDegBuffer[i][6] = rpa_uint16.s6;
			outDegBuffer[i][7] = rpa_uint16.s7;
			outDegBuffer[i][8] = rpa_uint16.s8;
			outDegBuffer[i][9] = rpa_uint16.s9;
			outDegBuffer[i][10] = rpa_uint16.sa;
			outDegBuffer[i][11] = rpa_uint16.sb;
			outDegBuffer[i][12] = rpa_uint16.sc;
			outDegBuffer[i][13] = rpa_uint16.sd;
			outDegBuffer[i][14] = rpa_uint16.se;
			outDegBuffer[i][15] = rpa_uint16.sf;
	}
	//printf("fpga ck 1\n");
	for(int i = 0; i < blkActiveVertexNum[iteration]; i++){
		int vertexIdx = read_channel_altera(activeVertexCh);
		int bufIdx = vertexIdx - srcStart;
		int start = rpaStartBuffer[bufIdx >> 4][bufIdx & 0xf];
		int num = rpaNumBuffer[bufIdx >> 4][bufIdx & 0xf];
		int deg = outDegBuffer[bufIdx >> 4][bufIdx & 0xf];

		for(int j = start >> 4; j < (start + num) >> 4; j+=4){

			int16 ngbVidx0 = blkCia[j];
			int16 ngbVidx1 = blkCia[j+1];
			int16 ngbVidx2 = blkCia[j+2];
			int16 ngbVidx3 = blkCia[j+3];

			EDGE_INFO edgeInfo;
			edgeInfo.vertexIdx = vertexIdx;
			edgeInfo.ngbVidx0 = ngbVidx0;
			edgeInfo.ngbVidx1 = ngbVidx1;
			edgeInfo.ngbVidx2 = ngbVidx2;
			edgeInfo.ngbVidx3 = ngbVidx3;
			//printf("bufIdx %d, num %d, vertexIdx %d, ngbVidx %d\n",bufIdx, num, vertexIdx,ngbVidx);
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
		__global const int* restrict iterNum,
		__global const int* restrict srcRange,
		__global const int* restrict sinkRange
		)
{	
	#if SW == 1
		PROP_TYPE * vPropBuffer0 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBuffer1 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBuffer2 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBuffer3 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBuffer4 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBuffer5 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBuffer6 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBuffer7 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBuffer8 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBuffer9 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBuffera 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBufferb 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBufferc 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBufferd 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBuffere 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);
		PROP_TYPE * vPropBufferf 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/4);


		PROP_TYPE * tmpVPropBuffer0 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer2 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer3 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer4 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer5 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer6 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer7 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer8 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer9 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffera 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBufferb 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBufferc 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBufferd 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffere 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBufferf 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);



		PROP_TYPE * tmpVPropBuffer10 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer11 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer12 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer13 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer14 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer15 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer16 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer17 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer18 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer19 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1a 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1b 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1c 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1d 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1e 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1f 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);


		PROP_TYPE * tmpVPropBuffer20 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer21 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer22 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer23 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer24 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer25 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer26 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer27 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer28 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer29 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer2a 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer2b 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer2c 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer2d 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer2e 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer2f 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);

		PROP_TYPE * tmpVPropBuffer30 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer31 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer32 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer33 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer34 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer35 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer36 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer37 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer38 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer39 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer3a 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer3b 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer3c 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer3d 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer3e 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer3f 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);

	#else 
	PROP_TYPE vPropBuffer0[VERTEX_MAX >> 4];
	PROP_TYPE vPropBuffer1[VERTEX_MAX >> 4];
	PROP_TYPE vPropBuffer2[VERTEX_MAX >> 4];
	PROP_TYPE vPropBuffer3[VERTEX_MAX >> 4];
	PROP_TYPE vPropBuffer4[VERTEX_MAX >> 4];
	PROP_TYPE vPropBuffer5[VERTEX_MAX >> 4];
	PROP_TYPE vPropBuffer6[VERTEX_MAX >> 4];
	PROP_TYPE vPropBuffer7[VERTEX_MAX >> 4];
	PROP_TYPE vPropBuffer8[VERTEX_MAX >> 4];
	PROP_TYPE vPropBuffer9[VERTEX_MAX >> 4];
	PROP_TYPE vPropBuffera[VERTEX_MAX >> 4];
	PROP_TYPE vPropBufferb[VERTEX_MAX >> 4];
	PROP_TYPE vPropBufferc[VERTEX_MAX >> 4];
	PROP_TYPE vPropBufferd[VERTEX_MAX >> 4];
	PROP_TYPE vPropBuffere[VERTEX_MAX >> 4];
	PROP_TYPE vPropBufferf[VERTEX_MAX >> 4];


	PROP_TYPE tmpVPropBuffer0[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer1[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer2[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer3[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer4[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer5[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer6[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer7[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer8[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer9[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffera[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBufferb[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBufferc[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBufferd[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffere[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBufferf[VERTEX_MAX >> LOG2_BRAM_BANK];

	PROP_TYPE tmpVPropBuffer10[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer11[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer12[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer13[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer14[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer15[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer16[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer17[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer18[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer19[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer1a[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer1b[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer1c[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer1d[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer1e[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer1f[VERTEX_MAX >> LOG2_BRAM_BANK];

	PROP_TYPE tmpVPropBuffer20[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer21[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer22[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer23[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer24[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer25[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer26[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer27[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer28[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer29[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer2a[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer2b[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer2c[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer2d[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer2e[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer2f[VERTEX_MAX >> LOG2_BRAM_BANK];

	PROP_TYPE tmpVPropBuffer30[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer31[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer32[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer33[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer34[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer35[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer36[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer37[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer38[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer39[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer3a[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer3b[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer3c[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer3d[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer3e[VERTEX_MAX >> LOG2_BRAM_BANK];
	PROP_TYPE tmpVPropBuffer3f[VERTEX_MAX >> LOG2_BRAM_BANK];
	#endif

	int  endFlag = 0;
	bool validData = 0;
	bool validFlag = 0;

	//int baseAddr = iterNum[0] << 13; // <<17 >>4
	int iteration= iterNum[0];
	int srcStart = srcRange[iteration << 1];
	int srcEnd   = srcRange[(iteration << 1) + 1];
	int srcNum   = srcEnd - srcStart;
	int dstStart = sinkRange[(iteration << 1)];
	int dstEnd   = sinkRange[(iteration << 1) + 1];
	int dstNum   = dstEnd - dstStart;

		for(int i = 0; i < (srcNum >> 4); i++){
			float16 prop_uint16 = vertexProp[(srcStart >> 4) + i];
			vPropBuffer0[i]=  prop_uint16.s0;
			vPropBuffer1[i]=  prop_uint16.s1;
			vPropBuffer2[i]=  prop_uint16.s2;
			vPropBuffer3[i]=  prop_uint16.s3;
			vPropBuffer4[i]=  prop_uint16.s4;
			vPropBuffer5[i]=  prop_uint16.s5;
			vPropBuffer6[i]=  prop_uint16.s6;
			vPropBuffer7[i]=  prop_uint16.s7;
			vPropBuffer8[i]=  prop_uint16.s8;
			vPropBuffer9[i]=  prop_uint16.s9;
			vPropBuffera[i] = prop_uint16.sa;
			vPropBufferb[i] = prop_uint16.sb;
			vPropBufferc[i] = prop_uint16.sc;
			vPropBufferd[i] = prop_uint16.sd;
			vPropBuffere[i] = prop_uint16.se;
			vPropBufferf[i] = prop_uint16.sf;
		}
		for(int i = 0; i < (dstNum >> 4); i+=4){
			float16 prop_uint16 = tmpVertexProp[(dstStart >> 4) + i];
			tmpVPropBuffer0[i]= 	prop_uint16.s0;
			tmpVPropBuffer1[i]= 	prop_uint16.s1;
			tmpVPropBuffer2[i]= 	prop_uint16.s2;
			tmpVPropBuffer3[i]= 	prop_uint16.s3;
			tmpVPropBuffer4[i]= 	prop_uint16.s4;
			tmpVPropBuffer5[i]= 	prop_uint16.s5;
			tmpVPropBuffer6[i]= 	prop_uint16.s6;
			tmpVPropBuffer7[i]= 	prop_uint16.s7;
			tmpVPropBuffer8[i]= 	prop_uint16.s8;
			tmpVPropBuffer9[i]= 	prop_uint16.s9;
			tmpVPropBuffera[i] = 	prop_uint16.sa;
			tmpVPropBufferb[i] = 	prop_uint16.sb;
			tmpVPropBufferc[i] = 	prop_uint16.sc;
			tmpVPropBufferd[i] = 	prop_uint16.sd;
			tmpVPropBuffere[i] = 	prop_uint16.se;
			tmpVPropBufferf[i] = 	prop_uint16.sf;

			float16 prop1_uint16 = tmpVertexProp[(dstStart >> 4) + i + 1];
			tmpVPropBuffer10[i]= 	prop1_uint16.s0;
			tmpVPropBuffer11[i]= 	prop1_uint16.s1;
			tmpVPropBuffer12[i]= 	prop1_uint16.s2;
			tmpVPropBuffer13[i]= 	prop1_uint16.s3;
			tmpVPropBuffer14[i]= 	prop1_uint16.s4;
			tmpVPropBuffer15[i]= 	prop1_uint16.s5;
			tmpVPropBuffer16[i]= 	prop1_uint16.s6;
			tmpVPropBuffer17[i]= 	prop1_uint16.s7;
			tmpVPropBuffer18[i]= 	prop1_uint16.s8;
			tmpVPropBuffer19[i]= 	prop1_uint16.s9;
			tmpVPropBuffer1a[i] = 	prop1_uint16.sa;
			tmpVPropBuffer1b[i] = 	prop1_uint16.sb;
			tmpVPropBuffer1c[i] = 	prop1_uint16.sc;
			tmpVPropBuffer1d[i] = 	prop1_uint16.sd;
			tmpVPropBuffer1e[i] = 	prop1_uint16.se;
			tmpVPropBuffer1f[i] = 	prop1_uint16.sf;

			float16 prop2_uint16 = tmpVertexProp[(dstStart >> 4) + i + 2];
			tmpVPropBuffer20[i]= 	prop2_uint16.s0;
			tmpVPropBuffer21[i]= 	prop2_uint16.s1;
			tmpVPropBuffer22[i]= 	prop2_uint16.s2;
			tmpVPropBuffer23[i]= 	prop2_uint16.s3;
			tmpVPropBuffer24[i]= 	prop2_uint16.s4;
			tmpVPropBuffer25[i]= 	prop2_uint16.s5;
			tmpVPropBuffer26[i]= 	prop2_uint16.s6;
			tmpVPropBuffer27[i]= 	prop2_uint16.s7;
			tmpVPropBuffer28[i]= 	prop2_uint16.s8;
			tmpVPropBuffer29[i]= 	prop2_uint16.s9;
			tmpVPropBuffer2a[i] = 	prop2_uint16.sa;
			tmpVPropBuffer2b[i] = 	prop2_uint16.sb;
			tmpVPropBuffer2c[i] = 	prop2_uint16.sc;
			tmpVPropBuffer2d[i] = 	prop2_uint16.sd;
			tmpVPropBuffer2e[i] = 	prop2_uint16.se;
			tmpVPropBuffer2f[i] = 	prop2_uint16.sf;

			float16 prop3_uint16 = tmpVertexProp[(dstStart >> 4) + i + 3];
			tmpVPropBuffer30[i]= 	prop3_uint16.s0;
			tmpVPropBuffer31[i]= 	prop3_uint16.s1;
			tmpVPropBuffer32[i]= 	prop3_uint16.s2;
			tmpVPropBuffer33[i]= 	prop3_uint16.s3;
			tmpVPropBuffer34[i]= 	prop3_uint16.s4;
			tmpVPropBuffer35[i]= 	prop3_uint16.s5;
			tmpVPropBuffer36[i]= 	prop3_uint16.s6;
			tmpVPropBuffer37[i]= 	prop3_uint16.s7;
			tmpVPropBuffer38[i]= 	prop3_uint16.s8;
			tmpVPropBuffer39[i]= 	prop3_uint16.s9;
			tmpVPropBuffer3a[i] = 	prop3_uint16.sa;
			tmpVPropBuffer3b[i] = 	prop3_uint16.sb;
			tmpVPropBuffer3c[i] = 	prop3_uint16.sc;
			tmpVPropBuffer3d[i] = 	prop3_uint16.sd;
			tmpVPropBuffer3e[i] = 	prop3_uint16.se;
			tmpVPropBuffer3f[i] = 	prop3_uint16.sf;
		}

	while(true){
		EDGE_INFO edgeInfo = read_channel_nb_altera(edgeInfoCh, &validData);
		if(validData){
			int srcVidx    = edgeInfo.vertexIdx;
			int16 dstVidx0 = edgeInfo.ngbVidx0;
			int16 dstVidx1 = edgeInfo.ngbVidx1;
			int16 dstVidx2 = edgeInfo.ngbVidx2;
			int16 dstVidx3 = edgeInfo.ngbVidx3;

			PROP_TYPE eProp  = 0x1;
			int outDeg = edgeInfo.outDeg;
			int count = 0;
			int srcBufIdx = srcVidx - srcStart;
			PROP_TYPE srcProp;
		
			switch(srcBufIdx & 0xf){
				case 0 : srcProp = vPropBuffer0[srcBufIdx >> 4]; break;
				case 1 : srcProp = vPropBuffer1[srcBufIdx >> 4]; break;
				case 2 : srcProp = vPropBuffer2[srcBufIdx >> 4]; break;
				case 3 : srcProp = vPropBuffer3[srcBufIdx >> 4]; break;
				case 4 : srcProp = vPropBuffer4[srcBufIdx >> 4]; break;
				case 5 : srcProp = vPropBuffer5[srcBufIdx >> 4]; break;
				case 6 : srcProp = vPropBuffer6[srcBufIdx >> 4]; break;
				case 7 : srcProp = vPropBuffer7[srcBufIdx >> 4]; break;
				case 8 : srcProp = vPropBuffer8[srcBufIdx >> 4]; break;
				case 9 : srcProp = vPropBuffer9[srcBufIdx >> 4]; break;
				case 10: srcProp = vPropBuffera[srcBufIdx >> 4]; break;
				case 11: srcProp = vPropBufferb[srcBufIdx >> 4]; break;
				case 12: srcProp = vPropBufferc[srcBufIdx >> 4]; break;
				case 13: srcProp = vPropBufferd[srcBufIdx >> 4]; break;
				case 14: srcProp = vPropBuffere[srcBufIdx >> 4]; break;
				case 15: srcProp = vPropBufferf[srcBufIdx >> 4]; break;
			}
			//int dstBufIdx = dstVidx - dstStart;
			//PROP_TYPE srcProp = vertexPropBuffer[srcVidx - srcStart];
			//PROP_TYPE dstProp = tmpVertexPropBuffer[dstVidx - dstStart];
 
					compute(srcVidx, dstVidx0.s0, dstStart, outDeg, tmpVPropBuffer0, srcProp);
					compute(srcVidx, dstVidx0.s1, dstStart, outDeg, tmpVPropBuffer1, srcProp);
					compute(srcVidx, dstVidx0.s2, dstStart, outDeg, tmpVPropBuffer2, srcProp);
					compute(srcVidx, dstVidx0.s3, dstStart, outDeg, tmpVPropBuffer3, srcProp);
					compute(srcVidx, dstVidx0.s4, dstStart, outDeg, tmpVPropBuffer4, srcProp);
					compute(srcVidx, dstVidx0.s5, dstStart, outDeg, tmpVPropBuffer5, srcProp);
					compute(srcVidx, dstVidx0.s6, dstStart, outDeg, tmpVPropBuffer6, srcProp);
					compute(srcVidx, dstVidx0.s7, dstStart, outDeg, tmpVPropBuffer7, srcProp);
					compute(srcVidx, dstVidx0.s8, dstStart, outDeg, tmpVPropBuffer8, srcProp);
					compute(srcVidx, dstVidx0.s9, dstStart, outDeg, tmpVPropBuffer9, srcProp);
					compute(srcVidx, dstVidx0.sa, dstStart, outDeg, tmpVPropBuffera, srcProp);
					compute(srcVidx, dstVidx0.sb, dstStart, outDeg, tmpVPropBufferb, srcProp);
					compute(srcVidx, dstVidx0.sc, dstStart, outDeg, tmpVPropBufferc, srcProp);
					compute(srcVidx, dstVidx0.sd, dstStart, outDeg, tmpVPropBufferd, srcProp);
					compute(srcVidx, dstVidx0.se, dstStart, outDeg, tmpVPropBuffere, srcProp);
					compute(srcVidx, dstVidx0.sf, dstStart, outDeg, tmpVPropBufferf, srcProp);

					compute(srcVidx, dstVidx1.s0, dstStart, outDeg, tmpVPropBuffer10, srcProp);
					compute(srcVidx, dstVidx1.s1, dstStart, outDeg, tmpVPropBuffer11, srcProp);
					compute(srcVidx, dstVidx1.s2, dstStart, outDeg, tmpVPropBuffer12, srcProp);
					compute(srcVidx, dstVidx1.s3, dstStart, outDeg, tmpVPropBuffer13, srcProp);
					compute(srcVidx, dstVidx1.s4, dstStart, outDeg, tmpVPropBuffer14, srcProp);
					compute(srcVidx, dstVidx1.s5, dstStart, outDeg, tmpVPropBuffer15, srcProp);
					compute(srcVidx, dstVidx1.s6, dstStart, outDeg, tmpVPropBuffer16, srcProp);
					compute(srcVidx, dstVidx1.s7, dstStart, outDeg, tmpVPropBuffer17, srcProp);
					compute(srcVidx, dstVidx1.s8, dstStart, outDeg, tmpVPropBuffer18, srcProp);
					compute(srcVidx, dstVidx1.s9, dstStart, outDeg, tmpVPropBuffer19, srcProp);
					compute(srcVidx, dstVidx1.sa, dstStart, outDeg, tmpVPropBuffer1a, srcProp);
					compute(srcVidx, dstVidx1.sb, dstStart, outDeg, tmpVPropBuffer1b, srcProp);
					compute(srcVidx, dstVidx1.sc, dstStart, outDeg, tmpVPropBuffer1c, srcProp);
					compute(srcVidx, dstVidx1.sd, dstStart, outDeg, tmpVPropBuffer1d, srcProp);
					compute(srcVidx, dstVidx1.se, dstStart, outDeg, tmpVPropBuffer1e, srcProp);
					compute(srcVidx, dstVidx1.sf, dstStart, outDeg, tmpVPropBuffer1f, srcProp);

					compute(srcVidx, dstVidx2.s0, dstStart, outDeg, tmpVPropBuffer20, srcProp);
					compute(srcVidx, dstVidx2.s1, dstStart, outDeg, tmpVPropBuffer21, srcProp);
					compute(srcVidx, dstVidx2.s2, dstStart, outDeg, tmpVPropBuffer22, srcProp);
					compute(srcVidx, dstVidx2.s3, dstStart, outDeg, tmpVPropBuffer23, srcProp);
					compute(srcVidx, dstVidx2.s4, dstStart, outDeg, tmpVPropBuffer24, srcProp);
					compute(srcVidx, dstVidx2.s5, dstStart, outDeg, tmpVPropBuffer25, srcProp);
					compute(srcVidx, dstVidx2.s6, dstStart, outDeg, tmpVPropBuffer26, srcProp);
					compute(srcVidx, dstVidx2.s7, dstStart, outDeg, tmpVPropBuffer27, srcProp);
					compute(srcVidx, dstVidx2.s8, dstStart, outDeg, tmpVPropBuffer28, srcProp);
					compute(srcVidx, dstVidx2.s9, dstStart, outDeg, tmpVPropBuffer29, srcProp);
					compute(srcVidx, dstVidx2.sa, dstStart, outDeg, tmpVPropBuffer2a, srcProp);
					compute(srcVidx, dstVidx2.sb, dstStart, outDeg, tmpVPropBuffer2b, srcProp);
					compute(srcVidx, dstVidx2.sc, dstStart, outDeg, tmpVPropBuffer2c, srcProp);
					compute(srcVidx, dstVidx2.sd, dstStart, outDeg, tmpVPropBuffer2d, srcProp);
					compute(srcVidx, dstVidx2.se, dstStart, outDeg, tmpVPropBuffer2e, srcProp);
					compute(srcVidx, dstVidx2.sf, dstStart, outDeg, tmpVPropBuffer2f, srcProp);


					compute(srcVidx, dstVidx3.s0, dstStart, outDeg, tmpVPropBuffer30, srcProp);
					compute(srcVidx, dstVidx3.s1, dstStart, outDeg, tmpVPropBuffer31, srcProp);
					compute(srcVidx, dstVidx3.s2, dstStart, outDeg, tmpVPropBuffer32, srcProp);
					compute(srcVidx, dstVidx3.s3, dstStart, outDeg, tmpVPropBuffer33, srcProp);
					compute(srcVidx, dstVidx3.s4, dstStart, outDeg, tmpVPropBuffer34, srcProp);
					compute(srcVidx, dstVidx3.s5, dstStart, outDeg, tmpVPropBuffer35, srcProp);
					compute(srcVidx, dstVidx3.s6, dstStart, outDeg, tmpVPropBuffer36, srcProp);
					compute(srcVidx, dstVidx3.s7, dstStart, outDeg, tmpVPropBuffer37, srcProp);
					compute(srcVidx, dstVidx3.s8, dstStart, outDeg, tmpVPropBuffer38, srcProp);
					compute(srcVidx, dstVidx3.s9, dstStart, outDeg, tmpVPropBuffer39, srcProp);
					compute(srcVidx, dstVidx3.sa, dstStart, outDeg, tmpVPropBuffer3a, srcProp);
					compute(srcVidx, dstVidx3.sb, dstStart, outDeg, tmpVPropBuffer3b, srcProp);
					compute(srcVidx, dstVidx3.sc, dstStart, outDeg, tmpVPropBuffer3c, srcProp);
					compute(srcVidx, dstVidx3.sd, dstStart, outDeg, tmpVPropBuffer3d, srcProp);
					compute(srcVidx, dstVidx3.se, dstStart, outDeg, tmpVPropBuffer3e, srcProp);
					compute(srcVidx, dstVidx3.sf, dstStart, outDeg, tmpVPropBuffer3f, srcProp);
				
		}
		int tmpEndFlag = read_channel_nb_altera(edgeInfoChEof, &validFlag);
		if(validFlag) endFlag = tmpEndFlag;	
		if(endFlag == EOF_FLAG && !validData && !validFlag) break;
	}
		//for(int i = 0; i < dstNum; i++){
			//tmpVertexProp[i + dstStart] = tmpVertexPropBuffer[i];
		//}
		for(int i = 0; i < (dstNum >> 4); i+=4){
			float16 prop_uint16;
			prop_uint16.s0 = tmpVPropBuffer0[i]; 
			prop_uint16.s1 = tmpVPropBuffer1[i]; 
			prop_uint16.s2 = tmpVPropBuffer2[i]; 
			prop_uint16.s3 = tmpVPropBuffer3[i]; 
			prop_uint16.s4 = tmpVPropBuffer4[i]; 
			prop_uint16.s5 = tmpVPropBuffer5[i]; 
			prop_uint16.s6 = tmpVPropBuffer6[i]; 
			prop_uint16.s7 = tmpVPropBuffer7[i]; 
			prop_uint16.s8 = tmpVPropBuffer8[i]; 
			prop_uint16.s9 = tmpVPropBuffer9[i]; 
			prop_uint16.sa = tmpVPropBuffera[i]; 
			prop_uint16.sb = tmpVPropBufferb[i]; 
			prop_uint16.sc = tmpVPropBufferc[i]; 
			prop_uint16.sd = tmpVPropBufferd[i]; 
			prop_uint16.se = tmpVPropBuffere[i]; 
			prop_uint16.sf = tmpVPropBufferf[i]; 
			tmpVertexProp[(dstStart >> 4) + i] = prop_uint16;
			float16 prop1_uint16;
			prop1_uint16.s0 = tmpVPropBuffer10[i]; 
			prop1_uint16.s1 = tmpVPropBuffer11[i]; 
			prop1_uint16.s2 = tmpVPropBuffer12[i]; 
			prop1_uint16.s3 = tmpVPropBuffer13[i]; 
			prop1_uint16.s4 = tmpVPropBuffer14[i]; 
			prop1_uint16.s5 = tmpVPropBuffer15[i]; 
			prop1_uint16.s6 = tmpVPropBuffer16[i]; 
			prop1_uint16.s7 = tmpVPropBuffer17[i]; 
			prop1_uint16.s8 = tmpVPropBuffer18[i]; 
			prop1_uint16.s9 = tmpVPropBuffer19[i]; 
			prop1_uint16.sa = tmpVPropBuffer1a[i]; 
			prop1_uint16.sb = tmpVPropBuffer1b[i]; 
			prop1_uint16.sc = tmpVPropBuffer1c[i]; 
			prop1_uint16.sd = tmpVPropBuffer1d[i]; 
			prop1_uint16.se = tmpVPropBuffer1e[i]; 
			prop1_uint16.sf = tmpVPropBuffer1f[i]; 
			tmpVertexProp[(dstStart >> 4) + i + 1] = prop1_uint16;
			float16 prop2_uint16;
			prop2_uint16.s0 = tmpVPropBuffer20[i]; 
			prop2_uint16.s1 = tmpVPropBuffer21[i]; 
			prop2_uint16.s2 = tmpVPropBuffer22[i]; 
			prop2_uint16.s3 = tmpVPropBuffer23[i]; 
			prop2_uint16.s4 = tmpVPropBuffer24[i]; 
			prop2_uint16.s5 = tmpVPropBuffer25[i]; 
			prop2_uint16.s6 = tmpVPropBuffer26[i]; 
			prop2_uint16.s7 = tmpVPropBuffer27[i]; 
			prop2_uint16.s8 = tmpVPropBuffer28[i]; 
			prop2_uint16.s9 = tmpVPropBuffer29[i]; 
			prop2_uint16.sa = tmpVPropBuffer2a[i]; 
			prop2_uint16.sb = tmpVPropBuffer2b[i]; 
			prop2_uint16.sc = tmpVPropBuffer2c[i]; 
			prop2_uint16.sd = tmpVPropBuffer2d[i]; 
			prop2_uint16.se = tmpVPropBuffer2e[i]; 
			prop2_uint16.sf = tmpVPropBuffer2f[i]; 
			tmpVertexProp[(dstStart >> 4) + i + 2] = prop2_uint16;
			float16 prop3_uint16;
			prop3_uint16.s0 = tmpVPropBuffer30[i]; 
			prop3_uint16.s1 = tmpVPropBuffer31[i]; 
			prop3_uint16.s2 = tmpVPropBuffer32[i]; 
			prop3_uint16.s3 = tmpVPropBuffer33[i]; 
			prop3_uint16.s4 = tmpVPropBuffer34[i]; 
			prop3_uint16.s5 = tmpVPropBuffer35[i]; 
			prop3_uint16.s6 = tmpVPropBuffer36[i]; 
			prop3_uint16.s7 = tmpVPropBuffer37[i]; 
			prop3_uint16.s8 = tmpVPropBuffer38[i]; 
			prop3_uint16.s9 = tmpVPropBuffer39[i]; 
			prop3_uint16.sa = tmpVPropBuffer3a[i]; 
			prop3_uint16.sb = tmpVPropBuffer3b[i]; 
			prop3_uint16.sc = tmpVPropBuffer3c[i]; 
			prop3_uint16.sd = tmpVPropBuffer3d[i]; 
			prop3_uint16.se = tmpVPropBuffer3e[i]; 
			prop3_uint16.sf = tmpVPropBuffer3f[i]; 
			tmpVertexProp[(dstStart >> 4) + i + 3] = prop3_uint16;

		}
}


