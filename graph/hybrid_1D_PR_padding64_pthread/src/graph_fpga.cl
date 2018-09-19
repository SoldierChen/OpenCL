//#define SW 
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
	PAD_TYPE ngbVidx0;
	PAD_TYPE ngbVidx1;
	PAD_TYPE ngbVidx2;
	PAD_TYPE ngbVidx3;
	int outDeg;
} EDGE_INFO;

channel int activeVertexCh    __attribute__((depth(1024)));
channel EDGE_INFO edgeInfoCh  __attribute__((depth(1024)));
channel int edgeInfoChEof     __attribute__((depth(4)));

__attribute__((always_inline)) void compute(int srcVidx, int dstVidx, int outDeg,
	PROP_TYPE* tmpVPropBuffer, PROP_TYPE srcProp){

	#ifdef PR 
			if(dstVidx >= 0) {
				int idx = dstVidx >> LOG2_BRAM_BANK;
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
		__global int*  restrict blkNum,
		__global PAD_TYPE*  restrict blkCia,
		__global PAD_TYPE* restrict blkEdgeProp,
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
	#else
	int rpaStartBuffer[VERTEX_MAX];
	int rpaNumBuffer[VERTEX_MAX];
	int outDegBuffer[VERTEX_MAX];
	//int ciaBuffer[EDGE_MAX];
	//PROP_TYPE edgePropBuffer[EDGE_MAX];
	#endif
	int srcStart = srcRange[0];
#pragma unroll 16
	for(int i = 0; i < blkVertexNum[0]; i++){
		outDegBuffer[i] = outDeg[srcStart + i];
	}
#pragma unroll 16
	for(int i = 0; i < blkVertexNum[0]; i++){
		rpaStartBuffer[i] = blkRpa[i];
	}
#pragma unroll 16
	for(int i = 0; i < blkVertexNum[0]; i++){
		rpaNumBuffer[i] = blkNum[i];
	}

	for(int i = 0; i < blkActiveVertexNum[0]; i++){
		int vertexIdx = read_channel_altera(activeVertexCh);
		int bufIdx = vertexIdx - srcStart;
		int start = rpaStartBuffer[bufIdx];
		int num = rpaNumBuffer[bufIdx];
		int deg = outDegBuffer[bufIdx];

		for(int j = start >> 4; j < ((start + num) >> 4); j += 4){
			PAD_TYPE ngbVidx0 = blkCia[j];
			PAD_TYPE ngbVidx1 = blkCia[j+1];
			PAD_TYPE ngbVidx2 = blkCia[j+2];
			PAD_TYPE ngbVidx3 = blkCia[j+3];
			//printf("%d %d %d %d \n",ngbVidx.s0, ngbVidx.s1, ngbVidx.s2, ngbVidx.s3 );
			//int4 eProp = edgePropBuffer[start + j];
			EDGE_INFO edge_info;
			edge_info.vertexIdx= vertexIdx;
			edge_info.ngbVidx0 = ngbVidx0;
			edge_info.ngbVidx1 = ngbVidx1;
			edge_info.ngbVidx2 = ngbVidx2;
			edge_info.ngbVidx3 = ngbVidx3;
		//	if(bufIdx == 100)
		//	printf("bufIdx %d, num %d, vertexIdx %d\n",bufIdx, num, vertexIdx);
			edge_info.outDeg = deg;
			write_channel_altera(edgeInfoCh, edge_info);
		}
	}
	write_channel_altera(edgeInfoChEof, EOF_FLAG);
}

__kernel void __attribute__((task)) processEdge(
		__global float* restrict vertexProp,
		__global float16* restrict tmpVertexProp,
		__global const int* restrict eop,
		__global const int* restrict itNum,
		__global const int* restrict srcRange,
		__global const int* restrict sinkRange
		)
{	
	#ifdef SW
		PROP_TYPE * vPropBuffer 	= 	 (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX);

		PROP_TYPE * tmpVertexPropBuffer0=  (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1=  (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2=  (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3=  (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer4=  (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer5=  (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer6=  (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer7=  (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer8=  (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer9=  (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffera = (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBufferb = (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBufferc = (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBufferd = (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffere = (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBufferf = (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer10= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer11= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer12= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer13= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer14= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer15= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer16= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer17= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer18= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer19= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1a =(PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1b =(PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1c =(PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1d =(PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1e =(PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1f =(PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);

		PROP_TYPE * tmpVertexPropBuffer20= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer21= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer22= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer23= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer24= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer25= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer26= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer27= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer28= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer29= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2a = (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2b = (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2c = (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2d = (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2e = (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2f = (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer30= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer31= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer32= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer33= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer34= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer35= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer36= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer37= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer38= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer39= (PROP_TYPE *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3a = (PROP_TYPE *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3b = (PROP_TYPE *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3c = (PROP_TYPE *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3d = (PROP_TYPE *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3e = (PROP_TYPE *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3f = (PROP_TYPE *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
	#else 
		//PROP_TYPE vPropBuffer[VERTEX_MAX];
		PROP_TYPE vPropBuffer[VERTEX_MAX];
	
		PROP_TYPE tmpVertexPropBuffer0[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer1[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer2[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer3[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer4[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer5[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer6[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer7[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer8[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer9[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffera[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBufferb[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBufferc[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBufferd[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffere[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBufferf[VERTEX_MAX >> LOG2_BRAM_BANK];
	
		PROP_TYPE tmpVertexPropBuffer10[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer11[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer12[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer13[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer14[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer15[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer16[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer17[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer18[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer19[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer1a[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer1b[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer1c[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer1d[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer1e[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer1f[VERTEX_MAX >> LOG2_BRAM_BANK];
	
		PROP_TYPE tmpVertexPropBuffer20[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer21[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer22[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer23[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer24[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer25[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer26[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer27[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer28[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer29[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer2a[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer2b[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer2c[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer2d[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer2e[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer2f[VERTEX_MAX >> LOG2_BRAM_BANK];
	
		PROP_TYPE tmpVertexPropBuffer30[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer31[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer32[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer33[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer34[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer35[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer36[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer37[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer38[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer39[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer3a[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer3b[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer3c[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer3d[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer3e[VERTEX_MAX >> LOG2_BRAM_BANK];
		PROP_TYPE tmpVertexPropBuffer3f[VERTEX_MAX >> LOG2_BRAM_BANK];
	#endif

	int  endFlag = 0;
	bool validData = 0;
	bool validFlag = 0;

	int srcStart = srcRange[0];
	int srcEnd = srcRange[1];
	int srcNum = srcEnd - srcStart;
	int dstStart = sinkRange[0];
	int dstEnd = sinkRange[1];
	int dstNum = dstEnd - dstStart;
	#pragma unroll 16
		for(int i = 0; i < (srcNum); i++){
			vPropBuffer[i] = vertexProp[(srcStart) + i];
		}

	for(int i = 0; i < (dstNum >> 4); i += 4){
			float16 prop_uint16 = tmpVertexProp[(dstStart >> 4) + i];
			tmpVertexPropBuffer0[i]= 	prop_uint16.s0;
			tmpVertexPropBuffer1[i]= 	prop_uint16.s1;
			tmpVertexPropBuffer2[i]= 	prop_uint16.s2;
			tmpVertexPropBuffer3[i]= 	prop_uint16.s3;
			tmpVertexPropBuffer4[i]= 	prop_uint16.s4;
			tmpVertexPropBuffer5[i]= 	prop_uint16.s5;
			tmpVertexPropBuffer6[i]= 	prop_uint16.s6;
			tmpVertexPropBuffer7[i]= 	prop_uint16.s7;
			tmpVertexPropBuffer8[i]= 	prop_uint16.s8;
			tmpVertexPropBuffer9[i]= 	prop_uint16.s9;
			tmpVertexPropBuffera[i]= 	prop_uint16.sa;
			tmpVertexPropBufferb[i]= 	prop_uint16.sb;
			tmpVertexPropBufferc[i]= 	prop_uint16.sc;
			tmpVertexPropBufferd[i]= 	prop_uint16.sd;
			tmpVertexPropBuffere[i]= 	prop_uint16.se;
			tmpVertexPropBufferf[i]= 	prop_uint16.sf;

			float16 prop1_uint16 = tmpVertexProp[(dstStart >> 4) + i + 1];
			tmpVertexPropBuffer10[i] = 	prop1_uint16.s0;
			tmpVertexPropBuffer11[i] = 	prop1_uint16.s1;
			tmpVertexPropBuffer12[i] = 	prop1_uint16.s2;
			tmpVertexPropBuffer13[i] = 	prop1_uint16.s3;
			tmpVertexPropBuffer14[i] = 	prop1_uint16.s4;
			tmpVertexPropBuffer15[i] = 	prop1_uint16.s5;
			tmpVertexPropBuffer16[i] = 	prop1_uint16.s6;
			tmpVertexPropBuffer17[i] = 	prop1_uint16.s7;
			tmpVertexPropBuffer18[i] = 	prop1_uint16.s8;
			tmpVertexPropBuffer19[i] = 	prop1_uint16.s9;
			tmpVertexPropBuffer1a[i] = 	prop1_uint16.sa;
			tmpVertexPropBuffer1b[i] = 	prop1_uint16.sb;
			tmpVertexPropBuffer1c[i] = 	prop1_uint16.sc;
			tmpVertexPropBuffer1d[i] = 	prop1_uint16.sd;
			tmpVertexPropBuffer1e[i] = 	prop1_uint16.se;
			tmpVertexPropBuffer1f[i] = 	prop1_uint16.sf;

			float16 prop2_uint16 = tmpVertexProp[(dstStart >> 4) + i + 2];
			tmpVertexPropBuffer20[i] = 	prop2_uint16.s0;
			tmpVertexPropBuffer21[i] = 	prop2_uint16.s1;
			tmpVertexPropBuffer22[i] = 	prop2_uint16.s2;
			tmpVertexPropBuffer23[i] = 	prop2_uint16.s3;
			tmpVertexPropBuffer24[i] = 	prop2_uint16.s4;
			tmpVertexPropBuffer25[i] = 	prop2_uint16.s5;
			tmpVertexPropBuffer26[i] = 	prop2_uint16.s6;
			tmpVertexPropBuffer27[i] = 	prop2_uint16.s7;
			tmpVertexPropBuffer28[i] = 	prop2_uint16.s8;
			tmpVertexPropBuffer29[i] = 	prop2_uint16.s9;
			tmpVertexPropBuffer2a[i] = 	prop2_uint16.sa;
			tmpVertexPropBuffer2b[i] = 	prop2_uint16.sb;
			tmpVertexPropBuffer2c[i] = 	prop2_uint16.sc;
			tmpVertexPropBuffer2d[i] = 	prop2_uint16.sd;
			tmpVertexPropBuffer2e[i] = 	prop2_uint16.se;
			tmpVertexPropBuffer2f[i] = 	prop2_uint16.sf;

			float16 prop3_uint16 = tmpVertexProp[(dstStart >> 4) + i + 3];
			tmpVertexPropBuffer30[i] = 	prop3_uint16.s0;
			tmpVertexPropBuffer31[i] = 	prop3_uint16.s1;
			tmpVertexPropBuffer32[i] = 	prop3_uint16.s2;
			tmpVertexPropBuffer33[i] = 	prop3_uint16.s3;
			tmpVertexPropBuffer34[i] = 	prop3_uint16.s4;
			tmpVertexPropBuffer35[i] = 	prop3_uint16.s5;
			tmpVertexPropBuffer36[i] = 	prop3_uint16.s6;
			tmpVertexPropBuffer37[i] = 	prop3_uint16.s7;
			tmpVertexPropBuffer38[i] = 	prop3_uint16.s8;
			tmpVertexPropBuffer39[i] = 	prop3_uint16.s9;
			tmpVertexPropBuffer3a[i] = 	prop3_uint16.sa;
			tmpVertexPropBuffer3b[i] = 	prop3_uint16.sb;
			tmpVertexPropBuffer3c[i] = 	prop3_uint16.sc;
			tmpVertexPropBuffer3d[i] = 	prop3_uint16.sd;
			tmpVertexPropBuffer3e[i] = 	prop3_uint16.se;
			tmpVertexPropBuffer3f[i] = 	prop3_uint16.sf;
		}

	while(true){
		EDGE_INFO edgeInfo = read_channel_nb_altera(edgeInfoCh, &validData);
		if(validData){
			int srcVidx    = edgeInfo.vertexIdx;
			int16 dstVidx0 = edgeInfo.ngbVidx0;
			int16 dstVidx1 = edgeInfo.ngbVidx1;
			int16 dstVidx2 = edgeInfo.ngbVidx2;
			int16 dstVidx3 = edgeInfo.ngbVidx3;
			PROP_TYPE eProp= 0x1;
			int outDeg     = edgeInfo.outDeg;
			int count = 0;
			int srcBufIdx = srcVidx - srcStart;

			PROP_TYPE srcProp = vPropBuffer[srcBufIdx];//vertexProp[srcVidx]; // 
			//printf("srcBufIdx %d \n", srcBufIdx);
			if(outDeg){

				if(dstVidx0.s0 != -1) tmpVertexPropBuffer0[dstVidx0.s0 - dstStart] += srcProp/outDeg;
				if(dstVidx0.s1 != -1) tmpVertexPropBuffer1[dstVidx0.s1 - dstStart] += srcProp/outDeg;
				if(dstVidx0.s2 != -1) tmpVertexPropBuffer2[dstVidx0.s2 - dstStart] += srcProp/outDeg;
				if(dstVidx0.s3 != -1) tmpVertexPropBuffer3[dstVidx0.s3 - dstStart] += srcProp/outDeg;
				if(dstVidx0.s4 != -1) tmpVertexPropBuffer4[dstVidx0.s4 - dstStart] += srcProp/outDeg;
				if(dstVidx0.s5 != -1) tmpVertexPropBuffer5[dstVidx0.s5 - dstStart] += srcProp/outDeg;
				if(dstVidx0.s6 != -1) tmpVertexPropBuffer6[dstVidx0.s6 - dstStart] += srcProp/outDeg;
				if(dstVidx0.s7 != -1) tmpVertexPropBuffer7[dstVidx0.s7 - dstStart] += srcProp/outDeg;
				if(dstVidx0.s8 != -1) tmpVertexPropBuffer8[dstVidx0.s8 - dstStart] += srcProp/outDeg;
				if(dstVidx0.s9 != -1) tmpVertexPropBuffer9[dstVidx0.s9 - dstStart] += srcProp/outDeg;
				if(dstVidx0.sa != -1) tmpVertexPropBuffera[dstVidx0.sa - dstStart] += srcProp/outDeg;
				if(dstVidx0.sb != -1) tmpVertexPropBufferb[dstVidx0.sb - dstStart] += srcProp/outDeg;
				if(dstVidx0.sc != -1) tmpVertexPropBufferc[dstVidx0.sc - dstStart] += srcProp/outDeg;
				if(dstVidx0.sd != -1) tmpVertexPropBufferd[dstVidx0.sd - dstStart] += srcProp/outDeg;
				if(dstVidx0.se != -1) tmpVertexPropBuffere[dstVidx0.se - dstStart] += srcProp/outDeg;
				if(dstVidx0.sf != -1) tmpVertexPropBufferf[dstVidx0.sf - dstStart] += srcProp/outDeg;
				if(dstVidx1.s0 != -1) tmpVertexPropBuffer10[dstVidx1.s0- dstStart] += srcProp/outDeg;
				if(dstVidx1.s1 != -1) tmpVertexPropBuffer11[dstVidx1.s1- dstStart] += srcProp/outDeg;
				if(dstVidx1.s2 != -1) tmpVertexPropBuffer12[dstVidx1.s2- dstStart] += srcProp/outDeg;
				if(dstVidx1.s3 != -1) tmpVertexPropBuffer13[dstVidx1.s3- dstStart] += srcProp/outDeg;
				if(dstVidx1.s4 != -1) tmpVertexPropBuffer14[dstVidx1.s4- dstStart] += srcProp/outDeg;
				if(dstVidx1.s5 != -1) tmpVertexPropBuffer15[dstVidx1.s5- dstStart] += srcProp/outDeg;
				if(dstVidx1.s6 != -1) tmpVertexPropBuffer16[dstVidx1.s6- dstStart] += srcProp/outDeg;
				if(dstVidx1.s7 != -1) tmpVertexPropBuffer17[dstVidx1.s7- dstStart] += srcProp/outDeg;
				if(dstVidx1.s8 != -1) tmpVertexPropBuffer18[dstVidx1.s8- dstStart] += srcProp/outDeg;
				if(dstVidx1.s9 != -1) tmpVertexPropBuffer19[dstVidx1.s9- dstStart] += srcProp/outDeg;
				if(dstVidx1.sa != -1) tmpVertexPropBuffer1a[dstVidx1.sa- dstStart] += srcProp/outDeg;
				if(dstVidx1.sb != -1) tmpVertexPropBuffer1b[dstVidx1.sb- dstStart] += srcProp/outDeg;
				if(dstVidx1.sc != -1) tmpVertexPropBuffer1c[dstVidx1.sc- dstStart] += srcProp/outDeg;
				if(dstVidx1.sd != -1) tmpVertexPropBuffer1d[dstVidx1.sd- dstStart] += srcProp/outDeg;
				if(dstVidx1.se != -1) tmpVertexPropBuffer1e[dstVidx1.se- dstStart] += srcProp/outDeg;
				if(dstVidx1.sf != -1) tmpVertexPropBuffer1f[dstVidx1.sf- dstStart] += srcProp/outDeg;
				if(dstVidx2.s0 != -1) tmpVertexPropBuffer20[dstVidx2.s0- dstStart] += srcProp/outDeg;
				if(dstVidx2.s1 != -1) tmpVertexPropBuffer21[dstVidx2.s1- dstStart] += srcProp/outDeg;
				if(dstVidx2.s2 != -1) tmpVertexPropBuffer22[dstVidx2.s2- dstStart] += srcProp/outDeg;
				if(dstVidx2.s3 != -1) tmpVertexPropBuffer23[dstVidx2.s3- dstStart] += srcProp/outDeg;
				if(dstVidx2.s4 != -1) tmpVertexPropBuffer24[dstVidx2.s4- dstStart] += srcProp/outDeg;
				if(dstVidx2.s5 != -1) tmpVertexPropBuffer25[dstVidx2.s5- dstStart] += srcProp/outDeg;
				if(dstVidx2.s6 != -1) tmpVertexPropBuffer26[dstVidx2.s6- dstStart] += srcProp/outDeg;
				if(dstVidx2.s7 != -1) tmpVertexPropBuffer27[dstVidx2.s7- dstStart] += srcProp/outDeg;
				if(dstVidx2.s8 != -1) tmpVertexPropBuffer28[dstVidx2.s8- dstStart] += srcProp/outDeg;
				if(dstVidx2.s9 != -1) tmpVertexPropBuffer29[dstVidx2.s9- dstStart] += srcProp/outDeg;
				if(dstVidx2.sa != -1) tmpVertexPropBuffer2a[dstVidx2.sa- dstStart] += srcProp/outDeg;
				if(dstVidx2.sb != -1) tmpVertexPropBuffer2b[dstVidx2.sb- dstStart] += srcProp/outDeg;
				if(dstVidx2.sc != -1) tmpVertexPropBuffer2c[dstVidx2.sc- dstStart] += srcProp/outDeg;
				if(dstVidx2.sd != -1) tmpVertexPropBuffer2d[dstVidx2.sd- dstStart] += srcProp/outDeg;
				if(dstVidx2.se != -1) tmpVertexPropBuffer2e[dstVidx2.se- dstStart] += srcProp/outDeg;
				if(dstVidx2.sf != -1) tmpVertexPropBuffer2f[dstVidx2.sf- dstStart] += srcProp/outDeg;
				if(dstVidx3.s0 != -1) tmpVertexPropBuffer30[dstVidx3.s0- dstStart] += srcProp/outDeg;
				if(dstVidx3.s1 != -1) tmpVertexPropBuffer31[dstVidx3.s1- dstStart] += srcProp/outDeg;
				if(dstVidx3.s2 != -1) tmpVertexPropBuffer32[dstVidx3.s2- dstStart] += srcProp/outDeg;
				if(dstVidx3.s3 != -1) tmpVertexPropBuffer33[dstVidx3.s3- dstStart] += srcProp/outDeg;
				if(dstVidx3.s4 != -1) tmpVertexPropBuffer34[dstVidx3.s4- dstStart] += srcProp/outDeg;
				if(dstVidx3.s5 != -1) tmpVertexPropBuffer35[dstVidx3.s5- dstStart] += srcProp/outDeg;
				if(dstVidx3.s6 != -1) tmpVertexPropBuffer36[dstVidx3.s6- dstStart] += srcProp/outDeg;
				if(dstVidx3.s7 != -1) tmpVertexPropBuffer37[dstVidx3.s7- dstStart] += srcProp/outDeg;
				if(dstVidx3.s8 != -1) tmpVertexPropBuffer38[dstVidx3.s8- dstStart] += srcProp/outDeg;
				if(dstVidx3.s9 != -1) tmpVertexPropBuffer39[dstVidx3.s9- dstStart] += srcProp/outDeg;
				if(dstVidx3.sa != -1) tmpVertexPropBuffer3a[dstVidx3.sa- dstStart] += srcProp/outDeg;
				if(dstVidx3.sb != -1) tmpVertexPropBuffer3b[dstVidx3.sb- dstStart] += srcProp/outDeg;
				if(dstVidx3.sc != -1) tmpVertexPropBuffer3c[dstVidx3.sc- dstStart] += srcProp/outDeg;
				if(dstVidx3.sd != -1) tmpVertexPropBuffer3d[dstVidx3.sd- dstStart] += srcProp/outDeg;
				if(dstVidx3.se != -1) tmpVertexPropBuffer3e[dstVidx3.se- dstStart] += srcProp/outDeg;
				if(dstVidx3.sf != -1) tmpVertexPropBuffer3f[dstVidx3.sf- dstStart] += srcProp/outDeg;
			}

		}				
		int tmpEndFlag = read_channel_nb_altera(edgeInfoChEof, &validFlag);
		if(validFlag) endFlag = tmpEndFlag;	
		if(endFlag == EOF_FLAG && !validData && !validFlag) break;
	}
				for(int i = 0; i < (dstNum >> 4); i += 4){	
			float16 prop_uint16;
			prop_uint16.s0 = tmpVertexPropBuffer0[i]; 
			prop_uint16.s1 = tmpVertexPropBuffer1[i]; 
			prop_uint16.s2 = tmpVertexPropBuffer2[i]; 
			prop_uint16.s3 = tmpVertexPropBuffer3[i]; 
			prop_uint16.s4 = tmpVertexPropBuffer4[i]; 
			prop_uint16.s5 = tmpVertexPropBuffer5[i]; 
			prop_uint16.s6 = tmpVertexPropBuffer6[i]; 
			prop_uint16.s7 = tmpVertexPropBuffer7[i]; 
			prop_uint16.s8 = tmpVertexPropBuffer8[i]; 
			prop_uint16.s9 = tmpVertexPropBuffer9[i]; 
			prop_uint16.sa = tmpVertexPropBuffera[i]; 
			prop_uint16.sb = tmpVertexPropBufferb[i]; 
			prop_uint16.sc = tmpVertexPropBufferc[i]; 
			prop_uint16.sd = tmpVertexPropBufferd[i]; 
			prop_uint16.se = tmpVertexPropBuffere[i]; 
			prop_uint16.sf = tmpVertexPropBufferf[i]; 
			

			float16 prop1_uint16;
			prop1_uint16.s0 = tmpVertexPropBuffer10[i]; 
			prop1_uint16.s1 = tmpVertexPropBuffer11[i]; 
			prop1_uint16.s2 = tmpVertexPropBuffer12[i]; 
			prop1_uint16.s3 = tmpVertexPropBuffer13[i]; 
			prop1_uint16.s4 = tmpVertexPropBuffer14[i]; 
			prop1_uint16.s5 = tmpVertexPropBuffer15[i]; 
			prop1_uint16.s6 = tmpVertexPropBuffer16[i]; 
			prop1_uint16.s7 = tmpVertexPropBuffer17[i]; 
			prop1_uint16.s8 = tmpVertexPropBuffer18[i]; 
			prop1_uint16.s9 = tmpVertexPropBuffer19[i]; 
			prop1_uint16.sa = tmpVertexPropBuffer1a[i]; 
			prop1_uint16.sb = tmpVertexPropBuffer1b[i]; 
			prop1_uint16.sc = tmpVertexPropBuffer1c[i]; 
			prop1_uint16.sd = tmpVertexPropBuffer1d[i]; 
			prop1_uint16.se = tmpVertexPropBuffer1e[i]; 
			prop1_uint16.sf = tmpVertexPropBuffer1f[i]; 

			float16 prop2_uint16;
			prop2_uint16.s0 = tmpVertexPropBuffer20[i]; 
			prop2_uint16.s1 = tmpVertexPropBuffer21[i]; 
			prop2_uint16.s2 = tmpVertexPropBuffer22[i]; 
			prop2_uint16.s3 = tmpVertexPropBuffer23[i]; 
			prop2_uint16.s4 = tmpVertexPropBuffer24[i]; 
			prop2_uint16.s5 = tmpVertexPropBuffer25[i]; 
			prop2_uint16.s6 = tmpVertexPropBuffer26[i]; 
			prop2_uint16.s7 = tmpVertexPropBuffer27[i]; 
			prop2_uint16.s8 = tmpVertexPropBuffer28[i]; 
			prop2_uint16.s9 = tmpVertexPropBuffer29[i]; 
			prop2_uint16.sa = tmpVertexPropBuffer2a[i]; 
			prop2_uint16.sb = tmpVertexPropBuffer2b[i]; 
			prop2_uint16.sc = tmpVertexPropBuffer2c[i]; 
			prop2_uint16.sd = tmpVertexPropBuffer2d[i]; 
			prop2_uint16.se = tmpVertexPropBuffer2e[i]; 
			prop2_uint16.sf = tmpVertexPropBuffer2f[i]; 
			float16 prop3_uint16;
			prop3_uint16.s0 = tmpVertexPropBuffer30[i]; 
			prop3_uint16.s1 = tmpVertexPropBuffer31[i]; 
			prop3_uint16.s2 = tmpVertexPropBuffer32[i]; 
			prop3_uint16.s3 = tmpVertexPropBuffer33[i]; 
			prop3_uint16.s4 = tmpVertexPropBuffer34[i]; 
			prop3_uint16.s5 = tmpVertexPropBuffer35[i]; 
			prop3_uint16.s6 = tmpVertexPropBuffer36[i]; 
			prop3_uint16.s7 = tmpVertexPropBuffer37[i]; 
			prop3_uint16.s8 = tmpVertexPropBuffer38[i]; 
			prop3_uint16.s9 = tmpVertexPropBuffer39[i]; 
			prop3_uint16.sa = tmpVertexPropBuffer3a[i]; 
			prop3_uint16.sb = tmpVertexPropBuffer3b[i]; 
			prop3_uint16.sc = tmpVertexPropBuffer3c[i]; 
			prop3_uint16.sd = tmpVertexPropBuffer3d[i]; 
			prop3_uint16.se = tmpVertexPropBuffer3e[i]; 
			prop3_uint16.sf = tmpVertexPropBuffer3f[i]; 

			tmpVertexProp[(dstStart >> 4) + i] = prop_uint16;
			tmpVertexProp[(dstStart >> 4) + i + 1] = prop1_uint16;
			tmpVertexProp[(dstStart >> 4) + i + 2] = prop2_uint16;
			tmpVertexProp[(dstStart >> 4) + i + 3] = prop3_uint16;
			//tmpVertexProp[i + dstStart] = tmpVertexPropBuffer[i];
		}	

			/*
			if(outDeg) {
				
					compute(srcVidx, dstVidx0.s0 - dstStart, outDeg, tmpVertexPropBuffer0, srcProp);
					compute(srcVidx, dstVidx0.s1 - dstStart, outDeg, tmpVertexPropBuffer1, srcProp);
					compute(srcVidx, dstVidx0.s2 - dstStart, outDeg, tmpVertexPropBuffer2, srcProp);
					compute(srcVidx, dstVidx0.s3 - dstStart, outDeg, tmpVertexPropBuffer3, srcProp);
					compute(srcVidx, dstVidx0.s4 - dstStart, outDeg, tmpVertexPropBuffer4, srcProp);
					compute(srcVidx, dstVidx0.s5 - dstStart, outDeg, tmpVertexPropBuffer5, srcProp);
					compute(srcVidx, dstVidx0.s6 - dstStart, outDeg, tmpVertexPropBuffer6, srcProp);
					compute(srcVidx, dstVidx0.s7 - dstStart, outDeg, tmpVertexPropBuffer7, srcProp);
					compute(srcVidx, dstVidx0.s8 - dstStart, outDeg, tmpVertexPropBuffer8, srcProp);
					compute(srcVidx, dstVidx0.s9 - dstStart, outDeg, tmpVertexPropBuffer9, srcProp);
					compute(srcVidx, dstVidx0.sa - dstStart, outDeg, tmpVertexPropBuffera, srcProp);
					compute(srcVidx, dstVidx0.sb - dstStart, outDeg, tmpVertexPropBufferb, srcProp);
					compute(srcVidx, dstVidx0.sc - dstStart, outDeg, tmpVertexPropBufferc, srcProp);
					compute(srcVidx, dstVidx0.sd - dstStart, outDeg, tmpVertexPropBufferd, srcProp);
					compute(srcVidx, dstVidx0.se - dstStart, outDeg, tmpVertexPropBuffere, srcProp);
					compute(srcVidx, dstVidx0.sf - dstStart, outDeg, tmpVertexPropBufferf, srcProp);


					compute(srcVidx, dstVidx1.s0 - dstStart, outDeg, tmpVertexPropBuffer10, srcProp);
					compute(srcVidx, dstVidx1.s1 - dstStart, outDeg, tmpVertexPropBuffer11, srcProp);
					compute(srcVidx, dstVidx1.s2 - dstStart, outDeg, tmpVertexPropBuffer12, srcProp);
					compute(srcVidx, dstVidx1.s3 - dstStart, outDeg, tmpVertexPropBuffer13, srcProp);
					compute(srcVidx, dstVidx1.s4 - dstStart, outDeg, tmpVertexPropBuffer14, srcProp);
					compute(srcVidx, dstVidx1.s5 - dstStart, outDeg, tmpVertexPropBuffer15, srcProp);
					compute(srcVidx, dstVidx1.s6 - dstStart, outDeg, tmpVertexPropBuffer16, srcProp);
					compute(srcVidx, dstVidx1.s7 - dstStart, outDeg, tmpVertexPropBuffer17, srcProp);
					compute(srcVidx, dstVidx1.s8 - dstStart, outDeg, tmpVertexPropBuffer18, srcProp);
					compute(srcVidx, dstVidx1.s9 - dstStart, outDeg, tmpVertexPropBuffer19, srcProp);
					compute(srcVidx, dstVidx1.sa - dstStart, outDeg, tmpVertexPropBuffer1a, srcProp);
					compute(srcVidx, dstVidx1.sb - dstStart, outDeg, tmpVertexPropBuffer1b, srcProp);
					compute(srcVidx, dstVidx1.sc - dstStart, outDeg, tmpVertexPropBuffer1c, srcProp);
					compute(srcVidx, dstVidx1.sd - dstStart, outDeg, tmpVertexPropBuffer1d, srcProp);
					compute(srcVidx, dstVidx1.se - dstStart, outDeg, tmpVertexPropBuffer1e, srcProp);
					compute(srcVidx, dstVidx1.sf - dstStart, outDeg, tmpVertexPropBuffer1f, srcProp);

					compute(srcVidx, dstVidx2.s0 - dstStart, outDeg, tmpVertexPropBuffer20, srcProp);
					compute(srcVidx, dstVidx2.s1 - dstStart, outDeg, tmpVertexPropBuffer21, srcProp);
					compute(srcVidx, dstVidx2.s2 - dstStart, outDeg, tmpVertexPropBuffer22, srcProp);
					compute(srcVidx, dstVidx2.s3 - dstStart, outDeg, tmpVertexPropBuffer23, srcProp);
					compute(srcVidx, dstVidx2.s4 - dstStart, outDeg, tmpVertexPropBuffer24, srcProp);
					compute(srcVidx, dstVidx2.s5 - dstStart, outDeg, tmpVertexPropBuffer25, srcProp);
					compute(srcVidx, dstVidx2.s6 - dstStart, outDeg, tmpVertexPropBuffer26, srcProp);
					compute(srcVidx, dstVidx2.s7 - dstStart, outDeg, tmpVertexPropBuffer27, srcProp);
					compute(srcVidx, dstVidx2.s8 - dstStart, outDeg, tmpVertexPropBuffer28, srcProp);
					compute(srcVidx, dstVidx2.s9 - dstStart, outDeg, tmpVertexPropBuffer29, srcProp);
					compute(srcVidx, dstVidx2.sa - dstStart, outDeg, tmpVertexPropBuffer2a, srcProp);
					compute(srcVidx, dstVidx2.sb - dstStart, outDeg, tmpVertexPropBuffer2b, srcProp);
					compute(srcVidx, dstVidx2.sc - dstStart, outDeg, tmpVertexPropBuffer2c, srcProp);
					compute(srcVidx, dstVidx2.sd - dstStart, outDeg, tmpVertexPropBuffer2d, srcProp);
					compute(srcVidx, dstVidx2.se - dstStart, outDeg, tmpVertexPropBuffer2e, srcProp);
					compute(srcVidx, dstVidx2.sf - dstStart, outDeg, tmpVertexPropBuffer2f, srcProp);


					compute(srcVidx, dstVidx3.s0 - dstStart, outDeg, tmpVertexPropBuffer30, srcProp);
					compute(srcVidx, dstVidx3.s1 - dstStart, outDeg, tmpVertexPropBuffer31, srcProp);
					compute(srcVidx, dstVidx3.s2 - dstStart, outDeg, tmpVertexPropBuffer32, srcProp);
					compute(srcVidx, dstVidx3.s3 - dstStart, outDeg, tmpVertexPropBuffer33, srcProp);
					compute(srcVidx, dstVidx3.s4 - dstStart, outDeg, tmpVertexPropBuffer34, srcProp);
					compute(srcVidx, dstVidx3.s5 - dstStart, outDeg, tmpVertexPropBuffer35, srcProp);
					compute(srcVidx, dstVidx3.s6 - dstStart, outDeg, tmpVertexPropBuffer36, srcProp);
					compute(srcVidx, dstVidx3.s7 - dstStart, outDeg, tmpVertexPropBuffer37, srcProp);
					compute(srcVidx, dstVidx3.s8 - dstStart, outDeg, tmpVertexPropBuffer38, srcProp);
					compute(srcVidx, dstVidx3.s9 - dstStart, outDeg, tmpVertexPropBuffer39, srcProp);
					compute(srcVidx, dstVidx3.sa - dstStart, outDeg, tmpVertexPropBuffer3a, srcProp);
					compute(srcVidx, dstVidx3.sb - dstStart, outDeg, tmpVertexPropBuffer3b, srcProp);
					compute(srcVidx, dstVidx3.sc - dstStart, outDeg, tmpVertexPropBuffer3c, srcProp);
					compute(srcVidx, dstVidx3.sd - dstStart, outDeg, tmpVertexPropBuffer3d, srcProp);
					compute(srcVidx, dstVidx3.se - dstStart, outDeg, tmpVertexPropBuffer3e, srcProp);
					compute(srcVidx, dstVidx3.sf - dstStart, outDeg, tmpVertexPropBuffer3f, srcProp);
			}
		}
		int tmpEndFlag = read_channel_nb_altera(edgeInfoChEof, &validFlag);
		if(validFlag) endFlag = tmpEndFlag;	
		if(endFlag == EOF_FLAG && !validData && !validFlag) break;
	}
		//for(int i = 0; i < dstNum; i++){
			//tmpVertexProp[i + dstStart] = tmpVertexPropBuffer[i];
		//}
		for(int i = 0; i < (dstNum >> 4); i += 4){	
			float16 prop_uint16;
			prop_uint16.s0 = tmpVertexPropBuffer0[i]; 
			prop_uint16.s1 = tmpVertexPropBuffer1[i]; 
			prop_uint16.s2 = tmpVertexPropBuffer2[i]; 
			prop_uint16.s3 = tmpVertexPropBuffer3[i]; 
			prop_uint16.s4 = tmpVertexPropBuffer4[i]; 
			prop_uint16.s5 = tmpVertexPropBuffer5[i]; 
			prop_uint16.s6 = tmpVertexPropBuffer6[i]; 
			prop_uint16.s7 = tmpVertexPropBuffer7[i]; 
			prop_uint16.s8 = tmpVertexPropBuffer8[i]; 
			prop_uint16.s9 = tmpVertexPropBuffer9[i]; 
			prop_uint16.sa = tmpVertexPropBuffera[i]; 
			prop_uint16.sb = tmpVertexPropBufferb[i]; 
			prop_uint16.sc = tmpVertexPropBufferc[i]; 
			prop_uint16.sd = tmpVertexPropBufferd[i]; 
			prop_uint16.se = tmpVertexPropBuffere[i]; 
			prop_uint16.sf = tmpVertexPropBufferf[i]; 
			

			float16 prop1_uint16;
			prop1_uint16.s0 = tmpVertexPropBuffer10[i]; 
			prop1_uint16.s1 = tmpVertexPropBuffer11[i]; 
			prop1_uint16.s2 = tmpVertexPropBuffer12[i]; 
			prop1_uint16.s3 = tmpVertexPropBuffer13[i]; 
			prop1_uint16.s4 = tmpVertexPropBuffer14[i]; 
			prop1_uint16.s5 = tmpVertexPropBuffer15[i]; 
			prop1_uint16.s6 = tmpVertexPropBuffer16[i]; 
			prop1_uint16.s7 = tmpVertexPropBuffer17[i]; 
			prop1_uint16.s8 = tmpVertexPropBuffer18[i]; 
			prop1_uint16.s9 = tmpVertexPropBuffer19[i]; 
			prop1_uint16.sa = tmpVertexPropBuffer1a[i]; 
			prop1_uint16.sb = tmpVertexPropBuffer1b[i]; 
			prop1_uint16.sc = tmpVertexPropBuffer1c[i]; 
			prop1_uint16.sd = tmpVertexPropBuffer1d[i]; 
			prop1_uint16.se = tmpVertexPropBuffer1e[i]; 
			prop1_uint16.sf = tmpVertexPropBuffer1f[i]; 

			float16 prop2_uint16;
			prop2_uint16.s0 = tmpVertexPropBuffer20[i]; 
			prop2_uint16.s1 = tmpVertexPropBuffer21[i]; 
			prop2_uint16.s2 = tmpVertexPropBuffer22[i]; 
			prop2_uint16.s3 = tmpVertexPropBuffer23[i]; 
			prop2_uint16.s4 = tmpVertexPropBuffer24[i]; 
			prop2_uint16.s5 = tmpVertexPropBuffer25[i]; 
			prop2_uint16.s6 = tmpVertexPropBuffer26[i]; 
			prop2_uint16.s7 = tmpVertexPropBuffer27[i]; 
			prop2_uint16.s8 = tmpVertexPropBuffer28[i]; 
			prop2_uint16.s9 = tmpVertexPropBuffer29[i]; 
			prop2_uint16.sa = tmpVertexPropBuffer2a[i]; 
			prop2_uint16.sb = tmpVertexPropBuffer2b[i]; 
			prop2_uint16.sc = tmpVertexPropBuffer2c[i]; 
			prop2_uint16.sd = tmpVertexPropBuffer2d[i]; 
			prop2_uint16.se = tmpVertexPropBuffer2e[i]; 
			prop2_uint16.sf = tmpVertexPropBuffer2f[i]; 
			float16 prop3_uint16;
			prop3_uint16.s0 = tmpVertexPropBuffer30[i]; 
			prop3_uint16.s1 = tmpVertexPropBuffer31[i]; 
			prop3_uint16.s2 = tmpVertexPropBuffer32[i]; 
			prop3_uint16.s3 = tmpVertexPropBuffer33[i]; 
			prop3_uint16.s4 = tmpVertexPropBuffer34[i]; 
			prop3_uint16.s5 = tmpVertexPropBuffer35[i]; 
			prop3_uint16.s6 = tmpVertexPropBuffer36[i]; 
			prop3_uint16.s7 = tmpVertexPropBuffer37[i]; 
			prop3_uint16.s8 = tmpVertexPropBuffer38[i]; 
			prop3_uint16.s9 = tmpVertexPropBuffer39[i]; 
			prop3_uint16.sa = tmpVertexPropBuffer3a[i]; 
			prop3_uint16.sb = tmpVertexPropBuffer3b[i]; 
			prop3_uint16.sc = tmpVertexPropBuffer3c[i]; 
			prop3_uint16.sd = tmpVertexPropBuffer3d[i]; 
			prop3_uint16.se = tmpVertexPropBuffer3e[i]; 
			prop3_uint16.sf = tmpVertexPropBuffer3f[i]; 

			tmpVertexProp[(dstStart >> 4) + i] = prop_uint16;
			tmpVertexProp[(dstStart >> 4) + i + 1] = prop1_uint16;
			tmpVertexProp[(dstStart >> 4) + i + 2] = prop2_uint16;
			tmpVertexProp[(dstStart >> 4) + i + 3] = prop3_uint16;
			//tmpVertexProp[i + dstStart] = tmpVertexPropBuffer[i];
		}
		*/
}


