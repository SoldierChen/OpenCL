//now is the padding 32 case
//#define SW 1
#define BFS
#define EOF_FLAG 0xffff
#define PROP_TYPE int
#define kDamp 108//(0.85 << 7)  // * 128

#define VERTEX_MAX  (256*1024)//262144//40960//40960//(128*1024)
#define EDGE_MAX    (2*1024*1024)//5610680////163840 // (1024*1024)
#define BRAM_BANK 32
#define LOG2_BRAM_BANK 5
#define PAD_TYPE int16
#define PAD_WITH 64

#define INT2FLOAT (pow(2,28))
int float2int(float a){
	return (int)(a * INT2FLOAT);
}

float int2float(int a){
	return ((float)a / INT2FLOAT);
}

typedef struct EdgeInfo{
	int vertexIdx;
	int16 ngbVidx0;
	int16 ngbVidx1;
	//PROP_TYPE eProp;
	int outDeg;
} EDGE_INFO;

channel int activeVertexCh    __attribute__((depth(1024)));
channel EDGE_INFO edgeInfoCh  __attribute__((depth(1024)));
channel int edgeInfoChEof     __attribute__((depth(4)));
channel int nextActiveVertexEof  __attribute__((depth(4)));
channel int nextActiveVertexCh[16]    __attribute__((depth(256)));

__attribute__((always_inline)) void compute(int srcVidx, int dstVidx, int dstStart, int srcVprop,
	PROP_TYPE* tmpVPropBuffer){
	#ifdef PR 
			if(dstVidx != -1) {
				int idx = (dstVidx - dstStart) >> LOG2_BRAM_BANK;
				PROP_TYPE operand1 = tmpVPropBuffer[idx];
			    tmpVPropBuffer[idx] = (outDeg) + operand1;
			}
	#endif
	#ifdef BFS
			if(dstVidx != -1) {
				int idx = (dstVidx - dstStart) >> LOG2_BRAM_BANK;
				int dstVprop = tmpVPropBuffer[idx];
				//printf("dstVprop %d srcVprop %d, tmpVPropBuffer %d\n", dstVprop, srcVprop, tmpVPropBuffer[idx]);
				tmpVPropBuffer[idx] = (dstVprop > (srcVprop + 1))? (srcVprop + 1) : dstVprop;
			}
	#endif
	#ifdef SSSP
			tmpVPropBuffer[dstVidx] = (dstVprop > srcVprop + eProp)? (srcVprop + eProp) : dstVprop;
	#endif
}

__attribute__((always_inline)) void apply(int srcVidx, int dstVidx, int dstStart, int outDeg,
	PROP_TYPE* tmpVPropBuffer){
	#ifdef PR 
			if(dstVidx != -1) {
				int idx = (dstVidx - dstStart) >> LOG2_BRAM_BANK;
				PROP_TYPE operand1 = tmpVPropBuffer[idx];
			    tmpVPropBuffer[idx] = (outDeg) + operand1;
			}
	#endif
	#ifdef BFS
			//tmpVPropBuffer[dstVidx] = (dstVprop > srcVprop + 1)? (srcVprop + 1) : dstVprop;
	#endif
	#ifdef SSSP
			tmpVPropBuffer[dstVidx] = (dstVprop > srcVprop + eProp)? (srcVprop + eProp) : dstVprop;
	#endif
}

void print_int16(char* name,int16 x){
	printf("%s \n", name);
	printf("%d \t", x.s0);
	printf("%d \t", x.s1);
	printf("%d \t", x.s2);
	printf("%d \t", x.s3);
	printf("%d \t", x.s4);
	printf("%d \t", x.s5);
	printf("%d \t", x.s6);
	printf("%d \t", x.s7);
	printf("%d \t", x.s8);
	printf("%d \t", x.s9);
	printf("%d \t", x.sa);
	printf("%d \t", x.sb);
	printf("%d \t", x.sc);
	printf("%d \t", x.sd);
	printf("%d \t", x.se);
	printf("%d \t", x.sf);
	printf("\n");
}

__kernel void __attribute__((task)) readActiveVertices(
		__global const int* restrict blkActiveVertices, 
		__global const int* restrict blkActiveVertexNum,
		__global const int* restrict iterNum
		)
{	

	int iteration = iterNum[0];
	int baseAddr = iteration << 18;
	for(int i = 0; i < blkActiveVertexNum[iteration]; i++){
		int vertexIdx = blkActiveVertices[i + baseAddr];
		write_channel_altera(activeVertexCh, vertexIdx);
	}
	//printf("row %d blkActiveVertexNum[%d] %d baseAddr %d , \n",iterNum[0], iterNum[0], blkActiveVertexNum[iteration], baseAddr);
}

__kernel void __attribute__((task)) readNgbInfo(
		__global int16*  restrict blkRpa,
		__global int*  restrict blkRpaLast,
		__global PAD_TYPE*  restrict blkCia,
		//__global PAD_TYPE* restrict blkEdgeProp,
		__global int16*  restrict outDeg,
		__global int*  restrict blkActiveVertexNum,
		//__global int*  restrict blkVertexNum,
		//__global int*  restrict blkEdgeNum,
		__global int*  restrict srcRange,
		__global int*  restrict iterNum
		)
{	
	//#if SW == 1
	//int (*rpaStartBuffer)[16]=(int(*)[16])malloc(sizeof(int)* VERTEX_MAX /16 *16);
	//int (*rpaNumBuffer)[16]=(int(*)[16])malloc(sizeof(int)* VERTEX_MAX /16 *16);
	//int (*outDegBuffer)[16]=(int(*)[16])malloc(sizeof(int)* VERTEX_MAX /16 * 16);
	//#else
	int rpaStartBuffer[VERTEX_MAX >> 4][16];
	int outDegBuffer[VERTEX_MAX >> 4][16];
	//#endif
	int iteration = iterNum[0];
	int baseAddr = iteration << 14; // <<17 >>4
	int srcStart = srcRange[iteration << 1];
	
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

	int lastRpa = blkRpaLast[iteration];

	for(int i = 0; i < blkActiveVertexNum[iteration]; i++){
		int vertexIdx = read_channel_altera(activeVertexCh);
		int bufIdx = vertexIdx - srcStart;
		int start = rpaStartBuffer[bufIdx >> 4][bufIdx & 0xf];
		int end = 0;
		
		if(bufIdx == VERTEX_MAX) {
			end = lastRpa;
		}
		else
			end = rpaStartBuffer[(bufIdx + 1) >> 4][(bufIdx + 1) & 0xf];

		int num = end - start;
		int deg = outDegBuffer[bufIdx >> 4][bufIdx & 0xf];

		for(int j = (start >> 5); j < (end >> 5); j++){
			int16 ngbVidx0 = blkCia[(j << 1)];
			int16 ngbVidx1 = blkCia[(j << 1) + 1];
			EDGE_INFO edgeInfo;
			edgeInfo.vertexIdx = vertexIdx;
			edgeInfo.ngbVidx0 = ngbVidx0;
			edgeInfo.ngbVidx1 = ngbVidx1;
			edgeInfo.outDeg = deg;
			write_channel_altera(edgeInfoCh, edgeInfo);
		}
	}
	write_channel_altera(edgeInfoChEof, EOF_FLAG);
}

__kernel void __attribute__((task)) processEdge(
		//__global int16* restrict vertexProp,
		__global int16* restrict tmpVertexProp,
		//__global const int* restrict eop,
		__global const int* restrict iterNum,
		__global const int* restrict srcRange,
		__global const int* restrict sinkRange
		)
{	
	#if SW == 1

		PROP_TYPE * tmpVPropBuffer0 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer2 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer3 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer4 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer5 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer6 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer7 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer8 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer9 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffera 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBufferb 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBufferc 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBufferd 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffere 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBufferf 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);

		PROP_TYPE * tmpVPropBuffer10 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer11 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer12 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer13 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer14 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer15 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer16 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer17 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer18 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer19 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1a 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1b 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1c 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1d 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1e 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVPropBuffer1f 	= (int *)malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);

		PROP_TYPE * vPropBuffer0 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBuffer1 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBuffer2 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBuffer3 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBuffer4 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBuffer5 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBuffer6 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBuffer7 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBuffer8 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBuffer9 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBuffera 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBufferb 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBufferc 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBufferd 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBuffere 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);
		PROP_TYPE * vPropBufferf 	= (int *) malloc (sizeof(PROP_TYPE) * VERTEX_MAX/16);


	#else 

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

	//printf("dstStart %d, srcEnd %d, num %d \n", dstStart, dstEnd, dstNum);
	
	for(int i = 0; i < (dstNum >> 5); i ++){
		int16 prop_uint16 = tmpVertexProp[(dstStart >> 4) + (i << 1)];
		tmpVPropBuffer0[i]= prop_uint16.s0;
		tmpVPropBuffer1[i]= prop_uint16.s1;
		tmpVPropBuffer2[i]= prop_uint16.s2;
		tmpVPropBuffer3[i]= prop_uint16.s3;
		tmpVPropBuffer4[i]= prop_uint16.s4;
		tmpVPropBuffer5[i]= prop_uint16.s5;
		tmpVPropBuffer6[i]= prop_uint16.s6; 
		tmpVPropBuffer7[i]= prop_uint16.s7;
		tmpVPropBuffer8[i]= prop_uint16.s8;
		tmpVPropBuffer9[i]= prop_uint16.s9;
		tmpVPropBuffera[i] = prop_uint16.sa;
		tmpVPropBufferb[i] = prop_uint16.sb;
		tmpVPropBufferc[i] = prop_uint16.sc;
		tmpVPropBufferd[i] = prop_uint16.sd;
		tmpVPropBuffere[i] = prop_uint16.se;
		tmpVPropBufferf[i] = prop_uint16.sf;
		int16 prop1_uint16 = tmpVertexProp[(dstStart >> 4) + (i << 1) + 1];
		tmpVPropBuffer10[i]= prop1_uint16.s0;
		tmpVPropBuffer11[i]= prop1_uint16.s1;
		tmpVPropBuffer12[i]= prop1_uint16.s2;
		tmpVPropBuffer13[i]= prop1_uint16.s3;
		tmpVPropBuffer14[i]= prop1_uint16.s4;
		tmpVPropBuffer15[i]= prop1_uint16.s5;
		tmpVPropBuffer16[i]= prop1_uint16.s6;
		tmpVPropBuffer17[i]= prop1_uint16.s7;
		tmpVPropBuffer18[i]= prop1_uint16.s8;
		tmpVPropBuffer19[i]= prop1_uint16.s9;
		tmpVPropBuffer1a[i]= prop1_uint16.sa;
		tmpVPropBuffer1b[i]= prop1_uint16.sb;
		tmpVPropBuffer1c[i]= prop1_uint16.sc;
		tmpVPropBuffer1d[i]= prop1_uint16.sd;
		tmpVPropBuffer1e[i]= prop1_uint16.se;
		tmpVPropBuffer1f[i]= prop1_uint16.sf;
	}

	//printf("\n idx %d \n", (268646 - dstStart) >> 5);
	//printf("sample %d \n\n", tmpVPropBuffer6[(268646 - dstStart) >> 5]);

	while(true){
		EDGE_INFO edgeInfo = read_channel_nb_altera(edgeInfoCh, &validData);
		if(validData){
			int srcVidx    = edgeInfo.vertexIdx;
			int16 dstVidx0 = edgeInfo.ngbVidx0;
			int16 dstVidx1 = edgeInfo.ngbVidx1;
			//print_int16("ngbVidx0", dstVidx0);
			//print_int16("ngbVidx1", dstVidx1);
			PROP_TYPE eProp  = 0x1;
			int outDeg = edgeInfo.outDeg;
			int count = 0;
			//int srcBufIdx = srcVidx - srcStart;
			//int dstBufIdx = dstVidx - dstStart;
			//PROP_TYPE srcProp = vertexPropBuffer[srcVidx - srcStart];
			//PROP_TYPE dstProp = tmpVertexPropBuffer[dstVidx - dstStart];
					compute(srcVidx, dstVidx0.s0, dstStart, outDeg, tmpVPropBuffer0);
					compute(srcVidx, dstVidx0.s1, dstStart, outDeg, tmpVPropBuffer1);
					compute(srcVidx, dstVidx0.s2, dstStart, outDeg, tmpVPropBuffer2);
					compute(srcVidx, dstVidx0.s3, dstStart, outDeg, tmpVPropBuffer3);
					compute(srcVidx, dstVidx0.s4, dstStart, outDeg, tmpVPropBuffer4);
					compute(srcVidx, dstVidx0.s5, dstStart, outDeg, tmpVPropBuffer5);
					compute(srcVidx, dstVidx0.s6, dstStart, outDeg, tmpVPropBuffer6);
					compute(srcVidx, dstVidx0.s7, dstStart, outDeg, tmpVPropBuffer7);
					compute(srcVidx, dstVidx0.s8, dstStart, outDeg, tmpVPropBuffer8);
					compute(srcVidx, dstVidx0.s9, dstStart, outDeg, tmpVPropBuffer9);
					compute(srcVidx, dstVidx0.sa, dstStart, outDeg, tmpVPropBuffera);
					compute(srcVidx, dstVidx0.sb, dstStart, outDeg, tmpVPropBufferb);
					compute(srcVidx, dstVidx0.sc, dstStart, outDeg, tmpVPropBufferc);
					compute(srcVidx, dstVidx0.sd, dstStart, outDeg, tmpVPropBufferd);
					compute(srcVidx, dstVidx0.se, dstStart, outDeg, tmpVPropBuffere);
					compute(srcVidx, dstVidx0.sf, dstStart, outDeg, tmpVPropBufferf);

					compute(srcVidx, dstVidx1.s0, dstStart, outDeg, tmpVPropBuffer10);
					compute(srcVidx, dstVidx1.s1, dstStart, outDeg, tmpVPropBuffer11);
					compute(srcVidx, dstVidx1.s2, dstStart, outDeg, tmpVPropBuffer12);
					compute(srcVidx, dstVidx1.s3, dstStart, outDeg, tmpVPropBuffer13);
					compute(srcVidx, dstVidx1.s4, dstStart, outDeg, tmpVPropBuffer14);
					compute(srcVidx, dstVidx1.s5, dstStart, outDeg, tmpVPropBuffer15);
					compute(srcVidx, dstVidx1.s6, dstStart, outDeg, tmpVPropBuffer16);
					compute(srcVidx, dstVidx1.s7, dstStart, outDeg, tmpVPropBuffer17);
					compute(srcVidx, dstVidx1.s8, dstStart, outDeg, tmpVPropBuffer18);
					compute(srcVidx, dstVidx1.s9, dstStart, outDeg, tmpVPropBuffer19);
					compute(srcVidx, dstVidx1.sa, dstStart, outDeg, tmpVPropBuffer1a);
					compute(srcVidx, dstVidx1.sb, dstStart, outDeg, tmpVPropBuffer1b);
					compute(srcVidx, dstVidx1.sc, dstStart, outDeg, tmpVPropBuffer1c);
					compute(srcVidx, dstVidx1.sd, dstStart, outDeg, tmpVPropBuffer1d);
					compute(srcVidx, dstVidx1.se, dstStart, outDeg, tmpVPropBuffer1e);
					compute(srcVidx, dstVidx1.sf, dstStart, outDeg, tmpVPropBuffer1f);
		}
		int tmpEndFlag = read_channel_nb_altera(edgeInfoChEof, &validFlag);
		if(validFlag) endFlag = tmpEndFlag;	
		if(endFlag == EOF_FLAG && !validData) break;
	}

		for(int i = 0; i < (dstNum >> 5); i ++){
			int16 prop_uint16;
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
			int16 prop1_uint16;
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
      		int ddr_idx = (dstStart >> 4) + (i << 1);
			tmpVertexProp[ddr_idx] = prop_uint16;
			tmpVertexProp[ddr_idx + 1] = prop1_uint16;
    }
		//mem_fence(CLK_GLOBAL_MEM_FENCE);
}

__kernel void __attribute__((task)) vertexApply(
		__global int* restrict vertexProp,
		__global int* restrict tmpVertexProp,
		//__global int* restrict activeVertices,
		//__global int* restrict outDeg,
		//__global int* restrict vertexScore,
		//__global int* restrict error,
		const int vertexNum
		//const int base_score
		)
{	
	#ifdef PR
	  int error_l[16] = {0};
#pragma unroll 16
	for(int i = 0; i < vertexNum; i++){
		int tProp = tmpVertexProp[i];
		int old_score = vertexProp[i];
		int out_deg = outDeg[i];
		int new_score = base_score + ((kDamp * tProp) >> 7);

		vertexProp[i] = new_score;
		error_l[i & 0xf] += (new_score - old_score) > 0? (new_score - old_score) : (old_score - new_score) ;
		if(out_deg) vertexScore[i] = new_score/out_deg;
	}

  int total_error = 0;
#pragma unroll 16
  for(int i = 0; i < 16; i ++)
	  total_error += error_l[i];
    
  error[0] = total_error;
  #endif


#ifdef BFS
    //int error_l[16] = {0};
  #pragma unroll 16
  	int idx = 0;
  	for(int i = 0; i < vertexNum; i++){
		PROP_TYPE vProp = vertexProp[i];
		PROP_TYPE tProp = tmpVertexProp[i];
		if(vProp != tProp){
			vertexProp[i] = tProp;
			write_channel_altera(nextActiveVertexCh[i & 0xf], i);
		}
	}
	write_channel_altera(nextActiveVertexEof, EOF_FLAG);
#endif
}

__kernel void __attribute__((task)) activeVertexOutput(
		__global int* restrict activeVertices,
		__global int* restrict error
		)
{		

	int  data_tmp[16];
	bool data_vald[16] = {0};
	bool end_vald = 0;
	int  end_tmp = 0;
	int  idx = 0;
	while(true){
	
		#pragma unroll 16
			for(int i = 0; i < 16; i ++){
				data_tmp[i] = read_channel_nb_altera(nextActiveVertexCh[i], &data_vald[i]);
			}
	
		for(int i = 0; i < 16; i ++)
			if(data_vald[i]) activeVertices[idx++] = data_tmp[i];
	
		int tmpEndFlag = read_channel_nb_altera(nextActiveVertexEof, &end_vald);


		bool all_vald = data_vald[0] | data_vald[1] | data_vald[2] | data_vald[3] 
					| data_vald[4] | data_vald[5] | data_vald[6] | data_vald[7] 
					| data_vald[8] | data_vald[9] | data_vald[10] | data_vald[11]
					| data_vald[12] | data_vald[13] | data_vald[14] | data_vald[15];

		if(end_vald) end_tmp = tmpEndFlag;	
		if(end_tmp == EOF_FLAG && !all_vald) break;
	}

	error[0] = idx;
}

