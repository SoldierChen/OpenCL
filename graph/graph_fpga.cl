//#include "config.h"
//type define
#define SW 1
#if SW == 1
	#define VERTEX_MAX  128*1024 //(128*1024)
	#define EDGE_MAX  (512*1024)
#else
	#define VERTEX_MAX  (128*1024)
	#define EDGE_MAX  (512*1024)
#endif
#define PROP_TYPE float
#define PR
#define kDamp 0.85f
#define BRAM_BANK 64
#define LOG2_BRAM_BANK 6
typedef struct EdgeInfo{
	int vertexIdx;
	int16 ngbVidx0;
	int16 ngbVidx1;
	int16 ngbVidx2;
	int16 ngbVidx3;
	//PROP_TYPE eProp;
	int outDeg;
} edge_info_t;

channel int activeVertexCh __attribute__((depth(1024)));
channel edge_info_t edgeInfoCh __attribute__((depth(512)));
channel int edgeInfoChEof __attribute__((depth(4)));
channel int nextFrontierCh __attribute__((depth(2048)));
channel int nextFrontierChEof __attribute__((depth(4)));

 // BFS // SSSP // PR // CC
__attribute__((always_inline)) void compute(int srcVidx, int dstVidx, int outDeg,
	PROP_TYPE* tmpVPropBuffer, PROP_TYPE srcProp){
		//PROP_TYPE srcVprop = vPropBuffer[srcVidx];
		//PROP_TYPE dstVprop = tmpVPropBuffer[(int)(dstVidx / 4)];
	#ifdef PR 
			if(dstVidx != -1) {
				int idx = dstVidx >> LOG2_BRAM_BANK;
				//int idx_src = srcVidx >> LOG2_BRAM_BANK;
				//PROP_TYPE operand0 = vPropBuffer[idx_src];
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
		__global const int16* restrict rpaStart,
		__global const int16* restrict rpaNum,
		__global const int16* restrict outDeg,
		__global const int16 * restrict cia,
		__global const PROP_TYPE* restrict edgeProp,
		__global const int* restrict activeVertexNum,
		const int vertexNum,
		const int edgeNum,
		__global const int* restrict itNum
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



	for(int i = 0; i < (vertexNum >> 4); i++){
			int16 rpa_uint16 = rpaStart[i];
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
	for(int i = 0; i < (vertexNum >> 4); i++){
			int16 rpa_uint16 = rpaNum[i];
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
	for(int i = 0; i < (vertexNum >> 4); i++){
			int16 rpa_uint16 = outDeg[i];
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
	//printf("ck1 \n");
	for(int i = 0; i < activeVertexNum[0]; i++){	
		int vertexIdx = read_channel_altera(activeVertexCh);
		int start = rpaStartBuffer[vertexIdx >> 4][vertexIdx & 0xf];
		int num = rpaNumBuffer[vertexIdx >> 4][vertexIdx & 0xf];
		int outDeg = outDegBuffer[vertexIdx >> 4][vertexIdx & 0xf];
		//if(vertexIdx < 20 )
		//printf("\n vertex %d num %d start %d \t ",vertexIdx, num, start);
		for(int j = (start >> 4); j < ((start + num) >> 4); j += 4){
			int16 ngbVidx0 = cia[j];
			int16 ngbVidx1 = cia[j+1];
			int16 ngbVidx2 = cia[j+2];
			int16 ngbVidx3 = cia[j+3];

			//printf("%d %d %d %d \n",ngbVidx.s0, ngbVidx.s1, ngbVidx.s2, ngbVidx.s3 );
			//int4 eProp = edgePropBuffer[start + j];
			edge_info_t edge_info;
			edge_info.vertexIdx = vertexIdx;
			edge_info.ngbVidx0 = ngbVidx0;
			edge_info.ngbVidx1 = ngbVidx1;
			edge_info.ngbVidx2 = ngbVidx2;
			edge_info.ngbVidx3 = ngbVidx3;
			//edge_info.eProp = eProp;
			edge_info.outDeg = outDeg;
			write_channel_altera(edgeInfoCh, edge_info);
		}
	}
	write_channel_altera(edgeInfoChEof, 0xffff);
}

__kernel void __attribute__((task)) processEdge(
		__global float16* restrict vertexProp,
		__global float16 * restrict tmpVertexProp,
		__global PROP_TYPE* restrict semaphore,
		const int vertexNum,
		__global const int* restrict itNum
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


		PROP_TYPE * tmpVertexPropBuffer0 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer4 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer5 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer6 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer7 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer8 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer9 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffera 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBufferb 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBufferc 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBufferd 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffere 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBufferf 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);



		PROP_TYPE * tmpVertexPropBuffer10 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer11 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer12 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer13 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer14 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer15 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer16 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer17 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer18 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer19 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1a 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1b 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1c 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1d 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1e 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer1f 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);


		PROP_TYPE * tmpVertexPropBuffer20 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer21 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer22 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer23 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer24 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer25 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer26 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer27 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer28 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer29 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2a 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2b 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2c 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2d 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2e 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer2f 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);

		PROP_TYPE * tmpVertexPropBuffer30 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer31 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer32 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer33 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer34 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer35 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer36 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer37 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer38 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer39 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3a 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3b 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3c 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3d 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3e 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);
		PROP_TYPE * tmpVertexPropBuffer3f 	= malloc (sizeof(PROP_TYPE) * VERTEX_MAX/BRAM_BANK);

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
	edge_info_t edge_info; 
	PROP_TYPE eProp = 0;
	int end_flag = 0;
	bool valid_data = 0;
	bool valid_flag = 0;
	//int count[1] = {vertexNum};
	for(int i = 0; i < (vertexNum >> 4); i++){
			
			float16 prop_uint16 = vertexProp[i];
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

	for(int i = 0; i < (vertexNum >> LOG2_BRAM_BANK); i++){
			tmpVertexPropBuffer0[i]=  0;
			tmpVertexPropBuffer1[i]=  0;
			tmpVertexPropBuffer2[i]=  0;
			tmpVertexPropBuffer3[i]=  0;
			tmpVertexPropBuffer4[i]=  0;
			tmpVertexPropBuffer5[i]=  0;
			tmpVertexPropBuffer6[i]=  0;
			tmpVertexPropBuffer7[i]=  0;
			tmpVertexPropBuffer8[i]=  0;
			tmpVertexPropBuffer9[i]=  0;
			tmpVertexPropBuffera[i] = 0;
			tmpVertexPropBufferb[i] = 0;
			tmpVertexPropBufferc[i] = 0;
			tmpVertexPropBufferd[i] = 0;
			tmpVertexPropBuffere[i] = 0;
			tmpVertexPropBufferf[i] = 0;

			tmpVertexPropBuffer10[i]=  0;
			tmpVertexPropBuffer11[i]=  0;
			tmpVertexPropBuffer12[i]=  0;
			tmpVertexPropBuffer13[i]=  0;
			tmpVertexPropBuffer14[i]=  0;
			tmpVertexPropBuffer15[i]=  0;
			tmpVertexPropBuffer16[i]=  0;
			tmpVertexPropBuffer17[i]=  0;
			tmpVertexPropBuffer18[i]=  0;
			tmpVertexPropBuffer19[i]=  0;
			tmpVertexPropBuffer1a[i] = 0;
			tmpVertexPropBuffer1b[i] = 0;
			tmpVertexPropBuffer1c[i] = 0;
			tmpVertexPropBuffer1d[i] = 0;
			tmpVertexPropBuffer1e[i] = 0;
			tmpVertexPropBuffer1f[i] = 0;

			tmpVertexPropBuffer20[i]=  0;
			tmpVertexPropBuffer21[i]=  0;
			tmpVertexPropBuffer22[i]=  0;
			tmpVertexPropBuffer23[i]=  0;
			tmpVertexPropBuffer24[i]=  0;
			tmpVertexPropBuffer25[i]=  0;
			tmpVertexPropBuffer26[i]=  0;
			tmpVertexPropBuffer27[i]=  0;
			tmpVertexPropBuffer28[i]=  0;
			tmpVertexPropBuffer29[i]=  0;
			tmpVertexPropBuffer2a[i] = 0;
			tmpVertexPropBuffer2b[i] = 0;
			tmpVertexPropBuffer2c[i] = 0;
			tmpVertexPropBuffer2d[i] = 0;
			tmpVertexPropBuffer2e[i] = 0;
			tmpVertexPropBuffer2f[i] = 0;

			tmpVertexPropBuffer30[i]=  0;
			tmpVertexPropBuffer31[i]=  0;
			tmpVertexPropBuffer32[i]=  0;
			tmpVertexPropBuffer33[i]=  0;
			tmpVertexPropBuffer34[i]=  0;
			tmpVertexPropBuffer35[i]=  0;
			tmpVertexPropBuffer36[i]=  0;
			tmpVertexPropBuffer37[i]=  0;
			tmpVertexPropBuffer38[i]=  0;
			tmpVertexPropBuffer39[i]=  0;
			tmpVertexPropBuffer3a[i] = 0;
			tmpVertexPropBuffer3b[i] = 0;
			tmpVertexPropBuffer3c[i] = 0;
			tmpVertexPropBuffer3d[i] = 0;
			tmpVertexPropBuffer3e[i] = 0;
			tmpVertexPropBuffer3f[i] = 0;
	}

	while(true){	

		//printf("--------------ck2 \n");

		edge_info = read_channel_nb_altera(edgeInfoCh, &valid_data);
		if(valid_data){
			int srcVidx = edge_info.vertexIdx;
			int16 dstVidx0 = edge_info.ngbVidx0;
			int16 dstVidx1 = edge_info.ngbVidx1;
			int16 dstVidx2 = edge_info.ngbVidx2;
			int16 dstVidx3 = edge_info.ngbVidx3;
			int outDeg = edge_info.outDeg;
			PROP_TYPE srcProp;
			
			switch (srcVidx & 0xf){
				case 0: srcProp =  vPropBuffer0[srcVidx >> 4]; break;
				case 1: srcProp =  vPropBuffer1[srcVidx >> 4]; break;
				case 2: srcProp =  vPropBuffer2[srcVidx >> 4]; break;
				case 3: srcProp =  vPropBuffer3[srcVidx >> 4]; break;
				case 4: srcProp =  vPropBuffer4[srcVidx >> 4]; break;
				case 5: srcProp =  vPropBuffer5[srcVidx >> 4]; break;
				case 6: srcProp =  vPropBuffer6[srcVidx >> 4]; break;
				case 7: srcProp =  vPropBuffer7[srcVidx >> 4]; break;
				case 8: srcProp =  vPropBuffer8[srcVidx >> 4]; break;
				case 9: srcProp =  vPropBuffer9[srcVidx >> 4]; break;
				case 10: srcProp = vPropBuffera[srcVidx >> 4]; break;
				case 11: srcProp = vPropBufferb[srcVidx >> 4]; break;
				case 12: srcProp = vPropBufferc[srcVidx >> 4]; break;
				case 13: srcProp = vPropBufferd[srcVidx >> 4]; break;
				case 14: srcProp = vPropBuffere[srcVidx >> 4]; break;
				case 15: srcProp = vPropBufferf[srcVidx >> 4]; break;
			}

			//if(srcVidx < 20)
			//printf("srcVidx %d readNgbInfo %d %d %d %d \n",srcVidx,dstVidx0.s0,dstVidx0.s1,dstVidx0.s2,dstVidx0.s3);
			if(outDeg) {
				
					compute(srcVidx, dstVidx0.s0, outDeg, tmpVertexPropBuffer0, srcProp);
					compute(srcVidx, dstVidx0.s1, outDeg, tmpVertexPropBuffer1, srcProp);
					compute(srcVidx, dstVidx0.s2, outDeg, tmpVertexPropBuffer2, srcProp);
					compute(srcVidx, dstVidx0.s3, outDeg, tmpVertexPropBuffer3, srcProp);
					compute(srcVidx, dstVidx0.s4, outDeg, tmpVertexPropBuffer4, srcProp);
					compute(srcVidx, dstVidx0.s5, outDeg, tmpVertexPropBuffer5, srcProp);
					compute(srcVidx, dstVidx0.s6, outDeg, tmpVertexPropBuffer6, srcProp);
					compute(srcVidx, dstVidx0.s7, outDeg, tmpVertexPropBuffer7, srcProp);
					compute(srcVidx, dstVidx0.s8, outDeg, tmpVertexPropBuffer8, srcProp);
					compute(srcVidx, dstVidx0.s9, outDeg, tmpVertexPropBuffer9, srcProp);
					compute(srcVidx, dstVidx0.sa, outDeg, tmpVertexPropBuffera, srcProp);
					compute(srcVidx, dstVidx0.sb, outDeg, tmpVertexPropBufferb, srcProp);
					compute(srcVidx, dstVidx0.sc, outDeg, tmpVertexPropBufferc, srcProp);
					compute(srcVidx, dstVidx0.sd, outDeg, tmpVertexPropBufferd, srcProp);
					compute(srcVidx, dstVidx0.se, outDeg, tmpVertexPropBuffere, srcProp);
					compute(srcVidx, dstVidx0.sf, outDeg, tmpVertexPropBufferf, srcProp);


					compute(srcVidx, dstVidx1.s0, outDeg, tmpVertexPropBuffer10, srcProp);
					compute(srcVidx, dstVidx1.s1, outDeg, tmpVertexPropBuffer11, srcProp);
					compute(srcVidx, dstVidx1.s2, outDeg, tmpVertexPropBuffer12, srcProp);
					compute(srcVidx, dstVidx1.s3, outDeg, tmpVertexPropBuffer13, srcProp);
					compute(srcVidx, dstVidx1.s4, outDeg, tmpVertexPropBuffer14, srcProp);
					compute(srcVidx, dstVidx1.s5, outDeg, tmpVertexPropBuffer15, srcProp);
					compute(srcVidx, dstVidx1.s6, outDeg, tmpVertexPropBuffer16, srcProp);
					compute(srcVidx, dstVidx1.s7, outDeg, tmpVertexPropBuffer17, srcProp);
					compute(srcVidx, dstVidx1.s8, outDeg, tmpVertexPropBuffer18, srcProp);
					compute(srcVidx, dstVidx1.s9, outDeg, tmpVertexPropBuffer19, srcProp);
					compute(srcVidx, dstVidx1.sa, outDeg, tmpVertexPropBuffer1a, srcProp);
					compute(srcVidx, dstVidx1.sb, outDeg, tmpVertexPropBuffer1b, srcProp);
					compute(srcVidx, dstVidx1.sc, outDeg, tmpVertexPropBuffer1c, srcProp);
					compute(srcVidx, dstVidx1.sd, outDeg, tmpVertexPropBuffer1d, srcProp);
					compute(srcVidx, dstVidx1.se, outDeg, tmpVertexPropBuffer1e, srcProp);
					compute(srcVidx, dstVidx1.sf, outDeg, tmpVertexPropBuffer1f, srcProp);

					compute(srcVidx, dstVidx2.s0, outDeg, tmpVertexPropBuffer20, srcProp);
					compute(srcVidx, dstVidx2.s1, outDeg, tmpVertexPropBuffer21, srcProp);
					compute(srcVidx, dstVidx2.s2, outDeg, tmpVertexPropBuffer22, srcProp);
					compute(srcVidx, dstVidx2.s3, outDeg, tmpVertexPropBuffer23, srcProp);
					compute(srcVidx, dstVidx2.s4, outDeg, tmpVertexPropBuffer24, srcProp);
					compute(srcVidx, dstVidx2.s5, outDeg, tmpVertexPropBuffer25, srcProp);
					compute(srcVidx, dstVidx2.s6, outDeg, tmpVertexPropBuffer26, srcProp);
					compute(srcVidx, dstVidx2.s7, outDeg, tmpVertexPropBuffer27, srcProp);
					compute(srcVidx, dstVidx2.s8, outDeg, tmpVertexPropBuffer28, srcProp);
					compute(srcVidx, dstVidx2.s9, outDeg, tmpVertexPropBuffer29, srcProp);
					compute(srcVidx, dstVidx2.sa, outDeg, tmpVertexPropBuffer2a, srcProp);
					compute(srcVidx, dstVidx2.sb, outDeg, tmpVertexPropBuffer2b, srcProp);
					compute(srcVidx, dstVidx2.sc, outDeg, tmpVertexPropBuffer2c, srcProp);
					compute(srcVidx, dstVidx2.sd, outDeg, tmpVertexPropBuffer2d, srcProp);
					compute(srcVidx, dstVidx2.se, outDeg, tmpVertexPropBuffer2e, srcProp);
					compute(srcVidx, dstVidx2.sf, outDeg, tmpVertexPropBuffer2f, srcProp);


					compute(srcVidx, dstVidx3.s0, outDeg, tmpVertexPropBuffer30, srcProp);
					compute(srcVidx, dstVidx3.s1, outDeg, tmpVertexPropBuffer31, srcProp);
					compute(srcVidx, dstVidx3.s2, outDeg, tmpVertexPropBuffer32, srcProp);
					compute(srcVidx, dstVidx3.s3, outDeg, tmpVertexPropBuffer33, srcProp);
					compute(srcVidx, dstVidx3.s4, outDeg, tmpVertexPropBuffer34, srcProp);
					compute(srcVidx, dstVidx3.s5, outDeg, tmpVertexPropBuffer35, srcProp);
					compute(srcVidx, dstVidx3.s6, outDeg, tmpVertexPropBuffer36, srcProp);
					compute(srcVidx, dstVidx3.s7, outDeg, tmpVertexPropBuffer37, srcProp);
					compute(srcVidx, dstVidx3.s8, outDeg, tmpVertexPropBuffer38, srcProp);
					compute(srcVidx, dstVidx3.s9, outDeg, tmpVertexPropBuffer39, srcProp);
					compute(srcVidx, dstVidx3.sa, outDeg, tmpVertexPropBuffer3a, srcProp);
					compute(srcVidx, dstVidx3.sb, outDeg, tmpVertexPropBuffer3b, srcProp);
					compute(srcVidx, dstVidx3.sc, outDeg, tmpVertexPropBuffer3c, srcProp);
					compute(srcVidx, dstVidx3.sd, outDeg, tmpVertexPropBuffer3d, srcProp);
					compute(srcVidx, dstVidx3.se, outDeg, tmpVertexPropBuffer3e, srcProp);
					compute(srcVidx, dstVidx3.sf, outDeg, tmpVertexPropBuffer3f, srcProp);

			}
		}

		int end_flag_tmp = read_channel_nb_altera(edgeInfoChEof, &valid_flag);
		if(valid_flag) end_flag = end_flag_tmp;		
		if(end_flag == 0xffff && !valid_data && !valid_flag) break;
	}

		for(int i = 0; i < (vertexNum >> LOG2_BRAM_BANK); i ++){	
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

			tmpVertexProp[i << 2] = prop_uint16;
			tmpVertexProp[(i<<2)+1] = prop1_uint16;
			tmpVertexProp[(i<<2)+2] = prop2_uint16;
			tmpVertexProp[(i<<2)+3] = prop3_uint16;
			//tmpVertexProp[i + dstStart] = tmpVertexPropBuffer[i];
		}
}
