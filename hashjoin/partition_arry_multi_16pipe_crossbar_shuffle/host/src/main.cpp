#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#include "cpu_parallel_radix_join.h"  /* parallel radix joins: RJ, PRO, PRH, PRHO */
#include "fpga_radix_join.h" 

using namespace aocl_utils;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue_relRead;
cl_command_queue queue_hashjoin;
cl_kernel kernel_hashjoin;
cl_kernel kernel_relRead;
cl_program program;
cl_int status;

unsigned int * rTable = NULL;
unsigned int * sTable = NULL;
unsigned int * HashTable = NULL;
unsigned int * matchedTable = NULL;
unsigned int * rHashCount = NULL;
unsigned int * rTableReadRange = NULL;
unsigned int * sTableReadRange = NULL;


int factor = 4;
unsigned int rTupleNum = 1024*256*1;//0x1000000/factor;//16318*1024; //16 * 1024 * 1204 ;
unsigned int sTupleNum = 1;//1024*256*1;//16318*1024; //16 * 1024 * 1024;
unsigned int rHashTableBucketNum = 4 * 1024 * 1024 / factor; //32*1024; //0x400000; //
unsigned int hashBucketSize      = 4*rTupleNum / rHashTableBucketNum;
unsigned int rTableSize = sizeof(unsigned int)*2*rTupleNum;
unsigned int sTableSize = sizeof(unsigned int)*2*sTupleNum;
unsigned int matchedTableSize = 400;//rTableSize + sTableSize;

#define RAND_RANGE(N) ((float)rand() / ((float)RAND_MAX + 1) * (N))

int setKernelEnv(){
    queue_relRead = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateCommandQueue.", status);
        freeResources();
        return 1;
    }
    queue_hashjoin = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateCommandQueue.", status);
        freeResources();
        return 1;
    }
    cl_int kernel_status;
    size_t binsize = 0;
    unsigned char * binary_file = loadBinaryFile("./shj.aocx", &binsize);
    if(!binary_file) {
        dump_error("Failed loadBinaryFile.", status);
        freeResources();
        return 1;
    }
    program = clCreateProgramWithBinary(
        context, 1, &device, &binsize, 
        (const unsigned char**)&binary_file, 
        &kernel_status, &status);

    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateProgramWithBinary.", status);
        freeResources();
        return 1;
    }
    delete [] binary_file;

    // build the program
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    if(status != CL_SUCCESS) {
        dump_error("Failed clBuildProgram.", status);
        freeResources();
        return 1;
    }
    return 0;
}
int create_relation_pk(unsigned int *tuple_addr, int num_tuples)
{
	int i;
	for (i = 0; i < num_tuples; i++) {
        tuple_addr[2*i] = (i+1);   //+1
		tuple_addr[2*i+1] = (i+2);   //+2
	}
  // shuffle tuples of the relation using Knuth shuffle
#if 0
    for (i = num_tuples - 1; i > 0; i--) {   //knuth_shuflle
    	  int  j  = RAND_RANGE(i);
        int tmp = tuple_addr[2*i];        //intkey_t tmp            = relation->tuples[i].key;
        tuple_addr[2*i] = tuple_addr[2*j];//relation->tuples[i].key = relation->tuples[j].key;
        tuple_addr[2*j] = tmp;            //relation->tuples[j].key = tmp;
    }
#endif
    return 0;
}
int64_t RJonCPU( tuple_t * rTable, tuple_t * sTable, uint rTupleNum, uint sTupleNum){
	relation_t  relR;
    relation_t  relS;
    int64_t result = 0;

    relR.tuples = rTable;
    relR.num_tuples = rTupleNum;

    relS.tuples = sTable;
    relS.num_tuples = sTupleNum;

    result = RJ(&relR, &relS, 1);

    return result;
}

int64_t RJonFPGA( tuple_t * rTable, tuple_t * sTable, uint rTupleNum, uint sTupleNum){
	relation_t  relR;
    relation_t  relS;
    int64_t result = 0;

    relR.tuples = rTable;
    relR.num_tuples = rTupleNum;

    relS.tuples = sTable;
    relS.num_tuples = sTupleNum;

    result = RJ_FPGA(&relR, &relS, 1);

    return result;
}

//static const size_t workSize = 128*1;

int main(int argc, char *argv[]) {
	cl_uint num_platforms;
	cl_uint num_devices;
	status = setHardwareEnv(num_platforms, num_devices);
	printf("Creating host buffers.\n");
	//rHashCount = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(unsigned int)*rHashTableBucketNum, 1024); 
	//tHist = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(unsigned int)*rHashTableBucketNum * 128, 1024); 
	rTable = (unsigned int*)clSVMAllocAltera(context, 0, rTableSize, 1024); 
	sTable = (unsigned int*)clSVMAllocAltera(context, 0, sTableSize, 1024);
	matchedTable = (unsigned int*)clSVMAllocAltera(context, 0,matchedTableSize,1024);
	sTableReadRange = (unsigned int *) clSVMAllocAltera(context, 0, sizeof(unsigned int) * 2, 1024);
	rTableReadRange = (unsigned int *) clSVMAllocAltera(context, 0, sizeof(unsigned int) * 2, 1024);
	//HashTable = (unsigned int*)clSVMAllocAltera(context,0,sizeof(unsigned int)*2*hashBucketSize * rHashTableBucketNum,1024);
	if(!rTable || !sTable || !matchedTable) {
		dump_error("Failed to allocate buffers.", status);
		freeResources();
		return 1;	
	}
	memset (matchedTable,0,matchedTableSize);
// creat relations
	create_relation_pk(rTable, rTupleNum);
	create_relation_pk(sTable, sTupleNum);
	
// Process on CPU
	const double cpu_start_time = getCurrentTimestamp();
	int64_t result = RJonCPU( (tuple_t *)rTable, (tuple_t *)sTable, rTupleNum, sTupleNum);
	const double cpu_end_time = getCurrentTimestamp();
	printf("[INFO] cpu processing result is %lu \n", result);
	printf("[INFO] cpu runtime is: %f\n", cpu_end_time - cpu_start_time);
	

// start the hardware
	setKernelEnv();
	printf("Creating hash join kernel\n");		
	creatKernels();
#if 0
	const double fpga_start_time = getCurrentTimestamp();
	int64_t result_fpga = RJonFPGA( (tuple_t *)rTable, (tuple_t *)sTable, rTupleNum, sTupleNum);
	const double fpga_end_time = getCurrentTimestamp();
	printf("[INFO] FPGA processing result is %lu \n", result_fpga);
	printf("[INFO] FPGA runtime is: %f\n", fpga_end_time - fpga_start_time);
#else
	{
    	int argvi = 0;
    	rTableReadRange[0] = 0;
    	rTableReadRange[1] = rTupleNum;
    	sTableReadRange[0] = 0;
    	sTableReadRange[1] = sTupleNum;
  /*  	clSetKernelArgSVMPointerAltera(kernel_relRead, argvi ++, (void*)rTable);
    	clSetKernelArgSVMPointerAltera(kernel_relRead, argvi ++, (void*)rTableReadRange);
    	clSetKernelArgSVMPointerAltera(kernel_relRead, argvi ++, (void*)sTable);
    	clSetKernelArgSVMPointerAltera(kernel_relRead, argvi ++, (void*)sTableReadRange);
    	argvi = 0;
    	clSetKernelArgSVMPointerAltera(kernel_hashjoin, argvi ++, (void*)matchedTable);
		clSetKernelArgSVMPointerAltera(kernel_hashjoin, argvi ++, (void*)rTableReadRange);
    	clSetKernelArgSVMPointerAltera(kernel_hashjoin, argvi ++, (void*)sTableReadRange);
		varMap();
	*/
    printf("Launching the hash join kernel...\n");

		cl_event event_hashjoin, event_relRead;
		//status = clEnqueueNDRangeKernel(queue,kernel,1,NULL,&gworkSize_build,&workSize,0,NULL,&event1);
		status = clEnqueueTask(queue_relRead, kernel_relRead, 0, NULL, &event_relRead);
		if (status != CL_SUCCESS) {
			dump_error("Failed to launch kernel.", status);
			freeResources();
			return 1;
		}
		status = clEnqueueTask(queue_hashjoin, kernel_hashjoin, 0, NULL, &event_hashjoin);
		if (status != CL_SUCCESS) {
			dump_error("Failed to launch kernel.", status);
			freeResources();
			return 1;
		}
		const double start_time = getCurrentTimestamp();
		clFinish(queue_relRead);
		clFinish(queue_hashjoin);
		const double end_time = getCurrentTimestamp();
		printf("kernel : finish building the hash table \n");
		double time = (end_time - start_time);
		printf("[INFO ]FPGA runtime is: %f\n", time);
		// caculate the time with OpenCL profiling 
		cl_ulong time_start, time_end;
		double total_time;
		clGetEventProfilingInfo(event_hashjoin, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event_hashjoin, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time = time_end - time_start;
		printf("[INFO] hash join kernel(len: 0x%x) time = %0.3f ms\n", rTupleNum, (total_time / 1000000.0) );
  
    clGetEventProfilingInfo(event_relRead, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event_relRead, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time = time_end - time_start;
		printf("[INFO] relation read kernel(len: 0x%x) time = %0.3f ms\n", rTupleNum, (total_time / 1000000.0) );


		for (int i  = 0; i < 100; i ++)
		printf("%d \t", matchedTable[i]);
	}
	//unsigned int * buck_p;int total_len = 0; int buck_len;
  // check the result
  /*  for (int i = 0; i < rHashTableBucketNum; i++){
				buck_len = 0;
				buck_p = HashTable + hashBucketSize*i*2;
				for (int j = 0; j < hashBucketSize; j++){
				  if (((buck_p[2*j] % rHashTableBucketNum) != i) && (buck_p[2*j] != 0) )
            printf("hash data with key: 0x%x to wrong bucket 0x%x\n", buck_p[2*j],i);
				  else if (((buck_p[2*j] % rHashTableBucketNum) == i ) && (buck_p[2*j] != 0))
					   buck_len++;
				  else
					   break;
				}
				if ( buck_len != (rTupleNum/rHashTableBucketNum) )
            printf("error bucket index: 0x%x is %d (not %d)\n", i, buck_len,(rTupleNum/rHashTableBucketNum));
				total_len += buck_len;
			}
			printf("hash table total_len is 0x%x\n", total_len);
      for(int i = 0; i < rTupleNum; i++) //(int i = rTupleNum-20; i < rTupleNum; i++)
      {
			if ( (rTable[2*i]&(rHashTableBucketNum-1)) == 0)
	          printf("id_0x%x: 0x%x\t", i, rTable[2*i] );
      }
	  	printf("\n");
*/
	varUnmap();
#endif
	
	freeResources();

	return 0;
}



