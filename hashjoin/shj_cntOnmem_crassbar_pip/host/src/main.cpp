#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
//using namespace std;
using namespace aocl_utils;
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_kernel kernel[2];
static cl_kernel b1;
static cl_kernel kernel_write;
static cl_program program;
static cl_int status;
unsigned int * rTable = NULL;
unsigned int * sTable = NULL;
unsigned int * HashTable = NULL;
unsigned int * tHist = NULL;
unsigned int * rHashCount = NULL;
unsigned int * WASTable = NULL;
unsigned int * WASReadyTable = NULL;
unsigned int * l_cnt = NULL;
#define RAND_RANGE(N) ((float)rand() / ((float)RAND_MAX + 1) * (N))
static void dump_error(const char *str, cl_int status) {
    printf("%s\n", str);
    printf("Error code: %d\n", status);
}
static void freeResources() {
 printf("freeResources!!!!\n");
  if(kernel[0]) 
    clReleaseKernel(kernel[0]);  
  if(kernel[1]) 
    clReleaseKernel(kernel[1]); 
  if(b1) 
    clReleaseKernel(b1);  
  if(kernel_write) 
    clReleaseKernel(kernel_write);      
  if(program) 
    clReleaseProgram(program);
  if(queue) 
    clReleaseCommandQueue(queue);
  if(rTable) 
   clSVMFreeAltera(context,rTable);
  if(sTable) 
   clSVMFreeAltera(context,sTable);
  if(WASTable)
    clSVMFreeAltera(context,WASTable);
  if(tHist)
    clSVMFreeAltera(context,tHist);
  if(rHashCount)
    clSVMFreeAltera(context,rHashCount);
  if(HashTable)
    clSVMFreeAltera(context,HashTable);
  if(context) 
    clReleaseContext(context);
  if(l_cnt)  
  clSVMFreeAltera(context,l_cnt);
}
void cleanup(){
}
unsigned int setHardwareEnv(
		cl_uint &num_platforms,
		cl_uint &num_devices
		){
	status = clGetPlatformIDs(1, &platform, &num_platforms);
	if(status != CL_SUCCESS) {
		dump_error("Failed clGetPlatformIDs.", status);
		freeResources();
		return 1;
	}
	if(num_platforms != 1) {
		printf("Found %d platforms!\n", num_platforms);
		freeResources();
		return 1;
	}
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
	if(status != CL_SUCCESS) {
		dump_error("Failed clGetDeviceIDs.", status);
		freeResources();
		return 1;
	}
	if(num_devices != 1) {
		printf("Found %d devices!\n", num_devices);
		freeResources();
		return 1;
	}
	context = clCreateContext(0, 1, &device, NULL, NULL, &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateContext.", status);
		freeResources();
		return 1;
	}
}
unsigned int setKernelEnv(){
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateCommandQueue.", status);
		freeResources();
		return 1;
	}
	cl_int kernel_status;
	size_t binsize = 0;
	unsigned char * binary_file = loadBinaryFile("./NDrange_pingpong_shj.aocx", &binsize);
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
}
int create_relation_pk(unsigned int *tuple_addr, int num_tuples)
{
    int i;
    for (i = 0; i < num_tuples; i++) {
        tuple_addr[2*i  ] = (i+1);   //+1
		    tuple_addr[2*i+1] = (i+2);   //+2
    }
#if 1
    for (i = num_tuples - 1; i > 0; i--) {   //knuth_shuflle
        int  j              = RAND_RANGE(i);
        int tmp = tuple_addr[2*i];        //intkey_t tmp            = relation->tuples[i].key;
        tuple_addr[2*i] = tuple_addr[2*j];//relation->tuples[i].key = relation->tuples[j].key;
        tuple_addr[2*j] = tmp;            //relation->tuples[j].key = tmp;
    }
#endif
    return 0;
}
#define  ETH_DATA_INPUT
static const size_t workSize = 128*1;
//static const size_t gworkSize = workSize*4*4;
static const size_t gworkSize_build = workSize*1;
static const size_t gworkSize_probe = workSize*4*4;

int main(int argc, char *argv[]) {
	cl_uint num_platforms;
	cl_uint num_devices;

	status = setHardwareEnv(num_platforms, num_devices);
	printf("Creating host buffers.\n");
	
  int factor = 4 ;
  // generally the stupleNum >> rtupleNum
  unsigned int rTupleNum = 0x1000000/factor;//16318*1024; //16 * 1024 * 1204 ;
	unsigned int sTupleNum = 0x1000000/factor;//16318*1024; //16 * 1024 * 1024;
	unsigned int rHashTableBucketNum = 4 * 1024 * 1024 / factor; //32*1024; //0x400000; //
	unsigned int hashBucketSize      = 4*rTupleNum / rHashTableBucketNum;
  unsigned int rTableSize = 2*rTupleNum;
  unsigned int sTableSize = 2*sTupleNum;
  unsigned int HashTableSize = 2*hashBucketSize*rHashTableBucketNum;
  unsigned int tHistSize = rHashTableBucketNum * 128;
  unsigned int rHashCountSize = 128*128;
  unsigned int WASTableSize = rHashTableBucketNum*hashBucketSize * 32;
  printf("WASTableSize is %d\n",WASTableSize);
  WASTable = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(unsigned int)*WASTableSize, 1024); 
  WASReadyTable = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(unsigned int)*rHashTableBucketNum * 2, 1024); 
  rHashCount = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(unsigned int)*128*128, 1024); 
  tHist = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(unsigned int)*rHashTableBucketNum * 128, 1024); 
  rTable = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(unsigned int)*2*rTupleNum, 1024); 
	sTable = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(unsigned int)*2*sTupleNum, 1024);
  HashTable = (unsigned int*)clSVMAllocAltera(context,0,sizeof(unsigned int)*2*hashBucketSize * rHashTableBucketNum,1024);
  l_cnt = (unsigned int*)clSVMAllocAltera(context,0,sizeof(unsigned int)* rHashTableBucketNum,1024);
  if(!rTable || !sTable || !HashTable) {
		dump_error("Failed to allocate buffers.", status);
		freeResources();
		return 1;	
	}
  memset (HashTable,0,HashTableSize*sizeof(unsigned int));
  memset (l_cnt,0,rHashTableBucketNum*sizeof(unsigned int));
#ifdef ETH_DATA_INPUT
  printf("using ETH_DATA_INPUT::::\n");
  create_relation_pk(rTable, rTupleNum);
#else  
  for (int i = 0, j = 0; i < rTupleNum; i++, j += 2){
    rTable[j]   = ( rand() % 256 ) * rTupleNum + i;
    rTable[j+1] = ( rand() % 256 ) * rTupleNum + i + 1;
  }
#endif 

  status = setKernelEnv();

  printf("Creating hash_build kernel\n");
	{
		kernel[0] = clCreateKernel(program, "hash_div_func", &status);
		if(status != CL_SUCCESS) {
			dump_error("Failed clCreateKernel.", status);
			freeResources();
			return 1;
		} 
   
    cl_uint fpga_offset = 0;
    cl_uint fpga_size   = rTupleNum;
		// set the arguments
		clSetKernelArgSVMPointerAltera(kernel[0], 0, (void*)rTable);
		clSetKernelArgSVMPointerAltera(kernel[0], 1, (void*)WASTable);
	//	clSetKernelArgSVMPointerAltera(kernel[0], 2, (void*)WASReadyTable);
  	clSetKernelArg(kernel[0],2,sizeof(cl_uint),(void*)&fpga_size);
		clSetKernelArg(kernel[0],3,sizeof(cl_uint),(void*)&rHashTableBucketNum);
		clSetKernelArgSVMPointerAltera(kernel[0], 4, (void*)rHashCount);
	
    printf("Launching the build Hash Table kernel...\n");
		status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, (void *)rTable, rTableSize, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMMap", status);
			freeResources();
			return 1;
		}

    status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
				(void *)WASTable, WASTableSize, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMMap", status);
			freeResources();
			return 1;
		}
    status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
				(void *)rHashCount, rHashCountSize, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMMap", status);
			freeResources();
			return 1;
		}

    cl_event event0;
    cl_event event1;
		// launch kernel
    printf("start time\n");
		const double start_time = getCurrentTimestamp();
  	status =	clEnqueueNDRangeKernel(queue,kernel[0],1,NULL,&gworkSize_build,&workSize,0,NULL,&event0);
		if (status != CL_SUCCESS) {
			dump_error("Failed to launch kernel.", status);
			freeResources();
			return 1;
		}
   // clFinish(queue);
    cl_ulong time_start, time_end;
    double total_time;
    clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    total_time = time_end - time_start;
    printf("kernel[0] time = %0.3f ms\n", (total_time / 1000000.0) );

   //kernel [1]
  //  setKernelEnv();
    kernel[1] = clCreateKernel(program, "update_table", &status);
		if(status != CL_SUCCESS) {
			dump_error("Failed clCreateKernel.", status);
			freeResources();
			return 1;
		} 
    clSetKernelArgSVMPointerAltera(kernel[1], 0, (void*)HashTable);
		clSetKernelArgSVMPointerAltera(kernel[1], 1, (void*)WASTable);
		//clSetKernelArgSVMPointerAltera(kernel[1], 2, (void*)WASReadyTable);
		clSetKernelArgSVMPointerAltera(kernel[1], 2, (void*)rHashCount);
  //	clSetKernelArg(kernel[1],4,sizeof(cl_uint),(void*)&rHashTableBucketNum);
  	clSetKernelArg(kernel[1],3,sizeof(cl_uint),(void*)&hashBucketSize);
		clSetKernelArgSVMPointerAltera(kernel[1], 4, (void*)l_cnt);
  	clSetKernelArg(kernel[1],5,sizeof(cl_uint),(void*)&fpga_size);
    printf("Launching the build Hash Table kernel...\n");
		status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
				(void *)HashTable, HashTableSize, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMMap", status);
			freeResources();
			return 1;
		}
    status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
				(void *)WASTable, WASTableSize, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMMap", status);
			freeResources();
			return 1;
		}
    status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
				(void *)rHashCount, rHashCountSize, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMMap", status);
			freeResources();
			return 1;
		}
    printf("start kernel[1] \n");
	  status =	clEnqueueNDRangeKernel(queue,kernel[1],1,NULL,&gworkSize_build,&workSize,0,NULL,&event1);
		if (status != CL_SUCCESS) {
			dump_error("Failed to launch kernel.", status);
			freeResources();
			return 1;
		}
    clFinish(queue);
    printf("kernel[1] is finished \n");
		const double end_time = getCurrentTimestamp();
    printf("kernel : finish building the hash table \n");
    
    clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    total_time = time_end - time_start;
    printf("hash table(len: 0x%x) time = %0.3f ms\n", rTupleNum, (total_time / 1000000.0) );

    unsigned int * buck_p;int total_len = 0; int buck_len;
  // check the result
    for (int i = 0; i < rHashTableBucketNum; i++){
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


     status = clEnqueueSVMUnmap(queue, (void *)WASTable, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMUnmap", status);
			freeResources();
			return 1;
		}
    status = clEnqueueSVMUnmap(queue, (void *)rHashCount, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMUnmap", status);
			freeResources();
			return 1;
		}
    status = clEnqueueSVMUnmap(queue, (void *)HashTable, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMUnmap", status);
			freeResources();
			return 1;
		}
				// Wall-clock time taken.
		double time = (end_time - start_time);
    printf("runtime is: %f\n", time);

	}
	freeResources();

	return 0;
}
