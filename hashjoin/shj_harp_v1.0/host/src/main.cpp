#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
using namespace aocl_utils;
static const char *kernel_name =  "buildHashTable";
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_kernel kernel;
static cl_kernel b1;
static cl_kernel kernel_write;
static cl_program program;
static cl_int status;
unsigned int * rTable = NULL;
unsigned int * sTable = NULL;
unsigned int * HashTable = NULL;
unsigned int * tHist = NULL;
unsigned int * rHashCount = NULL;
#define RAND_RANGE(N) ((float)rand() / ((float)RAND_MAX + 1) * (N))
static void dump_error(const char *str, cl_int status) {
	printf("%s\n", str);
	printf("Error code: %d\n", status);
}
static void freeResources() {
	if(kernel) 
		clReleaseKernel(kernel);  
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
	if(tHist)
		clSVMFreeAltera(context,tHist);
	if(rHashCount)
		clSVMFreeAltera(context,rHashCount);
	if(HashTable)
		clSVMFreeAltera(context,HashTable);
	if(context) 
		clReleaseContext(context);
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
    	int  j  = RAND_RANGE(i);
        int tmp = tuple_addr[2*i];        //intkey_t tmp            = relation->tuples[i].key;
        tuple_addr[2*i] = tuple_addr[2*j];//relation->tuples[i].key = relation->tuples[j].key;
        tuple_addr[2*j] = tmp;            //relation->tuples[j].key = tmp;
    }
#endif
    return 0;
}

static const size_t workSize = 128*1;
//static const size_t gworkSize = workSize*4*4;
static const size_t gworkSize_build = workSize*1;
static const size_t gworkSize_probe = workSize*4*4;

int main(int argc, char *argv[]) {
	printf("Creating host buffers.\n");
	cl_uint num_platforms;
	cl_uint num_devices;

	status = setHardwareEnv(num_platforms, num_devices);
	printf("Creating host buffers.\n");
	
	int factor = 4;
  // generally the stupleNum >> rtupleNum
  	unsigned int rTupleNum = 0x1000000/factor;//16318*1024; //16 * 1024 * 1204 ;
	unsigned int sTupleNum = 0x1000000/factor;//16318*1024; //16 * 1024 * 1024;
	unsigned int rHashTableBucketNum = 4 * 1024 * 1024 / factor; //32*1024; //0x400000; //
	unsigned int hashBucketSize      = 4*rTupleNum / rHashTableBucketNum;
	unsigned int rTableSize = sizeof(unsigned int)*2*rTupleNum;
	unsigned int sTableSize = sizeof(unsigned int)*2*sTupleNum;
	unsigned int HashTableSize = sizeof(unsigned int)*2*hashBucketSize*rHashTableBucketNum;
	unsigned int tHistSize = rHashTableBucketNum * 128;
	unsigned int rHashCountSize = rHashTableBucketNum;

	rHashCount = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(unsigned int)*rHashTableBucketNum, 1024); 
	tHist = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(unsigned int)*rHashTableBucketNum * 128, 1024); 
	rTable = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(unsigned int)*2*rTupleNum, 1024); 
	sTable = (unsigned int*)clSVMAllocAltera(context, 0, sizeof(unsigned int)*2*sTupleNum, 1024);
	HashTable = (unsigned int*)clSVMAllocAltera(context,0,sizeof(unsigned int)*2*hashBucketSize * rHashTableBucketNum,1024);
	if(!rTable || !sTable || !HashTable) {
		dump_error("Failed to allocate buffers.", status);
		freeResources();
		return 1;	
	}
	memset (HashTable,0,HashTableSize);
	create_relation_pk(rTable, rTupleNum);
	status = setKernelEnv();

	printf("Creating hash_build kernel\n");
	{
		kernel = clCreateKernel(program, "buildHashTable", &status);
		if(status != CL_SUCCESS) {
			dump_error("Failed clCreateKernel.", status);
			freeResources();
			return 1;
		} 
		cl_uint fpga_offset = 0;
		cl_uint fpga_size   = rTupleNum;
		// set the arguments
		clSetKernelArgSVMPointerAltera(kernel, 0, (void*)rTable);
		clSetKernelArgSVMPointerAltera(kernel, 1, (void*)HashTable);
		clSetKernelArg(kernel,2, sizeof(cl_uint), (void*)&fpga_offset);
		clSetKernelArg(kernel,3,sizeof(cl_uint),(void*)&fpga_size);
		clSetKernelArg(kernel,4,sizeof(cl_uint),(void*)&rHashTableBucketNum);
		clSetKernelArg(kernel,5,sizeof(cl_uint),(void*)&hashBucketSize);
		clSetKernelArgSVMPointerAltera(kernel, 6, (void*)tHist);
		clSetKernelArgSVMPointerAltera(kernel, 7, (void*)rHashCount);
		printf("Launching the build Hash Table kernel...\n");
		status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)rTable, rTableSize, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMMap", status);
			freeResources();
			return 1;
		}

		status = clEnqueueSVMMap(queue, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE, 
			(void *)sTable, sTableSize, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMMap", status);
			freeResources();
			return 1;
		}	
		status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)HashTable, HashTableSize, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMMap", status);
			freeResources();
			return 1;
		}
		status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 
			(void *)tHist, tHistSize, 0, NULL, NULL); 
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

		cl_event event1;
		const double start_time = getCurrentTimestamp();
		status =	clEnqueueNDRangeKernel(queue,kernel,1,NULL,&gworkSize_build,&workSize,0,NULL,&event1);
		if (status != CL_SUCCESS) {
			dump_error("Failed to launch kernel.", status);
			freeResources();
			return 1;
		}

		status = clEnqueueSVMUnmap(queue, (void *)rTable, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMUnmap", status);
			freeResources();
			return 1;
		}
		clFinish(queue);
		const double end_time = getCurrentTimestamp();
		printf("kernel : finish building the hash table \n");
		cl_ulong time_start, time_end;
		double total_time;
		clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
		total_time = time_end - time_start;
		printf("hash table(len: 0x%x) time = %0.3f ms\n", rTupleNum, (total_time / 1000000.0) );

		status = clEnqueueSVMUnmap(queue, (void *)sTable, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			dump_error("Failed clEnqueueSVMUnmap", status);
			freeResources();
			return 1;
		}
		status = clEnqueueSVMUnmap(queue, (void *)tHist, 0, NULL, NULL); 
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
	//double time = (end_time - start_time);
    //printf("runtime is: %f\n", time);

	}
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

	freeResources();

	return 0;
}



