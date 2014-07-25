#define RANK 1024
#define BLOCK_SIZE 128

__kernel __attribute__ ((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
void mmult(__global int* a, __global int* b, __global int* output, __local int* B)
{

	//r=i r=row , c=column/running = tmp/c=j/k = index

	int k, j;
  	int i = get_global_id(0);	
	int iloc = get_local_id(0);
	int nloc = get_local_size(0);
	
	int rank = get_global_size(0);
	int A[RANK];
	int tmp;						
			
	if (i<rank){

		for (k = 0; k < rank; k++)
		{
			A[k] = a[i*rank+k];
		}

		for (j = 0; j < rank; j++) 									
		{		
				for (k=iloc; k<rank; k+=nloc) 
					B[k] = b[k*rank+j];
				
				barrier(CLK_LOCAL_MEM_FENCE);
				tmp = 0;
				for (k = 0; k < rank; k++)
					tmp += A[k] * B[k];		
				output[i*rank + j] = tmp;
			
				barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
}
