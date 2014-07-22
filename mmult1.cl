__kernel// __attribute__ ((reqd_work_group_size(16, 16, 1)))
void mmult(__global int* a, __global int* b, __global int* output, __local int *B)
{
	int index, c;
  int r = get_global_id(0);		//r=i		r=row , c=column
	int iloc = get_local_id(0);
	int nloc = get_local_size(0);

	printf("[%d], [%d], [%d]\n",  get_local_size(0),  get_local_size(1),  get_local_size(2)); 
	int A[16];

  int rank = get_global_size(0);
	printf("get_global_size: %d\n", get_global_size(0));
	
	for (index = 0; index < rank; index++)
	{
		A[index] = A[r*rank+index];
	//	printf("A[index]: %d\n",A[index]);
	}

	for (c = 0; c < rank; c++) 									//c=j
	{		
			for (index=iloc; index<rank; index+=nloc) 	//index = k
			{	
				B[index] = B[index*rank+c];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			int running = 0;			//running = tmp
			for (index = 0; index < rank; index++)
			{
				running += A[index] * B[index];
			}
			output[r*rank + c] += running;
		//	printf("output: %d\n", running);
			barrier(CLK_LOCAL_MEM_FENCE);
	}
	//return;

}
