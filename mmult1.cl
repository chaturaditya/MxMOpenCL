__kernel //__attribute__ ((reqd_work_group_size(2048, 2048, 1)))
void mmult(__global int* a, __global int* b, __global int* output, __local float *C)
{
  int r = get_global_id(0);		//r=i			//r=row , c=column
  int rank = get_global_size(0);
	if(r < rank)
	{
		for (int c = 0; c < rank; c++) {		//c=j

			int running = 0;			//running = tmp
			for (int index=0; index<rank; index++) {		//index = k
				int aIndex = r*rank + index;
				int bIndex = index*rank + c;
				running +=  a[aIndex] * b[bIndex];
			}

			output[r*rank + c] = running;
		}
	}
  //return;
}
