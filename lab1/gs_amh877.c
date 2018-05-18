#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
/*** Skeleton for Lab 1 ***/
/*
Author: Andrew Huang

Given a system of equations, this code utilizes MPI to find the solutions in
a parallel manner with a certain margin of error.
*/

/***** Globals ******/
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */


/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */
/********************************/
/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/*
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix()
{
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;

  for(i = 0; i < num; i++)
  {
    sum = 0;
    aii = fabs(a[i][i]);

    for(j = 0; j < num; j++)
       if( j != i)
	 sum += fabs(a[i][j]);

    if( aii < sum)
    {
      printf("The matrix will not converge.\n");
      exit(1);
    }

    if(aii > sum)
      bigger++;

  }

  if( !bigger )
  {
     printf("The matrix will not converge\n");
     exit(1);
  }
}


/******************************************************/
/* Read input from file */
/* After this function returns:
 * a[][] will be filled with coefficients and you can access them using a[i][j] for element (i,j)
 * x[] will contain the initial values of x
 * b[] will contain the constants (i.e. the right-hand-side of the equations
 * num will have number of variables
 * err will have the absolute error that you need to reach
 */
void get_input(char filename[])
{
  FILE * fp;
  int i,j;

  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

 fscanf(fp,"%d ",&num);
 fscanf(fp,"%f ",&err);

 /* Now, time to allocate the matrices and vectors */
 a = (float**)malloc(num * sizeof(float*));
 if( !a)
  {
	printf("Cannot allocate a!\n");
	exit(1);
  }

 for(i = 0; i < num; i++)
  {
    a[i] = (float *)malloc(num * sizeof(float));
    if( !a[i])
  	{
		printf("Cannot allocate a[%d]!\n",i);
		exit(1);
  	}
  }

 x = (float *) malloc(num * sizeof(float));
 if( !x)
  {
	printf("Cannot allocate x!\n");
	exit(1);
  }


 b = (float *) malloc(num * sizeof(float));
 if( !b)
  {
	printf("Cannot allocate b!\n");
	exit(1);
  }

 /* Now .. Filling the blanks */

 /* The initial values of Xs */
 for(i = 0; i < num; i++)
	fscanf(fp,"%f ", &x[i]);

 for(i = 0; i < num; i++)
 {
   for(j = 0; j < num; j++)
     fscanf(fp,"%f ",&a[i][j]);

   /* reading the b element */
   fscanf(fp,"%f ",&b[i]);
 }

 fclose(fp);

}


/************************************************************/
float computeX(int idx, float *a, float b, float *x) {
    // finds xi given an index
    int i;
    float ans = 0;
    ans += b;
    for (i = 0; i < num; i++) {
        if (i != idx) {
            ans -= a[i] * x[i];
        }
    }
    return ans / a[idx];
}

int main(int argc, char *argv[]){
 int i,j,k;
 int nit = 0; /* number of iterations */
 FILE * fp;
 char output[100] ="";
 if( argc != 2){
   printf("Usage: ./gsref filename\n");
   exit(1);
 }

 /* Read the input file and fill the global data structure above */
 get_input(argv[1]);
 check_matrix();
 // MPI CODE - WRITE TO OUTPUT
 /*

 */
  //======================================
  int comm_sz;
  int my_rank;
  MPI_Init(&argc,&argv); // boilerplate
  /* Get the number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
   /* Get my rank among all the processes */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int per_proc = num / comm_sz;
  int leftover = num % comm_sz;
  int num_send[comm_sz]; // aka sendcounts
  int displs[comm_sz]; // displacements for scatter
  int recv[comm_sz];
  int disp = 0;
  for (i = 0; i < comm_sz; i++){
      if (i < leftover){
          num_send[i] = per_proc +1;
      } else {
          num_send[i] = per_proc;
      }
      recv[i] = num_send[i];
       displs[i] = disp;
       disp = disp + num_send[i];
  }
  float localx[num];
  int brk = 0;
  while (!brk){
      brk=1;
      // we iterate through each xi term and compute the loss, if it's less for this process,
      // break and know that for this xi we are done.
      for (i = 0; i < num_send[my_rank]; i++){
          localx[i + displs[my_rank]] = computeX(displs[my_rank]+i, a[displs[my_rank]+i], b[displs[my_rank]+i], x);
      }
      MPI_Allgatherv(&localx[displs[my_rank]], num_send[my_rank], MPI_FLOAT,
                     &localx, num_send, displs, MPI_FLOAT, MPI_COMM_WORLD);

    for (i = 0; i < num; i++){
        if (err < fabsf((x[i] - localx[i]) / localx[i])) {
                brk = 0;break;
        }
    }
    for (i = 0; i < num; i++) {
           x[i] = localx[i];
       }
      nit++;
  }
 //========================================
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
   printf("total number of iterations: %d\n", nit);
    /* Writing results to file */
   sprintf(output,"%d.sol",num);
   fp = fopen(output,"w");
   if(!fp)
   {
     printf("Cannot create the file %s\n", output);
     exit(1);
   }

   for( i = 0; i < num; i++)
     fprintf(fp,"%f\n",x[i]);
      fclose(fp);
free(a); // cleanup
free(b);
free(x);
 exit(0);

}
