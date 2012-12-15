#include <stdio.h>

#define blocksizex 16 
#define blocksizey 16 
#define N 40000
#define DOT(a,b) a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
#define DOTVA(a,b) a.x*b[0]+a.y*b[1]+a.z*b[2]
 typedef struct 
 {
   float x;
   float y;
   float z;
 } vertex;
  typedef struct 
 {
   vertex a;
   vertex n;
 } mata;
 
__global__ void RunICP(
int   validPoints[N],
vertex verProjD[N],
vertex verNormD[N],
float matRgPre[9],
float matTransltgPre[3],
const int focus,
vertex verProS[N],
vertex verNormS[N],
const float epsD,
const float epsN,
const float epsDs,
float Energy[N],
mata Ak[N],
float Bk[N],
int flag[N]
)
{
	int gidx=blockIdx.x*blockDim.x+threadIdx.x;
        int gidy=blockIdx.y*blockDim.y+threadIdx.y;
        int i=gidx+(gidy-1)*gridDim.y;//num of order
        float vertexProj[3];
        float normalProj[3];
        float disVdiff[3];
        float disNdiff[3];
        float error;

		if (validPoints[i] == true && i<N){
			//transformV  and trandformR
            
			vertexProj[0] = verProjD[i].x*matRgPre[0]+verProjD[i].y*matRgPre[1]+verProjD[i].z*matRgPre[2]+matTransltgPre[0];
			vertexProj[1] = verProjD[i].x*matRgPre[3]+verProjD[i].y*matRgPre[4]+verProjD[i].z*matRgPre[5]+matTransltgPre[1];
			vertexProj[2] = verProjD[i].x*matRgPre[6]+verProjD[i].y*matRgPre[7]+verProjD[i].z*matRgPre[8]+matTransltgPre[2];
			
			normalProj[0] = verNormD[i].x*matRgPre[0]+verNormD[i].y*matRgPre[1]+verNormD[i].z*matRgPre[2];
			normalProj[1] = verProjD[i].x*matRgPre[3]+verProjD[i].y*matRgPre[4]+verProjD[i].z*matRgPre[5];
			normalProj[2] = verProjD[i].x*matRgPre[6]+verProjD[i].y*matRgPre[7]+verProjD[i].z*matRgPre[8];
            

			//get index
			int x = (vertexProj[0] / vertexProj[2] * focus) + 320;

	        int y = (vertexProj[1] / vertexProj[2] * focus) + 240;

	        int index=y*640 + x;

			if(index < 0 || index >= N){

                                Energy[i]=0;
				flag[i]=0;
				return;

			}
			
			disVdiff[0]  = verProS[index].x-vertexProj[0];
			disNdiff[0]  = verNormS[index].x-normalProj[0];
		    disVdiff[1]  = verProS[index].y-vertexProj[1];
			disNdiff[1]  = verNormS[index].y-normalProj[1];
			disVdiff[2]  = verProS[index].y-vertexProj[2];
			disNdiff[2]  = verNormS[index].y-normalProj[2];


			float disV  = DOT(disVdiff, disVdiff);
			float disN  = DOT(disNdiff, disNdiff);

			if( disV < epsD && disN >= epsN && disV > epsDs){
				
				error=DOTVA(verNormS[index],disVdiff);

				Energy[i]= error*error/10;//remember to reduce sumEngry

  				Ak[i].a.x=vertexProj[1]*verNormS[index].z-vertexProj[2]*verNormS[index].y;
                Ak[i].a.y=vertexProj[2]*verNormS[index].x-vertexProj[0]*verNormS[index].z;
                Ak[i].a.z=vertexProj[0]*verNormS[index].y-vertexProj[1]*verNormS[index].x;         

				Ak[i].n.x=verNormS[index].x;
				Ak[i].n.x=verNormS[index].y;
				Ak[i].n.x=verNormS[index].z;

				Bk[i]=DOTVA(verNormS[index],disVdiff);
				flag[i]=1;
			}
           else
                            flag[i]=0;

	     }
             else
             {flag[i]=0;}
}
int main()
{
   //kernel invocation with N threads
   
    /*
   	int   validPoints[N];
   	float verProjD[N][3];
	float verNormD[N][3];
	float matRgPre[3];
	float matTransltgPre[3];

	float verProS[N][3];
	float verNormS[N][3];

	float Energy[N];
	float Ak[N][6];
	float Bk[N];
	int flag[N];
	*/
	const int focus=550;
	const float epsD=40000.0;
	const float epsN=0.6;
	const float epsDs=100.0;
   	//----------------------------------
	//allocation for host
        //------------------------------------
    size_t flsizeN =N*sizeof(float);
 	size_t flsize3N =3*N*sizeof(float);
	size_t flsize3 =3*sizeof(float);
        size_t flsize6N=6*N*sizeof(float);

	int* validPoints=(int*)malloc(flsizeN);
    vertex* verProjD=(vertex*)malloc(flsize3N);
	vertex* verNormD=(vertex*)malloc(flsize3N);
	float* matRgPre=(float*)malloc(9*sizeof(float));
	float* matTransltgPre=(float*)malloc(flsize3);

	vertex* verProS=(vertex*)malloc(flsize3N);
	vertex* verNormS=(vertex*)malloc(flsize3N);

	float* Energy=(float*)malloc(flsizeN);
	mata* Ak=(mata*)malloc(flsize6N);
	float* Bk=(float*)malloc(flsizeN);
	int* flag=(int*)malloc(flsizeN);

	if (validPoints == NULL || verProjD == NULL || verNormD == NULL || 
        matRgPre == NULL||matTransltgPre==NULL||verProS==NULL||verNormS==NULL
       ||Energy==NULL||Ak==NULL||Bk==NULL||flag==NULL)
      {
       fprintf(stderr, "Failed to allocate host vectors!\n");
       exit(EXIT_FAILURE);
      }
      
      //-----------------------------------
      //allocation for device
      //-----------------------------------
        cudaError_t err=cudaSuccess;
         
        int* d_validPoints=NULL;
        err=cudaMalloc((void **)&d_validPoints,flsizeN);
        
        if (err != cudaSuccess)
       {
        fprintf(stderr, "Failed to allocate device vector d_validPoints (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }


	vertex* d_verProjD=NULL;
        err=cudaMalloc((void **)&d_verProjD,flsize3N);
        
        if (err != cudaSuccess)
       {
        fprintf(stderr, "Failed to allocate device d_verProjD (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }


	vertex* d_verNormD=NULL;
        err=cudaMalloc((void **)&d_verNormD,flsize3N);
        
        if (err != cudaSuccess)
       {
        fprintf(stderr, "Failed to allocate device d_verNormD (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }

	float* d_matRgPre=NULL;
        err=cudaMalloc((void **)&d_matRgPre,9*sizeof(float));
        
        if (err != cudaSuccess)
       {
        fprintf(stderr, "Failed to allocate device d_matRgPre (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }


        float* d_matTransltgPre=NULL;
        err=cudaMalloc((void **)&d_matTransltgPre,flsize3);
        
        if (err != cudaSuccess)
       {
        fprintf(stderr, "Failed to allocate device d_matTransltgPre (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }

	
	vertex* d_verProS=NULL;
        err=cudaMalloc((void **)&d_verProS,flsize3N);
        
        if (err != cudaSuccess)
       {
        fprintf(stderr, "Failed to allocate device d_verProS (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }

   	vertex* d_verNormS=NULL;
        err=cudaMalloc((void **)&d_verNormS,flsize3N);
        
        if (err != cudaSuccess)
       {
        fprintf(stderr, "Failed to allocate device d_verNormS (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }

	
	float* d_Energy=NULL;
        err=cudaMalloc((void **)&d_Energy,flsizeN);
        
        if (err != cudaSuccess)
       {
        fprintf(stderr, "Failed to allocate device d_Energy (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }


	mata* d_Ak=NULL;
        err=cudaMalloc((void **)&d_Ak,flsize6N);
        
        if (err != cudaSuccess)
       {
        fprintf(stderr, "Failed to allocate device d_Ak (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }

	
	float* d_Bk=NULL;
        err=cudaMalloc((void **)&d_Bk,flsizeN);
        
        if (err != cudaSuccess)
       {
        fprintf(stderr, "Failed to allocate device d_Bk (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }

	int* d_flag=NULL;
        err=cudaMalloc((void **)&d_flag,flsizeN);
        
        if (err != cudaSuccess)
       {
        fprintf(stderr, "Failed to allocate device d_flag (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }

     //--------------------------------------------------------------------------
    // Copy the host input variables in host memory 
    //to the device input  variables in device memory
    //--------------------------------------------------------------------------
    err = cudaMemcpy(d_validPoints, validPoints, flsizeN,cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variables from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_verProjD, verProjD, flsize3N,cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variables from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_verNormD, verNormD, flsize3N,cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variables from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_matRgPre, matRgPre, 9*sizeof(float),cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variables from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_matTransltgPre, matTransltgPre, flsize3,cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variables from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_verProS, verProS, flsize3N,cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variables from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_verNormS, verNormS, flsize3N,cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variables from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //output ----------------------


    

   dim3 threadsPerBlock(blocksizex,blocksizey);
   dim3 numBlocks(N/threadsPerBlock.x+1,N/threadsPerBlock.y+1);

   RunICP<<<numBlocks,threadsPerBlock>>>(
   d_validPoints,
   d_verProjD,
   d_verNormD,
   d_matRgPre,
   d_matTransltgPre,
   focus,
   d_verProS,
   d_verNormS,
   epsD,
   epsN,
   epsDs,
   d_Energy,
   d_Ak,
   d_Bk,
   d_flag
   );

   //--------------------------------------------------------------------------
    // Copy the device result variables in device memory to the host result variables
    // in host memory.
    //--------------------------------------------------------------------------
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(Energy, d_Energy, flsizeN, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variable cube from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
	
    err = cudaMemcpy(Ak, d_Ak, flsize6N, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variable cube from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(Bk, d_Bk, flsizeN, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variable cube from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(flag, d_flag, flsizeN, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy variable cube from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    int sumEnergy=0;
     for(int i=0;i<N;i++)
     {
      if(flag[i]==1)
      {
      
       sumEnergy+=Energy[i];
      //Ak.push_back(d_Ak[i]);

      //Bk.push_back(d_Bk[i]);

      }

     }
}
