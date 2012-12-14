#include <stdio.h>
#include <cuda_runtime.h>


//#define POS3(p) p[1]+size_x*p[2]+size_x*size_y*p[3]
#define POS3(i,j,k) k+size_z*j+size_z*size_y*i
#define POS(n,x,y) x*n+y
//location on Rk 
#define DMAP(a) (a[0]+map_center_x)*map_size_y+(a[1]+map_center_y)
// For the CUDA runtime routines (prefixed with "cuda_")

#define DOT(a,b) a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
#define DOTV(a,b) a.x*b.x+a.y*b.y+a.z*b.z

//---------------------------------------------------------------------------
// Some paramenters------------->Define them
//---------------------------------------------------------------------------
//----- paramenters
   #define mu 1.0f//trunction length
   #define weightOweight 1.0f // W_Rk =weightOweight * cos(theta)/R_k
   #define W_eta 100.0f //trunction weight
// Constant
//----- in cube
   //unit length
   #define unit 0.1f
   // Size of cube
   #define size_x 10
   #define size_y 10
   #define size_z 10
   // The index of original point
   #define center_x 5
   #define center_y 5
   #define center_z 5
//----- in depth map
   // Size of depth map-------Define it
   #define map_size_x 9
   #define map_size_y 9
   // The index of original point-------Define it
   #define map_center_x 4
   #define map_center_y 4


//debug
#define numElements size_x*size_y*size_z


//---------------------------vertex
 typedef struct 
 {
   float x;
   float y;
   float z;
 } vertex;

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
TSDF(float *cube, float *cube_w, float *cube_k, float *cube_wk, float *Rgk, float *tgk, float *Rk, vertex *NRk)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float a[3]={1,2,3};
    vertex b={.x=1,.y=2,.z=3};
    if (i < numElements )
    {
        cube[i] = cube[i] + cube[i];
        cube[i] = DOTV(b,b);
        
    }/*
    
            int i = 0;
            int j = 0;
            int k = 0;
            
            //p---?? unit
            float p[3], pt[3];
            int x[2];
            p[0] = (i - center_x) *unit;
            p[1] = (j - center_y) *unit;
            p[2] = (k - center_z) *unit;
            //printf("p[2]=%f\n",p[2]);
            
            //T_gk^-1 * p
            pt[0]=p[0]-tgk[0];
            pt[1]=p[1]-tgk[1];
            pt[2]=p[2]-tgk[2];
            
            for ( int s=0 ; s<3 ; s++  )
               for ( int t=0 ; t<3 ; t++  )
               {
                  p[s] += Rgk[ POS(3,t,s) ] * pt[t];
               }
            
            //first part of eta
            float eta=0.0f;
            eta=sqrt(pt[0]*pt[0]+pt[1]*pt[1]+pt[2]*pt[2]);
            
            //pi( K * (T_gk^-1 * p) )
            //round to nearest pixel
            for ( int s=0 ; s<2 ; s++  )
              x[s]=int( p[s] + p[2] + 0.5f );
              
            if(-1 < x[0] < map_size_x && -1< x[1] < map_size_y)
            {
               //lambd=norm(K^-1*x)
               float lambd = 0.0f;
               for ( int s=0 ; s<2 ; s++  )
                  lambd += x[s] * x[s];

               lambd=sqrt( lambd + 1.0f );
               

               //second part of eta
               eta=eta / lambd + Rk[DMAP(x)];
               
               
               //F_Rk(p)
               if (eta>=-mu)
                  cube_k[ POS3(i,j,k) ] = min(1.0f,eta/mu)*(-1);
               //W_Rk(p)
               cube_wk[ POS3(i,j,k) ] = NRk[ DMAP(x) ].x * x[0] + NRk[ DMAP(x) ].y * x[1] + NRk[ DMAP(x) ].z;
               cube_wk[ POS3(i,j,k) ] = weightOweight * cube_wk[ POS3(i,j,k) ] / lambd / Rk[DMAP(x)];
               //F_k(p) W_k(p)
               cube[ POS3(i,j,k) ] = cube_w[ POS3(i,j,k) ] * cube[ POS3(i,j,k) ] + cube_wk[ POS3(i,j,k) ] * cube_k[ POS3(i,j,k) ];
               cube_w[ POS3(i,j,k) ] = min( cube_w[ POS3(i,j,k) ]+cube_wk[ POS3(i,j,k)], W_eta);
               cube[ POS3(i,j,k) ] /= cube_w[ POS3(i,j,k) ]+cube_wk[ POS3(i,j,k)];
            }
        */          
}


 
 
 
 /**
 * Host main routine
 */
 
int 
main(void)
{
   //---------------------------------------------------------------------------
   // Input
   //---------------------------------------------------------------------------
   //T_g,k
   float Rgk[9]={0.99957997,-0.003303149,-0.01823758,-0.0149635,1.0000104,0.010601612,0.018084152,-0.010273285,0.99978584};
   float tgk[3]={10.8916,11.8622,-9.30357};

   //--R and N_Rk
   float *Rk=(float *)malloc(map_size_x*map_size_y*sizeof(float));
   vertex *NRk=(vertex *)malloc(map_size_x*map_size_y*sizeof(vertex));
   
   
   
   //---------------------------------------------------------------------------
   // Allocation
   //---------------------------------------------------------------------------
   size_t size = size_x*size_y*size_z*sizeof(float);
   // Allocat *cube* vectors on host --may not malloc???
   //Fk
   float *cube=(float *)malloc(size);

   //Wk
   float *cube_w=(float *)malloc(size);

   //F_Rk
   float *cube_k=(float *)malloc(size);//???????????????????????????? No use

   //W_Rk
   float *cube_wk=(float *)malloc(size);

   // Verify that allocations succeeded
   if (cube == NULL || cube_w == NULL || cube_k == NULL || cube_wk == NULL)
   {
       fprintf(stderr, "Failed to allocate host vectors!\n");
       exit(EXIT_FAILURE);
   }


   //---------------------------------------------------------------------------
    // Allocate the device input vector cube   
   //---------------------------------------------------------------------------
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    float *d_cube = NULL;
    err = cudaMalloc((void **)&d_cube, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector cube (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector cube_w
    float *d_cube_w = NULL;
    err = cudaMalloc((void **)&d_cube_w, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector cube_w (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector cube_k
    float *d_cube_k = NULL;
    err = cudaMalloc((void **)&d_cube_k, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector cube_k (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Allocate the device output vector cube_wk
    float *d_cube_wk = NULL;
    err = cudaMalloc((void **)&d_cube_wk, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector cube_wk (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Allocate the device output vector Rgk
    float *d_Rgk = NULL;
    err = cudaMalloc((void **)&d_Rgk, 9*sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector Rgk (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Allocate the device output vector tgk
    float *d_tgk = NULL;
    err = cudaMalloc((void **)&d_tgk, 3*sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector tgk (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Allocate the device output vector Rk
    float *d_Rk = NULL;
    err = cudaMalloc((void **)&d_Rk, map_size_x*map_size_y*sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector Rk (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Allocate the device output vector NRk
    vertex *d_NRk = NULL;
    err = cudaMalloc((void **)&d_NRk, map_size_x*map_size_y*sizeof(vertex));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector Rk (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    //--------------------------------------------------------------------------
    // Copy the host input vectors in host memory to the device input vectors in
    // device memory
    //--------------------------------------------------------------------------    
    err = cudaMemcpy(d_cube, cube, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector cube from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_cube_w, cube_w, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector cube_w from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(d_cube_k, cube_k, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector cube_k from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(d_cube_wk, cube_wk, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector cube_wk from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(d_Rgk, Rgk, 9*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector Rgk from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    

    err = cudaMemcpy(d_tgk, tgk, 3*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector tgk from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(d_Rk, Rk, map_size_x*map_size_y*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector Rk from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(d_NRk, NRk, map_size_x*map_size_y*sizeof(vertex), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector NRk from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    
    
    
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(size_x*size_y*size_z + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    TSDF<<<blocksPerGrid, threadsPerBlock>>>(d_cube, d_cube_w, d_cube_k, d_cube_wk, d_Rgk, d_tgk, d_Rk, d_NRk);
    err = cudaGetLastError();
   // cube cube_w cube_k cube_wk Rgk tgk Rk NRk

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //--------------------------------------------------------------------------
    // Wait to be Kernel
    //--------------------------------------------------------------------------
   // The main loop ----------->kernel
   for (int i = 0 ; i<size_x ;i++)
      for (int j = 0 ; j<size_y ;j++)
         for (int k = 0 ; k<size_z ;k++)
         {
            //p---?? unit
            float p[3], pt[3];
            int x[2];
            p[0] = (i - center_x) * unit;
            p[1] = (j - center_y) *unit;
            p[2] = (k - center_z) *unit;
            //printf("p[2]=%f\n",p[2]);
            
            //T_gk^-1 * p
            pt[0]=p[0]-tgk[0];
            pt[1]=p[1]-tgk[1];
            pt[2]=p[2]-tgk[2];
            
            for ( int s=0 ; s<3 ; s++  )
               for ( int t=0 ; t<3 ; t++  )
               {
                  p[s] += Rgk[ POS(3,t,s) ] * pt[t];
               }
            
            //first part of eta
            float eta=sqrt(pt[0]*pt[0]+pt[1]*pt[1]+pt[2]*pt[2]);
            
            //pi( K * (T_gk^-1 * p) )
            //round to nearest pixel
            for ( int s=0 ; s<2 ; s++  )
              x[s]=int( p[s] + p[2] + 0.5f );
              
            if(-1 < x[0] < map_size_x && -1< x[1] < map_size_y)
            {
               //lambd=norm(K^-1*x)
               float lambd = 0;
               for ( int s=0 ; s<2 ; s++  )
                  lambd += x[s] * x[s];

               lambd=sqrt( lambd + 1.0f );
               

               //second part of eta
               eta=eta / lambd + Rk[DMAP(x)];
               
               //printf("cube[ POS3(i,j,k) ]=%f\n",cube_w[ POS3(i,j,k) ]+cube_wk[ POS3(i,j,k)]);
               //F_Rk(p)
               if (eta>=-mu)
                  cube_k[ POS3(i,j,k) ] = min(1.0f,eta/mu)*(-1);
               
               //W_Rk(p)
               cube_wk[ POS3(i,j,k) ] = NRk[ DMAP(x) ].x * x[0] + NRk[ DMAP(x) ].y * x[1] + NRk[ DMAP(x) ].z;
               cube_wk[ POS3(i,j,k) ] = weightOweight * cube_wk[ POS3(i,j,k) ] / lambd / Rk[DMAP(x)];
               //F_k(p) W_k(p)
               cube[ POS3(i,j,k) ] = cube_w[ POS3(i,j,k) ] * cube[ POS3(i,j,k) ] + cube_wk[ POS3(i,j,k) ] * cube_k[ POS3(i,j,k) ];
               cube_w[ POS3(i,j,k) ] = min( cube_w[ POS3(i,j,k) ]+cube_wk[ POS3(i,j,k)], W_eta);
               cube[ POS3(i,j,k) ] /= cube_w[ POS3(i,j,k) ]+cube_wk[ POS3(i,j,k)];
               
            }
         }









    

    //--------------------------------------------------------------------------
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //--------------------------------------------------------------------------
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(cube, d_cube, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector cube from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    


    // Verify that the result vector is correct
    for (int i = 0; i < 10 ; ++i)
    {
      printf("cube[0]=%f\n",cube[0]);
    }
    

    // Free device global memory


    // Free host memory

    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}


