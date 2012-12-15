#define dotPro(a,b) a(0)*b(0)+a(1)*b(1)+a(2)*b(2)

__global__ void RunICP(
int   validPoints[N],
float verProjD[N][3],
float verNormD[N][3],
float matRgPre[3],
float matTransltgPre[3],
const int focus,
float verProS[N][3],
float verNormS[N][3],
const float epsD,
const float epsN,
const float epsDs,
float Energy[N],
float Ak[N][6],
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
                       for(int k=1;k<3;k++)
			{
			vertexProj[k] = dotPro(verProjD[i], matRgPre)+matTransltgPre[k];
			
                        normalProj[k] = dotPro(verNormD[i], matRgPre);
			}

			//get index
			int x = (vertexProj[0] / vertexProj[2] * focus) + 320;

	                int y = (vertexProj[1] / vertexProj[2] * focus) + 240;

	                index=y*640 + x;

			if(index < 0 || index >= npts){

                                Energy[i]=0;
				flag[i]=0;
				return;

			}
			
                        for(k=0;k<3;k++){

			disVdiff[k]  = verProjS[index][k]-vertexProj[k];
			disNdiff[k]  = verNormS[index][k]-normalProj[k];

                        }

			disV  = dotPro(disVdiff, disVdiff);
			disN  = dotPro(disNdiff, disNdiff);

			if( disV < epsD && disN >= epsN && disV > epsDs){
				
				error=dotPro(disVdiff,verNormS[index]);

				Energy= error*error/10;//remember to reduce sumEngry

  				Ak[i][0]=vertexProj[1]*verNormS[index][2]-vertexProj[2]*verNormS[index][1];
                                Ak[i][1]=vertexProj[2]*verNormS[index][0]-vertexProj[0]*verNormS[index][2];
                                Ak[i][2]=vertexProj[0]*verNormS[index][1]-vertexProj[1]*verNormS[index][0];
                                
 				for(int k=0;k<3;k++)
                               {
                                
				Ak[i][k+3]=verNormS[index][k];

				Bk[i][k]=dotPro(verNormS[index],disVdiff);
				}
				flag[i]=1;
			}
                        else
                            flag[i]=0;

	     }
             else
             {flag[i]=0}
}
