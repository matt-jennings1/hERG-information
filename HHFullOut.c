#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cvode/cvode.h>
#include <cvode/cvode_dense.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_dense.h>
#include <sundials/sundials_types.h>
#include <nvector/nvector_serial.h>

/* Problem Constants */

#define NEQ   3               /* number of equations  */
#define C0    0.0     /* initial y components */
#define I0    0.0
#define O0    0.0
#define T0    0.0     /* initial time           */
#define T1    0.1     /* first output time      */
#define  TADD 0.1	/* time step */

/* Function to determine V as a function of t */
static double getV(realtype t, int PR, realtype* params) {
    FILE *VrefFile;
    double v;
    /* This shift is needed for simulated protocol to match the protocol recorded in experiment, which is shifted by 0.1ms as compared to the original input protocol. Consequently, each step is held for 0.1ms longer in this version of the protocol as compared to the input.*/
    double shift = 0.1;
    /* The number corresponding to the protocol to be simulated is passed at the front of the parameter values vector*/
    
    /* sine wave*/
    if (PR==0) {
        int l = floor(10*t);
        v = params[12+l];

    } if (PR==1) {
        int l = floor(10*t);
        v = params[12+l];

    } if (PR==2) {
        int l = floor(10*t);
        v = params[12+l];
    
    } if (PR==3) {
        int l = floor(10*t);
        v = params[12+l];
        
    } if (PR==4) {
        int l = floor(10*t);
        v = params[12+l];
        
    } if (PR==5) {
        int l = floor(10*t);
        v = params[12+l];

    } if (PR==6) {
        int l = floor(10*t);
        v = params[12+l];
        
    } if (PR==7) {
        double C[6] = {54, 26, 10, 0.007/(2*M_PI), 0.037/(2*M_PI), 0.19/(2*M_PI)};
        
        if (t>=0 && t<250+shift) {
            v = -80;
        }
        
        if (t>=250+shift && t<300+shift) {
            v = -120;
        }
        
        if (t>=300+shift && t<500+shift) {
            v = -80;
        }
        
        if (t>=500+shift && t<1500+shift) {
            v = 40;
        }
        
        
        if (t>=1500+shift && t<2000+shift) {
            v = -120;
        }
        if (t>=2000+shift && t<3000+shift) {
            v = -80;
        }
        if (t>=3000+shift && t<6500+shift) {
            v=-30+C[0]*(sin(2*M_PI*C[3]*(t-2500-shift))) + C[1]*(sin(2*M_PI*C[4]*(t-2500-shift))) + C[2]*(sin(2*M_PI*C[5]*(t-2500-shift)));
            
        }
        if (t>=6500+shift && t<7000+shift) {
            v=-120;
        }
        if (t>= 7000+shift && t<8000+shift) {
            v = -80;
        }
    } 
    
    return(v);
}
  
static double getiK(realtype t, realtype o, realtype GKr, int PR, realtype* params)
{
    realtype EK, iK, v;
    realtype F   = RCONST(96485.0);
    realtype T   = RCONST(294.65);
    realtype R   = RCONST(8314.0);
    realtype K_i = RCONST(130.0);
    realtype k_o = RCONST(4.0);

    EK = ((R*T)/F)*log(k_o/K_i);
    v = getV(t, PR, params);
    iK = GKr * o * (v - EK);

    return (iK);
}


static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data) 
{ 
    /*Ensures microscopic reversibility condition satisfied*/
    
    const double y1 = NV_Ith_S(y, 0);
    const double y2 = NV_Ith_S(y, 1);
    const double y3 = NV_Ith_S(y, 2);
    
    const double y4 = (1.0-y1-y2-y3);
    /* Model equations*/
    
    realtype *params;
    params = (double *) user_data;
    
    double P0 = params[0];
    double P1 = params[1];
    double P2 = params[2];
    double P3 = params[3];
    double P4 = params[4];
    double P5 = params[5];
    double P6 = params[6];
    double P7 = params[7];
    int PR = params[11];
    
    realtype v;
    v = getV(t, PR, params);
    
    const double k32 = P4*exp(P5*v);
    const double k23 = P6*exp(-P7*v);
    
    const double k43 = P0*exp(P1*v);
    const double k34 = P2*exp(-P3*v);
    
    const double k12 = k43;
    const double k21 = k34;
    
    const double k41 = k32;
    const double k14 = k23;
    
    
    NV_Ith_S(ydot, 0) = -k12*y1 + k21*y2 + k41*y4 - k14*y1;
    NV_Ith_S(ydot, 1) = -k23*y2 + k32*y3 + k12*y1 - k21*y2;
    NV_Ith_S(ydot, 2) = -k34*y3 + k43*y4 + k23*y2 - k32*y3;
    
    
    return 0;
}

/* Function to print the solver output, C I O, to the intermediate output file */

static void PrintOutput(realtype t, realtype v, realtype c, realtype i, realtype o, realtype ik, FILE *fptr)
{
    fprintf(fptr, "%0.4e %14.6e %14.6e %14.6e %14.6e %14.6e\n", t, v, c, i, o, ik);
    return;
}

int main() 
{
        
       /* Read parameters */
    /* First 8 values are P1--P8 and the ninth value is GKr. The tenth and eleventh values are RTOL and ATOL respectively. */

    FILE *paramsFile;
    paramsFile = fopen("input.txt", "r");
  
    /*read file into array, memory allocated for max possible size */
    realtype params[12+929016]; 
    int j;  
    for (j = 0; j < 11; j++)
    {
        fscanf(paramsFile, "%lf", &params[j]);
    }
    
    /* Parameters from input parameters */
    
    realtype GKr;

    GKr = params[8];

    realtype RTOL;

    RTOL = params[9];

    realtype ATOL;

    ATOL = params[10];
    
    /* Read protocol type */
                                           
    FILE *protocolFile;
    protocolFile = fopen("protocol.txt", "r");
    int PR;
    fscanf(protocolFile, "%d", &PR);
    params[11] = PR;
    fclose(protocolFile);
    
    FILE *VrefFile;

    if (PR==0) {
        VrefFile = fopen("protocols/PrX.txt", "r");
        realtype VprX[80000];
        int j;
        for (j = 0; j < 312000; j++) {
            fscanf(VrefFile, "%lf", &params[12+j]);
        }
        fclose(VrefFile);
    
    } if (PR==1) {
        VrefFile = fopen("protocols/Pr1.txt", "r");
        realtype Vpr1[312000];
        int j;
        for (j = 0; j < 312000; j++) {
            fscanf(VrefFile, "%lf", &params[12+j]);
        }
        fclose(VrefFile);
       
    } if (PR==2) {
        VrefFile = fopen("protocols/Pr2.txt", "r");
        realtype Vpr2[312000];
        int j;
        for (j = 0; j < 312000; j++) {
            fscanf(VrefFile, "%lf", &params[12+j]);
        }
        fclose(VrefFile);
        
    } if (PR==3) {
        VrefFile = fopen("protocols/Pr3.txt", "r");
        realtype Vpr3[578060];
        int j;
        for (j = 0; j < 578060; j++) {
            fscanf(VrefFile, "%lf", &params[12+j]);
        }
        fclose(VrefFile);
        
        
    } if (PR==4) {
        VrefFile = fopen("protocols/Pr4.txt", "r");
        realtype Vpr4[464096];
        int j;
        for (j = 0; j < 464096; j++) {
            fscanf(VrefFile, "%lf", &params[12+j]);
        }
        fclose(VrefFile);
        
        
    } if (PR==5) {
        VrefFile = fopen("protocols/Pr5.txt", "r");
        realtype Vpr5[929016];
        int j;
        for (j = 0; j < 929016; j++) {
            fscanf(VrefFile, "%lf", &params[12+j]);
        }
        fclose(VrefFile);
        

    } if (PR==6) {
        VrefFile = fopen("protocols/Pr6.txt", "r");
        realtype Vpr6[88245];
        int j;
        for (j = 0; j < 88245; j++) {
            fscanf(VrefFile, "%lf", &params[12+j]);
        }
        fclose(VrefFile);
        
        
    } if (PR==7) {
        }
    
    

    /* Output file */
    FILE *fptr = fopen("hh.out", "w");
    if (fptr == NULL)
    {
        printf("Could not open file");
        return 0;
    }
    
    N_Vector ydot;
    realtype t;
    N_Vector y;
    
    realtype v, c, i, o, ik;
    
    /* Define tolerances to be used*/
    N_Vector abstol = NULL;
    
    abstol = N_VNew_Serial(3);
    
    NV_Ith_S(abstol, 0) = 1e-8;
    NV_Ith_S(abstol, 1) = 1e-8;
    NV_Ith_S(abstol, 2) = 1e-8;
    realtype reltol = 1e-8;
    
    /* Set up CVode*/
    int flag, k;
    N_Vector y0 = NULL;
    void* cvode_mem = NULL;
    y0 = N_VNew_Serial(3);
    NV_Ith_S(y0, 0) = C0;
    NV_Ith_S(y0, 1) = I0;
    NV_Ith_S(y0, 2) = O0;
    
    /* Solver options*/
    cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
    
    /* Initialize the memory */
    flag = CVodeSetUserData(cvode_mem, params);
    
    flag = CVodeInit(cvode_mem, &f, T0, y0);
    
    /* Set tolerances and maximum step*/
    flag = CVodeSVtolerances(cvode_mem, reltol, abstol);
    flag = CVDense(cvode_mem, NEQ);
    flag = CVodeSetMaxStep(cvode_mem, 0.1);
    
    
    int iout = 0;  realtype tout = T1;
    while(1)
    {
        if (CVode(cvode_mem, tout, y0, &t, CV_NORMAL) < 0) 
        {
            fprintf(stderr, "Error in CVode: %d\n", flag);
            break;
        }
        
        /*Probability of being in open state is equal to 1-probability of being in any other state*/
        c = NV_Ith_S(y0, 0);
        i = NV_Ith_S(y0, 1);
        o = NV_Ith_S(y0, 2);
        
        v = getV(t, PR, params);
        ik = getiK(t, o, GKr, PR, params);
        
        PrintOutput(t, v, c, i, o, ik, fptr);
        
        iout++;
        tout += TADD;
        
        if (PR==0) {
        int NOUT = 80000;
        if (iout == NOUT) break;

	} if (PR==1) {
        int NOUT = 312000;
        if (iout == NOUT) break;
        
        } if (PR==2) {
        int NOUT = 312000;
        if (iout == NOUT) break;
        
        } if (PR==3) {
        int NOUT = 578060;
        if (iout == NOUT) break;
        
        } if (PR==4) {
        int NOUT = 464096;
        if (iout == NOUT) break;
        
        } if (PR==5) {
        int NOUT = 929016;
        if (iout == NOUT) break;
        
        } if (PR==6) {
        int NOUT = 88245;
        if (iout == NOUT) break;
        
        } if (PR==7) {
        int NOUT = 80000;
        if (iout == NOUT) break;
        } 
    }
    
    /*Free memory*/
    N_VDestroy_Serial(y0);
    CVodeFree(&cvode_mem);
    
    return 1;
    }
