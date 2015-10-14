/*
 *
 * Copyright (c) 2012, Neurasmus B.V., The Netherlands,
 * web: www.neurasmus.com email: info@neurasmus.com
 *
 * Any use or reproduction in whole or in parts is prohibited
 * without the written consent of the copyright owner.
 *
 * All Rights Reserved.
 *
 *
 * Author: Sebastian Isaza
 * Created: 19-01-2012
 * Modified: 07-08-2012
 *
 * Description: Top source file of the Inferior Olive model, originally written
 * in Matlab by Jornt De Gruijl. It contains the implementation of all functions.
 * The main function allocates the necessary memory, initializes the system
 * state and runs the model calculations.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include "infoli.h"

typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp ()
{
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

int main(int argc, char *argv[]){

    char *inFileName;
    char *outFileName = "InferiorOlive_Output.txt";
    FILE *pInFile;
    FILE *pOutFile;
    char *iAppBuf;
    const int iAppBufSize =  IAPP_MAX_CHARS*IO_NETWORK_DIM1*IO_NETWORK_DIM2+1;
    mod_prec iAppArray[IO_NETWORK_SIZE];
    int i, j, k, p, q, n;
    int simSteps = 0;
    int simTime = 0;
    int inputFromFile = 0;
    int initSteps;
    cellState ***cellStatePtr;
    cellCompParams **cellCompParamsPtr;
    int seedvar;
    char temp[100];//warning: this buffer may overflow
    mod_prec iApp;
    timestamp_t t0, t1, secs;
    //double secs;

    printf("Inferior Olive Model (%d x %d cell mesh)\n", IO_NETWORK_DIM1, IO_NETWORK_DIM2);

    //Process command line arguments
    if(argc == 1){
        inputFromFile = 0;
        printf("Warning: No input file has been specified. A one-pulse input will be used.\n");
    }else if(argc == 2){
        inputFromFile = 1;
        inFileName = argv[1];//comment out for a hardcoded name
        pInFile = fopen(inFileName,"r");
        if(pInFile==NULL){
            printf("Error: Couldn't open %s\n", inFileName);
            exit(EXIT_FAILURE);
        }
    }else{
        printf("Error: Too many arguments.\nUsage: ./InferiorOlive <Iapp_input_file> or ./InferiorOlive\n");
        exit(EXIT_FAILURE);
    }

    //Open output file
    pOutFile = fopen(outFileName,"w");
    if(pOutFile==NULL){
        printf("Error: Couldn't create %s\n", outFileName);
        exit(EXIT_FAILURE);
    }
    sprintf(temp, "#simSteps Time(ms) Input(Iapp) Output(V_axon)\n");
    fputs(temp, pOutFile);

    //Malloc for iAppBuffer holding iApp arrays, one 2D array (a single line in the file though) at the time
    printf("Malloc'ing memory...\n");
    printf("iAppBuf: %dB\n", iAppBufSize);
    iAppBuf = (char *)malloc(iAppBufSize);
    if(iAppBuf==NULL){
        printf("Error: Couldn't malloc for iAppBuf\n");
        exit(EXIT_FAILURE);
    }
    //Malloc for the array of cellStates and cellCompParams
    printf("cellStatePtr: %dB\n", 2*IO_NETWORK_SIZE*sizeof(cellState));
    //Two cell state structs are needed so as to avoid having to synchronize all consumers before they start rewriting the cell state.
    cellStatePtr = malloc(2*sizeof(cellState *));//current and next state
    if(cellStatePtr==NULL){
        printf("Error: Couldn't malloc for cellStatePtr\n");
        exit(EXIT_FAILURE);
    }
    cellStatePtr[0] = malloc(IO_NETWORK_DIM1*sizeof(cellState *));
    if(cellStatePtr[0]==NULL){
        printf("Error: Couldn't malloc for cellStatePtr[0]\n");
        exit(EXIT_FAILURE);
    }
    for(k=0;k<IO_NETWORK_DIM1;k++){
        cellStatePtr[0][k] = malloc(IO_NETWORK_DIM2*sizeof(cellState));
        if(cellStatePtr[0][k]==NULL){
            printf("Error: Couldn't malloc for cellStatePtr[0][k]\n");
            exit(EXIT_FAILURE);
        }
    }
    cellStatePtr[1] = malloc(IO_NETWORK_DIM1*sizeof(cellState));
    if(cellStatePtr[1]==NULL){
        printf("Error: Couldn't malloc for cellStatePt[1]r\n");
        exit(EXIT_FAILURE);
    }
    for(k=0;k<IO_NETWORK_DIM1;k++){
        cellStatePtr[1][k] = malloc(IO_NETWORK_DIM2*sizeof(cellState));
        if(cellStatePtr[1][k]==NULL){
            printf("Error: Couldn't malloc for cellStatePtr[1][k]\n");
            exit(EXIT_FAILURE);
        }
    }

    printf("cellCompParamsPtr: %dB\n", IO_NETWORK_SIZE*sizeof(cellCompParams));
    cellCompParamsPtr = malloc(IO_NETWORK_DIM1*sizeof(cellCompParams *));
    if(cellCompParamsPtr==NULL){
        printf("Error: Couldn't malloc for cellCompParamsPtr\n");
        exit(EXIT_FAILURE);
    }
    for(k=0;k<IO_NETWORK_DIM1;k++){
        cellCompParamsPtr[k] = malloc(IO_NETWORK_DIM2*sizeof(cellCompParams));
        if(cellCompParamsPtr[k]==NULL){
            printf("Error: Couldn't malloc for cellCompParamsPtr[k]\n");
            exit(EXIT_FAILURE);
        }
    }

    //Write initial state values
    InitState(cellStatePtr[0]);

    //Initialize g_CaL
    seedvar = 1;
    for(j=0;j<IO_NETWORK_DIM1;j++){
        for(k=0;k<IO_NETWORK_DIM2;k++){
            srand(seedvar++);   // use this for debugging, now there is difference
            cellStatePtr[1][j][k].soma.g_CaL = cellStatePtr[0][j][k].soma.g_CaL = 0.68;
            // Uncomment the next two lines to assign different soma conductances to each cell.
            //cellStatePtr[0][j][k].soma.g_CaL = 0.6+(0.2*(rand()%100)/100);
            //cellStatePtr[1][j][k].soma.g_CaL = cellStatePtr[0][j][k].soma.g_CaL;

        }
    }

    //Random initialization: put every cell in a different oscillation state
    if(RAND_INIT){
        seedvar=1;
        for(j=0;j<IO_NETWORK_DIM1;j++){
            for(k=0;k<IO_NETWORK_DIM2;k++){
                //Put each cell at a different random state
                //srand(time(NULL));//Initialize random seed - Too fast when called in a loop.
                srand(seedvar++);   // use this for debugging, now there is difference
                initSteps = rand()%(int)ceil(100/DELTA);
                initSteps = initSteps | 0x00000001;//make it odd, so that the final state is in prevCellState
                printf("%d iterations - ",initSteps);
                for(i=0;i<initSteps;i++){
                    //Arrange inputs
                    cellCompParamsPtr[j][k].iAppIn = 0;//No stimulus
                    cellCompParamsPtr[j][k].prevCellState = &cellStatePtr[i%2][j][k];
                    cellCompParamsPtr[j][k].newCellState = &cellStatePtr[(i%2)^1][j][k];
                    ComputeOneCell(&cellCompParamsPtr[j][k]);
                }
                printf("Random initialization of the cell states finished.\n");
            }
        }
    }

    t0 = get_timestamp();

    if(inputFromFile){
        simSteps = 0;
        //Read full lines until end of file. Every iteration (line) is one simulation step.
        while(ReadFileLine(iAppBuf, iAppBufSize, pInFile, iAppArray)){
            //Compute one sim step for all cells
            for(j=0;j<IO_NETWORK_DIM1;j++){
                for(k=0;k<IO_NETWORK_DIM2;k++){
                    //Compute one Cell...
                    //Arrange inputs
                    cellCompParamsPtr[j][k].iAppIn = iAppArray[j*IO_NETWORK_DIM1+k];
                    cellCompParamsPtr[j][k].prevCellState = &cellStatePtr[simSteps%2][j][k];
                    cellCompParamsPtr[j][k].newCellState = &cellStatePtr[(simSteps%2)^1][j][k];
                    ComputeOneCell(&cellCompParamsPtr[j][k]);
                    //Store results
                    sprintf(temp, "%d %.3f %.3f %.8f\n", simSteps, (float)simSteps/20000, cellCompParamsPtr[j][k].iAppIn, cellStatePtr[(simSteps%2)^1][j][k].axon.V_axon);
                    fputs(temp, pOutFile);
                }
            }
            simSteps++;
        }
    }else{
        simTime = SIMTIME; // in miliseconds
        simSteps = ceil(simTime/DELTA);
        for(i=0;i<simSteps;i++){
            //Compute one sim step for all cells
            //printf("simSteps: %d\n", i);
            if(i>20000-1 && i<20500-1){ iApp = 6;} // start @ 1 because skipping initial values
            else{ iApp = 0;}
            sprintf(temp, "%d %.2f %.1f ", i+1, i*0.05, iApp); // start @ 1 because skipping initial values
            fputs(temp, pOutFile);
            for(j=0;j<IO_NETWORK_DIM1;j++){
                for(k=0;k<IO_NETWORK_DIM2;k++){
                    //Get neighbors' voltage influence
/** ***************************************************************************************************************
 ********************BUGGED PART**********************************************************************************/
                    n = 0;
                    for(p=j-1;p<=j+1;p++){
                        for(q=k-1;q<=k+1;q++){
                            if(((p!=j)||(q!=k)) && ((p>=0)&&(q>=0)) && ((p<IO_NETWORK_DIM1)&&(q<IO_NETWORK_DIM2))){
                                cellCompParamsPtr[j][k].neighVdend[n++] = cellStatePtr[i%2][p][q].dend.V_dend;
			    }else if(p==j && q==k){ /** <<<<<<< THIS EXCEPTION FIXES THE BUG */
                                ;   // do nothing, this is the cell itself
                            }
                            else{
                                //store same V_dend so that Ic becomes zero by the subtraction
                                cellCompParamsPtr[j][k].neighVdend[n++] = cellStatePtr[i%2][j][k].dend.V_dend;
                            }
                        }
                    }
/** *****************END OF BUGGED PART****************************************************************************
 *****************************************************************************************************************/                    
                    //Hardcoded input pulse
                    //if(i>20000 && i<20500){ cellCompParamsPtr[j][k].iAppIn = 6;}
                    //else{ cellCompParamsPtr[j][k].iAppIn = 0;}
                    cellCompParamsPtr[j][k].iAppIn = iApp;
                    cellCompParamsPtr[j][k].prevCellState = &cellStatePtr[i%2][j][k];
                    cellCompParamsPtr[j][k].newCellState = &cellStatePtr[(i%2)^1][j][k];
                    //Compute one Cell...
                    ComputeOneCell(&cellCompParamsPtr[j][k]);
                    //Store results
                    //printf("V_dend, V_soma and V_axon at simStep %d are\t: %.8f\t %.8f\t%.8f\n", i, cellStatePtr[(i%2)^1][j][k].dend.V_dend, cellStatePtr[(i%2)^1][j][k].soma.V_soma, cellStatePtr[(i%2)^1][j][k].axon.V_axon);
                    //sprintf(temp, "%d %.3f %.3f %.8f\n", i, (float)i/20000, cellCompParamsPtr[j][k].iAppIn, cellStatePtr[(i%2)^1][j][k].axon.V_axon);
                    sprintf(temp, "%.8f ", cellStatePtr[(i%2)^1][j][k].axon.V_axon);
                    fputs(temp, pOutFile);
                }
            }
            sprintf(temp, "\n");
            fputs(temp, pOutFile);
        }
    }

    t1 = get_timestamp();
    secs = (t1 - t0);// / 1000000;
    printf("%d ms of brain time in %d simulation steps\n", simTime, simSteps);
    printf(" %lld usecs real time \n", secs);

    //Free up memory and close files
    free(cellStatePtr[0]);
    free(cellStatePtr[1]);
    free(cellStatePtr);
    free(cellCompParamsPtr);
    free(iAppBuf);
    fclose (pOutFile);
    if(inputFromFile){ fclose (pInFile);}

    return EXIT_SUCCESS;
}

void ComputeOneCell(cellCompParams *cellCompParamsPtr){

    //The three compartments can be computed concurrently but only across a single sim step
    CompDend(cellCompParamsPtr);
    CompSoma(cellCompParamsPtr);
    CompAxon(cellCompParamsPtr);

    return;
}

void CompDend(cellCompParams *cellCompParamsPtr){

    struct channelParams chPrms;
    struct dendCurrVoltPrms chComps;

    //printf("Dendrite ");

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->dend.V_dend;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->dend.Hcurrent_q;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->dend.Hcurrent_q;
    //Compute
    DendHCurr(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->dend.V_dend;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->dend.Calcium_r;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->dend.Calcium_r;
    //Compute
    DendCaCurr(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->dend.Potassium_s;
    chPrms.prevComp2 = &cellCompParamsPtr->prevCellState->dend.Ca2Plus;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->dend.Potassium_s;
    //Compute
    DendKCurr(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->dend.Ca2Plus;
    chPrms.prevComp2 = &cellCompParamsPtr->prevCellState->dend.I_CaH;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->dend.Ca2Plus;
    //Compute
    DendCal(&chPrms);

    chComps.iC = IcNeighbors(cellCompParamsPtr->neighVdend, cellCompParamsPtr->prevCellState->dend.V_dend);
    chComps.iApp = &cellCompParamsPtr->iAppIn;
    chComps.vDend = &cellCompParamsPtr->prevCellState->dend.V_dend;
    chComps.newVDend = &cellCompParamsPtr->newCellState->dend.V_dend;
    chComps.vSoma = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chComps.q = &cellCompParamsPtr->newCellState->dend.Hcurrent_q;
    chComps.r = &cellCompParamsPtr->newCellState->dend.Calcium_r;
    chComps.s = &cellCompParamsPtr->newCellState->dend.Potassium_s;
    chComps.newI_CaH = &cellCompParamsPtr->newCellState->dend.I_CaH;
    DendCurrVolt(&chComps);

    return;
}

void DendHCurr(struct channelParams *chPrms){

    mod_prec q_inf, tau_q, dq_dt, q_local;

    //Get inputs
    mod_prec prevV_dend = *chPrms->v;
    mod_prec prevHcurrent_q = *chPrms->prevComp1;

    // Update dendritic H current component
    q_inf = 1 /(1 + exp((prevV_dend + 80) / 4));
    tau_q = 1 /(exp(-0.086 * prevV_dend - 14.6) + exp(0.070 * prevV_dend - 1.87));
    dq_dt = (q_inf - prevHcurrent_q) / tau_q;
    q_local = DELTA * dq_dt + prevHcurrent_q;
    //Put result
    *chPrms->newComp1 = q_local;

    return;
}
void DendCaCurr(struct channelParams *chPrms){

    mod_prec alpha_r, beta_r, r_inf, tau_r, dr_dt, r_local;

    //Get inputs
    mod_prec prevV_dend = *chPrms->v;
    mod_prec prevCalcium_r = *chPrms->prevComp1;

    // Update dendritic high-threshold Ca current component
    alpha_r = 1.7 / (1 + exp( -(prevV_dend - 5) / 13.9));
    beta_r = 0.02 * (prevV_dend + 8.5) / (exp((prevV_dend + 8.5) / 5) - 1);
    r_inf = alpha_r / (alpha_r + beta_r);
    tau_r = 5 / (alpha_r + beta_r);
    dr_dt = (r_inf - prevCalcium_r) / tau_r;
    r_local = DELTA * dr_dt + prevCalcium_r;
    //Put result
    *chPrms->newComp1 = r_local;

    return;
}
void DendKCurr(struct channelParams *chPrms){

    mod_prec  alpha_s, beta_s, s_inf, tau_s, ds_dt, s_local;

    //Get inputs
    mod_prec prevPotassium_s = *chPrms->prevComp1;
    mod_prec prevCa2Plus = *chPrms->prevComp2;

    // Update dendritic Ca-dependent K current component
    alpha_s = min((0.00002*prevCa2Plus), 0.01);
    beta_s = 0.015;
    s_inf = alpha_s / (alpha_s + beta_s);
    tau_s = 1 / (alpha_s + beta_s);
    ds_dt = (s_inf - prevPotassium_s) / tau_s;
    s_local = DELTA * ds_dt + prevPotassium_s;
    //Put result
    *chPrms->newComp1 = s_local;

    return;
}
//Consider merging DendCal into DendKCurr since DendCal's output doesn't go to DendCurrVolt but to DendKCurr
void DendCal(struct channelParams *chPrms){

    mod_prec  dCa_dt, Ca2Plus_local;

    //Get inputs
    mod_prec prevCa2Plus = *chPrms->prevComp1;
    mod_prec prevI_CaH = *chPrms->prevComp2;

    // update Calcium concentration
    dCa_dt = -3 * prevI_CaH - 0.075 * prevCa2Plus;
    Ca2Plus_local = DELTA * dCa_dt + prevCa2Plus;
    //Put result
    *chPrms->newComp1 = Ca2Plus_local;//This state value is read in DendKCurr

    return;
}

void DendCurrVolt(struct dendCurrVoltPrms *chComps){

    //Loca variables
    mod_prec I_sd, I_CaH, I_K_Ca, I_ld, I_h, dVd_dt;

    //Get inputs
    mod_prec I_c = chComps->iC;
    mod_prec I_app = *chComps->iApp;
    mod_prec prevV_dend = *chComps->vDend;
    mod_prec prevV_soma = *chComps->vSoma;
    mod_prec q = *chComps->q;
    mod_prec r = *chComps->r;
    mod_prec s = *chComps->s;

    // DENDRITIC CURRENTS

    // Soma-dendrite interaction current I_sd
    I_sd   = (G_INT / (1 - P1)) * (prevV_dend - prevV_soma);
    // Inward high-threshold Ca current I_CaH
    I_CaH  =  G_CAH * r * r * (prevV_dend - V_CA);
    // Outward Ca-dependent K current I_K_Ca
    I_K_Ca =  G_K_CA * s * (prevV_dend - V_K);
    // Leakage current I_ld
    I_ld   =  G_LD * (prevV_dend - V_L);
    // Inward anomalous rectifier I_h
    I_h    =  G_H * q * (prevV_dend - V_H);

    dVd_dt = (-(I_CaH   + I_sd  + I_ld + I_K_Ca + I_c + I_h) + I_app) / C_M;

    //Put result (update V_dend)
    *chComps->newVDend = DELTA * dVd_dt + prevV_dend;
    *chComps->newI_CaH = I_CaH;//This is a state value read in DendCal
    return;
}
mod_prec IcNeighbors(mod_prec *neighVdend, mod_prec prevV_dend){

    int i;
    mod_prec f, V, I_c;
    //printf("Ic[0]= %f\n", neighVdend[0]);

    I_c = 0;
    for(i=0;i<8;i++){
        V = prevV_dend - neighVdend[i];
        f = 0.8 * exp(-1*pow(V, 2)/100) + 0.2;    // SCHWEIGHOFER 2004 VERSION
        I_c = I_c + (CONDUCTANCE * f * V);
    }

    return I_c;
}

void CompSoma(cellCompParams *cellCompParamsPtr){

    struct channelParams chPrms;
    struct somaCurrVoltPrms chComps;

    // update somatic components
    // SCHWEIGHOFER:

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->soma.Calcium_k;
    chPrms.prevComp2 = &cellCompParamsPtr->prevCellState->soma.Calcium_l;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->soma.Calcium_k;
    chPrms.newComp2 = &cellCompParamsPtr->newCellState->soma.Calcium_l;
    //Compute
    SomaCalcium(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->soma.Sodium_m;
    chPrms.prevComp2 = &cellCompParamsPtr->prevCellState->soma.Sodium_h;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->soma.Sodium_m;
    chPrms.newComp2 = &cellCompParamsPtr->newCellState->soma.Sodium_h;
    //Compute
    SomaSodium(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->soma.Potassium_n;
    chPrms.prevComp2 = &cellCompParamsPtr->prevCellState->soma.Potassium_p;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->soma.Potassium_n;
    chPrms.newComp2 = &cellCompParamsPtr->newCellState->soma.Potassium_p;
    //Compute
    SomaPotassium(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->soma.Potassium_x_s;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->soma.Potassium_x_s;
    //Compute
    SomaPotassiumX(&chPrms);

    chComps.g_CaL = &cellCompParamsPtr->prevCellState->soma.g_CaL;
    chComps.vDend = &cellCompParamsPtr->prevCellState->dend.V_dend;
    chComps.vSoma = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chComps.newVSoma = &cellCompParamsPtr->newCellState->soma.V_soma;
    chComps.vAxon = &cellCompParamsPtr->prevCellState->axon.V_axon;
    chComps.k = &cellCompParamsPtr->newCellState->soma.Calcium_k;
    chComps.l = &cellCompParamsPtr->newCellState->soma.Calcium_l;
    chComps.m = &cellCompParamsPtr->newCellState->soma.Sodium_m;
    chComps.h = &cellCompParamsPtr->newCellState->soma.Sodium_h;
    chComps.n = &cellCompParamsPtr->newCellState->soma.Potassium_n;
    chComps.x_s = &cellCompParamsPtr->newCellState->soma.Potassium_x_s;
    SomaCurrVolt(&chComps);

    return;
}
void SomaCalcium(struct channelParams *chPrms){

    mod_prec k_inf, l_inf, tau_k, tau_l, dk_dt, dl_dt, k_local, l_local;

    //Get inputs
    mod_prec prevV_soma = *chPrms->v;
    mod_prec prevCalcium_k = *chPrms->prevComp1;
    mod_prec prevCalcium_l = *chPrms->prevComp2;

    k_inf = (1 / (1 + exp(-1 * (prevV_soma + 61)   / 4.2)));
    l_inf = (1 / (1 + exp((     prevV_soma + 85.5) / 8.5)));
    tau_k = 1;
    tau_l = ((20 * exp((prevV_soma + 160) / 30) / (1 + exp((prevV_soma + 84) / 7.3))) +35);
    dk_dt = (k_inf - prevCalcium_k) / tau_k;
    dl_dt = (l_inf - prevCalcium_l) / tau_l;
    k_local = DELTA * dk_dt + prevCalcium_k;
    l_local = DELTA * dl_dt + prevCalcium_l;
    //Put result
    *chPrms->newComp1= k_local;
    *chPrms->newComp2= l_local;

    return;
}
void SomaSodium(struct channelParams *chPrms){

    mod_prec m_inf, h_inf, tau_h, dh_dt, m_local, h_local;

    //Get inputs
    mod_prec prevV_soma = *chPrms->v;
    //mod_prec prevSodium_m = *chPrms->prevComp1;
    mod_prec prevSodium_h = *chPrms->prevComp2;

    // RAT THALAMOCORTICAL SODIUM:
    m_inf   = 1 / (1 + (exp((-30 - prevV_soma)/ 5.5)));
    h_inf   = 1 / (1 + (exp((-70 - prevV_soma)/-5.8)));
    tau_h   =       3 * exp((-40 - prevV_soma)/33);
    dh_dt   = (h_inf - prevSodium_h)/tau_h;
    m_local       = m_inf;
    h_local       = prevSodium_h + DELTA * dh_dt;
    //Put result
    *chPrms->newComp1 = m_local;
    *chPrms->newComp2 = h_local;

    return;
}
void SomaPotassium(struct channelParams *chPrms){

    mod_prec n_inf, p_inf, tau_n, tau_p, dn_dt, dp_dt, n_local, p_local;

    //Get inputs
    mod_prec prevV_soma = *chPrms->v;
    mod_prec prevPotassium_n = *chPrms->prevComp1;
    mod_prec prevPotassium_p = *chPrms->prevComp2;

    // NEOCORTICAL
    n_inf = 1 / (1 + exp( ( -3 - prevV_soma) /  10));
    p_inf = 1 / (1 + exp( (-51 - prevV_soma) / -12));
    tau_n =   5 + (  47 * exp( -(-50 - prevV_soma) /  900));
    tau_p = tau_n;
    dn_dt = (n_inf - prevPotassium_n) / tau_n;
    dp_dt = (p_inf - prevPotassium_p) / tau_p;
    n_local = DELTA * dn_dt + prevPotassium_n;
    p_local = DELTA * dp_dt + prevPotassium_p;
    //Put result
    *chPrms->newComp1 = n_local;
    *chPrms->newComp2 = p_local;

    return;
}
void SomaPotassiumX(struct channelParams *chPrms){

    mod_prec alpha_x_s, beta_x_s, x_inf_s, tau_x_s, dx_dt_s, x_s_local;

    //Get inputs
    mod_prec prevV_soma = *chPrms->v;
    mod_prec prevPotassium_x_s = *chPrms->prevComp1;

    // Voltage-dependent (fast) potassium
    alpha_x_s = 0.13 * (prevV_soma + 25) / (1 - exp(-(prevV_soma + 25) / 10));
    beta_x_s  = 1.69 * exp(-0.0125 * (prevV_soma + 35));
    x_inf_s   = alpha_x_s / (alpha_x_s + beta_x_s);
    tau_x_s   =         1 / (alpha_x_s + beta_x_s);
    dx_dt_s   = (x_inf_s - prevPotassium_x_s) / tau_x_s;
    x_s_local       = 0.05 * dx_dt_s + prevPotassium_x_s;
    //Put result
    *chPrms->newComp1 = x_s_local;

    return;
}
void SomaCurrVolt(struct somaCurrVoltPrms *chComps){

    //Local variables
    mod_prec I_ds, I_CaL, I_Na_s, I_ls, I_Kdr_s, I_K_s, I_as, dVs_dt;

    //Get inputs
    mod_prec g_CaL = *chComps->g_CaL;
    mod_prec prevV_dend = *chComps->vDend;
    mod_prec prevV_soma = *chComps->vSoma;
    mod_prec prevV_axon = *chComps->vAxon;
    mod_prec k = *chComps->k;
    mod_prec l = *chComps->l;
    mod_prec m = *chComps->m;
    mod_prec h = *chComps->h;
    mod_prec n = *chComps->n;
    mod_prec x_s = *chComps->x_s;

    // SOMATIC CURRENTS

    // Dendrite-soma interaction current I_ds
    I_ds  = (G_INT / P1) * (prevV_soma - prevV_dend);
    // Inward low-threshold Ca current I_CaL
    I_CaL = g_CaL * k * k * k * l * (prevV_soma - V_CA); //k^3
    // Inward Na current I_Na_s
    I_Na_s  = G_NA_S * m * m * m * h * (prevV_soma - V_NA);
    // Leakage current I_ls
    I_ls  = G_LS * (prevV_soma - V_L);
    // Outward delayed potassium current I_Kdr
    I_Kdr_s = G_KDR_S * n * n * n * n * (prevV_soma - V_K); // SCHWEIGHOFER
    // I_K_s
    I_K_s   = G_K_S * pow(x_s, 4) * (prevV_soma - V_K);
    // Axon-soma interaction current I_as
    I_as    = (G_INT / (1 - P2)) * (prevV_soma - prevV_axon);

    dVs_dt = (-(I_CaL   + I_ds  + I_as + I_Na_s + I_ls   + I_Kdr_s + I_K_s)) / C_M;
    *chComps->newVSoma = DELTA * dVs_dt + prevV_soma;

    return;
}
void CompAxon(cellCompParams *cellCompParamsPtr){

    struct channelParams chPrms;
    struct axonCurrVoltPrms chComps;

    // update somatic components
    // SCHWEIGHOFER:

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->axon.V_axon;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->axon.Sodium_h_a;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->axon.Sodium_h_a;
    chPrms.newComp2 = &cellCompParamsPtr->newCellState->axon.Sodium_m_a;
    //Compute
    AxonSodium(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->axon.V_axon;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->axon.Potassium_x_a;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->axon.Potassium_x_a;
    //Compute
    AxonPotassium(&chPrms);

    //Get inputs
    chComps.vSoma = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chComps.vAxon = &cellCompParamsPtr->prevCellState->axon.V_axon;
    chComps.newVAxon = &cellCompParamsPtr->newCellState->axon.V_axon;
    chComps.m_a = &cellCompParamsPtr->newCellState->axon.Sodium_m_a;
    chComps.h_a = &cellCompParamsPtr->newCellState->axon.Sodium_h_a;
    chComps.x_a = &cellCompParamsPtr->newCellState->axon.Potassium_x_a;
    AxonCurrVolt(&chComps);

    return;
}

void AxonSodium(struct channelParams *chPrms){

    mod_prec m_inf_a, h_inf_a, tau_h_a, dh_dt_a, m_a_local, h_a_local;

    //Get inputs
    mod_prec prevV_axon = *chPrms->v;
    mod_prec prevSodium_h_a = *chPrms->prevComp1;

    // Update axonal Na components
    // NOTE: current has shortened inactivation to account for high
    // firing frequencies in axon hillock
    m_inf_a   = 1 / (1 + (exp((-30 - prevV_axon)/ 5.5)));
    h_inf_a   = 1 / (1 + (exp((-60 - prevV_axon)/-5.8)));
    tau_h_a   =     1.5 * exp((-40 - prevV_axon)/33);
    dh_dt_a   = (h_inf_a - prevSodium_h_a)/tau_h_a;
    m_a_local = m_inf_a;
    h_a_local = prevSodium_h_a + DELTA * dh_dt_a;
    //Put result
    *chPrms->newComp1 = h_a_local;
    *chPrms->newComp2 = m_a_local;

    return;
}
void AxonPotassium(struct channelParams *chPrms){

    mod_prec alpha_x_a, beta_x_a, x_inf_a, tau_x_a, dx_dt_a, x_a_local;

    //Get inputs
    mod_prec prevV_axon = *chPrms->v;
    mod_prec prevPotassium_x_a = *chPrms->prevComp1;

    // D'ANGELO 2001 -- Voltage-dependent potassium
    alpha_x_a = 0.13 * (prevV_axon + 25) / (1 - exp(-(prevV_axon + 25) / 10));
    beta_x_a  = 1.69 * exp(-0.0125 * (prevV_axon + 35));
    x_inf_a   = alpha_x_a / (alpha_x_a + beta_x_a);
    tau_x_a   =         1 / (alpha_x_a + beta_x_a);
    dx_dt_a   = (x_inf_a - prevPotassium_x_a) / tau_x_a;
    x_a_local = 0.05 * dx_dt_a + prevPotassium_x_a;
    //Put result
    *chPrms->newComp1 = x_a_local;

    return;
}
void AxonCurrVolt(struct axonCurrVoltPrms *chComps){

    //Local variable
    mod_prec I_Na_a, I_la, I_sa, I_K_a, dVa_dt;

    //Get inputs
    mod_prec prevV_soma = *chComps->vSoma;
    mod_prec prevV_axon = *chComps->vAxon;
    mod_prec m_a = *chComps->m_a;
    mod_prec h_a = *chComps->h_a;
    mod_prec x_a = *chComps->x_a;

    // AXONAL CURRENTS
    // Sodium
    I_Na_a  = G_NA_A  * m_a * m_a * m_a * h_a * (prevV_axon - V_NA);
    // Leak
    I_la    = G_LA    * (prevV_axon - V_L);
    // Soma-axon interaction current I_sa
    I_sa    = (G_INT / P2) * (prevV_axon - prevV_soma);
    // Potassium (transient)
    I_K_a   = G_K_A * pow(x_a, 4) * (prevV_axon - V_K);
    dVa_dt = (-(I_K_a + I_sa + I_la + I_Na_a)) / C_M;
    *chComps->newVAxon = DELTA * dVa_dt + prevV_axon;

    return;
}
void InitState(cellState **cellStatePtr){
    int j, k;
    cellState initState;
    //Initial dendritic parameters
    initState.dend.V_dend = -60;
    initState.dend.Calcium_r = 0.0112788;// High-threshold calcium
    initState.dend.Potassium_s = 0.0049291;// Calcium-dependent potassium
    initState.dend.Hcurrent_q = 0.0337836;// H current
    initState.dend.Ca2Plus = 3.7152;// Calcium concentration
    initState.dend.I_CaH   = 0.5;// High-threshold calcium current
    //Initial somatic parameters
    initState.soma.g_CaL = 0.68; //default arbitrary value but it should be randomized per cell
    initState.soma.V_soma = -60;
    initState.soma.Sodium_m = 1.0127807;// Sodium (artificial)
    initState.soma.Sodium_h = 0.3596066;
    initState.soma.Potassium_n = 0.2369847;// Potassium (delayed rectifier)
    initState.soma.Potassium_p = 0.2369847;
    initState.soma.Potassium_x_s = 0.1;// Potassium (voltage-dependent)
    initState.soma.Calcium_k = 0.7423159;// Low-threshold calcium
    initState.soma.Calcium_l = 0.0321349;
    // Initial axonal parameters
    initState.axon.V_axon = -60;
    //sisaza: Sodium_m_a doesn't have a state, therefore this assignment doesn'thave any effect
    initState.axon.Sodium_m_a = 0.003596066;// Sodium (thalamocortical)
    initState.axon.Sodium_h_a = 0.9;
    initState.axon.Potassium_x_a = 0.2369847;// Potassium (transient)

    //Copy init sate to all cell states
    for(j=0;j<IO_NETWORK_DIM1;j++){
        for(k=0;k<IO_NETWORK_DIM2;k++){
            memcpy(&cellStatePtr[j][k], &initState, sizeof(cellState));
        }
    }

    return;
}

int ReadFileLine(char *iAppBuf, int iAppBufSize, FILE *pInFile, mod_prec *iAppArray){
    //FIXME: make this function more robust
    char *strNumber;
    int i = 0;
    //Get one line
    if(fgets(iAppBuf, iAppBufSize, pInFile)){
        //Convert the ASCII string of one element to a double precision floating point value
        strNumber = strtok(iAppBuf," ");
        i = 0;
        //printf("Line:\n");
        while ((strNumber != NULL) && (i<IO_NETWORK_SIZE)){
            iAppArray[i] = atof(strNumber);//atof() should change if using integers or fixed point
            //printf ("(%s) %0.2f ", strNumber, iAppArray[i]);
            strNumber = strtok(NULL, " ");
            i++;
        }
        //printf("i: %d\n", i);
        if(i<IO_NETWORK_SIZE){
            //BUG: if only one element is missing but the line ends in a space, the error is not detected
            printf("Error: Input line doesn't have enough elements, only %d\n", i);
            exit(EXIT_FAILURE);
        }
        return 1;//success
    }else{
        if(!feof(pInFile)){
        printf("Error: Reading from input file didn't finish successfully\n");
        exit(EXIT_FAILURE);
        }
        return 0;//end of file
    }
}

inline mod_prec min(mod_prec a, mod_prec b){
    return (a < b) ? a : b;
}
