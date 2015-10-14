/*
 *
 * Copyright (c) 2012, Neurasmus B.V., The Netherlands,
 * web: www.neurasmus.com email: info@neurasmus.com
 * Copyright (c) 2013, Computer Engineering lab, TU Delft, The Netherlands,
 * web: www.ce.ewi.tudelft.nl
 * Any use or reproduction in whole or in parts is prohibited
 * without the written consent of the copyright owner.
 *
 * All Rights Reserved.
 *
 * Implementation in C by Sebastian Isaza
 * Author: Sebastian Isaza
 * Created: 19-01-2012
 * Modified: 07-08-2012
 *
 * Implementation in CUDA by Du Nguyen Hoang Anh
 * Author: Du Nguyen Hoang Anh
 * Created: 19-04-2013
 * Modified: 09-11-2013
 * Description: Top source file of the Inferior Olive model, originally written
 * in Matlab by Jornt De Gruijl. It contains the implementation of all functions.
 * The main function allocates the necessary memory, initializes the system
 * state and runs the model calculations.
 *
 */


#ifndef MAIN_H_
#define MAIN_H_
/*** MACROS ***/
#define RAND_INIT 0 // make it zero to facilitate debugging
#define SIMTIME 6000 // in ms, for when no input file is provided
//IO network size is IO_NETWORK_DIM1*IO_NETWORK_DIM2
#define IO_NETWORK_DIM1 8
#define IO_NETWORK_DIM2 6
#define IO_NETWORK_SIZE IO_NETWORK_DIM1*IO_NETWORK_DIM2

#define IAPP_MAX_CHARS 6 //2 integer, the dot, 2 decimals and the delimiter

// Cell properties
#define DELTA 0.05
//Conductance for neighbors' coupling
#define CONDUCTANCE 0.04
// Capacitance
#define C_M 1
// Somatic conductances (mS/cm2)
#define G_NA_S 150      // Na gate conductance (=90 in Schweighofer code, 70 in paper) 120 too little
#define G_KDR_S 9.0    // K delayed rectifier gate conductance (alternative value: 18)
#define G_K_S 5      // Voltage-dependent (fast) potassium
#define G_LS 0.016  // Leak conductance (0.015)
// Dendritic conductances (mS/cm2)
#define G_K_CA 35       // Potassium gate conductance (35)
#define G_CAH 4.5     // High-threshold Ca gate conductance (4.5)
#define G_LD 0.016   // Dendrite leak conductance (0.015)
#define G_H 0.125    // H current gate conductance (1.5) (0.15 in SCHWEIGHOFER 2004)
// Axon hillock conductances (mS/cm2)
#define G_NA_A 240      // Na gate conductance (according to literature: 100 to 200 times as big as somatic conductance)
#define G_NA_R 0      // Na (resurgent) gate conductance
#define G_K_A 20      // K voltage-dependent
#define G_LA 0.016  // Leak conductance
// Cell morphology
#define P1 0.25        // Cell surface ratio soma/dendrite (0.2)
#define P2 0.15      // Cell surface ratio axon(hillock)/soma (0.1)
#define G_INT 0.13       // Cell internal conductance (0.13)
// Reversal potentials
#define V_NA 55       // Na reversal potential (55)
#define V_K -75       // K reversal potential
#define V_CA 120       // Ca reversal potential (120)
#define V_H -43       // H current reversal potential
#define V_L 10       // leak current


/*** TYPEDEFS AND STRUCTS***/
//typedef double mod_prec;
//typedef float mod_prec;
#define BLOCKSIZEX 8
#define BLOCKSIZEY 6

#define STATE_SIZE 19
//#define CELL_STATE_SIZE 27
#define PARAM_SIZE 27
#define LOCAL_PARAM_SIZE 54
#define STATEADD 8
#define VNEIGHSTARTADD 1
#define PREVSTATESTARTADD 16
#define NEXTSTATESTARTADD 35
#define DEND_V 0
#define DEND_H 1
#define DEND_CAL 2
#define DEND_P 3
#define DEND_I 4
#define DEND_CA2 5
#define SOMA_G 6
#define SOMA_V 7
#define SOMA_SM 8
#define SOMA_SH 9
#define SOMA_CK 10
#define SOMA_CL 11
#define SOMA_PN 12
#define SOMA_PP 13
#define SOMA_PXS 14
#define AXON_V 15
#define AXON_SM 16
#define AXON_SH 17
#define AXON_P 18

struct dend{
	double V_dend;
	double Hcurrent_q;
	double Calcium_r;
	double Potassium_s;
	double I_CaH;
	double Ca2Plus;
};

struct soma{
	double g_CaL;
	double V_soma;
	double Sodium_m;
	double Sodium_h;
	double Calcium_k;
	double Calcium_l;
	double Potassium_n;
	double Potassium_p;
	double Potassium_x_s;
};

struct axon{
	double V_axon;
	double Sodium_m_a;
	double Sodium_h_a;
	double Potassium_x_a;
};

typedef struct cellState{
	struct dend dend;
	struct soma soma;
	struct axon axon;
}cellState;

typedef struct cellCompParams{
	double iAppIn;
	double neighVdend[15];
	cellState *prevCellState;
	cellState *newCellState;
}cellCompParams;

typedef struct channelParams{
	double *v;
	double *prevComp1, *prevComp2;
	double *newComp1, *newComp2;
}channelParams;

typedef struct dendCurrVoltPrms{
	double *iApp;
	double iC;
	double *vDend;
	double *vSoma;
	double *q, *r, *s;
	double *newVDend;
	double *newI_CaH;
}dendCurrVoltPrms;

typedef struct somaCurrVoltPrms{
	double *g_CaL;
	double *vSoma;
	double *vDend;
	double *vAxon;
	double *k, *l, *m, *h, *n, *x_s;
	double *newVSoma;
}somaCurrVoltPrms;

typedef struct axonCurrVoltPrms{
	double *vSoma;
	double *vAxon;
	double *m_a, *h_a, *x_a;
	double *newVAxon;
}axonCurrVoltPrms;

/*** FUNCTION PROTOTYPES ***/
int ReadFileLine(char *, int, FILE *, double *);


#endif /* MAIN_H_ */
