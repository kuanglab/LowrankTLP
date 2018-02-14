/**
* @file splatt.h
* @brief Entry point and main include of SPLATT API.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-10
*/



#ifndef SPLATT_SPLATT_H
#define SPLATT_SPLATT_H



/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#ifdef SPLATT_USE_MPI
#include <mpi.h>
#endif


/******************************************************************************
 * TYPES & MACROS
 *****************************************************************************/
#include "splatt/types.h"
#include "splatt/constants.h"
#include "splatt/structs.h"


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/
#include "splatt/api_csf.h"
#include "splatt/api_factorization.h"
#include "splatt/api_kernels.h"
#include "splatt/api_kruskal.h"
#include "splatt/api_mpi.h"
#include "splatt/api_options.h"
#include "splatt/api_version.h"

#endif
