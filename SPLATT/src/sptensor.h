#ifndef SPLATT_SPTENSOR_H
#define SPLATT_SPTENSOR_H


#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/

/**
* @brief Types of tensors supported by splatt.
*/
typedef enum
{
  SPLATT_3MODE,   /** Three-mode tensors. */
  SPLATT_NMODE,   /** Tensors of with arbitrary numbers of modes.
                      NOTE: support is minimal. */
} tt_type;


/**
* @brief The main data structure for representing sparse tensors in
*        coordinate format.
*/
typedef struct
{
  tt_type type;   /** Type of tensor represented */
  idx_t nmodes;   /** The number of modes in the tensor, denoted 'm'. */
  idx_t nnz;      /** The number of nonzeros in the tensor. */
  idx_t * dims;   /** An array containing the dimension of each mode. */
  idx_t ** ind;   /** An m x nnz matrix containing the coordinates of each
                      nonzero. The nth nnz is accessed via ind[0][n], ind[1][n],
                      ..., ind[m][n]. */
  val_t * vals;   /** An array containing the values of each nonzero. */
  int tiled;      /** Whether sptensor_t has been tiled. Used by ftensor_t. */

  idx_t * indmap[MAX_NMODES]; /** Maps local -> global indices. */
} sptensor_t;



/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "matrix.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


#define tt_read splatt_tt_read
/**
* @brief Load a sparse tensor from the file 'ifname'.
*
* @param ifname The file to read.
*
* @return A sparse tensor.
*/
sptensor_t * tt_read(
  char const * const ifname);


#define tt_alloc splatt_tt_alloc
/**
* @brief Allocate a sparse tensor.
*
* @param nnz The number of nonzeros to allocate.
* @param nmodes The number of modes to to allocate for.
*
* @return A pointer to the allocated tensor.
*/
sptensor_t * tt_alloc(
  idx_t const nnz,
  idx_t const nmodes);



#define tt_fill splatt_tt_fill
/**
* @brief Store inds/vals info into tt and determine dimension information.
*        NOTE: This only stores the pointers directly, does not allocate
*        memory!
*
* @param tt The tensor to fill.
* @param nnz The length of inds/vals.
* @param nmodes The number of modes in the tensor.
* @param inds A 2D array of indices. The x'th nnz is found at ind[0][x], ...
* @param vals An array of values.
*/
void tt_fill(
  sptensor_t * const tt,
  idx_t const nnz,
  idx_t const nmodes,
  idx_t ** const inds,
  val_t * const vals);



#define tt_get_slices splatt_tt_get_slices
/**
* @brief Return a list of unique slice ids found in mode m. Slice i will be
*        included if there is a nonzero in tt the mth index equal to i.
*
* @param tt The tensor to analyze.
* @param mode The mode to operate on.
* @param nunique The number of unique slices found.
*
* @return An array at least of size nunique containing the ids of each slice
*         found in tt.
*/
idx_t * tt_get_slices(
  sptensor_t const * const tt,
  idx_t const mode,
  idx_t * nunique);


#define tt_get_hist splatt_tt_get_hist
/**
* @brief Return a histogram counting nonzeros appearing in indices of a given
*        mode.
*
* @param tt The sparse tensor to make a histogram from.
* @param mode Which mode we are counting.
*
* @return An array of length tt->dims[m].
*/
idx_t * tt_get_hist(
  sptensor_t const * const tt,
  idx_t const mode);


#define tt_free splatt_tt_free
/**
* @brief Free the fields AND pointer of a tensor.
*
* @param tt The tensor to free. NOTE: the pointer will also be freed!
*/
void tt_free(
  sptensor_t * tt);


/**
* @brief Compute the density of a sparse tensor, defined by nnz/(I*J*K).
*
* @param tt The sparse tensor.
*
* @return The density of tt.
*/
double tt_density(
  sptensor_t const * const tt);

#define tt_remove_dups splatt_tt_remove_dups
/**
* @brief Remove the duplicate entries of a tensor. Duplicate values are
*        repeatedly averaged.
*
* @param tt The modified tensor to work on. NOTE: data structures are not
*           resized!
*
* @return The number of nonzeros removed.
*/
idx_t tt_remove_dups(
  sptensor_t * const tt);


#define tt_remove_empty splatt_tt_remove_empty
/**
* @brief Relabel tensor indices to remove empty slices. Local -> global mapping
*        is written to tt->indmap.
*
* @param tt The tensor to relabel.
*
* @return The number of empty slices removed.
*/
idx_t tt_remove_empty(
  sptensor_t * const tt);


#define tt_unfold splatt_tt_unfold
/**
* @brief Unfold a tensor to a sparse matrix in CSR format.
*
* @param tt The tensor to unfold.
* @param mode The mode unfolding to operate on.
*
* @return The unfolded tensor in CSR format. The matrix will be of dimension
*         dims[mode] x (dims[0] * dims[1] * ... * dims[mode-1] *
*         dims[mode+1] * ... * dims[m].
*/
spmatrix_t * tt_unfold(
  sptensor_t * const tt,
  idx_t const mode);


#define tt_normsq splatt_tt_normsq
/**
* @brief Calculate the Frobenius norm of tt, squared. This is the
*        sum-of-squares for all nonzero values.
*
* @param tv The tensor values to operate on.
*
* @return  The squared Frobenius norm.
*/
val_t tt_normsq(
  sptensor_t const * const tt);

#endif
