/**
* @file structs.h
* @brief Structures used by SPLATT.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-10
*/



#ifndef SPLATT_SPLATT_STRUCTS_H
#define SPLATT_SPLATT_STRUCTS_H



/******************************************************************************
 * DATA STRUCTURES
 *****************************************************************************/

/**
* @brief Kruskal tensors are the output of the CPD. Each mode of the tensor is
*        represented as a matrix with unit columns. Lambda is a vector whose
*        entries scale the columns of the matrix factors.
*/
typedef struct splatt_kruskal
{
  /** @brief The rank of the decomposition. */
  splatt_idx_t rank;

  /** @brief The row-major matrix factors for each mode. */
  splatt_val_t * factors[SPLATT_MAX_NMODES];

  /** @brief Length-for each column. */
  splatt_val_t * lambda;

  /** @brief The number of modes in the tensor. */
  splatt_idx_t nmodes;

  /** @brief The number of rows in each factor. */
  splatt_idx_t dims[SPLATT_MAX_NMODES];

  /** @brief The quality [0,1] of the CPD */
  double fit;
} splatt_kruskal;



/**
* @brief Struct describing a Tucker tensor, allocated and returned by
*        splatt_tucker_als.
*/
typedef struct splatt_tucker_t
{
  splatt_idx_t nmodes;                  /** Number of modes (i.e., factors[])*/
  splatt_idx_t rank[SPLATT_MAX_NMODES]; /** Number of columns in each factor */
  double fit;                           /** The quality [0,1] of the CPD */
  splatt_val_t * core;                  /** Core (dense) tensor */
  splatt_val_t * factors[SPLATT_MAX_NMODES];   /** Row-major matrices */
} splatt_tucker_t;




/**
* @brief The sparsity pattern of a CSF (sub-)tensor.
*/
typedef struct
{
  /** @brief The size of each fptr and fids array. */
  splatt_idx_t nfibs[SPLATT_MAX_NMODES];

  /** @brief The pointer structure for each sub-tree. fptr[f] marks the start
   *         of the children of node 'f'. This structure is a generalization of
   *         the 'rowptr' array used for CSR matrices. */
  splatt_idx_t * fptr[SPLATT_MAX_NMODES];

  /** @brief The index of each node. These map nodes back to the original
   *         tensor nonzeros. */
  splatt_idx_t * fids[SPLATT_MAX_NMODES];

  /** @brief The actual nonzero values. This array is of length
   *         nfibs[nmodes-1]. */
  splatt_val_t * vals;
} csf_sparsity;



/**
* @brief CSF tensors are the compressed storage format for performing fast
*        tensor computations in the SPLATT library.
*/
typedef struct splatt_csf
{
  /** @brief The number of nonzeros. */
  splatt_idx_t nnz;

  /** @brief The number of modes. */
  splatt_idx_t nmodes;

  /** @brief The dimension of each mode. */
  splatt_idx_t dims[SPLATT_MAX_NMODES];

  /** @brief The permutation of the tensor modes.
   *         dim_perm[0] is the mode stored at the root level and so on. */
  splatt_idx_t dim_perm[SPLATT_MAX_NMODES];

  /** @brief Which tiling scheme this tensor is stored as. */
  splatt_tile_type which_tile;

  /** @brief How many tiles there are. */
  splatt_idx_t ntiles;

  /** @brief For a dense tiling, how many tiles along each mode. */
  splatt_idx_t tile_dims[SPLATT_MAX_NMODES];

  /** @brief Sparsity structures -- one for each tile. */
  csf_sparsity * pt;
} splatt_csf;




/**
* @brief The well-known compressed sparse row format for matrices.
*/
typedef struct splatt_csr
{

  /** @brief The number of rows. */
  splatt_idx_t I;
  /** @brief The number of columns. */
  splatt_idx_t J;
  /** @brief The number of non-zeros. */
  splatt_idx_t nnz;

  /** @brief The start of each row, including row I+1. */
  splatt_idx_t * rowptr;
  /** @brief The column indices of each non-zero. */
  splatt_idx_t * colind;
  /** @brief The non-zero values. */
  splatt_val_t * vals;
} splatt_csr;



#endif
