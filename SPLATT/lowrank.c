

#include "src/base.h"
#include "src/sptensor.h"
#include "src/util.h"
#include "src/csf.h"
#include "src/io.h"
#include "src/timer.h"
#include "src/mttkrp.h"

#include <stdio.h>
#include <stdlib.h>



static splatt_val_t * p_multi_dotprod(
    splatt_val_t * A,
    splatt_val_t * B,
    splatt_idx_t const nrows,
    splatt_idx_t const ncols)
{
  splatt_val_t * rhs = splatt_malloc(ncols * sizeof(*rhs));
  for(splatt_idx_t r=0; r < ncols; ++r) {
    rhs[r] = 0.;
  }

  #pragma omp parallel
  {
    splatt_val_t * restrict buffer = splatt_malloc(ncols * sizeof(*buffer));
    for(splatt_idx_t r=0; r < ncols; ++r) {
      buffer[r] = 0.;
    }

    #pragma omp for schedule(static) nowait
    for(splatt_idx_t i=0; i < nrows; ++i) {
      splatt_val_t const * const restrict A_row = A + (i * ncols);
      splatt_val_t const * const restrict B_row = B + (i * ncols);

      for(splatt_idx_t r=0; r < ncols; ++r) {
        buffer[r] += A_row[r] * B_row[r];
      }
    }

    /* now combine buffers */
    #pragma omp critical
    {
      for(splatt_idx_t r=0; r < ncols; ++r) {
        rhs[r] += buffer[r];
      }
    }

    splatt_free(buffer);
  } /* end omp parallel */

  return rhs;
}


static void p_fill_result(
    sptensor_t const * const tensor,
    splatt_val_t * * mats,
    splatt_val_t const * const restrict rhs,
    splatt_idx_t const rank,
    splatt_val_t * const restrict result)
{
  splatt_idx_t const nmodes = tensor->nmodes;

  #pragma omp parallel
  {
    splatt_val_t * buffer = splatt_malloc(rank * sizeof(*buffer));

    #pragma omp for schedule(static)
    for(splatt_idx_t x=0; x < tensor->nnz; ++x) {
      /* initialize buffer with rhs */
      for(splatt_idx_t r=0; r < rank; ++r) {
        buffer[r] = rhs[r];
      }

      /* now fill in reproduced value from mats (compute a row of Q * rhs) */
      for(splatt_idx_t m=0; m < nmodes; ++m) {
        splatt_idx_t const row_id = tensor->ind[m][x];
        splatt_val_t const * const restrict mat_row = mats[m] + (row_id*rank);

        for(splatt_idx_t r=0; r < rank; ++r) {
          buffer[r] *= mat_row[r];
        }
      }

      /* sum values into result[x] */
      result[x] = 0.;
      for(splatt_idx_t r=0; r < rank; ++r) {
        result[x] += buffer[r];
      }
    } /* foreach nnz */

    splatt_free(buffer);
  } /* end omp parallel */
}



void lowrank_kernel(
    sptensor_t const * const tensor,
    splatt_csf const * const csf,
    double const * const splatt_opts,
    splatt_val_t * * mats,
    splatt_val_t const * const eigs,
    splatt_idx_t const rank,
    splatt_val_t * const restrict result)
{
  idx_t skipped_mode = 0;
  if(tensor->nmodes > 2) {
    skipped_mode = csf->dim_perm[0];
  }

  splatt_val_t * mttkrp =
      splatt_malloc(tensor->dims[skipped_mode] * rank * sizeof(*mttkrp));

  /* First do MTTKRP with all matrices but the top-level mode. */
  if(tensor->nmodes > 2) {
    splatt_mttkrp(skipped_mode, rank, csf, mats, mttkrp, splatt_opts);

  /* revert to simpler streaming MTTKRP for matrices */
  } else {
    /* setup some internal data structures */
    matrix_t * mttkrp_mats[MAX_NMODES+1];
    for(idx_t m=0; m < tensor->nmodes; ++m) {
      mttkrp_mats[m] = splatt_malloc(sizeof(**mttkrp_mats));
      mttkrp_mats[m]->I = tensor->dims[m];
      mttkrp_mats[m]->J = rank;
      mttkrp_mats[m]->vals = mats[m];
    }
    mttkrp_mats[MAX_NMODES] = splatt_malloc(sizeof(**mttkrp_mats));
    mttkrp_mats[MAX_NMODES]->I = tensor->dims[skipped_mode];
    mttkrp_mats[MAX_NMODES]->J = rank;
    mttkrp_mats[MAX_NMODES]->vals = mttkrp;

    /* do the operation */
    mttkrp_stream(tensor, mttkrp_mats, skipped_mode);

    for(idx_t m=0; m < tensor->nmodes; ++m) {
      splatt_free(mttkrp_mats[m]);
    }
    splatt_free(mttkrp_mats[MAX_NMODES]);
  }

  /* Finish Q^T * Y_0, which is simply columnwise inner products. */
  splatt_val_t * restrict rhs = p_multi_dotprod(mats[skipped_mode], mttkrp,
      tensor->dims[skipped_mode], rank);

  /* no longer needed */
  splatt_free(mttkrp);

  /* now scale RHS result by all eigenvalues */
  #pragma omp parallel for schedule(static)
  for(idx_t f=0; f < rank; ++f) {
    rhs[f] *= eigs[f];
  }


#if 1
  /* write 'f' vector */
  FILE * fout = fopen("f.txt", "w");
  for(idx_t f=0; f < rank; ++f) {
    fprintf(fout, "%0.20f\n", rhs[f]);
  }
  fclose(fout);
#endif


  /* Finally, fill in result following the sparsity pattern of 'tensor' */
  p_fill_result(tensor, mats, rhs, rank, result);

  splatt_free(rhs);
}





int main(
    int argc,
    char ** argv)
{
  if(argc < 3) {
    fprintf(stderr, "usage: %s <tensor> <rank> <evecs_base> <eigs> [output]\n", argv[0]);
    return EXIT_FAILURE;
  }

  /* Read tensor */
  sptensor_t * tensor = tt_read(argv[1]);

  /* Allocate CSF */
  double * splatt_opts = splatt_default_opts();
  splatt_opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  splatt_csf * csf = splatt_csf_alloc(tensor, splatt_opts);

  splatt_idx_t const rank = atoi(argv[2]);

  /* Allocate matrices and read matrices */
  splatt_val_t * mats[SPLATT_MAX_NMODES];
  for(splatt_idx_t m=0; m < tensor->nmodes; ++m) {
    mats[m] = splatt_malloc(tensor->dims[m] * rank * sizeof(**mats));

    /* Initialize to 0 in parallel -- this can improve performance on NUMA
     * systems due to first-touch allocation policies. */
    #pragma omp parallel for schedule(static)
    for(splatt_idx_t x=0; x < tensor->dims[m] * rank; ++x) {
      mats[m][x] = 0.;
    }

    /*
     * NOTE: The matrices need to be row major!
     */
    char buf[100];
    idx_t const m_adj = tensor->nmodes - m; /* reverse, 1-indexed */
    sprintf(buf, "%s%lu.txt", argv[3], m_adj);
    FILE *fin = fopen(buf, "r");
    for(idx_t i=0; i < tensor->dims[m]; ++i) {
      for(idx_t j=0; j < rank; ++j) {
        fscanf(fin, "%lf", &(mats[m][j + (i*rank)]));
      }
    }
    fclose(fin);
  } /* foreach mode */

  /* read eigenvalues */
  splatt_val_t * eigs = splatt_malloc(rank * sizeof(*eigs));
  FILE * fin = fopen(argv[4], "r");
  for(idx_t i=0; i < rank; ++i) {
    fscanf(fin, "%lf", &eigs[i]);
  }
  fclose(fin);

  /* allocate result */
  splatt_val_t * result = splatt_malloc(tensor->nnz * sizeof(*result));

  /* Call the actual computational kernel */
  sp_timer_t timer;
  timer_fstart(&timer);
  lowrank_kernel(tensor, csf, splatt_opts, mats, eigs, rank, result);
  timer_stop(&timer);

  printf("Kernel took %0.3fs\n", timer.seconds);

  /* write output */
  if(argc == 6) {
    #pragma omp parallel for schedule(static)
    for(idx_t x=0; x < tensor->nnz; ++x) {
      tensor->vals[x] += result[x];
    }
    splatt_tt_write(tensor, argv[5]);
  }

  /* cleanup */
  for(splatt_idx_t m=0; m < tensor->nmodes; ++m) {
    splatt_free(mats[m]);
  }
  splatt_free(eigs);

  splatt_csf_free(csf, splatt_opts);
  splatt_free_opts(splatt_opts);
  tt_free(tensor);
  splatt_free(result);

  return EXIT_SUCCESS;
}


