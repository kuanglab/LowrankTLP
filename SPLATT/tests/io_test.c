
#include "../src/io.h"
#include "../src/sort.h"

#include "ctest/ctest.h"

#include "splatt_test.h"

static char const * const TMP_FILE = "tmp.bin";


CTEST_DATA(io)
{
  idx_t ntensors;
  sptensor_t * tensors[MAX_DSETS];
};

CTEST_SETUP(io)
{
  data->ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(idx_t i=0; i < data->ntensors; ++i) {
    data->tensors[i] = tt_read(datasets[i]);
  }
}


CTEST_TEARDOWN(io)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    tt_free(data->tensors[i]);
  }
}



CTEST2(io, zero_index)
{
  sptensor_t * zero_tt = tt_read(DATASET(small4.tns));
  sptensor_t * one_tt  = tt_read(DATASET(small4_zeroidx.tns));

  ASSERT_EQUAL(one_tt->nnz, zero_tt->nnz);
  ASSERT_EQUAL(one_tt->nmodes, zero_tt->nmodes);

  for(idx_t m=0; m < one_tt->nmodes; ++m) {
    ASSERT_EQUAL(one_tt->dims[m], zero_tt->dims[m]);

    for(idx_t n=0; n < one_tt->nnz; ++n) {
      ASSERT_EQUAL(one_tt->ind[m][n], zero_tt->ind[m][n]);
    }
  }

  tt_free(zero_tt);
  tt_free(one_tt);
}




CTEST2(io, binary_io)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    sptensor_t * const gold = data->tensors[i];

    /* write to binary */
    tt_write_binary(gold, TMP_FILE);

    /* now read it back */
    sptensor_t * tt_bin = tt_read(TMP_FILE);

    /* now check for correctness */
    ASSERT_EQUAL(gold->nnz, tt_bin->nnz);
    ASSERT_EQUAL(gold->nmodes, tt_bin->nmodes);
    for(idx_t m=0; m < tt_bin->nmodes; ++m) {
      idx_t const * const gold_ind = gold->ind[m];
      idx_t const * const test_ind = tt_bin->ind[m];

      for(idx_t n=0; n < tt_bin->nnz; ++n) {
        ASSERT_EQUAL(gold_ind[n], test_ind[n]);
      }
    }

    /* values better be exact! */
    val_t const * const gold_vals = gold->vals;
    val_t const * const test_vals = tt_bin->vals;
    for(idx_t n=0; n < tt_bin->nnz; ++n) {
      ASSERT_DBL_NEAR_TOL(gold_vals[n], test_vals[n], 0.);
    }
  }

  /* delete temporary file */
  remove(TMP_FILE);
}


/* read a CSR matrix stored in COO form */
CTEST2(io, csr_coo)
{
  double * opts = splatt_default_opts();

  /* load the CSR */
  splatt_csr * csr = NULL;
  int ret = splatt_csr_load(DATASET(spmat.tns), &csr, opts);
  ASSERT_EQUAL(SPLATT_SUCCESS, ret);
  ASSERT_NOT_NULL(csr);

  sptensor_t * tt = tt_read_file(DATASET(spmat.tns));
  tt_sort(tt, 0, NULL);
  ASSERT_EQUAL(tt->nnz, csr->nnz);
  ASSERT_EQUAL(tt->dims[0], csr->I);
  ASSERT_EQUAL(tt->dims[1], csr->J);

  for(idx_t i=0; i < csr->I; ++i) {
    for(idx_t x=csr->rowptr[i]; x < csr->rowptr[i+1]; ++x) {
      ASSERT_EQUAL(tt->ind[0][x], i);
      ASSERT_EQUAL(tt->ind[1][x], csr->colind[x]);
      ASSERT_DBL_NEAR_TOL(tt->vals[x], csr->vals[x], 0.);
    }
  }

  splatt_free_csr(csr, opts);
  tt_free(tt);
  splatt_free_opts(opts);
}


