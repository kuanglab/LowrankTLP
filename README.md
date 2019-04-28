# LowrankTLP

## MATLAB implementation
We provide MATLAB codes for LowrankTLP algorithm in the following files:

- **LowrankTLP\_Task1.m:** LowrankTLP with known multi-relations as inputs.
- **LowrankTLP\_Task2.m:** LowrankTLP with pairwise relations as inputs.
- **greedy\_select\_topK_idx.m**: an eignevalue selection algorithm for LowrankTLP.

## Parallel scalable implementation

We implemented a parallel version of LowrankTLP using SPLATT<sup>[1](#splatt)</sup> library for parallel shared-memory tensor operations to increase the scalability by a large magnitude. We included SPLATT version 1.1.1 in our package, but it can also be obtained from [here](https://github.com/ShadenSmith/splatt "SPLATT").

In order to install it, run the following command, once inside the SPLATT folder:

```
$ ./configure && make
```

You can run it using the following command:

```
$ ./SPLATT/build/Linux-x86_64/bin/lowrank_bin <Y0_path> <num_modes> <eigvec_prefix_path> <eigvals_path> <output_path>
```

Where the parameters are the following:

- **Y0_path:** Path to the tensor `Y0`. The file should have the same format as this [example](SPLATT/tests/tensors/small.tns "Small tensor").
- **num_modes:** Number of modes in the tensor.
- **eigvec\_prefix\_path:** Prefix of the path to the eigenvectors files. If your tensor has 4 modes, you will need 4 eigenvectors file, let's say, `example/Q1.txt`, ..., `example/Q4.txt`. Therefore, you should provide `example/Q` as the  `eigvec_prefix_path` parameter. Each file should contain a matrix whose columns are separated by space character, and one row per line. Eigenvectors are the columns of the file.
- **eigvals\_path:** Path to a file containing the eigenvalues, separated by space character.
-  **output\_path:** Path to the output file.

Example:

```
$ ./SPLATT/build/Linux-x86_64/bin/lowrank_bin ./tmp/S.tns 10 ./tmp/q ./tmp/e.txt ./tmp/output.txt
```

References
------
<a name="splatt">1</a>: Smith, Shaden, and George Karypis. "SPLATT: The Surprisingly ParalleL spArse Tensor Toolkit." (2016).
