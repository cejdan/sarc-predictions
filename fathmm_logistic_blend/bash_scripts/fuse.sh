#!/bin/bash -l


sbatch -A montgomery --partition serial --mem 5000 -J fuse --wrap "Rscript 2021-04-14_fuse_FATHMM_pval_top_10.R"
sbatch -A montgomery --partition serial --mem 5000 -J fuse --wrap "Rscript 2021-04-14_fuse_FATHMM_pval_top_100.R"
sbatch -A montgomery --partition serial --mem 5000 -J fuse --wrap "Rscript 2021-04-14_fuse_FATHMM_pval_top_200.R"
sbatch -A montgomery --partition serial --mem 5000 -J fuse --wrap "Rscript 2021-04-14_fuse_FATHMM_pval_top_500.R"
sbatch -A montgomery --partition serial --mem 5000 -J fuse --wrap "Rscript 2021-04-14_fuse_FATHMM_pval_top_1000.R"
sbatch -A montgomery --partition serial --mem 10000 -J fuse --wrap "Rscript 2021-04-14_fuse_FATHMM_pval_top_2000.R"