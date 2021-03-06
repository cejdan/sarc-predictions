

library(tidyverse, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")
setwd("/s/cgm/Cejda/_thesis/fathmm_pval")

#If running manually use: 
#setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
#library(tidyverse)


fathmm_10 = read.table("fathmm_pval_top10.tsv", sep = "\t", header = T)


fathmm_10$Chrom_Pos_Alt = paste0("X", fathmm_10$ID, "_", fathmm_10$variant)
fathmm_10$Chrom_Pos_Alt = str_replace(fathmm_10$Chrom_Pos_Alt, ":", "\\.")


# Hmm, there are duplicates in the fathmm file. Not sure how or why.

fathmm_10 = fathmm_10[!duplicated(fathmm_10),]



top10_train = read.table("fathmm_pval_top_10_matrix_train_clean.raw", header = T)
sum(colnames(top10_train) %in% fathmm_10$Chrom_Pos_Alt) #Using the Alt allele, NOT all 10 found!!

# Hmm... what do I do about this?
# It's not correct to switch the score to 1 - score if I switch the alleles.
# switch allele but leave score alone?
# Easiest thing to do is ignore the problem ones and just drop them.

# Keep the Sex and Phenotype columns as well.
top10_fathmm = top10_train[,c(5,6,which(colnames(top10_train) %in% fathmm_10$Chrom_Pos_Alt))] #762 variables


for (i in seq(from = 3, to = length(colnames(top10_fathmm)), by = 1)) {
  for (j in seq(from = 1, to = length(top10_fathmm$PHENOTYPE), by = 1)) {
    
    colname_current = colnames(top10_fathmm)[i]
    
    if (top10_fathmm[j,i] == 2) {
      top10_fathmm[j,i] = 2 * fathmm_10$c2[which(str_detect(fathmm_10$Chrom_Pos_Alt,colname_current))]
    }
    
    else if (top10_fathmm[j,i] == 1) {
      top10_fathmm[j,i] = fathmm_10$c2[which(str_detect(fathmm_10$Chrom_Pos_Alt,colname_current))]
    }
    
    else {
      top10_fathmm[j,i] = 0
    }
  }
}


write.table(top10_fathmm, "fathmm_pval_top_10_scores_train.csv", sep = ",", row.names = F, quote = F)



# Now repeat the process for the TEST data. We need it to be in the same format.
top10_test = read.table("fathmm_pval_top_10_matrix_test_clean.raw", header = T)

sum(!colnames(top10_train) %in% colnames(top10_test))

# first, we need to ensure the two tables have the same column names:
# and that the major and minor alleles are the same.

columns_to_fix = top10_test[,!colnames(top10_train) %in% colnames(top10_test), drop = F]
column_names_to_fix = colnames(top10_test[,!colnames(top10_train) %in% colnames(top10_test),drop=FALSE])
correct_column_names = colnames(top10_train[,!colnames(top10_train) %in% colnames(top10_test),drop=FALSE])

if (length(columns_to_fix) > 0) {
  
  for (i in seq(from = 1, to = length(columns_to_fix), by = 1)) {
    for (j in seq(from = 1, to = length(columns_to_fix[,i]), by = 1)) {
      if (columns_to_fix[,i][j] == 0) {
        columns_to_fix[,i][j] = 2
      }
      else if (columns_to_fix[,i][j] == 2) {
        columns_to_fix[,i][j] = 0
      }
    }
  }
  for (i in seq(from = 1, to = length(columns_to_fix), by = 1)) {
    names(top10_test)[names(top10_test) == column_names_to_fix[i]] = correct_column_names[i]
    top10_test[,names(top10_test) == correct_column_names[i]] = columns_to_fix[,i]
  }
}

sum(!colnames(top10_train) %in% colnames(top10_test))
# Ok cool. Test data columns now match train data columns.
# We can safely apply the same method as before to the TEST data
# to generate the new modified score.
#################################



sum(colnames(top10_test) %in% fathmm_10$Chrom_Pos_Alt) #same as above

# Keep the Sex and Phenotype columns as well.
top10_fathmm_test = top10_test[,c(5,6,which(colnames(top10_test) %in% fathmm_10$Chrom_Pos_Alt))]

for (i in seq(from = 3, to = length(colnames(top10_fathmm_test)), by = 1)) {
  for (j in seq(from = 1, to = length(top10_fathmm_test$PHENOTYPE), by = 1)) {
    
    colname_current = colnames(top10_fathmm_test)[i]
    
    if (top10_fathmm_test[j,i] == 2) {
      top10_fathmm_test[j,i] = 2 * fathmm_10$c2[which(str_detect(fathmm_10$Chrom_Pos_Alt,colname_current))]
    }
    
    else if (top10_fathmm_test[j,i] == 1) {
      top10_fathmm_test[j,i] = fathmm_10$c2[which(str_detect(fathmm_10$Chrom_Pos_Alt,colname_current))]
    }
    
    else {
      top10_fathmm_test[j,i] = 0
    }
  }
}

write.table(top10_fathmm_test, "fathmm_pval_top_10_scores_test.csv", sep = ",", row.names = F, quote = F)


