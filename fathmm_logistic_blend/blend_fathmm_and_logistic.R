

setwd("/s/cgm/Cejda/_thesis/fathmm_pval")
#setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")


#fathmm = read.table("full_fathmm-xf_results_for_AA_no_NA.tsv", header = T, sep = "\t")
#logistic = read.table("sorted_logistic.logistic", header = T)

# Start with subsets, so I can fit it into memory.
fathmm = read.table("sorted_fathmm.tsv", sep = "\t", header = T)
logistic = read.table("sorted_logistic.logistic", header = T)



both = merge(fathmm, logistic, by.x = "ID", by.y = "SNP")
both = both[!duplicated(both),]

# Quick check to ensure that the ALT allele is the A1 in Plink. 
val = 0
for (i in seq(from=1, to=length(row.names(both)), by = 1)) {
  if (both$variant[i] != both$A1[i]) {
    val = val + 1
  }
  
}
val
#YES. Good. So having 2 A1 alleles should be multiplied by 2 cadd score.


both = both[,c(1,4,5,6,16)] # SNPID, REF, ALT, FATHMM_score, logistic_pval

#c1 = 2
both$c2 = log(1/both$P, 2) + both$score
#c2 = 10
both$c10 = log(1/both$P, 10) + both$score
#c3 - 20
both$c20 = log(1/both$P, 20) + both$score


write.table(both, "FATHMM_Logistic_blend_2.tsv", sep = "\t", row.names = F, quote = F)
