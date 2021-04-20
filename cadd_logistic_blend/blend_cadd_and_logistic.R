
setwd("/s/cgm/Cejda/_thesis/cadd_pval")

# Load the small datasets into my personal memory as a test.
# Do the full datasets on the cluster which has WAY bigger memory.
cadd = read.table("sorted_cadd_scores.tsv", header = T, sep = "\t")
logistic = read.table("sorted_logistic.logistic", header = T)


cadd$SNP = paste0(cadd$CHROM, ":",cadd$POS)

both = merge(cadd, logistic, by = "SNP")

# Quick check to ensure that the ALT allele is the A1 in Plink. 
val = 0
for (i in seq(from=1, to=length(row.names(both)), by = 1)) {
  if (both$ALT[i] != both$A1[i]) {
    val = val + 1
  }
  
}
val
#YES. Good. So having 2 A1 alleles should be multiplied by 2 cadd score.



both = both[,c(1,4,5,7,15)]


# The theory here is as c increases in log base c, the importance of p-value decreases.
# We are blending the p-value and the PHRED score to create a new score.
# We will use this blended score in our models.

#c1 = 2
both$c2 = log(1/both$P, 2) + both$PHRED
#c2 = 10
both$c10 = log(1/both$P, 10) + both$PHRED
#c3 - 20
both$c20 = log(1/both$P, 20) + both$PHRED


write.table(both, "CADD_Logistic_blend.tsv", sep = "\t", row.names = F, quote = F)
