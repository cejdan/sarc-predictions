library(tidyverse)


# Fathmm --
setwd("/s/cgm/Cejda/_thesis/fathmm_pval")
setwd("//jetsam/scratch/cgm/Cejda/_thesis/fathmm_pval")
fathmm_blend = read.table("fathmm_logistic_log2_sort_small.tsv", sep = "\t", header = T)

fathmm_blend = fathmm_blend[,c(1,4)]
colnames(fathmm_blend) = c("SNP", "fathmm_score")

setwd("/s/cgm/Cejda/_thesis/cadd_pval")
setwd("//jetsam/scratch/cgm/Cejda/_thesis/cadd_pval")
cadd_blend = read.table("cadd_logistic_log2_sort_small.tsv", sep = "\t", header = T)
cadd_blend = cadd_blend[,c(1,4)]
colnames(cadd_blend) = c("SNP", "cadd_score")

normalized = (cadd_blend$cadd_score-min(cadd_blend$cadd_score))/(max(cadd_blend$cadd_score)-min(cadd_blend$cadd_score))
cadd_blend[,2] = normalized


both = merge(fathmm_blend, cadd_blend, by = "SNP")
both = both[!duplicated(both),]


scores = pivot_longer(both, cols = ends_with("score"), names_to = "name", values_to = "Functional_score")

setwd("//jetsam/scratch/cgm/Cejda/_thesis/cadd_pval/functional_scores_density_plots")
write.table(both, "both_functional_scores.tsv", row.names = F, quote = F)

write.table(scores, "both_functional_scores_pivot.tsv", row.names = F, quote = F)


both = read.table("both_functional_scores.tsv", sep = " ", header =T)
num_snps = length(row.names(both))
scores = pivot_longer(both, cols = ends_with("score"), names_to = "name", values_to = "Functional_score")



tiff("SNP_distribution_histogram.tiff", width = 862, height = 660, compression = "none")

ggplot(data = scores, aes(x = Functional_score, fill = name)) +
  geom_density(alpha = 0.5, size = 1) +
  xlab("Functional Score") +
  labs(fill = "Functional score type") +
  theme(axis.text = element_text(face = "bold", color = "black")) +
  ggtitle(label = "SNPs Functional Scores", subtitle = paste0("Plotting ", num_snps, " SNPs with scores in common"))



dev.off()


