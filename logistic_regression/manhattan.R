

args=commandArgs(trailingOnly=T)
path=as.character(args[1])
datname=as.character(args[2])


library(qqman, lib.loc = "/net/qlotsam.lan.omrf.org/qlotsam/rc/homes/cejdan/R/x86_64-pc-linux-gnu-library/4.0")

data=read.table(paste0(path,"/",datname,".assoc.logistic"),header=T,stringsAsFactors = F)
data$CHR=as.numeric(as.character(data$CHR))
data$SNP=as.character(data$SNP)
data$BP=as.numeric(as.character(data$BP))
data$A1=as.character(data$A1)
data$OR=as.numeric(as.character(data$OR))
data$P=as.numeric(as.character(data$P))
toss=which(data$P==0 | is.na(data$P))
if(length(toss)!=0) data2=data[-toss,] else data2=data
rm(data)

pdf(paste0(path,"/",datname,".pdf"))
manhattan(data2,main=paste0(datname))
dev.off()







