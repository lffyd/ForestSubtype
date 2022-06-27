##Molecular typing with the genefu package
#if(!requireNamespace("BiocManager",quietly = TRUE)){
#  install.packages("BiocManager")
#}
#BiocManager::install("tensorflow2")

library(genefu)
library(org.Hs.eg.db)
library(stringr)
data(pam50.robust)
# c("scmgene", "scmod1", "scmod2","pam50", "ssp2006", "ssp2003", "intClust", "AIMS","claudinLow")

#Change of workspace
getwd()
setwd("D:/RProgram/2")
getwd()

ddata <- t(read.table("idconvert.txt",header = T,row.names=1) )
ddata[1:4,1:4]

#dannot <-read.table("idAnno.txt",header = T)
s=colnames(ddata);head(s);tail(s) ##Obtaining a genetic name
ls("package:org.Hs.eg.db")
class(org.Hs.egSYMBOL)
s2g=toTable(org.Hs.egSYMBOL)
# Using the match function, return the location information (or NA if there is no match)
g=s2g[match(s,s2g$symbol),1]
# Then make a dataframe
dannot=data.frame(probe=s,
                  "Gene.Symbol" =s,
                  "EntrezGene.ID"=g)

# Remove the NA line in ddata and dannot below
ddata=ddata[,!is.na(dannot$EntrezGene.ID)] #ID conversion
dim(ddata)
dannot=dannot[!is.na(dannot$EntrezGene.ID),]

# Look at the gene annotations and expression matrix after removing NA, you must ensure that the gene IDs of the annotations and the gene IDs of the expression matrix correspond to each other
head(dannot)
ddata[1:4,1:4]

s<-molecular.subtyping(sbt.model = "pam50",data=ddata,
                       annot=dannot,do.mapping=TRUE)

str(s)
table(s$subtype)
##Save typing results to data box
tmp=as.data.frame(s$subtype)
tmp
subtypes=as.character(s$subtype)

#----------------------------------------------------------------------------------------
#------------------------------------Personal handling of data dividers----------------------------------
#----------------------------------------------------------------------------------------

#-------------------------------------data, target merge------------------------------------------
#Aggregate data
#Store the value of pam50 in Finaldata
Finaldata <- data.frame(
  PAM50 = str_squish(tmp[,1]) ,            #stringr package to remove extraneous spaces str_squish()
  SAMPLE.ID = str_squish(rownames(tmp))
)  
#Screening for genes
ddata <- ddata[,colSums(ddata)>0]   #Screening for all expressed genes with a sum greater than 0
dim(ddata)
ddata <- ddata[rowSums(ddata>=1) >=18000,colSums(ddata>=1) >=500]   #Screening for samples expressing at least 18,000 genes and genes expressed in at least 500 samples, at this point listed as genes, behavioural samples
dim(ddata)
#min-max standardised treatment
ddata=(ddata-min(ddata))/(max(ddata)-min(ddata))
#Z-score normalisation
ddata=scale(ddata)
#SAMPLE.ID <- Finaldata[,"SAMPLE.ID"]
#pam50 <- Finaldata[,"PAM50"]
ddata <- cbind(Finaldata,ddata)
#Save
write.csv(ddata,
          "D:/breast_1211_23900.csv",
          row.names = FALSE,
          col.names = TRUE) #Save ddata as breast_1211.csv

dim(ddata)