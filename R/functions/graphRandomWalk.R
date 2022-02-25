## Spreading activation using Katz walks
# The following script computes a random walk similar to the Katz 
# centrality procedure (see for example Newman (2010) "Networks An Introduction"
# to provide a simple implementation of spreading activation.
# It depends on a single paramter alpha, which determines the contribution
# of longer paths (values of alpha should be > 0 and < 1). 
#
# It will create a path-augmented graph G.rw, which include weighted sum
# of all potential paths. This new graph can then be used to derive
# similarities from.  Pointwise mutual information is used to weigh 
# associative response frequency. 
# For more information see De Deyne, Navarro, Perfors &  Storms (2018).
#
# Input: 
# The input should be an adjacency file formatted as i j f, where i and j 
# refer to a cue and response coordinate, and f to the frequency of 
# response j to cue i
# In addition, a file with labels should also be provided where the labels
# correspond to the indices i and j in the adjacency file
#
# Typically the graph corresponding to the adjacency matrix is restricted 
# to the largest strongly connected component and loops are removed
# Output: 
# G.rw: graph with indirect paths, renormalized and ppmi weighted
# S.rw: dense similarity matrix for the graph. 
#
# Notes:
# Alpha. Throughout most experiments alpha = .75 performs reasonably well. 
# To control degrees of freedom this has been taken as a default.
#
# PPMI. PPMI is known to have a bias for rare events, which does not affect
# typical word associations graphs with n < 12,000 words, but becomes
# a concern for larger graphs (Turney & Pantel, 2010). 
# In such cases, weighted PPMI versions can be considered (see for example
# Levy,Goldberg & Dagan, 2015, p 215)
#
# S.rw: calculating the cosine similarity for all possible pairwise combinations
# is memory intensive, only consider doing this when your system has
# sufficient RAM. Otherwise, consider multiplying vectors instead.
#
# Total processing time on an i7 with 32Gb is about 96 seconds.
#
# References:
# De Deyne, S., Navarro, D., Perfors, A., Storms, G. (2016). Structure at 
# every scale: A semantic network account of the similarities between 
# unrelated concepts. Journal of Experimental Psychology. 
# General, 145, 1228-1254.
#
# Levy, O., Goldberg, Y., & Dagan, I. (2015). Improving distributional 
# similarity with lessons learned from word embeddings. Transactions of the 
# Association for Computational Linguistics, 3, 211-225.
#
# Newman, M. (2010). Networks: an introduction. Oxford university press.
#
# Turney, P. D., & Pantel, P. (2010). From frequency to meaning: 
# Vector space models of semantics. Journal of artificial intelligence 
# research, 37, 141-188.
#
# Questions / comments: 
# Simon De Deyne, simon2d@gmail.com
# Last changed: 12/06/2019
#
# See creataeRandomWalk.m for a more efficient version

library('here')
setwd(here())
library('Matrix')
library('tictoc')
library('tidyverse')
library('igraph')
library(stopwords)


source('./R/functions/importDataFunctions.R')
source('./R/functions/networkFunctions.R')
source('./R/functions/similarityFunctions.R')


# Construct similarity matrices for SWOW based on the primary (R1) responses 
# or choose 'R123' to include all responses

# default value for alpha 
#alpha = 0.75

# Load the data 
#dataFile.SWOWEN     = './data/2018/processed/SWOW-EN.R100.csv'
#SWOW.R1             = importDataSWOW(dataFile.SWOWEN,'R1')

# Generate the weighted graphs
#G                   = list()
#G$R1$Strength       = weightMatrix(SWOW.R1,'strength')
#G$R1$PPMI           = weightMatrix(SWOW.R1,'PPMI')

#tic()
#G$R1$RW             = weightMatrix(SWOW.R1,'RW',alpha)
#toc()


# Compute the cosine similarity matrix
#S = cosineMatrix(G$R1$RW)

#write.csv(S,'./output/2018/S_RW.R1.csv')


#rm(list = ls())

source('./R/functions/importDataFunctions.R')
source('./R/functions/networkFunctions.R')
source('./R/functions/similarityFunctions.R')

alpha = 0.75

# Load the data 
# File that I generated: /home/ecain/data/diachronic_lang_change/data_output/swow_eng_filtered.csv
# dataFile.SWOWEN     = './data/2018/processed/SWOW-EN.R100.csv'
dataFile.SWOWEN       = '/home/ecain/data/diachronic_lang_change/data_output/swow_eng_filtered.csv'
output_dir            = '/home/ecain/data/diachronic_lang_change/data_output/swow_katz_rsm/'
SWOW.R123             = importDataSWOW(dataFile.SWOWEN,'R123')
histwords_vocab       = read_csv("/home/ecain/data/diachronic_lang_change/data/histwords_vocab.csv")

# Running for 10y bins
by_num                = 10
age_groups            = c(seq(20,70,by=by_num), 100)

age_init_1 = age_groups[1]
age_init_2 = age_groups[2]

cue_cab <- dataset |> 
  filter((relative_age >= age_init_1) & (relative_age < age_init_2)) |>  #Seed age group, filter dataset to age group
  filter(!grepl(" ", cue) & #Select only single words cues
           !grepl("null", cue) & #Remove null due to later issues
           (nchar(cue)>1) & #Remove single char
           (response %nin% stopwords("en", source = "nltk"))) |>  #Remove stopwords
  select(cue)

cue_cab <- unique(cue_cab)

for (ii in 2:(length(age_groups)-1)){
  age_lower <- age_groups[ii]
  age_upper <- age_groups[ii+1]
  
  new_cue_cab <- dataset |> 
    filter((relative_age >= age_init_1) & (relative_age < age_init_2)) |>  #Seed age group, filter dataset to age group
    filter(!grepl(" ", cue) & #Select only single words cues
            !grepl("null", cue) & #Remove null due to later issues
            (nchar(cue)>1) & #Remove single char
            (response %nin% stopwords("en", source = "nltk"))) |>  #Remove stopwords
    select(cue)
  
  cue_cab <- intersect(cue_cab, unique(cue_cab))
}

cue_cab <- intersect(cue_cab, histwords_vocab$word)
cue_cab <- unique(cue_cab)

write.csv(cue_cab, paste("/home/ecain/data/diachronic_lang_change/data_output/cue_vocab_by",by_num,".csv",sep = ""), row.names=FALSE)

for (ii in 1:(length(age_groups)-1)){
  age_lower <- age_groups[ii]
  age_upper <- age_groups[ii+1]

  subset <- SWOW.R123 %>% filter((relative_age >= age_lower) & (relative_age < age_upper) & (cue %in% cue_cab))
  # Generate the weighted graphs
  G                   = list()
  G$R123$Strength       = weightMatrix(SWOW.R123,'strength')
  G$R123$PPMI           = weightMatrix(SWOW.R123,'PPMI')

  tic()
  G$R123$RW             = weightMatrix(subset,'RW',alpha)
  toc()


  # Compute the cosine similarity matrix
  S = cosineMatrix(G$R123$RW)
 
  output_filename <- paste0(output_dir,'S_RW_',age_lower,'-',age_upper,'.csv')
  write.csv(S,output_filename)
  rm(subset, G, S)
}


# REDO by 5 year bins

by_num            = 5
age_groups        = c(seq(20, 70, by= by_num), 100)

age_init_1 = age_groups[1]
age_init_2 = age_groups[2]

cue_cab <- dataset |> 
  filter((relative_age >= age_init_1) & (relative_age < age_init_2)) |>  #Seed age group, filter dataset to age group
  filter(!grepl(" ", cue) & #Select only single words cues
           !grepl("null", cue) & #Remove null due to later issues
           (nchar(cue)>1) & #Remove single char
           (response %nin% stopwords("en", source = "nltk"))) |>  #Remove stopwords
  select(cue)

cue_cab <- unique(cue_cab)

for (ii in 2:(length(age_groups)-1)){
  age_lower <- age_groups[ii]
  age_upper <- age_groups[ii+1]
  
  new_cue_cab <- dataset |> 
    filter((relative_age >= age_init_1) & (relative_age < age_init_2)) |>  #Seed age group, filter dataset to age group
    filter(!grepl(" ", cue) & #Select only single words cues
            !grepl("null", cue) & #Remove null due to later issues
            (nchar(cue)>1) & #Remove single char
            (response %nin% stopwords("en", source = "nltk"))) |>  #Remove stopwords
    select(cue)
  
  cue_cab <- intersect(cue_cab, unique(cue_cab))
}

cue_cab <- intersect(cue_cab, histwords_vocab$word)
cue_cab <- unique(cue_cab)

write.csv(cue_cab, paste("/home/ecain/data/diachronic_lang_change/data_output/cue_vocab_by",by_num,".csv",sep = ""), row.names=FALSE)

for (ii in 1:(length(age_groups)-1)){
  age_lower <- age_groups[ii]
  age_upper <- age_groups[ii+1]

  subset <- SWOW.R123 %>% filter((relative_age >= age_lower) & (relative_age < age_upper) & (cue %in% cue_cab))

  G                   = list()
  G$R123$Strength       = weightMatrix(SWOW.R123,'strength')
  G$R123$PPMI           = weightMatrix(SWOW.R123,'PPMI')

  tic()
  G$R123$RW             = weightMatrix(subset,'RW',alpha)
  toc()


  # Compute the cosine similarity matrix
  S = cosineMatrix(G$R123$RW)
 
  output_filename <- paste0(output_dir,'S_RW_',age_lower,'-',age_upper,'.csv')
  write.csv(S,output_filename)
  rm(subset, G, S)
}
