data_001e<-read.csv("C:\\Users\\ASUS\\Desktop\\Useful\\graduation\\data_analysis\\q_dataenglish.csv",
                    head=TRUE,encoding = "UTF-8")


##十进制转二进制
library(R.utils)
data_001e$env_type=intToBin(data_001e$env_type)

#处理缺失值
library(randomForest)
data_001e[3:32] <- lapply(data_001e[3:32], as.numeric)
data_001e[3:32] <- na.roughfix(as.data.frame(data_001e[3:32]))

data_001e$RS=data_001e$Medical.Examination+data_001e$Search.for.snails+data_001e$Risk.Surveillance
data_001e$Treat=data_001e$treat
data_001e$AC=data_001e$Arable.culling
data_001e$PC=data_001e$Preventive.chemotherapy
data_001e$LC=data_001e$Livestock.chemotherapy
data_001e$MSC=data_001e$Medication.for.snail.control
data_001e$EMSC=data_001e$Environmental.modification.for.snail.control
data_001e$HE=data_001e$Health.Education
data_001e$BT=data_001e$Build.toilets

data_001e=data_001e[c(6,33:41)]

##改变量名
library(reshape2)##重命名
data_001e=rename(data_001e,c(Number.of.schistosomiasis.patients="prevalence"))

#不滞后
##计算相关系数与P值
liu.cor <- function(data_001e) {
  z <- data.frame(rep(0, 4))
  for (i in 2: ncol(data_001e)) {
    cor <- data.frame(data_001e[,1], data_001e[,i])
    cor <- na.omit(cor)
    temp_results <- cor.test(cor[,1], cor[,2])
    v <- c(temp_results$estimate, temp_results$p.value, temp_results$conf.int)
    names(v) <- c('r', 'p.value', 'conf.int.l', 'conf.int.u')
    v <- as.data.frame(v)
    colnames(v) <- colnames(data_001e)[i]
    z <- cbind(z, v)
  }
  z <- z[,-1]
  z <- t(z)
  y1 <- rep(colnames(data_001e)[1], (ncol(data_001e)-1))
  y2 <- colnames(data_001e)[-1]
  cor_results <- data.frame(x=y1, y=y2,z)
  return(cor_results)
}
cor.results <- liu.cor(data_001e)
#write.csv(cor.results, './correlation_ZNF683.csv')

## 画森林图
cor.results$p.group <- factor(ifelse(cor.results$p.value > 0.1, 'ns', 
                                     ifelse(cor.results$p.value>0.05, '*', 
                                            ifelse(cor.results$p.value>0.01, '**', '***'))),
                              levels = c('ns', '*','**', '***'))
this_title <- paste0('Correlation analysis with ',colnames(data_001e)[1])

library(ggplot2)
library(tidyverse)
dfcor=cor.results %>% dplyr::mutate(
  y=factor(y,levels =rev(c("AC","HE","BT","MSC","LC","PC","Treat","EMSC","RS")) )
)
ggplot(dfcor, aes(x = r, y = y, size=p.group)) + 
  geom_vline(xintercept = c( -.8, -.5, -.3, 0.3,0.5,0.8), linetype = 'dashed', color ='grey')+
  geom_point()+
  theme(panel.grid.major.y =element_line(color = 'grey', linetype = 'dashed'), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line.x = element_line(colour = "black", size = 1 ),
        axis.text = element_text(color = 'black'),
        axis.ticks.x = element_line(color = 'black'), 
        axis.ticks.y = element_blank())+
  geom_vline(xintercept = 0, linetype = "solid", color ='black',size = 1)+
  
  annotate('segment', x=0, xend = cor.results$r, 
           y = cor.results$y, yend = cor.results$y)+
  xlab(expression(paste('correlation coefficients:',rho)))+ ylab('')+
  xlim(-1,1) +
  ggtitle(this_title) +
  scale_size_manual(values = c(0,2, 5, 8), name = 'p.value')



##健康教育滞后
library(mets)
lag1=dlag(data_001e[,c(3)],k=1:5,combine = TRUE,simplify = TRUE)
data_002e=data.frame(
  data_001e[,1],lag1
)

library(reshape)
data_002e=rename(data_002e,c(data_001e...1.="prevalence"))
data_002e=rename(data_002e,c(data.1="Lag1HE"))
data_002e=rename(data_002e,c(data.2="Lag2HE"))
data_002e=rename(data_002e,c(data.3="Lag3HE"))
data_002e=rename(data_002e,c(data.4="Lag4HE"))
data_002e=rename(data_002e,c(data.5="Lag5HE"))

#处理缺失值
library(randomForest)
data_002e[1:6] <- lapply(data_002e[1:6], as.numeric)
data_002e[1:6] <- na.roughfix(as.data.frame(data_002e[1:6]))


#data_002e=data_002e %>% 
 # rename("Number of schistosomiasis patients"=1,
  #       "Health education lag1"=2,"Health education lag2"=3,
   #      "Health education lag3"=4,"Health education lag4"=5,"Health education lag5"=6)

#计算相关系数
liu.cor <- function(data_002e) {
  z <- data.frame(rep(0, 4))
  for (i in 2: ncol(data_002e)) {
    cor <- data.frame(data_002e[,1], data_002e[,i])
    cor <- na.omit(cor)
    temp_results <- cor.test(cor[,1], cor[,2])
    v <- c(temp_results$estimate, temp_results$p.value, temp_results$conf.int)
    names(v) <- c('r', 'p.value', 'conf.int.l', 'conf.int.u')
    v <- as.data.frame(v)
    colnames(v) <- colnames(data_002e)[i]
    z <- cbind(z, v)
  }
  z <- z[,-1]
  z <- t(z)
  y1 <- rep(colnames(data_002e)[1], (ncol(data_002e)-1))
  y2 <- colnames(data_002e)[-1]
  cor_results <- data.frame(x=y1, y=y2,z)
  return(cor_results)
}
cor.results <- liu.cor(data_002e)

## 画森林图
cor.results$p.group <- factor(ifelse(cor.results$p.value > 0.4, 'ns', 
                                     ifelse(cor.results$p.value>0.05, '*', 
                                            ifelse(cor.results$p.value>0.01, '**', '***'))),
                              levels = c('ns', '*','**', '***'))
this_title <- paste0('Correlation analysis with ',colnames(data_002e)[1])
library(ggplot2)
g1 <- ggplot(cor.results, aes(x = r, y = y, size=p.group)) + 
  geom_vline(xintercept = c( -.8, -.5, -.3, 0.3,0.5,0.8), linetype = 'dashed', color ='grey')+
  geom_point()+
  theme(panel.grid.major.y =element_line(color = 'grey', linetype = 'dashed'), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line.x = element_line(colour = "black", size = 1 ),
        axis.text = element_text(color = 'black'),
        axis.ticks.x = element_line(color = 'black'), 
        axis.ticks.y = element_blank())+
  geom_vline(xintercept = 0, linetype = "solid", color ='black',size = 1)+
  annotate('segment', x=0, xend = cor.results$r, 
           y = cor.results$y, yend = cor.results$y)+
  xlab(expression(paste('correlation coefficients:',rho)))+ ylab('')+
  xlim(-1,1) +
  ggtitle(this_title) +
  scale_size_manual(values = c(0,2, 5, 8), name = 'p.value')
print(g1)




###环境改造灭螺滞后
library(mets)
lag1=dlag(data_001e[,c(9)],k=1:5,combine = TRUE,simplify = TRUE)
data_003e=data.frame(
  data_001e[,1],lag1
)


data_003e=data_003e %>% 
  rename("Prevalence"=1,
         "Lag1EMSC"=2,
         "Lag2EMSC"=3,
         "Lag3EMSC"=4,
         "Lag4EMSC"=5,
         "Lag5EMSC"=6)


#计算相关系数
liu.cor <- function(data_003e) {
  z <- data.frame(rep(0, 4))
  for (i in 2: ncol(data_003e)) {
    cor <- data.frame(data_003e[,1], data_003e[,i])
    cor <- na.omit(cor)
    temp_results <- cor.test(cor[,1], cor[,2])
    v <- c(temp_results$estimate, temp_results$p.value, temp_results$conf.int)
    names(v) <- c('r', 'p.value', 'conf.int.l', 'conf.int.u')
    v <- as.data.frame(v)
    colnames(v) <- colnames(data_003e)[i]
    z <- cbind(z, v)
  }
  z <- z[,-1]
  z <- t(z)
  y1 <- rep(colnames(data_003e)[1], (ncol(data_003e)-1))
  y2 <- colnames(data_003e)[-1]
  cor_results <- data.frame(x=y1, y=y2,z)
  return(cor_results)
}
cor.results <- liu.cor(data_003e)

## 画森林图
cor.results$p.group <- factor(ifelse(cor.results$p.value > 0.1, 'ns', 
                                     ifelse(cor.results$p.value>0.05, '*', 
                                            ifelse(cor.results$p.value>0.01, '**', '***'))),
                              levels = c('ns', '*','**', '***'))
this_title <- paste0('Correlation analysis with ',colnames(data_003e)[1])
library(ggplot2)
g2 <- ggplot(cor.results, aes(x = r, y = y, size=p.group)) + 
  geom_vline(xintercept = c( -.8, -.5, -.3, 0.3,0.5,0.8), linetype = 'dashed', color ='grey')+
  geom_point()+
  theme(panel.grid.major.y =element_line(color = 'grey', linetype = 'dashed'), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line.x = element_line(colour = "black", size = 1 ),
        axis.text = element_text(color = 'black'),
        axis.ticks.x = element_line(color = 'black'), 
        axis.ticks.y = element_blank())+
  geom_vline(xintercept = 0, linetype = "solid", color ='black',size = 1)+
  
  annotate('segment', x=0, xend = cor.results$r, 
           y = cor.results$y, yend = cor.results$y)+
  xlab(expression(paste('correlation coefficients:',rho)))+ ylab('')+
  xlim(-1,1) +
  ggtitle(this_title) +
  scale_size_manual(values = c(0,2, 5, 8), name = 'p.value')
print(g2)
