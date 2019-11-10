rm(list = ls())
library(tidyverse)
library(randomForest)
library(Hmisc)
library(stringi)
library(naniar)
library(mice)
library(hnp)
library(ROCR)
library(xgboost)

Mode = function(x, na.rm = FALSE) {
  if(na.rm){
    x = x[!is.na(x)]
  }
  
  ux = unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

theme_set(theme_minimal())

train = read.csv("data/train.csv",header = T,na.strings = "")
Survived = train$Survived
train$Survived = NULL

test = read.csv("data/test.csv",header = T, na.strings = "")

completo = bind_rows(train,test) %>% 
  mutate(type = c(rep("train",dim(train)[1]),rep("test",dim(test)[1])))

completo = completo %>% dplyr::select(-c(Name)) %>% 
  mutate(Pclass = factor(Pclass,levels = c(1,2,3),ordered = T),
         Cabin_ind = case_when(is.na(Cabin)~"0",
                               TRUE~"1"),
         Cabin_letra = str_sub(completo$Cabin,end=1),
         Cabin_count =  str_count(completo$Cabin, " ")+1) %>% 
  mutate(Cabin_letra = case_when(is.na(Cabin_letra)~"XY",
                                 TRUE~Cabin_letra),
         Cabin_count = case_when(is.na(Cabin_count)~0,
                                 TRUE~Cabin_count)) %>% 
  mutate(Ticket_letra = stri_match(completo$Ticket, regex='(.* )?([0-9]+)' )[,2]) %>% 
  mutate(Ticket_letra = case_when(is.na(Ticket_letra)~"0",
                                  TRUE~"1")) %>% 
  dplyr::select(-c(Cabin, Ticket)) %>% 
  mutate(Cabin_ind   = as.factor(Cabin_ind),
         Cabin_letra = as.factor(Cabin_letra),
         Ticket_letra = as.factor(Ticket_letra))
  

gg_miss_var(completo) + xlab("Variables") + ylab("Missing")
# Age, Embarked e Fare apresentam observações ausentes

# Usando o pacote Mice para realizar as imputações
set.seed(3578420)
imput            = mice(completo)
age_missing      = apply(imput$imp$Age,1,median)
Embarked_missing = apply(imput$imp$Embarked,1,Mode)
Fare_missing     = apply(imput$imp$Fare,1,median)

completo$Age[which(is.na(completo$Age))]           = age_missing
completo$Embarked[which(is.na(completo$Embarked))] = Embarked_missing  
completo$Fare[which(is.na(completo$Fare))]         = Fare_missing

sum(is.na(completo))

# nao ha mais valores ausentes, os dados ja estao tratados,
# fazendo a redivisao em treino e teste

teste = completo %>% filter(type=="test") %>% dplyr::select(-type)
treino = completo %>% filter(type=="train") %>% dplyr::select(-type) %>% 
  mutate(Survived = Survived)

chisq.test(treino$Pclass,treino$Survived)$p.value
chisq.test(treino$Sex,treino$Survived)$p.value
wilcox.test(treino$Age,treino$Survived)$p.value
wilcox.test(treino$SibSp,treino$Survived)$p.value
wilcox.test(treino$Parch,treino$Survived)$p.value
wilcox.test(treino$Fare,treino$Survived)$p.value
chisq.test(treino$Embarked,treino$Survived)$p.value
chisq.test(treino$Cabin_ind,treino$Survived)$p.value
chisq.test(treino$Cabin_letra,treino$Survived,simulate.p.value = T)$p.value
wilcox.test(treino$Cabin_count,treino$Survived)$p.value
chisq.test(treino$Ticket_letra,treino$Survived)$p.value

#================================
# regressao logistica 
#================================

f = paste(names(treino[,-c(1,13)]),collapse =' + ')
f = as.formula(paste('Survived ~',f))
f

modelo_rg = glm(f,family = "binomial",data = treino)
summary(modelo_rg)

modelo_rg = step(modelo_rg,direction = "both",trace = F)
summary(modelo_rg)

# Cabin_letra nao significativa
modelo_rg = update(modelo_rg,.~.-Cabin_letra)
summary(modelo_rg)

pchisq(modelo_rg$deviance,df=modelo_rg$df.residual,lower.tail = F)
# deviance nao significativa, portanto, modelo adequado

#==================
# Pacote simulado
#==================

set.seed(5047774)
hnp(modelo_rg,halfnormal = F,paint.out = T,pch=16,sim=500,print.on = T)

#==================
# curva Roc
#==================

predictionTreino = predict(modelo_rg,treino,type="response")

pred = ROCR::prediction(predictionTreino, treino$Survived)

# Calcula verdadeiros positivos e falsos positivos
perf = performance(pred,"tpr","fpr")
plot(perf,colorize = TRUE,main="Curva ROC - Conj. de Treino")
abline(0, 1, lty = 2)

# Calcula area abaixo da curva
auc=(performance(pred,"auc")@y.values)[[1]]
auc

predictionTreino = predict(modelo_rg,treino,type="response")

# definindo um ponto de corte ideal
corte=seq(0.35,0.85,0.01)
acuracia = c()
for(i in 1:length(corte)){
  pred_treino = ifelse(predictionTreino>corte[i],1,0)
  acuracia[i]=1-ks::compare(pred_treino, treino$Survived)$error
}
plot(corte,acuracia,type="l")
max(acuracia) 
corte[which(acuracia==max(acuracia))]

# Encontrando as predicoes do conjunto de teste
predictionTeste = predict(modelo_rg,teste,type="response")

pred_teste = ifelse(predictionTeste>0.58,1,0) # definindo ponto de corte igual a 0.58
table(pred_teste)

submission_rg = data.frame(PassengerId=teste$PassengerId,Survived=pred_teste)
head(submission_rg)

# write.csv(submission_rg,file = "submission_rg.csv",row.names = F)

#================================
# random forest
#================================

# as variaveis no formato caracter devem ser codificadas 
# em fator para usar o randomforest

set.seed(51716)
modelo_rf = randomForest(f,data = treino)

predictionTreino = predict(modelo_rf,treino,type="response")

# definindo um ponto de corte ideal
corte=seq(0.35,0.85,0.01)
acuracia = c()
for(i in 1:length(corte)){
  pred_treino = ifelse(predictionTreino>corte[i],1,0)
  acuracia[i]=1-ks::compare(pred_treino, treino$Survived)$error
}
plot(corte,acuracia,type="l")
max(acuracia) 
corte[which(acuracia==max(acuracia))]

# Encontrando as predicoes do conjunto de teste
predictionTeste = predict(modelo_rf,teste,type="response")

pred_teste = ifelse(predictionTeste>0.48,1,0) # definindo ponto de corte

submission_rf = data.frame(PassengerId=teste$PassengerId,Survived=pred_teste)
head(submission_rg)

# write.csv(submission_rf,file = "submission_rf.csv",row.names = F)

#================================
# xgboost
#================================

# tirando ticket letra que foi extremamente nao significativo no qui-quadrado

treino = treino %>% dplyr::select(-Ticket_letra)
teste  = teste %>% dplyr::select(-Ticket_letra)

# a funcao xgboost exige algumas adequacoes na base de dados

Pclass_treino         = model.matrix(~Pclass-1,treino)
Sex_treino            = model.matrix(~Sex-1,treino)
Embarked_treino       = model.matrix(~Embarked-1,treino)
Cabin_ind_treino      = model.matrix(~Cabin_ind-1,treino)
Cabin_letra_treino    = model.matrix(~Cabin_letra-1,treino)

treino_xgboost = cbind(Pclass_treino, Sex_treino, Embarked_treino,
                            Cabin_ind_treino, Cabin_letra_treino, 
                            Age = treino$Age, SibSp = treino$SibSp,
                            Parch = treino$Parch, Fare = treino$Fare,
                            Cabin_count = treino$Cabin_count)
treino_xgboost = data.matrix(treino_xgboost)

Pclass_teste         = model.matrix(~Pclass-1,teste)
Sex_teste            = model.matrix(~Sex-1,teste)
Embarked_teste       = model.matrix(~Embarked-1,teste)
Cabin_ind_teste      = model.matrix(~Cabin_ind-1,teste)
Cabin_letra_teste    = model.matrix(~Cabin_letra-1,teste)

teste_xgboost = cbind(Pclass_teste, Sex_teste, Embarked_teste,
                      Cabin_ind_teste, Cabin_letra_teste, 
                       Age = teste$Age, SibSp = teste$SibSp,
                       Parch = teste$Parch, Fare = teste$Fare,
                       Cabin_count = teste$Cabin_count)
teste_xgboost = data.matrix(teste_xgboost)

# O passo final é converter as matrizes em objetos dmatrix.

dtrain = xgb.DMatrix(data = treino_xgboost, label= treino$Survived)
dtest  = xgb.DMatrix(data  = teste_xgboost)

set.seed(345923)
# documentação do xgboost:
# https://www.rdocumentation.org/packages/xgboost/versions/0.90.0.2/topics/xgb.train
model_xg = xgboost(data = dtrain,
                objective = 'binary:logistic',
                nround = 50,
                max.depth = 5,# tamanho máximo das arvores
                early_stopping_rounds = 3,# interromperá o treinamento se não tivermos melhorias em um certo número de rodadas de treinamento.
                eval_metric = "error"
                )

importance_matrix = xgb.importance(model = model_xg) 
xgb.plot.importance(importance_matrix)

predictionTreino = predict(model_xg,dtrain,type="response")

# definindo um ponto de corte ideal
corte=seq(0.35,0.85,0.01)
acuracia = c()
for(i in 1:length(corte)){
  pred_treino = ifelse(predictionTreino>corte[i],1,0)
  acuracia[i]=1-ks::compare(pred_treino, treino$Survived)$error
}
plot(corte,acuracia,type="l")
max(acuracia) 
corte[which(acuracia==max(acuracia))]

# Encontrando as predicoes do conjunto de teste
predictionTeste = predict(model_xg,dtest,type="response")

pred_teste = ifelse(predictionTeste>0.47,1,0) # definindo ponto de corte

submission_xg = data.frame(PassengerId=teste$PassengerId,Survived=pred_teste)
head(submission_xg) # 0.78947

# write.csv(submission_xg,file = "submission_xg.csv",row.names = F) 
