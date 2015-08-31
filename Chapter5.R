# 5.1
# (a)
library(fpp)

# (b)

# (c)
log.fancy=log(fancy)
saledata=as.data.frame(log.fancy)
saledata$year=rep(c(1987,1988,1989,1990,1991,1992,1993),each=12)
saledata$sales_volume=saledata$x
saledata$trend=c(1:84)
saledata$season=rep(c(1:12),7)
saledata$season=factor(saledata$season)
a=c("no","no","no","no","no","no","no","no","no","no","no","no")
b=rep(c("no","no","yes","no","no","no","no","no","no","no","no","no"),6)
saledata$surfing_festival=c(a,b)
#saledata$surfing_festival=factor(saledata$surfing_festival)
fitlm=lm(sales_volume~trend+season+surfing_festival,data=saledata)
summary(fitlm)

# (d)
res=residuals(fitlm)
plot(res,ylab="Residuals",xlab="Time")
abline(h=0)
plot(fitted.values(fitlm),res,ylab="Residuals",xlab="Fitted Values")
abline(h=0)

# (e)
boxplot(res~saledata$season,ylab="Residuals",xlab="Month")

# (g)
dwtest(fitlm,alt="two.sided")

# (h)
x=c(85:120)
y=factor(rep(c(1:12),3))
z=rep(c("no","no","yes","no","no","no","no","no","no","no","no","no"),3)
fcast=forecast(fitlm,newdata=data.frame(trend=x,season=y,surfing_festival=z))
fcast
plot(fcast$mean,type="l") #plot the prediction against time

# (i)
fcast$mean=exp(fcast$mean)


# 5.2
library(fpp)
# (a)
plot(texasgas$consumption~texasgas$price, ylab="Consumption", xlab="Price")

# (b)
fit1=lm(log(consumption)~price,data=texasgas)
summary(fit1)

pricep=pmax(texasgas$price-60,0)
fit2a=lm(consumption~price+pricep,data=texasgas)
summary(fit2a)

## alternative
newtexasgas=as.data.frame(texasgas$consumption)
newtexasgas$dummy1=ifelse(texasgas$price<=60,1,0)
newtexasgas$dummy2=ifelse(texasgas$price<=60,texasgas$price,0)
newtexasgas$dummy3=ifelse(texasgas$price>60,1,0)
newtexasgas$dummy4=ifelse(texasgas$price>60,texasgas$price,0)
fit2b=lm(texasgas$consumption~dummy1+dummy2+dummy3+dummy4-1,data=newtexasgas)
summary(fit2b)

fit3=lm(consumption~price+I(price^2),data=texasgas)
summary(fit3)

# (d)
CV(fit1)
CV(fit2a)
CV(fit2b)
CV(fit3)

# (d)
x=c(40,60,80,100,120)
y=pmax(x-60,0)
fcast=forecast(fit2a,newdata=data.frame(price=x,pricep=y))
fcast

# (f)

