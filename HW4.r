## HW 4 

day=c(1,2,3,4,5,6,7,8,9,10,11,12)
mwh=c(16.3,16.8,15.5,18.2,15.2,17.5,19.8,19.0,17.5,16.0,19.6,18.0)
temp=c(29.3,21.7,23.7,10.4,29.7,11.9,9.0,23.4,17.8,30.0,8.6,11.8)

library(fpp)
plot(elec$mwh~elec$temp)
abline(20.19,-0.14)
fit=lm(mwh~temp,data=elec)
summary(fit)
plot(residuals(fit)~temp)
forecast(fit,newdata=data.frame(temp=c(10,35)))


plot(olympic$time~olympic$Year)
fit=lm(time~Year,data=olympic)
summary(fit)
plot(residuals(fit)~olympic$Year)
forecast(fit,newdata=data.frame(Year=c(2000,2004,2008,2012)))

log.fancy=log(fancy)
plot(log.fancy)

fit.fancy=tslm(log.fancy~trend)
lines(fitted(fit.fancy),col="blue")

log.fancy2=window(log.fancy,start=1987,end=1994,frequency=1)
fit=tslm(log.fancy2~trend+season)
summary(fit)


