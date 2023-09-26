//
// main.c
// adaptive
//
// Created by Ben on 27/10/2022.
//
#include <stdio.h>
#include <math.h>
# include <complex.h>
# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include "normal.h"


#define m 0.01 // minimum step scale factor
#define M 1 // maximum step scale factor
#define ds 0.1 // artificial time stepsize
#define T 100 // final (real) time
#define gamma 0.1 // friction coefficient
#define tau 0.01 // 'temperature'
#define numruns 10000 // total number of trajectories
double q, p, g, f;
double qnew, pnew, fnew, gnew, tnew;
double h, hnew;

double U(double x)
{
return (0.5/(x*x)+x*x);
}

double Up(double x)
{
return (-1.0/(x*x*x)+2*x);
}

double gfun(double x)
{
double x6,xi,temp;
x6 = pow(x,6.);
xi = sqrt(x6+m*m);
temp =1./(1./M + 1./xi);
return(temp);
}

//double gpfun(double x)
//{
// double x6, xi;
// x6 = pow(x,6);
// xi = sqrt(x6+m*m);
// return(-((1.0/(xi*xi))*0.5*6*pow(x,5)/xi));
//}

void one_step(void)
{
extern double q,p,g,h,f,qnew,pnew,gnew,hnew,fnew;
//
// BAOAB integrator
//
double C;
p += 0.5*h*f;
q += 0.5*h*pnew;
C = exp(-h*gamma);
p = C*p + sqrt((1.-C*C)*tau)*r8_normal_01();
q += 0.5*h*p;
f = -Up(q);
p += 0.5*h*f;
// update g and h
g = gfun(q);
h = ds*g;
}


int main(int argc, const char * argv[]) {
// insert code here...
int nt;
double t;
for(nt = 0; nt<numruns; nt++)
{
// initialize the trajectory
t=0;
q = 2;
p = 0;
f = -Up(q); // force
g = gfun(q); // step scale factor
h = g*ds;
printf("%lf\n", h);
while(t<T)
{
one_step();
t += h;
printf("%lf\n", h);
}
}
return 0;
}