


//
//  main.c
//  adaptive
//
//  Created by Ben on 27/10/2022.
//

#include <stdio.h>
#include <math.h>
# include <complex.h>
# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include "normal.h"

#undef DIVTERM
#define m               0.01            // minimum step scale factor
#define M               1               // maximum step scale factor
#define ds              0.2            // artificial time stepsize
#define T               100              // final (real) time
#define gamma           0.1             // friction coefficient
#define tau             0.1            // 'temperature'
#define printskip       10
#define numruns         10000           // total number of trajectories


double q, p, g, gp,  f;
double h;

double U(double x)
{
    return (0.5/(x*x)+x*x);
}
double Up(double x)
{
    return (-1.0/(x*x*x)+2*x);
}

#ifndef DIVTERM
double gfun(double x)
{
    double x6,xi,temp;
    x6 = pow(x,6.);
    xi = sqrt(x6+m*m);
    temp =1./(1./M + 1./xi);

    
    return(temp);
}
#else
void gfun(double x, double * g, double * gp)
{
    double x5, x6,xi,temp;
    x6 = pow(x,6.); x5 = x6/x;
    xi = sqrt(x6+m*m);
    temp =1./(1./M + 1./xi);
    (*g) = temp;
    (*gp)= 3*x5*pow((*g),2)/pow(xi,3);
}
#endif

void one_step(void)
{
    extern double q,p,g,h,f;
    //
    // BAOAB integrator
    //
    double C;

    
    p += 0.5*h*f;
#ifdef DIVTERM
    p += 0.5*h*sqrt(tau)*gp/g;
#endif
    q += 0.5*h*p;
    C = exp(-h*gamma);
    p = C*p + sqrt((1-C*C)*tau)*r8_normal_01();
    q += 0.5*h*p;
    f = -Up(q);
#ifdef DIVTERM
    gfun(q, &g, &gp);
#else
    g= gfun(q);
#endif
    h = ds*g;

#ifndef DIVTERM
    p += 0.5*h*f;
#else
    p += 0.5*h*f;
    p += 0.5*h*sqrt(tau)*gp/g;
#endif

 


}

int main(int argc, const char * argv[]) {
    // insert code here...

    
    int nt,counter;
    double t;

    

   
    for(nt = 0; nt<numruns; nt++)
    {
        // initialize the trajectory
        t=0;
        q = 2;
        p = 0;
        f = -Up(q);   // force
        #ifdef DIVTERM
            gfun(q, &g, &gp);
        #else
            g= gfun(q);
        #endif

        
        h = g*ds;

        
#ifndef DIVTERM
        printf("%lf %lf %lf %lf %lf %lf\n",t,q,p,g,h,U(q));
#else
        printf("%lf %lf %lf %lf %lf %lf %lf\n",t,q,p,g,h,U(q),gp);
#endif
        counter =0;
        while(t<T)
            {
                one_step();
                t += h;

                
                if(++counter ==printskip+1)
                {
#ifndef DIVTERM
        printf("%lf %lf %lf %lf %lf %lf\n",t,q,p,g,h,U(q));
#else
        printf("%lf %lf %lf %lf %lf %lf %lf\n",t,q,p,g,h,U(q),gp);
#endif
                    counter = 0;
                }

                
            }

        }
    return 0;
}