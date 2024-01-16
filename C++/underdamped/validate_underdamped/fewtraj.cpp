

// Anisotropic 1 dimension, sample few trajectories 

#include <cstring>
#include <stdio.h>
#include <random>
#include <cmath>
#include <string>
#include <list>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <iterator>
#include <iomanip>
#include <boost/random/random_device.hpp> //boost function
#include <boost/random/normal_distribution.hpp> //include normal distribution
#include <boost/random/mersenne_twister.hpp>
#include <boost/multi_array.hpp>
#include <chrono>

using namespace std::chrono;
 
using namespace std;

// # include <math.h>
// # include <complex.h>
// # include <stdlib.h>

// # include <stdio.h>
// # include <time.h>
// # include "normal.h"



using namespace std;
#define numsam          5          // number of sample
#define T               100         // total number of trajectories
#define dt              0.17
#define tau             1.            
#define numruns         T/dt         // total number of trajectories
#define gamma           1.            // friction coefficient
// #define PATH            "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/validate/fewtraj/dt1"
// #define PATH            "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/validate/fewtraj/dt2"
// #define PATH            "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/validate/fewtraj/dt3"



/////////////////////// DEFINE POTENTIAL //////////////////////////////
#define r     0.1
#define d     5
#define m1    0.01
#define M1    1/2 //max value that can be taken by dt



double Up(double x)
{
    double res = -2*x+2*1/pow((abs(x)-5),3)*abs(x)/x;
    return res;
}

double getg(double x)
{
    double f,f2,xi,den,g;
    f=1/(abs(x)-5);
    f2=f*f;
    xi=sqrt(1+m1*f2);
    den=M1*xi+sqrt(f2);
    g=xi/den;
    return(g);

}

double getgprime(double x)
{
    // double f,f2,fp,xi,gp,M1;
    // M1=dt/max;
    // f=r*((cos(1+d*x))+x*x*x)*((cos(1+d*x))+x*x*x);
    // f2=f*f;
    // fp=r*2*(cos(1+d*x)+x*x*x)*(3*x*x-d*d*sin(1+d*x));
    // xi=sqrt(1+m1*f2);
    // gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
    return(0);
}


// double Up(double x)
// {
//     double res = x*x*x+d*cos(1+d*x);
//     return res;
// }

// double getg(double x)
// {
//     double f,f2,xi,den,g,M1;
//     M1=dt/max;
//     f=r*((cos(1+d*x))+x*x*x)*((cos(1+d*x))+x*x*x);
//     f2=f*f;
//     xi=sqrt(1+m1*f2);
//     den=M1*xi+sqrt(f2);
//     g=xi/den;
//     return(g);

// }

// double getgprime(double x)
// {
//     double f,f2,fp,xi,gp,M1;
//     M1=dt/max;
//     f=r*((cos(1+d*x))+x*x*x)*((cos(1+d*x))+x*x*x);
//     f2=f*f;
//     fp=r*2*(cos(1+d*x)+x*x*x)*(3*x*x-d*d*sin(1+d*x));
//     xi=sqrt(1+m1*f2);
//     gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
//     return(gp);
// }


/////////////////////////////////
// Non adaptive one step function //
/////////////////////////////////

int one_step(void)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double q,p,f,g,gp,gdt,C;
    int ns,nt,nsp;
    // Savethe values 
    vector<double> q_list(numruns,0);
    vector<vector<double>> vec_q(numsam,q_list);
    vector<vector<double>> vec_p(numsam,q_list);

    #pragma omp parallel private(q,p,f,C,nt,gdt,nsp) shared(ns,vec_q,vec_p)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        q = .5;
        p = 1.;
        f = -Up(q);   // force
        // g = getg(q);
        // gp = getgprime(q); 
        // gdt = g*dt; 
        nsp=0;
        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            p += 0.5*dt*f;
            // #ifdef DIVTERM
            // p += 0.5*dt*tau*gp;
            // // #endif

            //**********
            //* STEP A *
            //**********
            q += 0.5*dt*p;

            //**********
            //* STEP O *
            //**********
            C = exp(-dt*gamma);
            p = C*p + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            q += 0.5*dt*p;

            //**********
            //* STEP B *
            //**********
            f = -Up(q);
            // g = getg(q);
            // gdt = dt*g;
            p += 0.5*dt*f;
            // If divterm is defined then add the gprime correction term
            // #ifdef DIVTERM
            // gp=getgprime(q);
            // p += 0.5*dt*tau*gp;
        
        // save the value value
            vec_q[ns][nt]=q;
            vec_p[ns][nt]=p;

        }
    }
    
    fstream file;
    string file_name;
    string path=PATH;
    for(int nsps = 0; nsps<numsam; nsps++){
        file_name=path+"/vec_noada"+to_string(nsps)+".txt";
        file.open(file_name,ios_base::out);
        ostream_iterator<double> out_itr(file, "\n");
        file<<"q\n";
        copy(vec_q[nsps].begin(), vec_q[nsps].end(), out_itr);
        file<<"p\n";
        copy(vec_p[nsps].begin(), vec_p[nsps].end(), out_itr);
        file.close();
        }

return 0;
}



int one_step_tr(void)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double q,p,f,g,gp,gdt,C,g0,g1;
    int ns,nt,nsp;
    // Savethe values 
    vector<double> q_list(numruns,0);
    vector<vector<double>> vec_q(numsam,q_list);
    vector<vector<double>> vec_p(numsam,q_list);
    vector<vector<double>> vec_g(numsam,q_list);

    #pragma omp parallel private(q,p,f,C,nt,gdt,g,nsp,g0,g1) shared(ns,vec_q,vec_p,vec_g)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        q = 0.5;
        p = 1.;
        g = getg(q);
        gdt = dt*g;
        gp=getgprime(q);
        f = -Up(q);   // force

        for(nt = 0; nt<numruns; nt++)
        {
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            p += 0.5*gdt*f;
            p += 0.5*dt*tau*gp;

            //**********
            //* STEP A *
            //**********
            // fixed point iteration
            g0=getg(q+dt/4*p*g);
            g1=getg(q+dt/4*p*g0);
            g0=getg(q+dt/4*p*g1);
            g1=getg(q+dt/4*p*g0);
            gdt=g1*dt;

            q += 0.5*gdt*p;

            //**********
            //* STEP O *
            //**********
            g = getg(q);
            gdt = dt*g;
            C = exp(-gdt*gamma);
            p = C*p + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            // fixed point iteration
            g0=getg(q+dt/4*p*g);
            g1=getg(q+dt/4*p*g0);
            g0=getg(q+dt/4*p*g1);
            g1=getg(q+dt/4*p*g0);
            gdt=g1*dt;
            q += 0.5*gdt*p;

            //**********
            //* STEP B *
            //**********
            f = -Up(q);
            g = getg(q);
            gp=getgprime(q);
            gdt = dt*g;
            p += 0.5*gdt*f;
            p += 0.5*dt*tau*gp;

        
            // save the value every %nsnapshot value
            vec_q[ns][nt]=q;
            vec_p[ns][nt]=p;
            vec_g[ns][nt]=g;
        }
    }
    fstream file;
    string file_name;
    // set up the path 
    string path=PATH;

    for(int nsps = 0; nsps<numsam; nsps++){
        file_name=path+"/vec_tr"+to_string(nsps)+".txt";
        file.open(file_name,ios_base::out);
        ostream_iterator<double> out_itr(file, "\n");
        file<<"q\n";
        copy(vec_q[nsps].begin(), vec_q[nsps].end(), out_itr);
        file<<"p\n";
        copy(vec_p[nsps].begin(), vec_p[nsps].end(), out_itr);
        file<<"g\n";
        copy(vec_g[nsps].begin(), vec_g[nsps].end(), out_itr);
        file.close();
        }
        
    return 0;
    }



int main(void) {    

    //Non adaptive step 
    int run_noada= one_step();

    //Transformed step
    int returnsmthg=one_step_tr();
return 0;
}


































// /////////////////////////////////
// // Spring potential definition //
// /////////////////////////////////
// // Spring potential 
// //parameters of the potential 
// #define a               20.0
// #define b               0.1
// #define x0              0.1
// #define c               0.1

// long double Up(double x)
// {
//    long double xx02= (x-x0)*(x-x0);
//    long double wx =b/(b/a+xx02);
//     return (wx*wx+c)*x;
// }

// double getg(double x)
// {
//     double wx,f,xi,g;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     xi = f+m;
//     g = 1/(1/M+1/sqrt(xi));
//     return(g);

// }

// double getgprime(double x)
// {
//     double wx,f,fp,xi,gprime;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     fp = 4*(x-x0)*((b/a)+pow(x-x0,2))/(b*b);
//     xi=sqrt(f+m*m);
//     gprime= M*M*fp/(2*xi*(xi+M)*(xi+M));
//     return(gprime);
// }

/////////////////////////////////
// Square definition //
/////////////////////////////////


// long double Up(double x)
// {
//    return x;
// }

// double getg(double x)
// {
//     return 1;
// }

// double getgprime(double x)
// {
//     return 0;
// }


/////////////////////////////////
// Using the usual f(x) = 4(x^2-1)
///////////////////////////////////
// double getg(double x)
// {
//     double f,g,xi,fabs;
//     f=4*(x*x-1);
//     fabs=abs(f);
//     xi=fabs*fabs*r+m*m;
//     g=1/M+1/sqrt(xi);
//     return(g);
// }

// double getgprime(double___ x)
// {
//     double f,fp,xi,den1,gprime;
//     f =4*(x*x-1);
//     fp = 8*s*x;
//     xi=sqrt(r*f*f+m*m);
//     den1=xi*xi*xi;
//     gprime= r*f*fp/(den1);
//     return(gprime);
// }
///////////////////////////////////

////////////////////////////////////7
// with the definition 1/M+1/f+m and hessian
///////////////////////////////////

// double getg(double x)
// {
//     double f,g,xi,fabs;
//     f=(8*x);
//     fabs=abs(f);
//     xi=fabs*fabs*r+m*m;
//     g=1/M+1/sqrt(xi);
//     return(g);
// }

// double getgprime(double x)
// {
//     double f,fp,xi,den1,gprime;
//     f =8*x;
//     fp = 8;
//     xi=sqrt(r*f*f+m*m);
//     den1=xi*xi*xi;
//     gprime= -r*f*fp/(den1);
//     return(gprime);
// }


////////////////////////////////////7
// with the definition 1/(1/M+1/f+m) 
///////////////////////////////////

// double getg(double x)
// {
//     double f,g,xi,fabs,fabs2;
//     f=1/((x*x-1));
//     fabs=abs(f);
//     fabs2=fabs*fabs;
//     xi=fabs2*fabs2*r+m*m;
//     g=1/(1/M+1/sqrt(xi));
//     return(g);
// }

// double getgprime(double x)
// {
//     double f,fp,xi,den1,den2,gprime,xx21,num;
//     xx21=x*x-1;
//     f =1/((x*x-1));
//     //fp=(1-3*x*x)/(x*x*xx21*xx21);
//     fp = 2*x/(xx21*xx21);
//     xi=sqrt(r*f*f*f*f+m*m);
//     den1=xi*xi*xi;
//     den2=(1/xi+1/M);
//     num = 2*f*fp*f*f;
//     gprime=num/(den1*den2*den2);
//     return(gprime);
// }
