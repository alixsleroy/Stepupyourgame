

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

// #define DIVTERM          //define to use
#define m               0.01           // minimum step scale factor
#define M               1.5             // maximum step scale factor
#define dt              0.0005           // artificial time stepsize
// #define T               500.            // final (real) time
#define gamma           0.1            // friction coefficient
#define tau             0.1            // 'temperature'
// #define printskip       10
#define numruns        8000          // total number of trajectories
#define numsam         10        // total number of trajectories
// #define nsnapshot      5
// double printskip=int(numruns/nsnapshot);
/////////////////////////////////
// Anisotropic 1 d definition //
/////////////////////////////////
// Spring potential
//parameters of the potential 
#define s               25
#define r               1

long double Up(double x)
{
    double res=4*s*x*(x*x-1);
    return res;
}

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

double getg(double x)
{
    double f,g,xi,fabs,fabs2;
    f=1/((x*x-1));
    fabs=abs(f);
    fabs2=fabs*fabs;
    xi=fabs2*fabs2*r+m*m;
    g=1/(1/M+1/sqrt(xi));
    return(g);
}

double getgprime(double x)
{
    double f,fp,xi,den1,den2,gprime,xx21,num;
    xx21=x*x-1;
    f =1/((x*x-1));
    //fp=(1-3*x*x)/(x*x*xx21*xx21);
    fp = 2*x/(xx21*xx21);
    xi=sqrt(r*f*f*f*f+m*m);
    den1=xi*xi*xi;
    den2=(1/xi+1/M);
    num = 2*f*fp*f*f;
    gprime=num/(den1*den2*den2);
    return(gprime);
}
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
        q = 0.;
        p = 0.;
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
    string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/1d";
    for(int nsps = 0; nsps<numsam; nsps++){
        file_name=path+"/onetraj/vec_noada"+to_string(nsps)+".txt";
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
    double q,p,f,g,gp,gdt,C;
    int ns,nt,nsp;
    // Savethe values 
    vector<double> q_list(numruns,0);
    vector<vector<double>> vec_q(numsam,q_list);
    vector<vector<double>> vec_p(numsam,q_list);
    vector<vector<double>> vec_g(numsam,q_list);

    #pragma omp parallel private(q,p,f,C,nt,gdt,g,nsp) shared(ns,vec_q,vec_p,vec_g)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        q = 0.;
        p = 0.;
        // f = -Up(q);   // force
        // g = getg(q);
        // gp = getgprime(q); 
        // gdt = g*dt; 

        nsp=0; // initialise the snapshot to 0
        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //
            g = getg(q);
            gdt = dt*g;
            gp=getgprime(q);
            f = -Up(q);   // force
            //**********
            //* STEP B *
            //**********
            p += 0.5*gdt*f;
            // #ifdef DIVTERM
            p += 0.5*dt*tau*gp;
            // #endif

            //**********
            //* STEP A *
            //**********
            q += 0.5*gdt*p;

            //**********
            //* STEP O *
            //**********
            C = exp(-gdt*gamma);
            p = C*p + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            q += 0.5*gdt*p;

            //**********
            //* STEP B *
            //**********
            f = -Up(q);
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
    string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/1d";

    for(int nsps = 0; nsps<numsam; nsps++){
        file_name=path+"/onetraj/vec_tr"+to_string(nsps)+".txt";
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
