
// -g "/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/Python/accuracy/accuracy_openmp_c/main_accuracy.cpp" -o "/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/Python/accuracy/accuracy_openmp_c/main_accuracy" -Ofast -fopenmp
//
//  main.c
//  adaptive
//
//  Created by Alix on 18/05/2022.
//   
//  
# include <math.h>
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

using namespace std;
#define m1              0.001          // minimum step scale factor
#define M1              1./1.5 
#define low             10           // how low the adaptive function goes in between the well
#define numsam          5           // number of sample
#define numruns         1000
#define ds              0.07
#define T               100       // final time of all simulations 
#define tau             0.115  
#define printskip       10

// PATH 1 
//#define PATH     "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/data_overdamped_ani/dt1";
// PATH 2 
//#define PATH     "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/data_overdamped_ani/dt2";
// PATH 3 
#define PATH     "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/data_overdamped_ani/dt3";

// /////////////////////////////////
// // Double well
// /////////////////////////////////
// #define a 1. // parameter of how steep the double well is 
// #define c .6 //parameter that determines how high the step size goes in between well, the highest the lowest it goes 

// double U(double x){
//     return ((x+a)*(x+a)-0.0001)*pow((x-c),4);
//     }

// double Up(double x){
//     double xc =x-c;
//     double xa=x+a;
//     double v=2*xc*xc*xc*(xa*xc+2*xa*xa-0.0002);
//     return v;}


// //g depends on (x-c)^3
// ///////////////////////////////////

// double getg(double x)
// {
//     double f,f2,xi,den,g;
//     f=pow(x-c,3);
//     f2=f*f;
//     xi=sqrt(1+m1*f2);
//     den=M1*xi+abs(f);
//     g=xi/den;
//     return(g);
// }

// double getgprime(double x)
// {
//     double xc,xa,f,f2,xi,fp,gp;
//     f=pow(x-c,3);
//     f2=f*f;
//     fp=3*pow(x-c,2);
//     xi=sqrt(1+m1*f2);
//     gp=-f*fp/(sqrt(f2)*xi*pow(M1*xi+abs(f),2));
//     return(gp);
//     }

// //g depends on (x-c)^32(x+a)^2
// ///////////////////////////////////

// double getg(double x)
// {
//     double f,f2,xi,den,g;
//     f=pow(x-c,3)*2*pow(x+a,2)*low;
//     f2=f*f;
//     xi=sqrt(1+m1*f2);
//     den=M1*xi+abs(f);
//     g=xi/den;
//     return(g);
// }

// double getgprime(double x)
// {
//     double xc,xa,f,f2,xi,fp,gp;
//     f=low*pow(x-c,3)*2*pow(x+a,2);
//     f2=f*f;
//     fp=low*(x+a)*pow(x-c,2)*(3*a-2*c+5*x)*2;
//     xi=sqrt(1+m1*f2);
//     gp=-f*fp/(sqrt(f2)*xi*pow(M1*xi+abs(f),2));
//     return(gp);
//     }


///////////////////////////////
//Anisotropic
///////////////////////////////
#define s 2. // parameter of how steep the double well is 
#define c 0.1 //parameter that determines how high the step size goes in between well, the highest the lowest it goes 

double Up(double x)
{
    double res=4*s*x*(x*x-1); //-3*x*x;
    return res;
}

//g depends on 1/((x-c)^3)
///////////////////////////////////

double getg(double x)
{
    double xc,xa,f,f2,xi,den,g;
    xc=x-1;
    xa=x+1;
    f=abs(c*s*xa*xa*xc*xc);
    f2=f*f;
    xi=sqrt(1+m1*f2);
    den=M1*xi+f;
    g=xi/den;
    return(g);
}

double getgprime(double x)
{
    double xc,xa,f,f2,xi,fp,gp;
    xc=x-1;
    xa=x+1;
    f=c*s*xa*xa*xc*xc;
    f2=f*f;
    fp=c*4*s*(x*x-1)*x;
    xi=sqrt(1+m1*f2);
    gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
    return(gp);
    }



/////////////////////////////////
// Spring potential definition //
/////////////////////////////////
//parameters of the potential 
// #define a               5.0
// #define b               0.1
// #define x0              0.1
// #define c               0.1

// square potential to test things out 
// double Up(double x)
// {
//     return 4*x*x*x;
// }

// // Spring potential 
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


//////////////////////////
// Double well potential 
/////////////////////////

// #define a 0.45
// #define c 1.35



// long double Up(double x)
// {
//     double xc =x-c;
//     double xa=x+a;
//     double v=2*xc*xc*xc*(xa*xc+2*xa*xa-0.0002);
//     return v;
// }

//g depends on 1/((x-c)^3 (x-a)^2)
///////////////////////////////////


// double getg(double x)
// {
//     double xc,xa,f,f2,xi,den,g;
//     xc=x-c;
//     xa=x+a;
//     f=0.5*abs(pow(xc,3)*pow(xa,2));
//     f2=f*f;
//     xi=sqrt(1+m1*f2);
//     den=M1*xi+f;
//     g=xi/den;
//     return(g);
// }

// double getgprime(double x)
// {
//     double xc,xa,f,f2,xi,fp,gp;
//     xc=x-c;
//     xa=x+a;
//     f=0.5*abs(pow(xc,3)*pow(xa,2));
//     f2=f*f;
//     fp=0.5*pow(xa,3)*pow(xc,5)*(3*a-2*c+5*x)/abs(f);
//     xi=sqrt(1+m1*f2);
//     gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
//     return(gp);
//     }

// //g depends on 1/((x-c)^3)
// ///////////////////////////////////

// double getg(double x)
// {
//     double xc,xa,f,f2,xi,den,g;
//     xc=x-c;
//     xa=x+a;
//     f=0.5*abs(pow(xc,3));
//     f2=f*f;
//     xi=sqrt(1+m1*f2);
//     den=M1*xi+f;
//     g=xi/den;
//     return(g);
// }

// double getgprime(double x)
// {
//     double xc,xa,f,f2,xi,fp,gp;
//     xc=x-c;
//     xa=x+a;
//     f=0.5*abs(pow(xc,3));
//     f2=f*f;
//     fp=1.5*pow(xc,2)*f/abs(f);
//     xi=sqrt(1+m1*f2);
//     gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
//     return(gp);
//     }


// //g depends on 1/((x-c)^3 ((x-a)^2+0.1)
// // so that the increase in step size in the smaller well is smaller
// ///////////////////////////////////
// double sign(double x){
// double res=0;
// if (x > 0){res=1;}
// if (x < 0) {res=-1;}
// return res;   
// }

// double getg(double x)
// {
//     double xc,xa,f,f2,xi,den,g;
//     xc=x-c;
//     xa=x+a;
//     f=0.5*abs(pow(xc,3)*(pow(xa,2)+0.15));
//     f2=f*f;
//     xi=sqrt(1+m1*f2);
//     den=M1*xi+f;
//     g=xi/den;
//     return(g);
// }

// double getgprime(double x)
// {
//     double xc,xa,f,f2,xi,fp,gp;
//     xc=x-c;
//     xa=x+a;
//     f=0.5*abs(pow(xc,3)*(pow(xa,2)+0.1));
//     f2=f*f;
//     fp=sign(f)*xc*xc*(2*xa*xc+3*xa*xa+0.45);
//     xi=sqrt(1+m1*f2);
//     gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
//     return(gp);
//     }


/////////////////////////////
// EM step - no adaptivity //
/////////////////////////////
int nt_steps_no_ada()
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    

    //extern int iseed;
    // vector<double> vec(numsam,0);
    // Savethe values 
    vector<double> q_list(numruns+1,0);
    vector<vector<double>> vec(numsam+1,q_list);


    double y0,y1;
    int nt,ns;
    // #pragma omp parallel private(y0,y1) shared(vec,ns)
    // #pragma omp for
    for(int ns = 0; ns <= numsam; ns++){ // run the loop for ns samples
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        y0=0.5;
        // cout<<"\n \n New sim \n\n";
        for(int nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
            y1=y0 -Up(y0) * ds + sqrt(ds * tau * 2) * normal(generator);
            y0=y1;
            vec[ns][nt]=y1;

        }
    }
    
    fstream file;
    string file_name;
    string path=PATH;
    for(int nsps = 0; nsps<numsam; nsps++){
        file_name=path+"/vec_noada"+to_string(nsps)+".txt";
        file.open(file_name,ios_base::out);
        ostream_iterator<double> out_itr(file, "\n");
        // file<<"y\n";
        copy(vec[nsps].begin(), vec[nsps].end(), out_itr);
        file.close();
        }

return 0;
}


///////////////////////////
// EM step - transformed //
///////////////////////////
int nt_steps_tr()
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    
      // Savethe values 
    vector<double> q_list(numruns+1,0);
    vector<vector<double>> vec(numsam+1,q_list);
    vector<vector<double>> vec_g(numsam+1,q_list);

    // extern int iseed;
    // vector<double> vec(numsam,0);
    double y1,gdt,gpdt,y0;
    int nt,ns;
    // #pragma omp parallel private(y0,y1,gdt,gpdt,nt) shared(vec,ns)
    // #pragma omp for
    for(ns = 0; ns <= numsam; ns++){ // run the loop for ns samples

        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        y0=0.5;
        for(nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
            gdt=getg(y0)*ds;
            gpdt=getgprime(y0)*ds;
            y1 = y0-Up(y0)*gdt+gpdt*tau+sqrt(2*gdt*tau)*normal(generator);
            y0 = y1;
            vec[ns][nt]=y1;
            vec_g[ns][nt]=gdt/ds;

        }
    }

    // Save values of t
    fstream file;
    string file_name;
    string path=PATH;
    for(int nsps = 0; nsps<numsam; nsps++){
        file_name=path+"/vec_tr"+to_string(nsps)+".txt";
        file.open(file_name,ios_base::out);
        ostream_iterator<double> out_itr(file, "\n");
        // file<<"y\n";
        copy(vec[nsps].begin(), vec[nsps].end(), out_itr);
        file.close();
        }

    // save values of vec g 
    for(int nsps = 0; nsps<numsam; nsps++){
        file_name=path+"/vec_g"+to_string(nsps)+".txt";
        file.open(file_name,ios_base::out);
        ostream_iterator<double> out_itr(file, "\n");
        // file<<"y\n";
        copy(vec_g[nsps].begin(), vec_g[nsps].end(), out_itr);
        file.close();
        }

return 0;
}


/////////////////////////
// EM step - rescaled ///
////////////////////////

int nt_steps_re()
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    

    //extern int iseed;
    // vector<double> vec(numsam,0);

    // Savethe values 
    vector<double> q_list(numruns+1,0);
    vector<vector<double>> vec(numsam+1,q_list);

    double y1,gdt,gp;
    // #pragma omp parallel
    // #pragma omp for
    for(int ns = 0; ns <= numsam; ns++){ // run the loop for ns samples

        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        double y0=0.5;
        for(int nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
           
            gdt= getg(y0)*ds;
            y1 = y0-Up(y0)*gdt+sqrt(2*gdt*tau)*normal(generator);
            y0 = y1;
            vec[ns][nt]=y1;
        }
    // vec[ns]=y1;
    }
    fstream file;
    string file_name;
    string path=PATH; 
    for(int nsps = 0; nsps<numsam; nsps++){
        file_name=path+"/vec_re"+to_string(nsps)+".txt";
        file.open(file_name,ios_base::out);
        ostream_iterator<double> out_itr(file, "\n");
        // file<<"y\n";
        copy(vec[nsps].begin(), vec[nsps].end(), out_itr);
        file.close();
        }

return 0;
}



///////////////////////////
////////// MAIN  //////////
///////////////////////////

int main(){
    
    // no adaptivity 
    int vec_noada=nt_steps_no_ada();

    // transformed 
    int vec_tr=nt_steps_tr();

    // rescaled 
    // int vec_re=nt_steps_re();

    return 0;
}



