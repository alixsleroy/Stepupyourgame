
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


#define m1              0.01          // minimum step scale factor
#define M1              1./1.1     
#define numsam          5000           // number of sample
#define numruns         10000
#define ds              0.01
#define T               100       // final time of all simulations 
#define tau             0.1  
#define printskip       10
#define PATH     "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/data_overdamped_dw_one_sample"

/////////////////////////////////
// Double well
/////////////////////////////////
#define a 1. // parameter of how steep the double well is 
#define c 1.3 //parameter that determines how high the step size goes in between well, the highest the lowest it goes 

double U(double x){
    return ((x+a)*(x+a)-0.0001)*pow((x-c),4);
    }

double Up(double x){
    double xc =x-c;
    double xa=x+a;
    double v=2*xc*xc*xc*(xa*xc+2*xa*xa-0.0002);
    return v;}


//g depends on (x-c)^3
///////////////////////////////////

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

//g depends on (x-c)^32(x+a)^2
///////////////////////////////////

double getg(double x)
{
    double f,f2,xi,den,g;
    f=pow(x-c,3)*2*pow(x+a,2);
    f2=f*f;
    xi=sqrt(1+m1*f2);
    den=M1*xi+abs(f);
    g=xi/den;
    return(g);
}

double getgprime(double x)
{
    double xc,xa,f,f2,xi,fp,gp;
    f=pow(x-c,3)*2*pow(x+a,2);
    f2=f*f;
    fp=(x+a)*pow(x-c,2)*(3*a-2*c+5*x)*2;
    xi=sqrt(1+m1*f2);
    gp=-f*fp/(sqrt(f2)*xi*pow(M1*xi+abs(f),2));
    return(gp);
    }

/////////////////////////////
// EM step - no adaptivity //
/////////////////////////////
vector<double> nt_steps_no_ada()
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    

    //extern int iseed;
    vector<double> vec(numsam,0);
    double y0,y1;
    int nt,ns;
    #pragma omp parallel private(y0,y1) shared(vec,ns)
    #pragma omp for
    for(int ns = 0; ns <= numsam; ns++){ // run the loop for ns samples
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        y0=1.5;
        // cout<<"\n \n New sim \n\n";
        for(int nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
            y1=y0 -Up(y0) * ds + sqrt(ds * tau * 2) * normal(generator);
            y0=y1;
        }
        // save the final value
        vec[ns]=y0;
    }
return vec;
}


///////////////////////////
// EM step - transformed //
///////////////////////////
vector<double> nt_steps_tr()
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    

    //extern int iseed;
    vector<double> vec(numsam,0);
    double y1,gdt,gpdt,y0;
    int nt,ns;
    #pragma omp parallel private(y0,y1,gdt,gpdt,nt) shared(vec,ns)
    #pragma omp for
    for(ns = 0; ns <= numsam; ns++){ // run the loop for ns samples

        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        y0=1.5;
        for(nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
            gdt=getg(y0)*ds;
            gpdt=getgprime(y0)*ds;
            y1 = y0-Up(y0)*gdt+gpdt*tau+sqrt(2*gdt*tau)*normal(generator);
            y0 = y1;
        }
    vec[ns]=y1;
    }
return vec;
}


/////////////////////////
// EM step - rescaled ///
////////////////////////

vector<double> nt_steps_re()
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    

    //extern int iseed;
    vector<double> vec(numsam,0);
    double y1,gdt,gp;
    #pragma omp parallel
    #pragma omp for
    for(int ns = 0; ns <= numsam; ns++){ // run the loop for ns samples

        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        double y0=1.5;
        for(int nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
           
            gdt= getg(y0)*ds;
            y1 = y0-Up(y0)*gdt+sqrt(2*gdt*tau)*normal(generator);
            y0 = y1;
        }
    vec[ns]=y1;
    }
return vec;
}



///////////////////////////
////////// MAIN  //////////
///////////////////////////

int main(){

    
    // no adaptivity 
    vector<double> vec_noada=nt_steps_no_ada();
    // copy the value in a txt file
    fstream file;
    file << fixed << setprecision(16) << endl;
    string path=PATH;
    string file_name=path+"/vec_noada.txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    copy(vec_noada.begin(), vec_noada.end(), out_itr);
    file.close();


    // transformed 
    vector<double> vec_tr=nt_steps_tr();
    // copy the value in a txt file
    file << fixed << setprecision(16) << endl;
    // list_para="dt="+to_string(ds).substr(0, 5); //+'-M='+to_string(M)+'m='+to_string(m)+"-Nt="+to_string(numruns)+"-Ns="+to_string(numsam);
    file_name=path+"/vec_tr.txt";
    file.open(file_name,ios_base::out);
    copy(vec_tr.begin(), vec_tr.end(), out_itr);
    file.close();

    // rescaled 
    vector<double> vec_re=nt_steps_re();
    //copy the value in a txt file
    file << fixed << setprecision(16) << endl;
    // list_para="dt="+to_string(ds).substr(0, 5); //+'-M='+to_string(M)+'m='+to_string(m)+"-Nt="+to_string(numruns)+"-Ns="+to_string(numsam);
    file_name=path+"/vec_re.txt";
    file.open(file_name,ios_base::out);
    copy(vec_re.begin(), vec_re.end(), out_itr);
    file.close();

    return 0;
}



