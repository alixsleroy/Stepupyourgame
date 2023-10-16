

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
#define gamma           0.1            // friction coefficient
#define tau             0.1            // 'temperature'
#define T               100          // Time to integrate to
#define numsam          50000        // total number of trajectories
#define printskip       100

vector<double> dtlist = {exp(-4.5),exp(-4.21),exp(-3.93),exp(-3.64),exp(-3.36),exp(-3.07),exp(-2.79),exp(-2.5),exp(-2.21)}; //,exp(-1.93),exp(-1.64),exp(-1.36),exp(-1.07), exp(-0.79), exp(-0.5)};

/////////////////////////////////
// Double well 

#define a 1.
#define c 1.
#define d 10.


long double Up(double x)
{
    double xc = x-c;
    double xa=x+a;
    double v=2*xc*xc*xc*(xa*xc+2*xa*xa-0.2);
    return v;
}


////////////////////////////////////7
// with the definition 1/(1/M+1/f+m) 
///////////////////////////////////

double getg(double x)
{
    double xc = x-c;
    double xa=x+a;
    double f =d* 1/(2*xc*xc*xc*(xa*xc+2*xa*xa-0.2));
    double g=1/(1/M+1/sqrt(f*f+m*m));
    return g;
}

double getgprime(double x)
{
    double xc = x-c;
    double xa=x+a;
    double f = d*1/(2*xc*xc*xc*(xa*xc+2*xa*xa-0.2));
    double fpnum=-3*a*a+4*a*c-10*a*x-0.5*c*c+5*c*x-7.5*x*x+0.3;
    double den1=(-2*a*a+a*c-5*a*x+c*x-3*x*x+0.2);
    double fpden=xc*xc*xc*xc*den1*den1;
    double fp=d*-fpnum/fpden;
    double xi=sqrt(f*f+m*m);
    double gp = M*M*f*fp/((xi+M)*(xi+M)*xi);
    return(gp);
}
/////////////////////////////////
// Non adaptive one step function //
/////////////////////////////////

vector<double> one_step(double dt, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double q,p,f,g,gp,gdt,C;
    int ns,nt,nsp;
    // Save the values 
    vector<double> vec_q(numsam/printskip,0);
    vector<double> vec_p(numsam/printskip,0);

    // Compute the moments, so its done
    vector<double> moments(8,0);
    nsp=0;

    #pragma omp parallel private(q,p,f,C,nt,gdt) shared(nsp,ns,vec_q,vec_p,moments)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        q = 0.;
        p = 0.;
        f = -Up(q);  
        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            p += 0.5*dt*f;

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
            p += 0.5*dt*f;
        }

    // compute the moments for p and for q 
    // compute the moments for p 
    moments[0]+=p;
    moments[1]+=p*p;
    moments[2]+=p*p*p;
    moments[3]+=p*p*p*p;

    // compute the moments for q 
    moments[4]+=q;
    moments[5]+=q*q;
    moments[6]+=q*q*q;
    moments[7]+=q*q*q*q;

    // Save every printskip values    
    if(ns%printskip==0){
        vec_q[nsp]=q;
        vec_p[nsp]=p;
        nsp+=1;
        }
    }

// rescale the moments 
moments[0]=moments[0]/numsam;
moments[1]=moments[1]/numsam;
moments[2]=moments[2]/numsam;
moments[3]=moments[3]/numsam;
moments[4]=moments[4]/numsam;
moments[5]=moments[5]/numsam;
moments[6]=moments[6]/numsam;
moments[7]=moments[7]/numsam;

// save the some of the values generated. 
string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/1d";
fstream file;
file << fixed << setprecision(16) << endl;
string list_para="i="+to_string(i); 
string file_name=path+"/vec_noada_q"+list_para+".txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec_q.begin(), vec_q.end(), out_itr);
file.close();

file_name=path+"/vec_noada_p"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_p.begin(), vec_p.end(), out_itr);
file.close();

return moments;
}



vector<double> one_step_tr(double dt, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double q,p,f,g,gp,gdt,C;
    int ns,nt,nsp;

    // Savethe values 
    vector<double> vec_q((numsam/printskip),0);
    vector<double> vec_p((numsam/printskip),0);
    vector<double> vec_g((numsam/printskip),0);

    // Compute the moments, so its done
    vector<double> moments(8,0);

    // Initialise snapshot
    nsp=0;
    #pragma omp parallel private(q,p,f,C,nt,gdt,g) shared(ns,vec_q,vec_p,vec_g,moments,nsp)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        q = 0.;
        p = 0.;
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
            p += 0.5*dt*tau*gp;

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
            // p += 0.5*dt*tau*gp;
        }

    
    // compute the moments for p and for q 
    // compute the moments for p 
    moments[0]+=p;
    moments[1]+=p*p;
    moments[2]+=p*p*p;
    moments[3]+=p*p*p*p;

    // compute the moments for q 
    moments[4]+=q;
    moments[5]+=q*q;
    moments[6]+=q*q*q;
    moments[7]+=q*q*q*q;

    // Save every printskip values    
    if(ns%printskip==0){
        vec_q[nsp]=q;
        vec_p[nsp]=p;
        vec_g[nsp]=g;
        nsp+=1;
        }
    
    }

    // rescale the moments 
    moments[0]=moments[0]/numsam;
    moments[1]=moments[1]/numsam;
    moments[2]=moments[2]/numsam;
    moments[3]=moments[3]/numsam;
    moments[4]=moments[4]/numsam;
    moments[5]=moments[5]/numsam;
    moments[6]=moments[6]/numsam;
    moments[7]=moments[7]/numsam;


    // save the some of the values generated. 
    string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/1d";
    fstream file;
    file << fixed << setprecision(16) << endl;
    string list_para="i="+to_string(i); 
    string file_name=path+"/vec_tr_q"+list_para+".txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    copy(vec_q.begin(), vec_q.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_p"+list_para+".txt";
    file.open(file_name,ios_base::out);
    copy(vec_p.begin(), vec_p.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_g"+list_para+".txt";
    file.open(file_name,ios_base::out);
    copy(vec_g.begin(), vec_g.end(), out_itr);
    file.close();

    // return the saved moments 
    return moments;
    }



int main(void) {    

    // Compute how much time it takes
    auto start = high_resolution_clock::now();
    using namespace std;
    vector<double> moments_1(dtlist.size(),0);
    vector<double> moments_2(dtlist.size(),0);
    vector<double> moments_3(dtlist.size(),0);
    vector<double> moments_4(dtlist.size(),0);

    vector<double> moments_tr_1(dtlist.size(),0);
    vector<double> moments_tr_2(dtlist.size(),0);
    vector<double> moments_tr_3(dtlist.size(),0);
    vector<double> moments_tr_4(dtlist.size(),0);


    for(int i = 0; i < dtlist.size(); i++){ // run the loop for ns samples

        double dti = dtlist[i];
        double ni = T/dti;

        // no adaptivity 
        vector<double> moments_di=one_step(dti,ni,i);
        moments_1[i]=moments_di[4];
        moments_2[i]=moments_di[5];
        moments_3[i]=moments_di[6];
        moments_4[i]=moments_di[7];


        // transformed 
        moments_di=one_step_tr(dti,ni,i);
        moments_tr_1[i]=moments_di[4];
        moments_tr_2[i]=moments_di[5];
        moments_tr_3[i]=moments_di[6];
        moments_tr_4[i]=moments_di[7];
 
    }

       // * SAVE THE COMPUTED MOMENTS IN A FILE
    /////////////////////////////////////////
    string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/1d";

    // NON ADAPTIVE
    fstream file;
    file << fixed << setprecision(16) << endl;
    string file_name=path+"/noada_moment1.txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    copy(moments_1.begin(), moments_1.end(), out_itr);
    file.close();

    file_name=path+"/noada_moment2.txt";
    file.open(file_name,ios_base::out);
    copy(moments_2.begin(), moments_2.end(), out_itr);
    file.close();

    file_name=path+"/noada_moment3.txt";
    file.open(file_name,ios_base::out);
    copy(moments_3.begin(), moments_3.end(), out_itr);
    file.close();

    file_name=path+"/noada_moment4.txt";
    file.open(file_name,ios_base::out);
    copy(moments_4.begin(), moments_4.end(), out_itr);
    file.close();

    // TRANSFORMED
    file_name=path+"/tr_moment1.txt";
    file.open(file_name,ios_base::out);
    copy(moments_tr_1.begin(), moments_tr_1.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment2.txt";
    file.open(file_name,ios_base::out);
    copy(moments_tr_2.begin(), moments_tr_2.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment3.txt";
    file.open(file_name,ios_base::out);
    copy(moments_tr_3.begin(), moments_tr_3.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment4.txt";
    file.open(file_name,ios_base::out);
    copy(moments_tr_4.begin(), moments_tr_4.end(), out_itr);
    file.close();

    // * SAVE THE TIME AND PARAMETERS OF THE SIMULATION IN A INFO FILE
    ///////////////////////////////////////////////////////////////////
    // find time by subtracting stop and start timepoints 
    auto stop = high_resolution_clock::now();
    auto duration_m = duration_cast<minutes>(stop - start);
    auto duration_s = duration_cast<seconds>(stop - start);
    auto duration_ms = duration_cast<microseconds>(stop - start);
    // save the parameters in a file info
    string parameters="M="+to_string(M)+"-m="+to_string(m)+"-Ns="+to_string(numsam)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    string information=path+"/parameters_used.txt";
    file.open(information,ios_base::out);
    file << parameters;
    file <<"\n";
    file <<"list of dt";
    copy(dtlist.begin(), dtlist.end(), out_itr);
    file.close();

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
