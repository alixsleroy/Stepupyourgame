
// -g "/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/Python/accuracy/accuracy_openmp_c/main_accuracy.cpp" -o "/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/Python/accuracy/accuracy_openmp_c/main_accuracy" -Ofast -fopenmp
//
//  main.c
//  adaptive
//
//  Created by Alix on 18/05/2022.
//  This is the working code to compute samples from underdamped using splitting scheme 
//  Baoab. This code implements Euler-Maruyama for the transformed SDE. 
//  
#include <math.h>
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
#include <chrono>

using namespace std::chrono;
 
using namespace std;

#define m1              0.1          // minimum step scale factor
#define M1              1./2. 
#define low             10           // how low the adaptive function goes in between the well  
#define numsam          100000           // number of sample
#define T               100       // final time of all simulations 
#define tau             0.1  
#define printskip       10
#define PATH     "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/data_overdamped_dw"

//vector<double> dtlist = {0.001 , 0.0039, 0.005 , 0.0065, 0.0084, 0.0109, 0.014 , 0.0181,0.0234, 0.0302};
vector<double> dtlist = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 };
/////////////////////////////////
// Double well
/////////////////////////////////
#define a 1. // parameter of how steep the double well is 
#define c 0.6 //parameter that determines how high the step size goes in between well, the highest the lowest it goes 

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
    f=pow(x-c,3)*2*pow(x+a,2)*low;
    f2=f*f;
    xi=sqrt(1+m1*f2);
    den=M1*xi+abs(f);
    g=xi/den;
    return(g);
}

double getgprime(double x)
{
    double xc,xa,f,f2,xi,fp,gp;
    f=low*pow(x-c,3)*2*pow(x+a,2);
    f2=f*f;
    fp=low*(x+a)*pow(x-c,2)*(3*a-2*c+5*x)*2;
    xi=sqrt(1+m1*f2);
    gp=-f*fp/(sqrt(f2)*xi*pow(M1*xi+abs(f),2));
    return(gp);
    }




/////////////////////////////
// EM step - no adaptivity //
/////////////////////////////
vector<double> nt_steps_no_ada(double ds, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    

    //extern int iseed;
    vector<double> vec((numsam/printskip),0);
    vector<double> moments(4,0);

    double y0,y1;
    int nt,ns,nsp;
    nsp=0;
    #pragma omp parallel private(nt,y0,y1) shared(vec,ns,moments,nsp)
    #pragma omp for
    for(int ns = 0; ns <= numsam; ns++){ // run the loop for ns samples

        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        y0=0;
        // cout<<"\n \n New sim \n\n";
        for(int nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
            y1=y0-Up(y0) * ds + sqrt(ds * tau * 2) * normal(generator);
            y0=y1;
        }
        // save the values every printskip
        if(ns%printskip==0){
            vec[nsp]=y1;
            nsp+=1;

        }
    

        // compute the moments
        moments[0]+=y1;
        moments[1]+=y1*y1;
        moments[2]+=y1*y1*y1;
        moments[3]+=y1*y1*y1*y1;

    }

// rescale the moments 
moments[0]=moments[0]/numsam;
moments[1]=moments[1]/numsam;
moments[2]=moments[2]/numsam;
moments[3]=moments[3]/numsam;


// set up the path
string path=PATH;
fstream file;
file << fixed << setprecision(16) << endl;
string list_para="i="+to_string(i); //+'-M='+to_string(M)+'m='+to_string(m)+"-Nt="+to_string(numruns)+"-Ns="+to_string(numsam);
string file_name=path+"/vec_noada"+list_para+".txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec.begin(), vec.end(), out_itr);
file.close();

return moments;
}

///////////////////////////
// EM step - transformed //
///////////////////////////
vector<double> nt_steps_tr(double ds, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    

    // create vectors
    vector<double> vec((numsam/printskip),0);
    vector<double> vec_g((numsam/printskip),0);
    vector<double> moments(5,0);


    // vector<vector<double>> vec_g(snapshot,vec);
    double y1,gdt,gpdt,y0,g_av;
    int nt,ns,nsp;
    nsp=0;
    #pragma omp parallel private(y1,gdt,gpdt,nt,y0,g_av) shared(vec,ns,moments,nsp)
    #pragma omp for
    for(ns = 0; ns <= numsam; ns++){ // run the loop for ns samples

        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        y0=0;
        g_av=0.;
        for(nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
            gdt=getg(y0)*ds;
            gpdt=getgprime(y0)*ds;

            y1 = y0-Up(y0)*gdt+gpdt*tau+sqrt(2*gdt*tau)*normal(generator);
            y0 = y1;
            
            // * Save the values generated for g, so we can get an average
            g_av+=gdt/ds;
        }

    // save the values every printskip
    if(ns%printskip==0){
        vec[nsp]=y1;
        vec_g[nsp]=gdt/ds;
        nsp+=1;
    }
 

    // compute the moments
    moments[0]+=y1;
    moments[1]+=y1*y1;
    moments[2]+=y1*y1*y1;
    moments[3]+=y1*y1*y1*y1;
    moments[4]+=g_av/numruns;

    }

// rescale the moments 
moments[0]=moments[0]/numsam;
moments[1]=moments[1]/numsam;
moments[2]=moments[2]/numsam;
moments[3]=moments[3]/numsam;
moments[4]=moments[4]/numsam;


// copy the value in a txt file
string path=PATH;
fstream file;
file << fixed << setprecision(16) << endl;
string list_para="i="+to_string(i); 
string file_name=path+"/vec_tr"+list_para+".txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec.begin(), vec.end(), out_itr);
file.close();

file_name=path+"/vec_g"+list_para+".txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr3(file, "\n");
copy(vec_g.begin(), vec_g.end(), out_itr3);
file.close();

return moments;
}

///////////////////////////
////////// MAIN  //////////
///////////////////////////

int main(){


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
    vector<double> moments_tr_5(dtlist.size(),0);


    for(int i = 0; i < dtlist.size(); i++){ // run the loop for ns samples

        double dti = dtlist[i];
        double ni = T/dti;

        // transformed 
        vector<double> moments_di=nt_steps_tr(dti,ni,i);
        moments_tr_1[i]=moments_di[0];
        moments_tr_2[i]=moments_di[1];
        moments_tr_3[i]=moments_di[2];
        moments_tr_4[i]=moments_di[3];
        moments_tr_5[i]=moments_di[4];

        double g_av = moments_di[4];

        cout<<"\ni\n";
        cout<<i;
        cout<<"\n gav\n";
        cout<<g_av;

        // no adaptivity 
        moments_di=nt_steps_no_ada(g_av*dti,ni,i);
        moments_1[i]=moments_di[0];
        moments_2[i]=moments_di[1];
        moments_3[i]=moments_di[2];
        moments_4[i]=moments_di[3];

    }

    // * SAVE THE COMPUTED MOMENTS IN A FILE
    /////////////////////////////////////////
    string path=PATH;

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

    file_name=path+"/tr_moment5.txt";
    file.open(file_name,ios_base::out);
    copy(moments_tr_5.begin(), moments_tr_5.end(), out_itr);
    file.close();

    // * SAVE THE TIME AND PARAMETERS OF THE SIMULATION IN A INFO FILE
    ///////////////////////////////////////////////////////////////////
    // find time by subtracting stop and start timepoints 
    auto stop = high_resolution_clock::now();
    auto duration_m = duration_cast<minutes>(stop - start);
    auto duration_s = duration_cast<seconds>(stop - start);
    auto duration_ms = duration_cast<microseconds>(stop - start);
    // save the parameters in a file info
    string parameters="Spring-M1="+to_string(M1)+"-m1="+to_string(m1)+"-Ns="+to_string(numsam)+"-a="+to_string(a)+"-c="+to_string(c)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    string information=path+"/parameters_used.txt";
    file.open(information,ios_base::out);
    file << parameters;
    file <<"\n";
    file <<"list of dt";
    copy(dtlist.begin(), dtlist.end(), out_itr);
    file.close();


    return 0;
}


