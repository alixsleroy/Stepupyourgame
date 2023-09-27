
// -g "/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/Python/accuracy/accuracy_openmp_c/main_accuracy.cpp" -o "/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/Python/accuracy/accuracy_openmp_c/main_accuracy" -Ofast -fopenmp
//
//  main.c
//  adaptive
//
//  Created by Alix on 18/05/2022.
//  This is the working code to compute samples from underdamped using splitting scheme 
//  Baoab. This code implements Euler-Maruyama for the transformed SDE. 
//  
// #include <math.h>
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
#include <boost/math/special_functions/sign.hpp>
#include <chrono>

using namespace std::chrono;
 
using namespace std;


#define m1              .005           // minimum step scale factor
#define M1              1./1.   
// #define m               0.9           // minimum step scale factor
// #define M               1.2              // maximum step scale factor
#define numsam          10000           // number of sample
#define T               100       // final time of all simulations 
#define tau             0.1  
#define printskip       1
#define PATH            "./overdamped_2d"

//////////////////
// Anisotropic  //
////////////////// 
//vector<double> dtlist = {0.02 , 0.023, 0.027, 0.03 , 0.033, 0.037, 0.04 , 0.043, 0.047,0.05};
vector<double> dtlist = {0.003 , 0.0039, 0.005 , 0.0065, 0.0084, 0.0109, 0.014 , 0.0181,0.0234, 0.0302};

#define s 15. // parameter of how steep the double well is 
#define c 1. //parameter that determines how high the step size goes in between well, the highest the lowest it goes 


double U(double x, double y)
{
    double res=s*(x*x+y*y-1)*(x*x+y*y-1);
    return res;
}

int sign(float x){
    return (x > 0) - (x < 0);
}

double Upx(double x,double y)
{
    double res=4*x*s*(x*x+y*y-1);
    return res;
}

double Upy(double x,double y)
{
    double res=4*y*s*(x*x+y*y-1);
    return res;
}

double getg(double x, double y)
{
    double xc,xa,f,f2,xi,den,g;
    f=(c*s*pow(x*x+y*y-1,2)*(x*x+y*y));
    f2=f*f;
    xi=sqrt(1+m1*f2);
    den=M1*xi+abs(f);
    g=xi/den;
    return(g);
}


double getgprime_x(double x,double y)
{
    double f,f2,fp,xi,gp;
    f=(c*s*pow(x*x+y*y-1,2)*(x*x+y*y));
    f2=f*f;
    fp=c*s*2*(x*x+y*y-1)*(3*x*x+3*y*y-1)*x;
    xi=sqrt(1+m1*f2);
    gp=-sign(f)*fp/(xi*pow(M1*xi+abs(f),2));
    return(gp);
    }

double getgprime_y(double x,double y)
{
    double f,f2,fp,xi,gp;
    f=(c*s*pow(x*x+y*y-1,2)*(x*x+y*y));
    f2=f*f;
    fp=c*s*2*(x*x+y*y-1)*(3*x*x+3*y*y-1)*y;
    xi=sqrt(1+m1*f2);
    gp=-sign(f)*fp/(xi*pow(M1*xi+abs(f),2));
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
    vector<double> vec_x((numsam/printskip),0);
    vector<double> vec_y((numsam/printskip),0);
    vector<double> moments(4,0);

    double y0,y1,x0,x1;
    int nt,ns,nsp;
    nsp=0;
    #pragma omp parallel private(nt,y0,x0,x1,y1) shared(vec_x,vec_y,ns,moments,nsp)
    #pragma omp for

    for(int ns = 0; ns <= numsam; ns++){ // run the loop for ns samples
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        y0=1.;
        x0=1.;
        for(int nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
            x1=x0-Upx(x0,y0) * ds + sqrt(ds * tau * 2) * normal(generator);
            y1=y0-Upy(x0,y0) * ds + sqrt(ds * tau * 2) * normal(generator);
            x0=x1;
            y0=y1;

        }

        // save the values every printskip
        if(ns%printskip==0){
            vec_x[nsp]=x1;
            vec_y[nsp]=y1;
            nsp+=1;

        }
    

        // compute the moments
        if (std::isnan(x1)==true or std::isnan(y1)==true){x1=pow(10,16); y1=pow(10,16);moments[3]+=1;}
        moments[0]+=x1;
        moments[1]+=y1;
        moments[2]+=U(x1,y1);

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
string file_name=path+"/vec_noada_x"+list_para+".txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec_x.begin(), vec_x.end(), out_itr);
file.close();

file_name=path+"/vec_noada_y"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_y.begin(), vec_y.end(), out_itr);
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
    vector<double> vec_x((numsam/printskip),0);
    vector<double> vec_y((numsam/printskip),0);
    vector<double> vec_g((numsam/printskip),0);
    vector<double> moments(4,0);


    // vector<vector<double>> vec_g(snapshot,vec);
    double gdt,gpdt_x,gpdt_y,y0,y1,x0,x1,g_av;
    int nt,ns,nsp;
    nsp=0;
    #pragma omp parallel private(y1,gdt,gpdt_x,gpdt_y,nt,y0,x0,x1,g_av) shared(vec_x,vec_y,ns,moments,nsp)
    #pragma omp for
    for(ns = 0; ns <= numsam; ns++){ // run the loop for ns samples
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        y0=1.;
        x0=1.;
        g_av=0.;
        for(nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
            gdt=getg(x0,y0)*ds;
            gpdt_x=getgprime_x(x0,y0)*ds;
            gpdt_y=getgprime_y(x0,y0)*ds;

            x1 = x0-Upx(x0,y0)*gdt+gpdt_x*tau+sqrt(2*gdt*tau)*normal(generator);
            y1 = y0-Upy(x0,y0)*gdt+gpdt_y*tau+sqrt(2*gdt*tau)*normal(generator);

            y0 = y1;
            x0 = x1;
            // * Save the values generated for g, so we can get an average
            g_av+=gdt/ds;
        }

    // save the values every printskip
    if(ns%printskip==0){
        vec_x[nsp]=x1;
        vec_y[nsp]=y1;
        vec_g[nsp]=gdt/ds;
        nsp+=1;
    }
 

    // compute the moments
    if (std::isnan(x1)==true or std::isnan(y1)==true){x1=pow(10,16); y1=pow(10,16);moments[3]+=1;}
    moments[0]+=x1;
    moments[1]+=y1;
    moments[2]+=U(x1,y1);
    moments[3]+=g_av/numruns;

    }

// rescale the moments 
moments[0]=moments[0]/numsam;
moments[1]=moments[1]/numsam;
moments[2]=moments[2]/numsam;
moments[3]=moments[3]/numsam;

// copy the value in a txt file
string path=PATH;
fstream file;
file << fixed << setprecision(16) << endl;
string list_para="i="+to_string(i); 
string file_name=path+"/vec_tr_x"+list_para+".txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec_x.begin(), vec_x.end(), out_itr);
file.close();

file_name=path+"/vec_tr_y"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_y.begin(), vec_y.end(), out_itr);
file.close();

file_name=path+"/vec_g"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_g.begin(), vec_g.end(), out_itr);
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


    for(int i = 0; i < dtlist.size(); i++){ 
        
        // run the loop for ns samples
        double dti = dtlist[i];
        double ni = T/dti;

        // transformed 
        vector<double> moments_di=nt_steps_tr(dti,ni,i);
        moments_tr_1[i]=moments_di[0];
        moments_tr_2[i]=moments_di[1];
        moments_tr_3[i]=moments_di[2];
        moments_tr_4[i]=moments_di[3]; //this vector contains the value taken by g overall
  

        double g_av=moments_di[3];
        cout<<"\n";
        cout<<g_av;
        cout<<"\n";

        double gdti=g_av*dti;
  
        moments_di=nt_steps_no_ada(gdti,ni,i);
        moments_1[i]=moments_di[0];
        moments_2[i]=moments_di[1];
        moments_3[i]=moments_di[2];
        moments_4[i]=moments_di[3]; // This vector contains zeros




        // * SAVE THE COMPUTED MOMENTS IN A FILE*//
        ///////////////////////////////////////////
        // We save file over an other            // 

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
    }



    // * SAVE THE TIME AND PARAMETERS OF THE SIMULATION IN A INFO FILE
    ///////////////////////////////////////////////////////////////////
    fstream file;
    string file_name;
    string path=PATH;
    // find time by subtracting stop and start timepoints 
    auto stop = high_resolution_clock::now();
    auto duration_m = duration_cast<minutes>(stop - start);
    auto duration_s = duration_cast<seconds>(stop - start);
    auto duration_ms = duration_cast<microseconds>(stop - start);
    // save the parameters in a file info
    string parameters="DW-M1="+to_string(M1)+"-m1="+to_string(m1)+"-Ns="+to_string(numsam)+"-c="+to_string(c)+"-s="+to_string(s)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    ostream_iterator<double> out_itr(file, "\n");
    string information=path+"/parameters_used.txt";
    file.open(information,ios_base::out);
    file << parameters;
    file <<"\n";
    file <<"list of dt";
    copy(dtlist.begin(), dtlist.end(), out_itr);
    file.close();


    return 0;
}


