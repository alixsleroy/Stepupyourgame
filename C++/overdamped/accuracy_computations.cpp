
// -g "/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/Python/accuracy/accuracy_openmp_c/main_accuracy.cpp" -o "/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/Python/accuracy/accuracy_openmp_c/main_accuracy" -Ofast -fopenmp
//
//  main.c
//  adaptive
//
//  Created by Alix on 18/05/2022.
//  This is the working code to compute samples from underdamped using splitting scheme 
//  Baoab. This code implements Euler-Maruyama for the transformed SDE. 
//  
# include <math.h>
#include <filesystem>
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

#define m               0.001           // minimum step scale factor
#define M               1.5              // maximum step scale factor
#define numsam          100000 //500000           // number of sample
#define T               100       // final time of all simulations 
#define tau             0.1  
          
//parameters of the vector of dt
// vector<double> dtlist = {0.01,0.03,0.05,0.07,0.09,0.1,0.2,0.3,0.4};
// vector<double> dtlist = {0.5};
// vector<double> dtlist = {0.5,0.2};


/////////////////////////////////
// Spring potential definition //
/////////////////////////////////
// // Spring potential 
// //parameters of the potential 
// #define a               1.0
// #define b               1.0
// #define x0              0.5
// #define c               0.1

// Spring potential 
//parameters of the potential 
#define a               3.0
#define b               0.1
#define x0              0.5
#define c               0.1
vector<double> dtlist = {0.002,0.006,0.018,0.05,0.13,0.36};



long double Up(double x)
{
   long double xx02= (x-x0)*(x-x0);
   long double wx =b/(b/a+xx02);
    return (wx*wx+c)*x;
}

double getg(double x)
{
    double wx,f,xi,g;
    wx =(b/a+pow(x-x0,2))/b;
    f = wx*wx;
    xi = f+m;
    g = 1/(1/M+1/sqrt(xi));
    return(g);

}

double getgprime(double x)
{
    double wx,f,fp,xi,gprime;
    wx =(b/a+pow(x-x0,2))/b;
    f = wx*wx;
    fp = 4*(x-x0)*((b/a)+pow(x-x0,2))/(b*b);
    xi=sqrt(f+m*m);
    gprime= M*M*fp/(2*xi*(xi+M)*(xi+M));
    return(gprime);
}

/////////////////////////////
// EM step - no adaptivity //
/////////////////////////////
vector<double> nt_steps_no_ada(double ds, double numruns)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    

    //extern int iseed;
    vector<double> vec(numsam,0);
    double y0,y1;
    int nt,ns;
    #pragma omp parallel private(nt,y0,y1) shared(vec,ns)
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
        // save the final value
        vec[ns]=y0;
    }
return vec;
}

///////////////////////////
// EM step - transformed //
///////////////////////////
vector<double> nt_steps_tr(double ds, double numruns)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    

    //extern int iseed;
    vector<double> vec(numsam,0);
    double y1,gdt,gpdt,y0;
    int nt,ns;
    #pragma omp parallel private(y1,gdt,gpdt,nt,y0) shared(vec,ns)
    #pragma omp for
    for(ns = 0; ns <= numsam; ns++){ // run the loop for ns samples

        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        y0=0;
        for(nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
            gdt=getg(y0)*ds;
            gpdt=getgprime(y0)*ds;
            y1 = y0-Up(y0)*gdt+gpdt*tau+sqrt(2*gdt*tau)*normal(generator);
            y0 = y1;
           
        }

    // save the final value
    vec[ns]=y1;
    }
return vec;
}


/////////////////////////
// EM step - rescaled ///
////////////////////////

vector<double> nt_steps_re(double ds, double numruns)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    

    //extern int iseed;
    vector<double> vec(numsam,0);
    double y1,gdt,gp,y0;
    int nt,ns;
    #pragma omp parallel private(y1,gdt,nt,y0) shared(vec,ns)
    #pragma omp for
    for(ns = 0; ns <= numsam; ns++){ // run the loop for ns samples

        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        y0=0;
        for(nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
           
            gdt= getg(y0)*ds;
            y1 = y0-Up(y0)*gdt+sqrt(2*gdt*tau)*normal(generator);
            y0 = y1;
        }

    // save the final value
    vec[ns]=y1;
    }
return vec;
}

///////////////////////////
////////// MAIN  //////////
///////////////////////////

int main(){

    // set up the path 
    string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/data_overdamped";

    // Compute how much time it takes
    auto start = high_resolution_clock::now();
    using namespace std;

    for(int i = 0; i < dtlist.size(); i++){ // run the loop for ns samples
        // cout<<"i=\n";
        // cout<<i;
        // cout<<"ni = \n";
        double dti = dtlist[i];
        double ni = T/dti;
        // cout<<ni;
        // cout<<"\n";


        // no adaptivity 
        vector<double> vec_noada=nt_steps_no_ada(dti,ni);
        // copy the value in a txt file
        fstream file;
        file << fixed << setprecision(16) << endl;
        string list_para="i="+to_string(i); //+'-M='+to_string(M)+'m='+to_string(m)+"-Nt="+to_string(numruns)+"-Ns="+to_string(numsam);
        string file_name=path+"/vec_noada"+list_para+".txt";
        file.open(file_name,ios_base::out);
        ostream_iterator<double> out_itr(file, "\n");
        copy(vec_noada.begin(), vec_noada.end(), out_itr);
        file.close();


        // transformed 
        vector<double> vec_tr=nt_steps_tr(dti,ni);
        // copy the value in a txt file
        file << fixed << setprecision(16) << endl;
        list_para="i="+to_string(i); //+'-M='+to_string(M)+'m='+to_string(m)+"-Nt="+to_string(numruns)+"-Ns="+to_string(numsam);
        file_name=path+"/vec_tr"+list_para+".txt";
        file.open(file_name,ios_base::out);
        copy(vec_tr.begin(), vec_tr.end(), out_itr);
        file.close();

        // // rescaled 
        // vector<double> vec_re=nt_steps_re(dti,ni);
        // //copy the value in a txt file
        // file << fixed << setprecision(16) << endl;
        // list_para="i="+to_string(i); //+'-M='+to_string(M)+'m='+to_string(m)+"-Nt="+to_string(numruns)+"-Ns="+to_string(numsam);
        // file_name=path+"/vec_re"+list_para+".txt";
        // file.open(file_name,ios_base::out);
        // copy(vec_re.begin(), vec_re.end(), out_itr);
        // file.close();
    }

    // Subtract stop and start timepoints 
    auto stop = high_resolution_clock::now();
    auto duration_m = duration_cast<minutes>(stop - start);
    auto duration_s = duration_cast<seconds>(stop - start);
    auto duration_ms = duration_cast<microseconds>(stop - start);

    // To get the value of duration use the count()
    // member function on the duration object
    // cout << chrono::duration_cast<chrono::seconds>(end - start).count();

    // save the parameters in a file info
    string parameters="Spring-M="+to_string(M)+"-m="+to_string(m)+"-Ns="+to_string(numsam)+"-a="+to_string(a)+"-b="+to_string(b)+"-c="+to_string(c)+"-x0="+to_string(x0)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    fstream file;
    file << fixed << setprecision(16) << endl;
    string information=path+"/parameters_used.txt";
    file.open(information,ios_base::out);
    file << parameters;
    file <<"\n";
    file <<"list of dt";
    ostream_iterator<double> out_itr(file, "\n");
    copy(dtlist.begin(), dtlist.end(), out_itr);
    file.close();


    return 0;
}


