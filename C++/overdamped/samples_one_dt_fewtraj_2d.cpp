
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
#define m1              .01        // minimum step scale factor
#define M1              1./1.4              // maximum step scale factor
#define numsam          10          // number of sample
#define T               50.         // total number of trajectories
#define ds              0.01
#define tau             0.1            
#define numruns         T/ds         // total number of trajectories


//////////////////
// Anisotropic  //
////////////////// 
//vector<double> dtlist = {0.02 , 0.023, 0.027, 0.03 , 0.033, 0.037, 0.04 , 0.043, 0.047,0.05};
vector<double> dtlist = {0.003 , 0.0039, 0.005 , 0.0065, 0.0084, 0.0109, 0.014 , 0.0181,0.0234, 0.0302};

#define s 15. // parameter of how steep the double well is 
#define c .4 //parameter that determines how high the step size goes in between well, the highest the lowest it goes 


double U(double x, double y)
{
    double res=s*(x*x+y*y-1)*(x*x+y*y-1);
    return res;
}

double sign(double x){
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
int nt_steps_no_ada()
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    

    //extern int iseed;
    // vector<double> vec(numsam,0);
    // Savethe values 
    vector<double> q_list(numruns+1,0);
    vector<vector<double>> vec_x(numsam+1,q_list);
    vector<vector<double>> vec_y(numsam+1,q_list);


    double y0,y1,x0,x1;
    int nt,ns;
    // #pragma omp parallel private(y0,y1) shared(vec,ns)
    // #pragma omp for
    for(int ns = 0; ns <= numsam; ns++){ // run the loop for ns samples
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        y0=0.;
        x0=1.;
        // cout<<"\n \n New sim \n\n";
        for(int nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
            x1=x0-Upx(x0,y0) * ds + sqrt(ds * tau * 2) * normal(generator);
            y1=y0-Upy(x0,y0) * ds + sqrt(ds * tau * 2) * normal(generator);
            x0=x1;
            y0=y1;
            vec_x[ns][nt]=x1;
            vec_y[ns][nt]=y1;


        }
    }
    
    fstream file;
    string file_name;
    string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/data_overdamped_fewtraj_2d";
    for(int nsps = 0; nsps<numsam; nsps++){
        file_name=path+"/vec_noada_x"+to_string(nsps)+".txt";
        file.open(file_name,ios_base::out);
        ostream_iterator<double> out_itr(file, "\n");
        copy(vec_x[nsps].begin(), vec_x[nsps].end(), out_itr);
        file.close();

        file_name=path+"/vec_noada_y"+to_string(nsps)+".txt";
        file.open(file_name,ios_base::out);
        copy(vec_y[nsps].begin(), vec_y[nsps].end(), out_itr);
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
    vector<vector<double>> vec_x(numsam+1,q_list);
    vector<vector<double>> vec_y(numsam+1,q_list);

    vector<vector<double>> vec_g(numsam+1,q_list);

    // extern int iseed;
    // vector<double> vec(numsam,0);
    double y1,gdt,gpdt_x,gpdt_y,y0,x0,x1;
    int nt,ns;
    // #pragma omp parallel private(y0,y1,gdt,gpdt,nt) shared(vec,ns)
    // #pragma omp for
    for(ns = 0; ns <= numsam; ns++){ // run the loop for ns samples

        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        y0=0.;
        x0=1.;
        for(nt = 0; nt<numruns; nt++) // run until time T (ie for Nt steps)
        {  
            gdt=getg(x0,y0)*ds;
            gpdt_x=getgprime_x(x0,y0)*ds;
            gpdt_y=getgprime_y(x0,y0)*ds;

            x1 = x0-Upx(x0,y0)*gdt+gpdt_x*tau+sqrt(2*gdt*tau)*normal(generator);
            y1 = y0-Upy(x0,y0)*gdt+gpdt_y*tau+sqrt(2*gdt*tau)*normal(generator);

            y0 = y1;
            x0 = x1;

            vec_x[ns][nt]=x1;
            vec_y[ns][nt]=y1;
            vec_g[ns][nt]=gdt/ds;

        }
    }

    // Save values of t
    fstream file;
    string file_name;
    string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/data_overdamped_fewtraj_2d";
    for(int nsps = 0; nsps<numsam; nsps++){
        file_name=path+"/vec_tr_x"+to_string(nsps)+".txt";
        file.open(file_name,ios_base::out);
        ostream_iterator<double> out_itr(file, "\n");
        copy(vec_x[nsps].begin(), vec_x[nsps].end(), out_itr);
        file.close();

        file_name=path+"/vec_tr_y"+to_string(nsps)+".txt";
        file.open(file_name,ios_base::out);
        copy(vec_y[nsps].begin(), vec_y[nsps].end(), out_itr);
        file.close();

        }

    // save values of vec g 
    for(int nsps = 0; nsps<numsam; nsps++){
        file_name=path+"/vec_g"+to_string(nsps)+".txt";
        file.open(file_name,ios_base::out);
        ostream_iterator<double> out_itr(file, "\n");
        copy(vec_g[nsps].begin(), vec_g[nsps].end(), out_itr);
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


    return 0;
}



