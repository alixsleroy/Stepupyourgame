

// anisotropic with two dimensions, with stepsize larger where potential steeper 

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

// Xtensor vector
// #include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"



using namespace std::chrono;
 
using namespace std;

// # include <math.h>
// # include <complex.h>
// # include <stdlib.h>

// # include <stdio.h>
// # include <time.h>
// # include "normal.h"

// #define DIVTERM          //define to use
#define m               0.001           // minimum step scale factor
#define M               1.2             // maximum step scale factor
#define dt              0.001           // artificial time stepsize
// #define T               500.            // final (real) time
#define gamma           0.1            // friction coefficient
#define tau             0.1            // 'temperature'
// #define printskip       10
#define numruns        1000 //1000000          // total number of trajectories
#define numsam         1000          // total number of trajectories


/////////////////////////////////
// Anisotropic 1 d definition //
/////////////////////////////////
// Spring potential
//parameters of the potential 
#define s               10
#define r               1


long double Upx(double x,double y)
{
    double res=4*x*s*(x*x+y*y-1);
    return res;
}

long double Upy(double x,double y)
{
    double res=4*y*s*(x*x+y*y-1);
    return res;
}

double getg(double x,double y)
{
    double f,g,xi,fabs;
    f =(4*(x*x+y*y-1));
    // f =(4*(abs(x)+abs(y)));

    fabs=abs(f);
    xi=fabs*fabs*r+m*m;
    g=1/M+1/sqrt(xi);
    return(g);
}

double getgprime_x(double x,double y)
{
    double f,fp,xi,den1,gprime;
    // f =(4*(abs(x)+abs(y)));
    // fp = 8*x;

    f =(4*(x*x+y*y-1));
    fp = 8*x;
    xi=sqrt(r*f*f+m*m);
    den1=xi*xi*xi;
    gprime= -r*f*fp/(den1);
    return(gprime);
}

double getgprime_y(double x,double y)
{
    double f,fp,xi,den1,gprime;
    // f =(4*(abs(x)+abs(y)));
    // fp = 8*x;
    f =(4*(x*x+y*y-1));
    fp = 8*x;
    xi=sqrt(r*f*f+m*m);
    den1=xi*xi*xi;
    gprime= -r*f*fp/(den1);
    return(gprime);
}


/////////////////////////////////
// Non adaptive one step function //
/////////////////////////////////


vector<vector<double>> one_step(void)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,gpx,C,fx,dwx;
    double qy,py,gpy,fy,dwy;
    
    vector<double> q_list(numsam,0);

    // savec the values of qx,px,qy,py
    vector<vector<double>> vec_qp(4,q_list);
    int ns,nt;

    #pragma omp parallel private(qx,qy,px,py,fx,fy,C,nt,dwx,dwy) shared(ns,vec_qp)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        // xt::xarray<normal_distribution<double>> dw{ 
        //       normal_distribution<double>{0.0, 1.0 },                                                     
        //       normal_distribution<double>{0.0, 1.0 } };

        qx = 0.;
        px = 0.;
        qy = 0.;
        py = 0.;

        fx = -Upx(qx,qy);
        fy = -Upy(qx,qy);    

        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            px+= 0.5*dt*fx;
            py+= 0.5*dt*fy;
         

            //**********
            //* STEP A *
            //**********
            qx += 0.5*dt*px;
            qy += 0.5*dt*py;

            //**********
            //* STEP O *
            //**********
        
            // dwx=dw(generator);
            // dw=dw(generator);
            // cout<<dwx<<"\n";
            // cout<<dwy<<"\n";

            C = exp(-dt*gamma);
            px = C*px + sqrt((1.-C*C)*tau)*normal(generator);
            py = C*py + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            qx += 0.5*dt*px;
            qy += 0.5*dt*py;

            //**********
            //* STEP B *
            //**********
            fx = -Upx(qx,qy);
            fy = -Upy(qx,qy);
            px += 0.5*dt*fx;
            py += 0.5*dt*fy;

    } 
    vec_qp[0][ns]=qx;
    vec_qp[1][ns]=px;
    vec_qp[2][ns]=qy;
    vec_qp[3][ns]=py;
    }
return vec_qp;
}



vector<vector<double>> one_step_tr(void)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,fx,g,gpx,gdt,C;
    double qy,py,fy,gpy;
    int ns,nt;
    vector<double> q_list(numsam,0);

    // save values of function g(x)
    // vector<double> g_vals((numruns-1)*(numsam-1),0);
    // 
    list<double> g_vals;
    // savec the values of qx,px,qy,py
    vector<vector<double>> vec_qp(4,q_list);

    #pragma omp parallel private(qx,qy,px,py,fx,fy,gpx,gpy,C,nt,gdt,g) shared(ns,vec_qp,g_vals)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        g = getg(qx,qy);
        gdt = g*dt; 

        qx = 0.;
        px = 0.;
        fx = -Upx(qx,qy);   
        gpx = getgprime_x(qx,qy); 

        qy = 0.;
        py = 0.;
        fy = -Upy(qx,qy);   
        gpy = getgprime_y(qx,qy); 


        for(nt = 0; nt<numruns; nt++)
        {

            //
            // BAOAB integrator
            //
            //**********
            //* STEP B *
            //**********
            px += 0.5*gdt*fx+0.5*dt*tau*gpx;
            py += 0.5*gdt*fy+0.5*dt*tau*gpy;


            //**********
            //* STEP A *
            //**********
            qx += 0.5*gdt*px;
            qy += 0.5*gdt*py;

            //**********
            //* STEP O *
            //**********
            C = exp(-gdt*gamma);
            px = C*px + sqrt((1.-C*C)*tau)*normal(generator);
            py = C*py + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            qx += 0.5*gdt*px;
            qy += 0.5*gdt*py;

            //**********
            //* STEP B *
            //**********
            fx = -Upx(qx,qy);
            fy = -Upy(qx,qy);

            g = getg(qx,qy);

            gdt = dt*g;
            gpx=getgprime_x(qx,qy);
            gpy=getgprime_y(qx,qy);
            px += 0.5*gdt*fx+0.5*dt*tau*gpx;
            py += 0.5*gdt*fy+0.5*dt*tau*gpy;

            // save gvals
            g_vals.push_back(g); //[(ns*(numsam-1)+nt)]=g;

    }
    // after running until final time, save the value
    vec_qp[0][ns]=qx;
    vec_qp[1][ns]=px;
    vec_qp[2][ns]=qy;
    vec_qp[3][ns]=py;
    }

// Save the value of g 

fstream file;
string file_name;
file << fixed << setprecision(16) << endl;
file_name="data_one_dt_2d/val_g_ada.txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
file<<"q\n";
copy(g_vals.begin(), g_vals.end(), out_itr);
file.close();


return vec_qp;
}


int main(void) {    

    vector<double> q_list(numsam,0);
    vector<vector<double>> vec_qp(4,q_list);

    // set up the path 
    string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/2d";

    //Non adaptive step 
    vec_qp= one_step();

    fstream file;
    string file_name;
    file << fixed << setprecision(16) << endl;
    file_name=path+"/vec_noada_x.txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    file<<"q\n";
    ostream_iterator<double> out_itr2(file, "\n");
    copy(vec_qp[0].begin(), vec_qp[0].end(), out_itr2);
    file<<"p\n";
    ostream_iterator<double> out_itr4(file, "\n");
    copy(vec_qp[1].begin(), vec_qp[1].end(), out_itr4);
    file.close();

    file << fixed << setprecision(16) << endl;
    file_name=path+"/vec_noada_y.txt";
    file.open(file_name,ios_base::out);
    file<<"q\n";
    copy(vec_qp[2].begin(), vec_qp[2].end(), out_itr2);
    file<<"p\n";
    copy(vec_qp[3].begin(), vec_qp[3].end(), out_itr4);
    file.close();

    //Transformed step 
    vec_qp= one_step_tr();

    file_name=path+"/vec_tr_x.txt";
    file.open(file_name,ios_base::out);
    // ostream_iterator<double> out_itr(file, "\n");
    file<<"q\n";
    // ostream_iterator<double> out_itr2(file, "\n");
    copy(vec_qp[0].begin(), vec_qp[0].end(), out_itr2);
    file<<"p\n";
    // ostream_iterator<double> out_itr4(file, "\n");
    copy(vec_qp[1].begin(), vec_qp[1].end(), out_itr4);
    file.close();
    
    file << fixed << setprecision(16) << endl;
    file_name=path+"/vec_tr_y.txt";
    file.open(file_name,ios_base::out);
    file<<"q\n";
    copy(vec_qp[2].begin(), vec_qp[2].end(), out_itr2);
    file<<"p\n";
    copy(vec_qp[3].begin(), vec_qp[3].end(), out_itr4);
    file.close();


return 0;
}