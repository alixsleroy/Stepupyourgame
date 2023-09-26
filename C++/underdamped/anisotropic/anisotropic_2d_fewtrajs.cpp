

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

using namespace std;
#define m1              0.1          // minimum step scale factor
#define M1              1./1.5              // maximum step scale factor
#define numsam          5          // number of sample
#define T               50         // total number of trajectories
#define dt              0.01

#define tau             0.1            
#define numruns         T/dt         // total number of trajectories
#define gamma           0.1            // friction coefficient



/////////////////////////////////
// Anisotropic 1 d definition //
/////////////////////////////////
// Spring potential
//parameters of the potential 



//////////////////
// Anisotropic  //
////////////////// 
#define s 10. // parameter of how steep the double well is 
#define c .5 //parameter that determines how high the step size goes in between well, the highest the lowest it goes 


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

//g depends on 1/((x^2+y^2-1)^2)
///////////////////////////////////



// double getg(double x, double y)
// {
//     double xc,xa,f,f2,xi,den,g;
//     f=abs(c*s*(x*x+y*y-1)*(x*x+y*y-1));
//     f2=f*f;
//     xi=sqrt(1+m1*f2);
//     den=M1*xi+f;
//     g=xi/den;
//     return(g);
// }


// double getgprime_x(double x,double y)
// {
//     double xc,xa,f,f2,xi,fp,gp;
//     f=abs(c*s*(x*x+y*y-1)*(x*x+y*y-1));
//     f2=f*f;
//     fp=c*s*4*(x*x+y*y-1)*x;
//     xi=sqrt(1+m1*f2);
//     gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
//     return(gp);
//     }

// double getgprime_y(double x,double y)
// {
//     double xc,xa,f,f2,xi,fp,gp;
//     f=abs(c*s*(x*x+y*y-1)*(x*x+y*y-1));
//     f2=f*f;
//     fp=c*s*4*(x*x+y*y-1)*y;
//     xi=sqrt(1+m1*f2);
//     gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
//     return(gp);
//     }


//g depends on 1/((x^2+y^2-1)^2)
///////////////////////////////////

double getg(double x, double y)
{
    double xc,xa,f,f2,xi,den,g;
    f=abs(c*s*(x*x+y*y-1)*(x*x+y*y-1)*(x*x+y*y));
    f2=f*f;
    xi=sqrt(1+m1*f2);
    den=M1*xi+f;
    g=xi/den;
    return(g);
}


double getgprime_x(double x,double y)
{
    double xc,xa,f,f2,xi,fp,gp;
    f=abs(c*s*(x*x+y*y-1)*(x*x+y*y-1)*(x*x+y*y));
    f2=f*f;
    fp=c*s*4*(x*x+y*y-1)*(3*x*x+3*y*y-1)*x;
    xi=sqrt(1+m1*f2);
    gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
    return(gp);
    }

double getgprime_y(double x,double y)
{
    double xc,xa,f,f2,xi,fp,gp;
    f=abs(c*s*(x*x+y*y-1)*(x*x+y*y-1));
    f2=f*f;
    fp=c*s*4*(x*x+y*y-1)*(3*x*x+3*y*y-1)*y;
    xi=sqrt(1+m1*f2);
    gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
    return(gp);
    }



/////////////////////////////////
// Non adaptive one step function //
/////////////////////////////////


int one_step(void)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,gpx,C,fx,dwx;
    double qy,py,gpy,fy,dwy;
    
    vector<double> q_list(numruns,0);
    vector<vector<double>> vec_qx(numsam,q_list);
    vector<vector<double>> vec_px(numsam,q_list);
    vector<vector<double>> vec_qy(numsam,q_list);
    vector<vector<double>> vec_py(numsam,q_list);

    int ns,nt;

    #pragma omp parallel private(qx,qy,px,py,fx,fy,C,nt,dwx,dwy) shared(ns,vec_qx,vec_px,vec_qy,vec_py)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        // xt::xarray<normal_distribution<double>> dw{ 
        //       normal_distribution<double>{0.0, 1.0 },                                                     
        //       normal_distribution<double>{0.0, 1.0 } };

        qx = 0.5;
        px = 0.;
        qy = 0.5;
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

            
            vec_qx[ns][nt]=qx;
            vec_px[ns][nt]=px;
            vec_qy[ns][nt]=qy;
            vec_py[ns][nt]=py;
        }
    }

fstream file;
string file_name;
string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/fewtraj2d";
for(int nsps = 0; nsps<numsam; nsps++){
    file_name=path+"/vec_noada_x"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    file<<"q\n";
    copy(vec_qx[nsps].begin(), vec_qx[nsps].end(), out_itr);
    file<<"p\n";
    copy(vec_px[nsps].begin(), vec_px[nsps].end(), out_itr);
    file.close();

    file_name=path+"/vec_noada_y"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    file<<"q\n";
    copy(vec_qy[nsps].begin(), vec_qy[nsps].end(), out_itr);
    file<<"p\n";
    copy(vec_py[nsps].begin(), vec_py[nsps].end(), out_itr);
    file.close();

    }

return 0;
}



int one_step_tr(void)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,fx,g,gpx,gdt,C;
    double qy,py,fy,gpy;
    int ns,nt;


    // Savethe values 
    vector<double> q_list(numruns,0);
    vector<vector<double>> vec_qx(numsam,q_list);
    vector<vector<double>> vec_px(numsam,q_list);
    vector<vector<double>> vec_qy(numsam,q_list);
    vector<vector<double>> vec_py(numsam,q_list);
    vector<vector<double>> vec_g(numsam,q_list);

    #pragma omp parallel private(qx,qy,px,py,fx,fy,gpx,gpy,C,nt,gdt,g) shared(ns,vec_qx,vec_qy,vec_px,vec_py,vec_g)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        g = getg(qx,qy);
        gdt = g*dt; 

        qx = 0.5;
        px = 0.;
        fx = -Upx(qx,qy);   
        gpx = getgprime_x(qx,qy); 

        qy = 0.5;
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

            // save the value every %nsnapshot value
            vec_qx[ns][nt]=qx;
            vec_px[ns][nt]=px;
            vec_qy[ns][nt]=qy;
            vec_py[ns][nt]=py;

            vec_g[ns][nt]=g;


        }
    }

fstream file;
string file_name;
string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/fewtraj2d";
for(int nsps = 0; nsps<numsam; nsps++){
    file_name=path+"/vec_tr_x"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    file<<"q\n";
    copy(vec_qx[nsps].begin(), vec_qx[nsps].end(), out_itr);
    file<<"p\n";
    copy(vec_px[nsps].begin(), vec_px[nsps].end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_y"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    file<<"q\n";
    copy(vec_qy[nsps].begin(), vec_qy[nsps].end(), out_itr);
    file<<"p\n";
    copy(vec_py[nsps].begin(), vec_py[nsps].end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_g"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    file<<"q\n";
    copy(vec_g[nsps].begin(), vec_g[nsps].end(), out_itr);
    file.close();
    


    }


return 0;
}


int main(void) {    

    //Non adaptive step 
    int out= one_step();

    //Transformed step 
    out= one_step_tr();

return 0;
}