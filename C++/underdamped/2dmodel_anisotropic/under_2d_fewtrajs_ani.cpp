

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
#define m1              0.001          // minimum step scale factor
#define M1              1./1.2              // maximum step scale factor
#define numsam          5          // number of sample
#define T               500         // total number of trajectories
#define dt              0.001

#define tau             0.1            
#define numruns         T/dt         // total number of trajectories
#define gamma           1.            // friction coefficient
#define PATH        "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/fewtraj2d";

/////////////////////////////////
// Square potential definition //
/////////////////////////////////
#define s 5.
#define c 0.4
// vector<double> dtlist ={0.01}; //  , 0.14 , 0.195, 0.273, 0.38 , 0.531, 0.741, 1.034, 1.443, 2.014};



double U(double x,double y)
{
    double res=(x*x+y*y);
    return res;
}

//g depends on 1/((x-1)^2(x+1)^2x^2)
///////////////////////////////////


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
    f=abs(c*s*(x*x+y*y-1)*(x*x+y*y-1));
    f2=f*f;
    xi=sqrt(1+m1*f2);
    den=M1*xi+f;
    g=xi/den;
    return(g);
}


double getgprime_x(double x,double y)
{
    double xc,xa,f,f2,xi,fp,gp;
    f=abs(c*s*(x*x+y*y-1)*(x*x+y*y-1));
    f2=f*f;
    fp=c*s*4*(x*x+y*y-1)*x;
    xi=sqrt(1+m1*f2);
    gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
    return(gp);
    }

double getgprime_y(double x,double y)
{
    double xc,xa,f,f2,xi,fp,gp;
    f=abs(c*s*(x*x+y*y-1)*(x*x+y*y-1));
    f2=f*f;
    fp=c*s*4*(x*x+y*y-1)*y;
    xi=sqrt(1+m1*f2);
    gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
    return(gp);
    }


/////////////////
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

        // X coordinates
        qx = 1.;
        px = 1.;

        // Y coordinates
        qy = 1.;
        py = 1.;

        // Values of dU/dx and dU/dy
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
            // -X coordinates
            px += 0.5*dt*fx;
            // -Y coordinates
            py += 0.5*dt*fy;

            //**********
            //* STEP A *
            //**********
            // -X coordinates
            qx += 0.5*dt*px;
            // -Y coordinates
            qy += 0.5*dt*py;


            //**********
            //* STEP O *
            //**********
            C = exp(-dt*gamma);
            // -X coordinates
            px = C*px + sqrt((1.-C*C)*tau)*normal(generator);
            // -Y coordinates
            py = C*py + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            // -X coordinates
            qx += 0.5*dt*px;
            // -Y coordinates
            qy += 0.5*dt*py;

            //**********
            //* STEP B *
            //**********
            // -X coordinates
            fx = -Upx(qx,qy);
            px += 0.5*dt*fx;
            // -Y coordinates
            fy = -Upy(qx,qy);
            py += 0.5*dt*fy;

            
            vec_qx[ns][nt]=qx;
            vec_px[ns][nt]=px;
            vec_qy[ns][nt]=qy;
            vec_py[ns][nt]=py;
        }
    }

fstream file;
string file_name;
string path=PATH;
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
    double qx,px,fx,g,gpx,gdt,C,g0,g1;
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

        // X- coordinates 
        qx =1.;
        px = 1.;

        // Y- coordinates 
        qy = 1.;
        py = 1.;

        // 
        gpx=getgprime_x(qx,qy);
        gpy=getgprime_y(qx,qy);

        fx = -Upx(qx,qy);   // force
        fy = -Upy(qx,qy);   // force

        // g_av=0.;
        g = getg(qx,qy);
        gdt = dt*g;

        for(nt = 0; nt<numruns; nt++)
        {

                      //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            // X- coordinates 
            px += 0.5*gdt*fx;
            // Y- coordinates 
            py += 0.5*gdt*fy;


            //**********
            //* STEP A *
            //**********
            // fixed point iteration
            g0=getg(qx+dt/4*px*g,qy+dt/4*py*g);
            g1=getg(qx+dt/4*px*g0,qy+dt/4*py*g0);
            g0=getg(qx+dt/4*px*g1,qy+dt/4*py*g1);
            g1=getg(qx+dt/4*px*g0,qy+dt/4*py*g0);
            gdt=g1*dt;
            
            // X- coordinates 
            qx += 0.5*gdt*px;
            // Y- coordinates 
            qy += 0.5*gdt*py;

            //**********
            //* STEP O *
            //**********
            g = getg(qx,qy);
            gdt = dt*g;
            C = exp(-gdt*gamma);
            gpx=getgprime_x(qx,qy);
            gpy=getgprime_y(qx,qy);
            // X- coordinates 
            px = C*px+(1.-C)*tau*gpx/(gamma*g) + sqrt((1.-C*C)*tau/gamma)*normal(generator);
             // Y- coordinates 
            py = C*py+(1.-C)*tau*gpy/(gamma*g) + sqrt((1.-C*C)*tau/gamma)*normal(generator);

            //**********
            //* STEP A *
            //**********
            // fixed point iteration
            g0=getg(qx+dt/4*px*g,qy+dt/4*py*g);
            g1=getg(qx+dt/4*px*g0,qy+dt/4*py*g0);
            g0=getg(qx+dt/4*px*g1,qy+dt/4*py*g1);
            g1=getg(qx+dt/4*px*g0,qy+dt/4*py*g0);
            gdt=g1*dt;
            
            // X- coordinates 
            qx += 0.5*gdt*px;
            // Y- coordinates 
            qy += 0.5*gdt*py;

            //**********
            //* STEP B *
            //**********
            // X- coordinates 
            fx = -Upx(qx,qy);   // force
            fy = -Upy(qx,qy);   // force           
            g = getg(qx,qy);
            gdt = dt*g;

            // X- coordinates 
            px += 0.5*gdt*fx;
            // Y- coordinates 
            py += 0.5*gdt*fy;


            //* Save values of g
            // g_av+=g;
            // cout<<"\ng\n";
            // cout<<g;

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
string path=PATH;
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