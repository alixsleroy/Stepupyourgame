

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
#define m               0.6
#define M               1.
#define m1              m*m         // minimum step scale factor
#define M1              1./M             // maximum step        // maximum step scale factor
#define numsam          250       // number of sample
vector<double> dtlist = {0.05,0.025,0.01,0.0075,0.005};
#define tau             1.5 

#define gamma           .1         // friction coefficient
#define printskip       100
#define Nt              100000
#define T               500
#define PATH   "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/hist_narrowpass"

#define d       2.2
#define R       8.3
#define p       0.001
#define K       .4
#define r       2.
#define ax      0.20
#define xw      0.15 // parameter of phi_1, determines how much of a double the potentila is
#define yw      0.1

///////////////////////////////////////
/// Bobsled Potential around x=4     //
///////////////////////////////////////
double s(double x){
    return 1./(1.+pow(x/d,6.));
    }
double phi1(double x,double y){
    return  p*pow((x-R)*(x+R),2.) + xw*x*x + yw*y*y;
    }
double phi2(double y){
    return 2.+K*y*y;
}
double phi1_x(double x){
    return x*(-4.*p*R*R+4.*p*x*x+2*xw);
}
double phi1_y(double y){
    return 0.2*y;
}
double phi2_y(double y){
    return 2.*K*y;
}
double s_x(double x){
    double x5,d6,res;
    x5=pow(x,5.);
    d6=pow(d,6.);
    res=-6.*x5*d6/(pow(x*x5+d6,2.));
    return res;
}
double Upx(double x, double y){
    double upx,sx_x;
    sx_x=s_x(x);
    upx=phi1_x(x)*(1.-s(x))+sx_x*(phi2(x)-phi1(x,y));
    return upx;
}

double Upy(double x, double y){
    double upy;
    double sx=s(x);
    upy = phi1_y(y)*(1.-sx)+sx*phi2_y(y);
    return upy;
    }



double getg(double x,double y){
    double f=1/(ax*pow(x,(2*r))); //+ay*y*y);  // a*pow((x+3)*(x-3),(2*r));  //(pow((x-d)*(x+d),2*r)*p/R+y*y*0.05*K);   
    double f2=f*f;
    double xi=sqrt(1.+m1*f2);
    double den=M1*xi+sqrt(f2);
    double g=xi/den;
    return(g);
    }

double getgprime_x(double x,double y){
    double f=1/(ax*pow(x,(2*r))); //+ay*y*y);  // a*pow((x+3)*(x-3),(2*r));  //(pow((x-d)*(x+d),2*r)*p/R+y*y*0.05*K);   
    double fp=-2*r*pow(x,-2*r-1)/ax;    ///-2*ax*r*pow(x,2*r-1)/pow(ax*pow(x,2*r)+ay*y*y,2);    //4*a*r*x*pow(x*x-9,2*r-1);                               //4*p*r*x*pow(x*x-d*d,2*r-1);   
    double f2=f*f;
    double xi=sqrt(1.+m1*f2);
    double sqf=sqrt(f2);
    double gp=-f*fp/(sqf*xi*pow(M1*xi+sqf,2.));
    return(gp);
}

double getgprime_y(double x,double y){
    double f=1/(ax*pow(x,(2*r))); //+ay*y*y);  // a*pow((x+3)*(x-3),(2*r));  //(pow((x-d)*(x+d),2*r)*p/R+y*y*0.05*K);   
    double fp=0; //-2*y*ay/pow(ax*pow(x,2*r)+ay*y*y,2); 
    double f2=f*f;
    double xi=sqrt(1.+m1*f2);
    double sqf=sqrt(f2);
    double gp=-f*fp/(sqf*xi*pow(M1*xi+sqf,2.));
    return(gp);
}
// double getg(double x,double y){
//     double f=1/(0.5*x*x+y*y);
//     double f2=f*f;
//     double xi=sqrt(1+m1*f2);
//     double den=M1*xi+f;
//     double g=xi/den;
//     return(g);
//     }

// double getgprime_x(double x,double y){
//     double f=phi1(x,y);   //1/(0.5*x*x+y*y);
//     double fp=phi1_x(x);   //-x*f*f;
//     double f2=f*f;
//     double xi=sqrt(1+m1*f2);
//     double gp=-f*xi*xi*fp/(sqrt(f2)*pow(xi,3)*pow(M1*xi+f,2));
//     return(gp);
// }

// double getgprime_y(double x, double y){
//     double f=phi1(x,y);  //1/(0.5*x*x+y*y);
//     double fp= phi1_y(y);  //-2*y*f*f;
//     double f2=f*f;
//     double xi=sqrt(1+m1*f2);
//     double gp=-xi*xi*fp/(sqrt(f2)*pow(xi,3)*pow(M1*xi+f,2));
//     return(gp);
// }

// There might be error on these definition 
// double getg(double x,double y){
//     double f=phi1(x,y);
//     double f2t=pow(f,2*t);
//     double xi=sqrt(r+m1*f2t);
//     double den=M1*xi+sqrt(f2t);
//     double g=xi/den;
//     return(g);
// }


// double getgprime_x(double x,double y){
//     double f=phi1(x,y);
//     double fp=phi1_x(x);
//     double f2t=pow(f,2*t);
//     double xi=sqrt(r+m1*f2t);
//     double num = r*t*sqrt(f2t)*fp;
//     double den=sqrt(f*f)*xi*pow(M1*xi+sqrt(f2t),2);
//     double g=-num/den;
//     return(g);}

// double getgprime_y(double x,double y){
//     double f=phi1(x,y);
//     double fp=phi1_y(y);
//     double f2t=pow(f,2*t);
//     double xi=sqrt(r+m1*f2t);
//     double num = r*t*sqrt(f2t)*fp;
//     double den=sqrt(f*f)*xi*pow(M1*xi+sqrt(f2t),2);
//     double g=-num/den;
//     return(g);}
/////////////////////////////////////
// Non adaptive one step function //
////////////////////////////////////


int one_step(double dt, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,gpx,C,fx,dwx,j;
    double qy,py,gpy,fy,dwy,signx,signx_c,ncrossing;
    
    // Save the values 
    // cout<<int(numsam*numruns/printskip);
    vector<double> vec_qx(int((numsam*numruns)/printskip),0);
    vector<double> vec_qy(int((numsam*numruns)/printskip),0);
    vector<double> vec_px(int((numsam*numruns)/printskip),0);
    vector<double> vec_py(int((numsam*numruns)/printskip),0);
    vector<double> vec_cross(numsam,0);

    int ns,nt,nsp;
    nsp=0;

    #pragma omp parallel private(qx,qy,px,py,fx,fy,C,nt,dwx,dwy,signx,signx_c,ncrossing) shared(ns,vec_qx,vec_px,vec_qy,vec_py,vec_cross)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        // xt::xarray<normal_distribution<double>> dw{ 
        //       normal_distribution<double>{0.0, 1.0 },                                                     
        //       normal_distribution<double>{0.0, 1.0 } };

        // X c0oordinates
        qx =-10.;
        px = 0.1;
        signx=(qx>0)-(qx<0);

        // Y coordinates
        qy = 0.;
        py = 0.1;

        // Values of dU/dx and dU/dy
        fx = -Upx(qx,qy);  
        fy = -Upy(qx,qy);  

        j=0;
        ncrossing=0;
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


            //************
            //* Crossing * 
            //************
            // evaluate the sign of qx 
            signx_c=(qx>0)-(qx<0);
            if (signx_c!=signx){
                ncrossing+=1;
                signx_c=signx;
            }

            // **************
            // * Save values
            // **************
            // burn in 1000
            if (nt%printskip==0 && nt>numruns/10){
                vec_qx[nsp]=qx;
                vec_px[nsp]=px;
                vec_qy[nsp]=qy;
                vec_py[nsp]=py;
                nsp+=1;
            }
           
        } 

     
        vec_cross[ns]=ncrossing;


    }

// Save the sample generated
string path=PATH;
fstream file;
file << fixed << setprecision(16) << endl;
string list_para="i="+to_string(i); 
string file_name=path+"/vec_noada_qx"+list_para+".txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec_qx.begin(), vec_qx.end(), out_itr);
file.close();

file_name=path+"/vec_noada_px"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_px.begin(), vec_px.end(), out_itr);
file.close();

file_name=path+"/vec_noada_qy"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_qy.begin(), vec_qy.end(), out_itr);
file.close();

file_name=path+"/vec_noada_py"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_py.begin(), vec_py.end(), out_itr);
file.close();

file_name=path+"/vec_noada_cross"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_cross.begin(), vec_cross.end(), out_itr);
file.close();

return 0;
}



double one_step_tr(double dt, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,fx,g,gpx,gdt,C,g0,g1,j,g_av;
    double qy,py,fy,gpy;
    int ns,nt,nsp,signx,signx_c,ncrossing;
    g_av=0;
    nsp=0;


    // Save the values 
    vector<double> vec_qx(int((numsam)*(numruns)/printskip),0);
    vector<double> vec_qy(int((numsam)*(numruns)/printskip),0);
    vector<double> vec_px(int((numsam)*(numruns)/printskip),0);
    vector<double> vec_py(int((numsam)*(numruns)/printskip),0);
    vector<double> vec_g(int((numsam)*(numruns)/printskip),0);
    vector<double> vec_cross(numsam,0);


    #pragma omp parallel private(qx,qy,px,py,fx,fy,gpx,gpy,C,nt,gdt,g,signx,signx_c,ncrossing) shared(ns,vec_qx,vec_qy,vec_px,vec_py,vec_g,g_av)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        // X- coordinates 
        qx =-10.;
        px = .1;
        signx=(qx>0)-(qx<0);

        // Y- coordinates 
        qy = 0.;
        py = .1;

        // 
        gpx=getgprime_x(qx,qy);
        gpy=getgprime_y(qx,qy);

        fx = -Upx(qx,qy);   // force
        fy = -Upy(qx,qy);   // force

        // g_av=0.;
        g = getg(qx,qy);
        gdt = dt*g;

        j=0;
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
            g0=getg(qx+gdt/4*px*g,qy+gdt/4*py*g);
            g1=getg(qx+gdt/4*px*g0,qy+gdt/4*py*g0);
            g0=getg(qx+gdt/4*px*g1,qy+gdt/4*py*g1);
            g1=getg(qx+gdt/4*px*g0,qy+gdt/4*py*g0);
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
            px = C*px+(1.-C)*tau*gpx/(gamma*g) + sqrt((1.-C*C)*tau)*normal(generator);
             // Y- coordinates 
            py = C*py+(1.-C)*tau*gpy/(gamma*g) + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            // fixed point iteration
            g0=getg(qx+gdt/4*px*g,qy+gdt/4*py*g);
            g1=getg(qx+gdt/4*px*g0,qy+gdt/4*py*g0);
            g0=getg(qx+gdt/4*px*g1,qy+gdt/4*py*g1);
            g1=getg(qx+gdt/4*px*g0,qy+gdt/4*py*g0);
            gdt=g1*gdt;
            
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


            // * Save values of g
            g_av+=g;

            //************
            //* Crossing * 
            //************
            // evaluate the sign of qx 
            signx_c=(qx>0)-(qx<0);
            if (signx_c!=signx){
                ncrossing+=1;
                signx_c=signx;
            }

            // **************
            // * Save values
            // **************
            if (nt%printskip==0){
                vec_qx[nsp]=qx;
                vec_px[nsp]=px;
                vec_qy[nsp]=qy;
                vec_py[nsp]=py;
                vec_g[nsp]=g;
                nsp+=1;
            }
           

        }
        vec_cross[ns]=ncrossing;
    

    }
g_av=g_av/(numsam*numruns);

// Save the sample generated
string path=PATH;
fstream file;
file << fixed << setprecision(16) << endl;
string list_para="i="+to_string(i); 
string file_name=path+"/vec_tr_qx"+list_para+".txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec_qx.begin(), vec_qx.end(), out_itr);
file.close();

file_name=path+"/vec_tr_px"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_px.begin(), vec_px.end(), out_itr);
file.close();

file_name=path+"/vec_tr_qy"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_qy.begin(), vec_qy.end(), out_itr);
file.close();

file_name=path+"/vec_tr_py"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_py.begin(), vec_py.end(), out_itr);
file.close();

file_name=path+"/vec_tr_g"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_g.begin(), vec_g.end(), out_itr);
file.close();

file_name=path+"/vec_tr_cross"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_cross.begin(), vec_cross.end(), out_itr);
file.close();

return g_av;
}


int main(void) { 
    for(int i = 0; i < dtlist.size(); i++){ // run the loop for ns samples
        double dti = dtlist[i];
        double nti = T/dti;
        double g_av= one_step_tr(dti,nti,i);
        int out= one_step(dti,nti,i);
    }
return 0;
}