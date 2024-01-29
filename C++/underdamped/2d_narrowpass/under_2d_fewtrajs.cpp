

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
#define m               0.1
#define M               1.
#define m1              m*m         // minimum step scale factor
#define M1              1./M             // maximum step scale factor
#define numsam          1          // number of sample

#define dt              .001
#define tau             1.5

#define numruns         1000000         // total number of trajectories
#define gamma           1.5            // friction coefficient
#define printskip       1

#define PATH   "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/fewtraj_narrowpass"

// #define d       6.5
// #define R       6.
// #define p       0.01
// #define K       4.

#define d       2.8
#define R       8.3
#define p       0.001
#define K       0.4
#define r       2.
#define ax      0.05
#define xw      0.15 // parameter of phi_1, determines how much of a double the potentila is
//vector<double> dtlist ={0.7 , 0.63, 0.56, 0.48, 0.41, 0.34, 0.27, 0.19, 
// vector<double> dtlist = {0.1,0.01,0.005};
//vector<double> dtlist = {0.001,0.0001,0.00001};

///////////////////////////////////////
/// Bobsled Potential around x=4     //
///////////////////////////////////////
double s(double x){
    return 1./(1.+pow(x/d,6.));
    }
double phi1(double x,double y){
    return  p*pow((x-R)*(x+R),2.) + xw*x*x + 0.1*y*y;
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
    double fp=-2*ax*r*pow(x,2*r-1)/pow(ax*pow(x,2*r),2);    ///-2*ax*r*pow(x,2*r-1)/pow(ax*pow(x,2*r)+ay*y*y,2);    //4*a*r*x*pow(x*x-9,2*r-1);                               //4*p*r*x*pow(x*x-d*d,2*r-1);   
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
//     double f=(pow((x-R)*(x+R),2*r)*p/R+y*y*0.05*K);   
//     double f2=f*f;
//     double xi=sqrt(1.+m1*f2);
//     double den=M1*xi+sqrt(f);
//     double g=xi/den;
//     return(g);
//     }

// double getgprime_x(double x,double y){
//     double f=pow((x-d)*(x+d),2*r)*p/R+y*y*K*0.05;   
//     double fp=4*p*r*x*pow(x*x-d*d,2*r-1);   
//     double f2=f*f;
//     double xi=sqrt(1.+m1*f2);
//     double sqf=sqrt(f2);
//     double gp=-f*fp/(sqf*xi*pow(M1*xi+sqf,2.));
//     return(gp);
// }

// double getgprime_y(double x,double y){
//     double f=pow((x-d)*(x+d),2*r)*p/R+y*y*K*0.05;   
//     double fp=2*r*pow(y,2*r-1)*K*0.05;   
//     double f2=f*f;
//     double xi=sqrt(1.+m1*f2);
//     double sqf=sqrt(f2);
//     double gp=-f*fp/(sqf*xi*pow(M1*xi+sqf,2.));
//     return(gp);
// }


// Problem with what is below
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

// double getg(double x,double y){
//     double f=1/(pow(x/R_w,p_r));
//     double f2=f*f;
//     double xi=sqrt(1+m1*f2);
//     double den=M1*xi+f;
//     double g=xi/den;
//     return(g);
//     }

// double getgprime_x(double x,double y){
//     double f=1/(pow(x/R_w,p_r));
//     double f2=f*f;
//     double fp=-p_r*pow(x/R_w,-p_r-1)*1/R_w;
//     double xi=sqrt(1+m1*f2);
//     double gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
//     return(gp);
// }

// double getgprime_y(double x, double y){
//     double f=0; //1/(0.5*x*x+y*y);
//     double f2=f*f;
//     double fp=-2*y*f*f;
//     double xi=sqrt(1+m1*f2);
//     double gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
//     return(0);
// }


/////////////////////////////////////
// Non adaptive one step function //
////////////////////////////////////


int one_step(double ds)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,gpx,C,fx,dwx,j;
    double qy,py,gpy,fy,dwy;
    
    vector<double> q_list(int(numruns/printskip),0);
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

        // X c0oordinates
        qx =-.1;
        px = 0.0;

        // Y coordinates
        qy = 0.;
        py = 0.0;

        // Values of dU/dx and dU/dy
        fx = -Upx(qx,qy);  
        fy = -Upy(qx,qy);  

        j=0;
        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            // -X coordinates
            px += 0.5*ds*fx;
            // -Y coordinates
            py += 0.5*ds*fy;

            //**********
            //* STEP A *
            //**********
            // -X coordinates
            qx += 0.5*ds*px;
            // -Y coordinates
            qy += 0.5*ds*py;


            //**********
            //* STEP O *
            //**********
            C = exp(-ds*gamma);
            // -X coordinates
            px = C*px + sqrt((1.-C*C)*tau)*normal(generator);
            // -Y coordinates
            py = C*py + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            // -X coordinates
            qx += 0.5*ds*px;
            // -Y coordinates
            qy += 0.5*ds*py;

            //**********
            //* STEP B *
            //**********
            // -X coordinates
            fx = -Upx(qx,qy);
            px += 0.5*ds*fx;
            // -Y coordinates
            fy = -Upy(qx,qy);
            py += 0.5*ds*fy;

            // To do later 
            if (nt%printskip==0){
            vec_qx[ns][j]=qx;
            vec_px[ns][j]=px;
            vec_qy[ns][j]=qy;
            vec_py[ns][j]=py;
            j=j+1;
            }
        }    vector<double> q_list(int(numruns/printskip),0);

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



double one_step_trO(void)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,fx,g,gpx,gdt,C,g0,g1,j,g_av;
    double qy,py,fy,gpy;
    int ns,nt;
    g_av=0;


    // Savethe values 
    vector<double> q_list(int(numruns/printskip),0);
    vector<vector<double>> vec_qx(numsam,q_list);
    vector<vector<double>> vec_px(numsam,q_list);
    vector<vector<double>> vec_qy(numsam,q_list);
    vector<vector<double>> vec_py(numsam,q_list);
    vector<vector<double>> vec_g(numsam,q_list);

    #pragma omp parallel private(qx,qy,px,py,fx,fy,gpx,gpy,C,nt,gdt,g) shared(ns,vec_qx,vec_qy,vec_px,vec_py,vec_g,g_av)

    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        // X- coordinates 
        qx =-.1;
        px = .0;

        // Y- coordinates 
        qy = 0.;
        py = .0;

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


            // * Save values of g
            g_av+=g;

            // save the value every %nsnapshot value
            if (nt%printskip==0){
            vec_qx[ns][j]=qx;
            vec_px[ns][j]=px;
            vec_qy[ns][j]=qy;
            vec_py[ns][j]=py;

            vec_g[ns][j]=g;
            j=j+1;
            }


        }
    }
g_av=g_av/(numsam*numruns);

fstream file;
string file_name;
string path=PATH;
for(int nsps = 0; nsps<numsam; nsps++){
    file_name=path+"/vec_trO_x"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    file<<"q\n";
    copy(vec_qx[nsps].begin(), vec_qx[nsps].end(), out_itr);
    file<<"p\n";
    copy(vec_px[nsps].begin(), vec_px[nsps].end(), out_itr);
    file.close();

    file_name=path+"/vec_trO_y"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    file<<"q\n";
    copy(vec_qy[nsps].begin(), vec_qy[nsps].end(), out_itr);
    file<<"p\n";
    copy(vec_py[nsps].begin(), vec_py[nsps].end(), out_itr);
    file.close();

    file_name=path+"/vec_trO_g"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    file<<"q\n";
    copy(vec_g[nsps].begin(), vec_g[nsps].end(), out_itr);
    file.close();

    }


return g_av;
}


double one_step_trB(void)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,fx,g,gpx,gdt,C,g0,g1,j,g_av;
    double qy,py,fy,gpy;
    int ns,nt;
    g_av=0;


    // Savethe values 
    vector<double> q_list(int(numruns/printskip),0);
    vector<vector<double>> vec_qx(numsam,q_list);
    vector<vector<double>> vec_px(numsam,q_list);
    vector<vector<double>> vec_qy(numsam,q_list);
    vector<vector<double>> vec_py(numsam,q_list);
    vector<vector<double>> vec_g(numsam,q_list);

    #pragma omp parallel private(qx,qy,px,py,fx,fy,gpx,gpy,C,nt,gdt,g) shared(ns,vec_qx,vec_qy,vec_px,vec_py,vec_g,g_av)

    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        // X- coordinates 
        qx =-.1;
        px = .0;

        // Y- coordinates 
        qy = 0.0;
        py = .0;

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
            px += 0.5*gdt*fx+0.5*dt*tau*gpx;
            // Y- coordinates 
            py += 0.5*gdt*fy+0.5*dt*tau*gpy;


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
            // X- coordinates 
            px = C*px+sqrt((1.-C*C)*tau)*normal(generator);
             // Y- coordinates 
            py = C*py+sqrt((1.-C*C)*tau)*normal(generator);

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
            //* STEP B *
            //**********
            // X- coordinates 
            fx = -Upx(qx,qy);   // force
            fy = -Upy(qx,qy);   // force 
            gpx= getgprime_x(qx,qy);
            gpy= getgprime_y(qx,qy);          
            g = getg(qx,qy);
            gdt = dt*g;

            // X- coordinates 
            px += 0.5*gdt*fx+0.5*dt*tau*gpx;
            // Y- coordinates 
            py += 0.5*gdt*fy+0.5*dt*tau*gpy;


            // * Save values of g
            g_av+=g;

            // save the value every %nsnapshot value
            if (nt%printskip==0){
            vec_qx[ns][j]=qx;
            vec_px[ns][j]=px;
            vec_qy[ns][j]=qy;
            vec_py[ns][j]=py;

            vec_g[ns][j]=g;
            j=j+1;
            }


        }
    }
g_av=g_av/(numsam*numruns);

fstream file;
string file_name;
string path=PATH;
for(int nsps = 0; nsps<numsam; nsps++){
    file_name=path+"/vec_trB_x"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    file<<"q\n";
    copy(vec_qx[nsps].begin(), vec_qx[nsps].end(), out_itr);
    file<<"p\n";
    copy(vec_px[nsps].begin(), vec_px[nsps].end(), out_itr);
    file.close();

    file_name=path+"/vec_trB_y"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    file<<"q\n";
    copy(vec_qy[nsps].begin(), vec_qy[nsps].end(), out_itr);
    file<<"p\n";
    copy(vec_py[nsps].begin(), vec_py[nsps].end(), out_itr);
    file.close();

    file_name=path+"/vec_trB_g"+to_string(nsps)+".txt";
    file.open(file_name,ios_base::out);
    file<<"q\n";
    copy(vec_g[nsps].begin(), vec_g[nsps].end(), out_itr);
    file.close();

    }


return g_av;
}



int main(void) {    
    double g_av= one_step_trO();
    cout<<g_av;
    cout<<"\n";
    g_av= one_step_trB();
    cout<<g_av;
    double newds=g_av*dt;
    //Non adaptive step 
    int out= one_step(dt);


return 0;
}