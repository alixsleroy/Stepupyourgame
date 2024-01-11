

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
#define m1              .001         // minimum step scale factor
#define M1              1.             // maximum step scale factor
#define numsam          100          // number of sample
#define tau             4. 
#define T               1000
#define gamma           .5            // friction coefficient
#define printskip       1

#define PATH   "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/fewtraj_narrowpass"
vector<double> dtlist = {0.001,0.0001,0.00001};
#define d       2.5
#define R       2
#define p       0.001
#define K       2


///////////////////////////////////////
/// Bobsled Potential around x=4     //
///////////////////////////////////////
double s(double x){
    return 1./(1.+pow(x/d,6));
    }
double phi1(double x,double y){
    return  p*pow((x-R)*(x+R),2) + 0.001*x*x + 0.01*y*y;
    }
double phi2(double y){
    return 2+K*y*y;
}
double phi1_x(double x){
    return x*(-4*p*R*R+4*p*x*x+0.002);
}
double phi1_y(double y){
    return 0.02*y;
}
double phi2_y(double y){
    return 2*K*y;
}
double s_x(double x){
    double x5,d6,res;
    x5=pow(x,5);
    d6=pow(d,6);
    res=-6*x5/(d6*pow(x5*x/d6+1,2));
    return res;
}
double Upx(double x, double y){

    double upx,sx_x;
    sx_x=s_x(x);
    upx=phi1_x(x)*(1-s(x))+sx_x*(phi2(x)-phi1(x,y));
    return upx;
}

double Upy(double x, double y){
    double upy;
    double sx=s(x);
    upy = phi1_y(y)*(1-sx)+sx*phi2_y(y);
    return upy;
    }

double getg(double x,double y){
    double f=1/(0.5*x*x+y*y);
    double f2=f*f;
    double xi=sqrt(1+m1*f2);
    double den=M1*xi+f;
    double g=xi/den;
    return(g);
    }

double getgprime_x(double x,double y){
    double f=1/(0.5*x*x+y*y);
    double f2=f*f;
    double fp=-x*f*f;
    double xi=sqrt(1+m1*f2);
    double gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
    return(gp);
}

double getgprime_y(double x, double y){
    double f=1/(0.5*x*x+y*y);
    double f2=f*f;
    double fp=-2*y*f*f;
    double xi=sqrt(1+m1*f2);
    double gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
    return(0);
}


/////////////////////////////////////
// Non adaptive one step function //
////////////////////////////////////


vector<double> one_step(double ds, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,gpx,C,fx,dwx,j,fmpt;
    double qy,py,gpy,fy,dwy;
    
    vector<double> q_list(int(numruns/printskip),0);
    vector<vector<double>> vec_qx(numsam,q_list);
    vector<vector<double>> vec_px(numsam,q_list);
    vector<vector<double>> vec_qy(numsam,q_list);
    vector<vector<double>> vec_py(numsam,q_list);

    int ns,nt,signx,signx_c,ncrossing;

    // Compute the number of crossing
    vector<double> info_to_return(2,0);

    #pragma omp parallel private(qx,qy,px,py,fx,fy,C,nt,dwx,dwy,signx,signx_c) shared(ns,vec_qx,vec_px,vec_qy,vec_py,ncrossing)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        // xt::xarray<normal_distribution<double>> dw{ 
        //       normal_distribution<double>{0.0, 1.0 },                                                     
        //       normal_distribution<double>{0.0, 1.0 } };

        // X c0oordinates
        qx = 1.;
        // Derive the sign of the x coordinate 
        signx=(qx>0)-(qx<0);
        px = 0.1;

        // Y coordinates
        qy = 0.;
        py = 0.1;

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

            // Check the sign of q x 
            signx_c=(qx>0)-(qx<0);
            if (signx!=signx_c){ //if sign changes save the value of numruns and exit the loop 
                ncrossing+=1;
                fmpt+=nt;
                signx=signx_c;
            }

            // To do later 
            if (nt%printskip==0){
            vec_qx[ns][j]=qx;
            vec_px[ns][j]=px;
            vec_qy[ns][j]=qy;
            vec_py[ns][j]=py;
            j=j+1;
            }
        }   

    }

fmpt=fmpt/ncrossing;
info_to_return[0]=fmpt;
info_to_return[1]=ncrossing;

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

return info_to_return;
}



vector<double> one_step_tr(double dt, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,fx,g,gpx,gdt,C,g0,g1,j,g_av,fmpt;
    double qy,py,fy,gpy;
    int ns,nt,signx,signx_c,ncrossing;
    g_av=0;
    fmpt=0;

    // Savethe values 
    vector<double> q_list(int(numruns/printskip),0);
    vector<vector<double>> vec_qx(numsam,q_list);
    vector<vector<double>> vec_px(numsam,q_list);
    vector<vector<double>> vec_qy(numsam,q_list);
    vector<vector<double>> vec_py(numsam,q_list);
    vector<vector<double>> vec_g(numsam,q_list);

    // Compute the number of crossing
    vector<double> info_to_return(3,0);

    #pragma omp parallel private(qx,qy,px,py,fx,fy,gpx,gpy,C,nt,gdt,g,signx,signx_c) shared(ns,vec_qx,vec_qy,vec_px,vec_py,vec_g,g_av,fmpt)
    #pragma omp for

    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        // X- coordinates 
        qx =1.;
        px = .1;
        // Derive the sign of the x coordinate 
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
            px = C*px+(1.-C)*tau*gpx/(gamma*g) + sqrt((1.-C*C)*tau)*normal(generator);
             // Y- coordinates 
            py = C*py+(1.-C)*tau*gpy/(gamma*g) + sqrt((1.-C*C)*tau)*normal(generator);

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


            // * Save values of g
            g_av+=g;

            // Check the sign of q x 
            signx_c=(qx>0)-(qx<0);
            if (signx!=signx_c){ //if sign changes save the value of numruns and exit the loop 
                ncrossing+=1;
                fmpt+=nt;
                signx=signx_c;
                cout<<1;
            }
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

// save the information on the values of g and the first mean passage time
// save the number of crossing
g_av=g_av/(numsam*numruns);
fmpt=fmpt/ncrossing;
info_to_return[0]=g_av;
info_to_return[1]=fmpt;
info_to_return[2]=ncrossing;

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


return info_to_return;
}


int main(void) {    


    // Start the clock to have the number of simulation 
    auto start = high_resolution_clock::now();
    using namespace std;
    //save values related to the transformed
    vector<double> g_av_list(dtlist.size(),0);
    vector<double> tr_fmpt_list(dtlist.size(),0);
    vector<double> tr_ncrossing_list(dtlist.size(),0);

    //save values related to the transformed
    vector<double> noada_fmpt_list(dtlist.size(),0);
    vector<double> noada_ncrossing_list(dtlist.size(),0);


    for(int i = 0; i < dtlist.size(); i++){ 
        double dti = dtlist[i];
        double ni = T/dti;

        vector<double> info= one_step_tr(dti,ni,i);
        g_av_list[i]=info[0];
        tr_fmpt_list[i]=info[1];
        tr_ncrossing_list[i]=info[2];

        info= one_step(dti,ni,i);
        noada_fmpt_list[i]=info[0];
        noada_ncrossing_list[i]=info[1];

    // * SAVE THE COMPUTED FMPT
    /////////////////////////////////////////
    string path=PATH;

    // NON ADAPTIVE-FMPT
    fstream file;
    file << fixed << setprecision(16) << endl;
    string file_name=path+"/noada_fmpt.txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    copy(noada_fmpt_list.begin(), noada_fmpt_list.end(), out_itr);
    file.close();

    file_name=path+"/noada_ncrossing.txt";
    file.open(file_name,ios_base::out);
    copy(noada_ncrossing_list.begin(), noada_ncrossing_list.end(), out_itr);
    file.close();

    // TRANSFORMED with corr in B 
    file_name=path+"/g_av_list.txt";
    file.open(file_name,ios_base::out);
    copy(g_av_list.begin(), g_av_list.end(), out_itr);
    file.close();

    file_name=path+"/tr_fmpt_list.txt";
    file.open(file_name,ios_base::out);
    copy(tr_fmpt_list.begin(), tr_fmpt_list.end(), out_itr);
    file.close();

    file_name=path+"/tr_ncrossing_list.txt";
    file.open(file_name,ios_base::out);
    copy(noada_ncrossing_list.begin(), noada_ncrossing_list.end(), out_itr);
    file.close();

        // * SAVE THE TIME AND PARAMETERS OF THE SIMULATION IN A INFO FILE
    ///////////////////////////////////////////////////////////////////
    // find time by subtracting stop and start timepoints 
    auto stop = high_resolution_clock::now();
    auto duration_m = duration_cast<minutes>(stop - start);
    auto duration_s = duration_cast<seconds>(stop - start);
    auto duration_ms = duration_cast<microseconds>(stop - start);
    // save the parameters in a file info
    // string parameters="M="+to_string(M)+"-m="+to_string(m)+"-gamma="+to_string(gamma)+"-tau="+to_string(tau)+"-a="+to_string(a)+"-b="+to_string(b)+"-x0="+to_string(x0)+"-c="+to_string(c)+"-Ns="+to_string(numsam)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    string parameters="M1="+to_string(M1)+"-m1="+to_string(m1)+"-gamma="+to_string(gamma)+"-tau="+to_string(tau)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    string information=path+"/parameters_used.txt";
    file.open(information,ios_base::out);
    file << parameters;
    file <<"\n";
    file <<"list of dt";
    copy(dtlist.begin(), dtlist.end(), out_itr);
    file.close();
    
    }



return 0;
}