# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:48:38 2018

@author: Shengjie Liu, stop68@foxmail.com

Calculate satellite position from Rinex navigation file

Test on Rinex v2.10, v3.02 with GPS navigation file

Requirements:
    numpy
    argparse
    
Example:
    python satpos.py --file=rinex302.18N
    
  Perform time correction with --timeCor=True
    python satpos.py --file=rinex210.18N --timeCor=True
  
  Use Householder's iteration instead of Newton's
    python satpos.py --file=rinex210.18N --iteration=Householder

"""
from dataclasses import dataclass, field
import numpy as np
import argparse
import georinex as gr
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
@dataclass
class datetime_t:
    week: int = 0
    sec: float = 0.0
@dataclass
class gpstime_t:
    y: int = 0
    m: int = 0
    d: int = 0
    hh: int = 0
    mm: int = 0
    sec: float = 0.0
@dataclass 
class Eph: 
    weight: int = None
    price: float = None
    vflg: int = None	#/*!< Valid Flag */
    t: datetime_t = field(default_factory=dict)
    toc: gpstime_t = field(default_factory=dict) #	/*!< Time of Clock */
    toe: gpstime_t  = field(default_factory=dict) #	/*!< Time of Ephemeris */
    iodc: int = None #	/*!< Issue of Data, Clock */
    iode: int = None #	/*!< Isuse of Data, Ephemeris */
    deltan: float = None #	/*!< Delta-N (radians/sec) */
    cuc: float	= None #/*!< Cuc (radians) */
    cus: float = None #/*!< Cus (radians) */
    cic: float = None #	/*!< Correction to inclination cos (radians) */
    cis: float = None #	/*!< Correction to inclination sin (radians) */
    crc: float = None#	/*!< Correction to radius cos (meters) */
    crs: float  = None #/*!< Correction to radius sin (meters) */
    ecc: float	= None #/*!< e Eccentricity */
    sqrta: float  = None #	/*!< sqrt(A) (sqrt(m)) */
    m0: float  = None #	/*!< Mean anamoly (radians) */
    omg0: float = None  #	/*!< Longitude of the ascending node (radians) */
    inc0: float = None  #	/*!< Inclination (radians) */
    aop: float  = None #
    omgdot: float = None  #	/*!< Omega dot (radians/s) */
    idot: float = None #	/*!< IDOT (radians/s) */
    af0: float  = None #	/*!< Clock offset (seconds) */
    af1: float  = None #	/*!< rate (sec/sec) */
    af2: float  = None #	/*!< acceleration (sec/sec^2) */
    tgd: float  = None #	/*!< Group delay L2 bias */
    svhlth: int = None
    codeL2: int = None
    #Working variables follow
    n: float = None #	/*!< Mean motion (Average angular velocity) */
    sq1e2: float = None #	/*!< sqrt(1-e^2) */
    A: float = None #	/*!< Semi-major axis */
    omgkdot: float = None # /*!< OmegaDot-OmegaEdot */
    def __post_init__(self):
        self.toe = gpstime_t()
        self.t = datetime_t()
        self.toc = gpstime_t()



parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--file', type=str, default = None)
parser.add_argument('--timeCor', type=bool, default = False)
parser.add_argument('--iteration', type=str, default = 'Newton')
args = parser.parse_args()
WGS84_RADIUS =	6378137.0
WGS84_ECCENTRICITY = 0.0818191908426
SECONDS_IN_WEEK = 604800.0
SECONDS_IN_HALF_WEEK =302400.0
OMEGA_EARTH = 7.2921151467e-5
GM = 3.986005*np.power(10.0,14)
c = 2.99792458*np.power(10.0,8)
omegae_dot = 7.2921151467*np.power(10.0,-5)

earth_rate = 2*np.pi/(60*60*24)


def checkSatVisibility(eph, g, xyz, elvMask, azel):

    llh=np.zeros(3)
    neu=np.zeros(3)
    pos=np.zeros(3)
    vel=np.zeros(3)
    clk=np.zeros(3)
    los=np.zeros(3)
    tmat=np.zeros(3,3)

    llh = xyz2llh(xyz)
    tmat = ltcmat(llh)

    pos, vel, clk = satpos(eph, g)
    #subVect(los, pos, xyz)
    #ecef2neu(los, tmat, neu)
    #neu2azel(azel, neu)

    #if azel[1]*180/np.pi > elvMask:
        #return 1 #Visible
    return 0

def satpos(eph, g):
   
    g = gpstime_t()
    pos=np.zeros(3)
    vel=np.zeros(3)
    clk=np.zeros(3)

    deltan = eph.deltan;
    tk = g.sec - eph.toe.sec;

    if tk > SECONDS_IN_HALF_WEEK:
        tk -= SECONDS_IN_WEEK
    elif tk<-SECONDS_IN_HALF_WEEK:
        tk += SECONDS_IN_WEEK

    #m0: mean anomaly(radians)
    #n: correction to mean motion(radian/sec)

    mk = eph.m0 + eph.n*tk
    ek = mk
    ekold = ek + 1.0

    OneMinusecosE = 0 # Suppress the uninitialized warning.
    while np.fabs(ek-ekold) > 1.0E-14:
    
        ekold = ek
        #ecc : eccentricity
        OneMinusecosE = 1.0-eph.ecc*np.cos(ekold)
        ek = ek + (mk-ekold+eph.ecc*np.sin(ekold))/OneMinusecosE
    
    #ek: eccentric anomaly
    sek = np.sin(ek)
    cek = np.cos(ek)

    #n: Mean motion (Average angular velocity)
    ekdot = eph.n/OneMinusecosE
    #sqrta : square root of semi-major axis
    relativistic = -4.442807633E-10*eph.ecc*eph.sqrta*sek;
    #aop : argument of perigee(rad) (lower    case    omega)
    #pk: true anomaly theta/2
    pk = np.arctan2(eph.sq1e2*sek,cek-eph.ecc) + eph.aop;
    pkdot = eph.sq1e2*ekdot/OneMinusecosE

    s2pk = np.sin(2.0*pk)
    c2pk = np.cos(2.0*pk)

    #cus,cuc: correction to argument in latitude (rad)
    #uk: true anomaly theta
    uk = pk + eph.cus*s2pk + eph.cuc*c2pk
    suk = np.sin(uk)
    cuk = np.cos(uk)
    ukdot = pkdot*(1.0 + 2.0*(eph.cus*c2pk - eph.cuc*s2pk))
    #rk: radius of true anomaly
    rk = eph.A*OneMinusecosE + eph.crc*c2pk + eph.crs*s2pk
    rkdot = eph.A*eph.ecc*sek*ekdot + 2.0*pkdot*(eph.crs*c2pk - eph.crc*s2pk)
    #ioc0: Inclination (radians)
    #idot: time derivative of inclination(rads/sec)
    ik = eph.inc0 + eph.idot*tk + eph.cic*c2pk + eph.cis*s2pk
    sik = np.sin(ik)
    cik = np.cos(ik)
    ikdot = eph.idot + 2.0*pkdot*(eph.cis*c2pk - eph.cic*s2pk)

    xpk = rk*cuk
    ypk = rk*suk
    xpkdot = rkdot*cuk - ypk*ukdot
    ypkdot = rkdot*suk + xpk*ukdot
    ok = eph.omg0 + tk*eph.omgkdot - OMEGA_EARTH*eph.toe.sec
    sok = np.sin(ok)
    cok = np.cos(ok)

    pos[0] = xpk*cok - ypk*cik*sok
    pos[1] = xpk*sok + ypk*cik*cok
    pos[2] = ypk*sik

    tmp = ypkdot*cik - ypk*sik*ikdot

    vel[0] = -eph.omgkdot*pos[1] + xpkdot*cok - tmp*sok
    vel[1] = eph.omgkdot*pos[0] + xpkdot*sok + tmp*cok
    vel[2] = ypk*cik*ikdot + ypkdot*sik

    # Satellite clock correction
    tk = g.sec - eph.toc.sec

    if tk>SECONDS_IN_HALF_WEEK:
        tk -= SECONDS_IN_WEEK
    elif tk <- SECONDS_IN_HALF_WEEK:
        tk += SECONDS_IN_WEEK

    clk[0] = eph.af0 + tk*(eph.af1 + tk*eph.af2) + relativistic # - eph.tgd 
    clk[1] = eph.af1 + 2.0*tk*eph.af2 

    return pos#, vel, clk

def ltcmat(llh):
    t = np.zeros(3,3)
    slat = np.sin(llh[0])
    clat = np.cos(llh[0])
    slon = np.sin(llh[1])
    clon = np.cos(llh[1])

    t[0][0] = -slat*clon
    t[0][1] = -slat*slon
    t[0][2] = clat
    t[1][0] = -slon
    t[1][1] = clon
    t[1][2] = 0.0
    t[2][0] = clat*clon
    t[2][1] = clat*slon
    t[2][2] = slat

    return t
def xyz2llh(xyz):

    a = WGS84_RADIUS
    e = WGS84_ECCENTRICITY

    eps = 1.0e-3
    e2 = e*e

    llh=np.zeros(3)
    if np.linalg.norm(xyz) < eps:
        #Invalid ECEF vector
        llh[0] = 0.0
        llh[1] = 0.0
        llh[2] = -a

        return llh

    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    rho2 = x*x + y*y
    dz = e2*z

    while 1:
        zdz = z + dz;
        nh = np.sqrt(rho2 + zdz*zdz);
        slat = zdz / nh;
        n = a / np.sqrt(1.0-e2*slat*slat);
        dz_new = n*e2*slat;

        if np.fabs(dz-dz_new) < eps:
            break

        dz = dz_new


    llh[0] = np.atan2(zdz, np.sqrt(rho2));
    llh[1] = np.atan2(y, x);
    llh[2] = nh - n;

    return llh

def llh2xyz(llh):
    a = WGS84_RADIUS
    e = WGS84_ECCENTRICITY
    e2 = e*e
    clat = np.cos(llh[0])
    slat = np.sin(llh[0])
    clon = np.cos(llh[1])
    slon = np.sin(llh[1])
    d = e*slat

    n = a/np.sqrt(1.0-d*d)
    nph = n + llh[2]

    tmp = nph*clat
    xyz = np.zeros(3)
    xyz[0] = tmp*clon
    xyz[1] = tmp*slon
    xyz[2]= ((1.0-e2)*n + llh[2])*slat

    return xyz

def calSatPos(data,time_tsv, timeCor=False, iteration='Newton'):
    sats = np.zeros(3)
           
    ## load variables
    A = np.power(data['sqrtA'],2) 
    toe = 259200 # Time of Ephemeris
    tsv = 259200+time_tsv
    tk = tsv - toe
    
    n0 = np.sqrt(GM/np.power(A,3)) #
    dn = data['DeltaN']
    n = n0 + dn
    m0 = data['M0']
    M = m0+n*tk
    
    af0 = data['SVclockBias']
    af1 = data['SVclockDrift']
    w = data['omega']
    cuc = data['Cuc'] 
    cus = data['Cus']
    crc = data['Crc']
    crs = data['Crs']
    cic = data['Cic']
    cis = data['Cis'] 
    i0 = data['Io']
    idot = data['IDOT']
    omg0 = data['Omega0']
    odot = data['OmegaDot'] 
    e = data['Eccentricity'] # Eccentricity
    #timeCor = True
    ## time correction
    if timeCor == True:
        NRnext = 0
        NR = 1
        m = 1
        while np.abs(NRnext-NR)>np.power(10.0,-16):
            NR = NRnext
            f = NR-e*np.sin(NR)-M
            f1 = 1-e*np.cos(NR)
            f2 = e*np.sin(NR)
            if iteration=='Householder':
                NRnext = NR - f/(f1-(f2*f/(2*f1)))
            else:
                NRnext = NR - f/f1
            m += 1
        
        E = NRnext
        
        F = -2*np.sqrt(GM)/np.power(c,2)
        delta_tr = F*e*np.sqrt(A)*np.sin(E) #relativistic
        delta_tsv = af0+af1*(tsv-toe)+delta_tr
        t = tsv-delta_tsv
        tk = t-toe
        M = m0+n*tk
    
    NRnext = 0
    NR = 1
    m = 1
    while np.abs(NRnext-NR)>np.power(10.0,-16):
        NR = NRnext
        f = NR-e*np.sin(NR)-M
        f1 = 1-e*np.cos(NR)
        f2 = e*np.sin(NR)
        if iteration=='Householder':
            NRnext = NR - f/(f1-(f2*f/(2*f1)))
        else:
            NRnext = NR - f/f1
        m += 1

    E = NRnext
    v = np.arctan2(np.sqrt(1-np.power(e,2))*np.sin(E),np.cos(E)-e)
    phi = v + w
    u = phi + cuc*np.cos(2*phi) + cus*np.sin(2*phi)
    r = A*(1-e*np.cos(E)) + crc*np.cos(2*phi) + crs*np.sin(2*phi)
    i = i0 + idot*tk + cic*np.cos(2*phi) + cis*np.sin(2*phi)   #inclinati0n
    #Omega = omg0 + (odot-omegae_dot)*tk - omegae_dot*toe
    Omega = omg0 + odot*tk - omegae_dot*toe
    x1 = np.cos(u)*r
    y1 = np.sin(u)*r
    
    sats[0] = x1*np.cos(Omega) - y1*np.cos(i)*np.sin(Omega)
    sats[1] = x1*np.sin(Omega) + y1*np.cos(i)*np.cos(Omega)
    sats[2] = y1*np.sin(i)
    return sats

def data2eph(data):
    eph=Eph()
    #eph.toe = gpstime_t()
    eph.deltan = data['DeltaN']
    eph.m0 = data['M0']
    eph.af0 = data['SVclockBias']
    eph.af1 = data['SVclockDrift']
    eph.af2 = 0.0
    eph.aop = data['omega']
    eph.cuc = data['Cuc'] 
    eph.cus = data['Cus']
    eph.crc = data['Crc']
    eph.crs = data['Crs']
    eph.cic = data['Cic']
    eph.cis = data['Cis'] 
    eph.inc0 = data['Io']
    eph.idot = data['IDOT']
    eph.omg0 = data['Omega0']
    eph.omgdot = data['OmegaDot'] 
    eph.ecc = data['Eccentricity'] # Eccentricity
    eph.sqrta = data['sqrtA']
    eph.A = np.power(eph.sqrta,2) 
    n0 = np.sqrt(GM/np.power(eph.A,3)) #
    eph.n = n0 + eph.deltan
    eph.sq1e2 = np.sqrt(1.0 - np.power(eph.ecc,2))
    eph.omgkdot = eph.omgdot - OMEGA_EARTH
   # eph.toe.sec = 0
    

    return eph
if __name__ == "__main__":


    
    args.file = "brdc0730.24n"
    print('\n--- Calculate satellite position ---\nN file:',args.file)
    print('Time correction =',args.timeCor,
          '\nIteration strategy =',args.iteration,'\n')
    
    data = gr.load(args.file)
    df = data.to_dataframe()
    satp = np.zeros([1,3])
    for i in range(len(satp)):

        g01 = df.iloc[0]
        #satp[i] = calSatPos(g01,timeCor=args.timeCor,iteration=args.iteration, time_tsv=i*60*40)
        eph = data2eph(g01)
        satp[i] = satpos(eph,g=gpstime_t())
    # load bluemarble with PIL
    bm = Image.open('bluemarble.jpg')
    # it's big, so I'll rescale it, convert to array, and divide by 256 to get RGB values that matplotlib accept

    bm = np.array(bm.resize([int(d/10) for d in bm.size]))/1024.
    # coordinates of the image - don't know if this is entirely accurate, but probably close

    lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180
    lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180

    # repeat code from one of the examples linked to in the question, except for specifying facecolors:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.outer(np.cos(lons), np.cos(lats)).T
    y = np.outer(np.sin(lons), np.cos(lats)).T
    z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors = bm)
    print(satp)
    for i in range(len(satp)):
        px=satp[i,0]/6357000.0
        py=satp[i,1]/6357000.0
        pz=satp[i,2]/6357000.0
        ax.plot([0,px],[0,py],[0,pz],'o')
        #ax.plot([0,satp[i ,0]/(6357000)],[0,satp[i, 1]/(6357000)],[0,satp[i ,2]/(6357000)])
    plt.show()

