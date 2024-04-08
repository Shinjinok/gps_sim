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
import math
import numpy as np
import argparse
import georinex as gr
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

CA_SEQ_LEN = 1023
N_DWRD_SBF = 10
N_SBF = 5
N_DWRD = (N_SBF+1)*N_DWRD_SBF
R2D = 57.2957795131

@dataclass
class gpstime_t:
    week: int = 0
    sec: float = 0.0
@dataclass
class datetime_t:
    y: int = 0
    m: int = 0
    d: int = 0
    hh: int = 0
    mm: int = 0
    sec: float = 0.0
@dataclass 
class Eph: 
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
@dataclass 
class range_t: 
    g: gpstime_t = field(default_factory=dict)
    range: float = None # pseudorange
    rate: float = None
    d: float = None #  // geometric distance
    azel: float = field(default_factory=dict)# 
    iono_delay: float = None # 
    def __post_init__(self):
        self.g = gpstime_t()
        self.azel = np.zeros(2)
@dataclass 
class channel_t:
    prn: int = None
    ca: int = field(default_factory=dict)
    f_carr: float = None
    f_code: float = None
    carr_phase: int = None
    carr_phasestep: int = None
    code_phase: float = None
    g0: gpstime_t = field(default_factory=dict)
    sbf: int = field(default_factory=dict)
    dwrd: int = field(default_factory=dict)
    iword: int = None
    ibit: int = None
    icode: int = None
    dataBit: int = None
    codeCA: int = None
    azel: float = field(default_factory=dict)
    rho0: range_t = field(default_factory=dict)
    def __post_init__(self):
        self.ca = np.zeros(CA_SEQ_LEN)
        self.g0 = gpstime_t()
        self.sbf = np.zeros((5,N_DWRD_SBF))
        self.dwrd = np.zeros(N_DWRD)
        self.azel = np.zeros(2)
        self.rho0 = range_t()

class ionoutc_t:
    enable: int = None
    vflg: int = None
    alpha0: float = None
    alpha1: float = None
    alpha2: float = None
    alpha3: float = None
    beta0: float = None
    beta1: float = None
    beta2: float = None
    beta3: float = None
    A0: float = None
    A1: float = None
    dtls: int = None
    tot: int = None
    wnt: int = None
    dtlsf: int = None
    dn: int = None
    wnlsf: int = None


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--file', type=str, default = None)
parser.add_argument('--timeCor', type=bool, default = False)
parser.add_argument('--iteration', type=str, default = 'Newton')
args = parser.parse_args()
WGS84_RADIUS =	6378137.0
WGS84_ECCENTRICITY = 0.0818191908426
SECONDS_IN_WEEK = 604800.0
SECONDS_IN_HALF_WEEK =302400.0
SECONDS_IN_DAY = 86400.0
SECONDS_IN_HOUR = 3600.0
SECONDS_IN_MINUTE = 60.0
OMEGA_EARTH = 7.2921151467e-5
GM_EARTH = 3.986005e14
MAX_SAT = 32
MAX_CHAN = 16
GM = 3.986005*np.power(10.0,14)
c = 2.99792458*np.power(10.0,8)
omegae_dot = 7.2921151467*np.power(10.0,-5)

earth_rate = 2*np.pi/(60*60*24)


SPEED_OF_LIGHT = 2.99792458e8
LAMBDA_L1 = 0.190293672798365
EPHEM_ARRAY_SIZE = 13

def computeRange(eph, ionoutc, grx, xyz):
    rho = range_t()

    pos, vel, clk = satpos(eph,grx)
    los = subVect(pos, xyz)
    tau = np.linalg.norm(los)/SPEED_OF_LIGHT

    # Extrapolate the satellite position backwards to the transmission time.
    pos[0] -= vel[0]*tau
    pos[1] -= vel[1]*tau
    pos[2] -= vel[2]*tau

    # Earth rotation correction. The change in velocity can be neglected.
    xrot = pos[0] + pos[1]*OMEGA_EARTH*tau
    yrot = pos[1] - pos[0]*OMEGA_EARTH*tau
    pos[0] = xrot
    pos[1] = yrot

    # New observer to satellite vector and satellite range.
    los = subVect(pos, xyz)
    range = np.linalg.norm(los)
    rho.d = range

    # Pseudorange.
    rho.range = range - SPEED_OF_LIGHT*clk[0]

    # Relative velocity of SV and receiver.
    rate = np.dot(vel, los)/range

    # Pseudorange rate.
    rho.rate = rate # - SPEED_OF_LIGHT*clk[1]

    # Time of application.
    rho.g = grx

    # Azimuth and elevation angles.
    llh = xyz2llh(xyz)
    tmat = ltcmat(llh)
    neu = ecef2neu(los, tmat)
    rho.azel = neu2azel(neu)

    # Add ionospheric delay
    rho.iono_delay = ionosphericDelay(ionoutc, grx, llh, rho.azel)
    rho.range += rho.iono_delay

    return rho

def ionosphericDelay(ionoutc, g, llh, azel):
    iono_delay = 0.0
    if ionoutc.enable == False:
        return(0.0)
    E = azel[1]/np.pi
    phi_u = llh[0]/np.pi
    lam_u = llh[1]/np.pi   
    F = 1.0 + 16.0*np.power((0.53 - E),3.0)
    if ionoutc.vflg==False:
        iono_delay = F*5.0e-9*SPEED_OF_LIGHT
    else:
        psi = 0.0137/(E + 0.11) - 0.022       
        phi_i = phi_u + psi*np.cos(azel[0])
        if phi_i>0.416:
            phi_i = 0.416
        elif phi_i<-0.416:
            phi_i = -0.416
    lam_i = lam_u + psi*np.sin(azel[0])/np.cos(phi_i*np.pi)
    phi_m = phi_i + 0.064*np.cos((lam_i - 1.617)*np.pi)
    phi_m2 = phi_m*phi_m
    phi_m3 = phi_m2*phi_m

    AMP = ionoutc.alpha0 + ionoutc.alpha1*phi_m + ionoutc.alpha2*phi_m2 + ionoutc.alpha3*phi_m3
    if AMP<0.0:
        AMP = 0.0
    PER = ionoutc.beta0 + ionoutc.beta1*phi_m + ionoutc.beta2*phi_m2 + ionoutc.beta3*phi_m3
    if PER<72000.0:
        PER = 72000.0

    # Local time (sec)
    t = SECONDS_IN_DAY/2.0*lam_i + g.sec
    while t>=SECONDS_IN_DAY:
        t -= SECONDS_IN_DAY
    while t<0:
        t += SECONDS_IN_DAY

    # Phase (radians)
    X = 2.0*np.pi*(t - 50400.0)/PER

    if np.fabs(X) < 1.57:
        X2 = X*X
        X4 = X2*X2
        iono_delay = F*(5.0e-9 + AMP*(1.0 - X2/2.0 + X4/24.0))*SPEED_OF_LIGHT
    else:
        iono_delay = F*5.0e-9*SPEED_OF_LIGHT
    
    return iono_delay
def codegen(prn):

    ca = np.zeros(CA_SEQ_LEN)
    delay = [ 5,   6,   7,   8,  17,  18, 139, 140, 141, 251,
		252, 254, 255, 256, 257, 258, 469, 470, 471, 472,
		473, 474, 509, 512, 513, 514, 515, 516, 859, 860,
		861, 862]

    if prn<1 or prn >32:
        return None
    
    r1 = np.full(N_DWRD_SBF,-1)
    r2 = np.full(N_DWRD_SBF,-1)
    g1 = np.zeros(CA_SEQ_LEN)
    g2 = np.zeros(CA_SEQ_LEN)

    for i in range(CA_SEQ_LEN):
        g1[i] = r1[9]
        g2[i] = r2[9]
        c1 = r1[2]*r1[9]
        c2 = r2[1]*r2[2]*r2[5]*r2[7]*r2[8]*r2[9]
        for j in reversed(range(9)):
            r1[j] = r1[j-1]
            r2[j] = r2[j-1]   
        r1[0] = c1
        r2[0] = c2

    for i in range(CA_SEQ_LEN):
        j = CA_SEQ_LEN-delay[prn-1] + i
        ca[i] = (1-g1[i]*g2[j%CA_SEQ_LEN])/2   

    return ca
                   
POW2_M5  =0.03125
POW2_M19 =1.907348632812500e-6
POW2_M29 =1.862645149230957e-9
POW2_M31 =4.656612873077393e-10
POW2_M33 =1.164153218269348e-10
POW2_M43 =1.136868377216160e-13
POW2_M55 =2.775557561562891e-17
POW2_M50 =8.881784197001252e-016
POW2_M30 =9.313225746154785e-010
POW2_M27 =7.450580596923828e-009
POW2_M24 =5.960464477539063e-008
PI = np.pi

def eph2sbf(eph, ionoutc):

    wn = int(0)
    toe = int(eph.toe.sec/16.0)
    toc = int(eph.toc.sec/16.0)
    iode = int(eph.iode)
    iodc = int(eph.iodc)
    deltan = int(eph.deltan/POW2_M43/PI)
    cuc = int(eph.cuc/POW2_M29)
    cus = int(eph.cus/POW2_M29)
    cic = int(eph.cic/POW2_M29)
    cis = int(eph.cis/POW2_M29)
    crc = int(eph.crc/POW2_M5)
    crs = int(eph.crs/POW2_M5)
    ecc = int(eph.ecc/POW2_M33)
    sqrta = int(eph.sqrta/POW2_M19)
    m0 = int(eph.m0/POW2_M31/PI)
    omg0 = int(eph.omg0/POW2_M31/PI)
    inc0 = int(eph.inc0/POW2_M31/PI)
    aop = int(eph.aop/POW2_M31/PI)
    omgdot = int(eph.omgdot/POW2_M43/PI)
    idot = int(eph.idot/POW2_M43/PI)
    af0 = int(eph.af0/POW2_M31)
    af1 = int(eph.af1/POW2_M43)
    af2 = int(eph.af2/POW2_M55)
    tgd = int(eph.tgd/POW2_M31)
    svhlth = int(eph.svhlth)
    codeL2 = int(eph.codeL2)

    wna = int(eph.toe.week%256)
    toa = int(eph.toe.sec/4096.0)

    alpha0 = int(round(ionoutc.alpha0/POW2_M30))
    alpha1 = int(round(ionoutc.alpha1/POW2_M27))
    alpha2 = int(round(ionoutc.alpha2/POW2_M24))
    alpha3 = int(round(ionoutc.alpha3/POW2_M24))
    beta0 = int(round(ionoutc.beta0/2048.0))
    beta1 = int(round(ionoutc.beta1/16384.0))
    beta2 = int(round(ionoutc.beta2/65536.0))
    beta3 = int(round(ionoutc.beta3/65536.0))
    A0 = int(round(ionoutc.A0/POW2_M30))
    A1 = int(round(ionoutc.A1/POW2_M50))
    dtls = int(ionoutc.dtls)
    tot = int(ionoutc.tot/4096)
    wnt = int(ionoutc.wnt%256)
    #// TO DO: Specify scheduled leap seconds in command options
    #// 2016/12/31 (Sat) -> WNlsf = 1929, DN = 7 (http://navigationservices.agi.com/GNSSWeb/)
    #// Days are counted from 1 to 7 (Sunday is 1).
    wnlsf = 1929%256
    dn = 7
    dtlsf = 18
    ura=0
    dataId = 1
    sbf4_page18_svId = 56
    sbf4_page25_svId = 63
    sbf5_page25_svId = 51

    sbf= np.zeros((5,10))
    # Subframe 1
    sbf[0][0] = 0x8B0000 << 6
    sbf[0][1] = 0x1<<8
    sbf[0][2] = ((wn&0x3FF)<<20) | ((codeL2&0x3)<<18) | ((ura&0xF)<<14) | ((svhlth&0x3F)<<8) | (((iodc>>8)&0x3)<<6)
    sbf[0][3] = 0
    sbf[0][4] = 0
    sbf[0][5] = 0
    sbf[0][6] = (tgd&0xFF)<<6
    sbf[0][7] = ((iodc&0xFF)<<22) | ((toc&0xFFFF)<<6)
    sbf[0][8] = ((af2&0xFF)<<22) | ((af1&0xFFFF)<<6)
    sbf[0][9] = (af0&0x3FFFFF)<<8

# Subframe 2
    sbf[1][0] = 0x8B0000<<6
    sbf[1][1] = 0x2<<8
    sbf[1][2] = ((iode&0xFF)<<22) | ((crs&0xFFFF)<<6)
    sbf[1][3] = ((deltan&0xFFFF)<<14) | (((m0>>24)&0xFF)<<6)
    sbf[1][4] = (m0&0xFFFFFF)<<6
    sbf[1][5] = ((cuc&0xFFFF)<<14) | (((ecc>>24)&0xFF)<<6)
    sbf[1][6] = (ecc&0xFFFFFF)<<6
    sbf[1][7] = ((cus&0xFFFF)<<14) | (((sqrta>>24)&0xFF)<<6)
    sbf[1][8] = (sqrta&0xFFFFFF)<<6
    sbf[1][9] = (toe&0xFFFF)<<14

    # Subframe 3
    sbf[2][0] = 0x8B0000<<6
    sbf[2][1] = 0x3<<8
    sbf[2][2] = ((cic&0xFFFF)<<14) | (((omg0>>24)&0xFF)<<6)
    sbf[2][3] = (omg0&0xFFFFFF)<<6
    sbf[2][4] = ((cis&0xFFFF)<<14) | (((inc0>>24)&0xFF)<<6)
    sbf[2][5] = (inc0&0xFFFFFF)<<6
    sbf[2][6] = ((crc&0xFFFF)<<14) | (((aop>>24)&0xFF)<<6)
    sbf[2][7] = (aop&0xFFFFFF)<<6
    sbf[2][8] = (omgdot&0xFFFFFF)<<6
    sbf[2][9] = ((iode&0xFF)<<22) | ((idot&0x3FFF)<<8)

    if ionoutc.vflg == True:

        # Subframe 4, page 18
        sbf[3][0] = 0x8B0000<<6
        sbf[3][1] = 0x4<<8
        sbf[3][2] = (dataId<<28) | (sbf4_page18_svId<<22) | ((alpha0&0xFF)<<14) | ((alpha1&0xFF)<<6)
        sbf[3][3] = ((alpha2&0xFF)<<22) | ((alpha3&0xFF)<<14) | ((beta0&0xFF)<<6)
        sbf[3][4] = ((beta1&0xFF)<<22) | ((beta2&0xFF)<<14) | ((beta3&0xFF)<<6)
        sbf[3][5] = (A1&0xFFFFFF)<<6
        sbf[3][6] = ((A0>>8)&0xFFFFFF)<<6
        sbf[3][7] = ((A0&0xFF)<<22) | ((tot&0xFF)<<14) | ((wnt&0xFF)<<6)
        sbf[3][8] = ((dtls&0xFF)<<22) | ((wnlsf&0xFF)<<14) | ((dn&0xFF)<<6)
        sbf[3][9] = (dtlsf&0xFF)<<22

    else:

        # Subframe 4, page 25
        sbf[3][0] = 0x8B0000<<6
        sbf[3][1] = 0x4<<8
        sbf[3][2] = (dataId<<28) | (sbf4_page25_svId<<22)
        sbf[3][3] = 0
        sbf[3][4] = 0
        sbf[3][5] = 0
        sbf[3][6] = 0
        sbf[3][7] = 0
        sbf[3][8] = 0
        sbf[3][9] = 0


    # Subframe 5, page 25
    sbf[4][0] = 0x8B0000<<6
    sbf[4][1] = 0x5<<8
    sbf[4][2] = (dataId<<28) | (sbf5_page25_svId<<22) | ((toa&0xFF)<<14) | ((wna&0xFF)<<6)
    sbf[4][3] = 0
    sbf[4][4] = 0
    sbf[4][5] = 0
    sbf[4][6] = 0
    sbf[4][7] = 0
    sbf[4][8] = 0
    sbf[4][9] = 0

    for i in range(5):
        for j in range(10):
            sbf[i][j] = int (sbf[i][j])
    return sbf

def countBits(v):
    S = [1, 2, 4, 8, 16]
    B = [0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF]

    c = v
    c = ((c >> S[0]) & B[0]) + (c & B[0])
    c = ((c >> S[1]) & B[1]) + (c & B[1])
    c = ((c >> S[2]) & B[2]) + (c & B[2])
    c = ((c >> S[3]) & B[3]) + (c & B[3])
    c = ((c >> S[4]) & B[4]) + (c & B[4])

    return c

def computeChecksum(source, nib):
    """ /*
    Bits 31 to 30 = 2 LSBs of the previous transmitted word, D29* and D30*
    Bits 29 to  6 = Source data bits, d1, d2, ..., d24
    Bits  5 to  0 = Empty parity bits
    */ 

    /*
    Bits 31 to 30 = 2 LSBs of the previous transmitted word, D29* and D30*
    Bits 29 to  6 = Data bits transmitted by the SV, D1, D2, ..., D24
    Bits  5 to  0 = Computed parity bits, D25, D26, ..., D30
    */ 

    /*
            1            2           3
    bit    12 3456 7890 1234 5678 9012 3456 7890
    ---    -------------------------------------
    D25    11 1011 0001 1111 0011 0100 1000 0000
    D26    01 1101 1000 1111 1001 1010 0100 0000
    D27    10 1110 1100 0111 1100 1101 0000 0000
    D28    01 0111 0110 0011 1110 0110 1000 0000
    D29    10 1011 1011 0001 1111 0011 0100 0000
    D30    00 1011 0111 1010 1000 1001 1100 0000
    */
 """
    bmask = [0x3B1F3480, 0x1D8F9A40, 0x2EC7CD00, 0x1763E680, 0x2BB1F340, 0x0B7A89C0]

    D=0
    d = source & 0x3FFFFFC0
    D29 = (source>>31)&0x1
    D30 = (source>>30)&0x1

    if nib: # Non-information bearing bits for word 2 and 10

    # Solve bits 23 and 24 to preserve parity check
    # with zeros in bits 29 and 30.
        

        if ((D30 + countBits(bmask[4] & d)) % 2):
            d ^= (0x1<<6)
        if ((D29 + countBits(bmask[5] & d)) % 2):
            d ^= (0x1<<7)

    D = d
    if D30:
        D ^= 0x3FFFFFC0

    D |= ((D29 + countBits(bmask[0] & d)) % 2) << 5
    D |= ((D30 + countBits(bmask[1] & d)) % 2) << 4
    D |= ((D29 + countBits(bmask[2] & d)) % 2) << 3
    D |= ((D30 + countBits(bmask[3] & d)) % 2) << 2
    D |= ((D30 + countBits(bmask[4] & d)) % 2) << 1
    D |= ((D29 + countBits(bmask[5] & d)) % 2)

    D &= 0x3FFFFFFF
    #D |= (source & 0xC0000000) // Add D29* and D30* from source data bits
    return D

def generateNavMsg(g, chan, init):
    g0 = gpstime_t()
    g0.week = g.week
    g0.sec = float((int(g.sec+0.5))/30) * 30.0 #// Align with the full frame length = 30 sec
    chan.g0 = g0 # Data bit reference time

    wn = g0.week%1024
    tow = int(g0.sec/6)

    if init == 1:
        prevwrd = int(0)
        for iwrd in range(N_DWRD_SBF):
            sbfwrd = int(chan.sbf[4][iwrd])

            # Add TOW-count message into HOW
            if iwrd == 1:
                sbfwrd |= (tow&0x1FFFF)<<13

            # Compute checksum
            sbfwrd = sbfwrd | ((prevwrd<<30) & 0xC0000000) # 2 LSBs of the previous transmitted word

            if iwrd==1 or iwrd == 9:
                nib = 1
            else:
                nib = 0
            #nib = ((iwrd==1)||(iwrd==9))?1:0 # Non-information bearing bits for word 2 and 10
            chan.dwrd[iwrd] = computeChecksum(sbfwrd, nib)

            prevwrd = int(chan.dwrd[iwrd])
    else:
        for iwrd in range(N_DWRD_SBF):
            chan.dwrd[iwrd] = int(chan.dwrd[N_DWRD_SBF*N_SBF+iwrd])
            prevwrd = int(chan.dwrd[iwrd])

    for isbf in range(N_SBF):
        tow += 1
        for iwrd in range(N_DWRD_SBF):
            sbfwrd = int(chan.sbf[isbf][iwrd])
            #  Add transmission week number to Subframe 1
            if isbf==0 and iwrd==2:
                sbfwrd |= (wn&0x3FF)<<20

            #  Add TOW-count message into HOW
            if iwrd==1:
                sbfwrd |= ((tow&0x1FFFF)<<13)

            #  Compute checksum
            sbfwrd |= (prevwrd<<30) & 0xC0000000 #  2 LSBs of the previous transmitted word
            #nib = ((iwrd==1)||(iwrd==9))?1:0; #  Non-information bearing bits for word 2 and 10
            if iwrd==1 or iwrd == 9:
                nib = 1
            else:
                nib = 0
            chan.dwrd[(isbf+1)*N_DWRD_SBF+iwrd] = computeChecksum(sbfwrd, nib)

            prevwrd = int(chan.dwrd[(isbf+1)*N_DWRD_SBF+iwrd])

    return chan
    
def allocateChannel(eph, ionoutc, grx, xyz, elvMask):
    chan = [channel_t() for _ in range(MAX_SAT)]
    for i in range(MAX_CHAN):
        chan[i].prn = 0
    azel = np.zeros(2)
    rho = range_t()
    ref = np.zeros(3)

    nsat = 0
    for sv in range(MAX_SAT):
        vis, azel = checkSatVisibility(eph[sv],grx,xyz,elvMask)
        if vis == 1:
            nsat += 1# // Number of visible satellites

            if allocatedSat[sv] == -1 : # // Visible but not allocated
#				// Allocated new satellite
                for i in range(MAX_CHAN):

                    if chan[i].prn == 0:

                        #// Initialize channel
                        chan[i].prn = sv+1
                        chan[i].azel[0] = azel[0]
                        chan[i].azel[1] = azel[1]

                       # C/A code generation
                        chan[i].ca = codegen(chan[i].prn)
                        
                        # Generate subframe
                        chan[i].sbf = eph2sbf(eph[sv], ionoutc)

						# Generate navigation message
                        chan[i] = generateNavMsg(grx, chan[i], 1)

					    # Initialize pseudorange
                        rho = computeRange(eph[sv], ionoutc, grx, xyz)
                        chan[i].rho0 = rho

                        # Initialize carrier phase
                        r_xyz = rho.range

                        rho = computeRange(eph[sv], ionoutc, grx, ref)
                        r_ref = rho.range

                        phase_ini = 0.0 # TODO: Must initialize properly
                        phase_ini = (2.0*r_ref - r_xyz)/LAMBDA_L1

                        phase_ini -= math.floor(phase_ini)
                        chan[i].carr_phase = int(512.0 * 65536.0 * phase_ini)

                        break

                #// Set satellite allocation channel
                if i < MAX_CHAN:
                    allocatedSat[sv] = i

        elif allocatedSat[sv] >= 0: # // Not visible but allocated
            
			#// Clear channel
            chan[allocatedSat[sv]].prn = 0

			#// Clear satellite allocation flag
            allocatedSat[sv] = -1
		
    return chan

def incGpsTime(g0, dt):
    g1 = gpstime_t()

    g1.week = g0.week
    g1.sec = g0.sec + dt
    g1.sec = round(g1.sec*1000.0)/1000.0 #// Avoid rounding error

    while  g1.sec >= SECONDS_IN_WEEK:
	
        g1.sec -= SECONDS_IN_WEEK
        g1.week += 1

    while  g1.sec < 0.0:

        g1.sec += SECONDS_IN_WEEK
        g1.week -= 1

    return g1

def checkSatVisibility(eph_, g, xyz, elvMask):

    if eph_.vflg !=1:
        return 0, 0
    llh=np.zeros(3)
    neu=np.zeros(3)
    pos=np.zeros(3)
    vel=np.zeros(3)
    clk=np.zeros(3)
    los=np.zeros(3)
    tmat=np.zeros((3,3))

    llh = xyz2llh(xyz)
    tmat = ltcmat(llh)

    pos, vel, clk = satpos(eph_, g)
    los = subVect(pos, xyz)
    neu = ecef2neu(los, tmat)
    azel = neu2azel(neu)

    if azel[1]*180/np.pi > elvMask:
        return 1 , azel#Visible
    return 0 , azel

def neu2azel(neu):
    azel = np.zeros(2)
    azel[0] = np.arctan2(neu[1],neu[0])
    if azel[0] < 0.0:
        azel[0] += (2.0*np.pi)

    ne = np.sqrt(neu[0]*neu[0] + neu[1]*neu[1])
    azel[1] = np.arctan2(neu[2], ne)
    return azel

def ecef2neu(xyz,t):
    neu = np.zeros(3)
    neu[0] = t[0][0]*xyz[0] + t[0][1]*xyz[1] + t[0][2]*xyz[2]
    neu[1] = t[1][0]*xyz[0] + t[1][1]*xyz[1] + t[1][2]*xyz[2]
    neu[2] = t[2][0]*xyz[0] + t[2][1]*xyz[1] + t[2][2]*xyz[2]
    return neu

def subVect(x1, x2):
    y = np.zeros(3)
    y[0] = x1[0]-x2[0]
    y[1] = x1[1]-x2[1]
    y[2] = x1[2]-x2[2]
    return y

def satpos(eph, g):
   
    pos=np.zeros(3)
    vel=np.zeros(3)
    clk=np.zeros(3)

    deltan = eph.deltan
    tk = g.sec - eph.toe.sec

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
    relativistic = -4.442807633E-10*eph.ecc*eph.sqrta*sek
    #aop : argument of perigee(rad) (lower    case    omega)
    #pk: true anomaly theta/2
    pk = np.arctan2(eph.sq1e2*sek,cek-eph.ecc) + eph.aop
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

    return pos, vel, clk

def ltcmat(llh):
    t = np.zeros((3,3))
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
        zdz = z + dz
        nh = np.sqrt(rho2 + zdz*zdz)
        slat = zdz / nh
        n = a / np.sqrt(1.0-e2*slat*slat)
        dz_new = n*e2*slat

        if np.fabs(dz-dz_new) < eps:
            break

        dz = dz_new


    llh[0] = np.arctan2(zdz, np.sqrt(rho2))
    llh[1] = np.arctan2(y, x)
    llh[2] = nh - n

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

def date2gps(t):
    g = gpstime_t()
    doy= [0,31,59,90,120,151,181,212,243,273,304,334]

    ye = t.y - 1980

	# Compute the number of leap days since Jan 5/Jan 6, 1980.
    lpdays = ye/4 + 1
    if ye%4 ==0 and t.m <=2:
        lpdays -=1

	# Compute the number of days elapsed since Jan 5/Jan 6, 1980.
    de = ye*365 + doy[t.m-1] + t.d + lpdays - 6

	#// Convert time to GPS weeks and seconds.
    g.week = int( de / 7 )
    g.sec = (de%7)*SECONDS_IN_DAY + t.hh*SECONDS_IN_HOUR + t.mm*SECONDS_IN_MINUTE + t.sec

    return g

def subGpsTime(g1,g0):
    dt = g1.sec - g0.sec
    dt += (g1.week - g0.week) * SECONDS_IN_WEEK
    return dt

def readRinexNavAll(fname):
    eph = [[Eph() for _ in range(MAX_SAT)] for _ in range(EPHEM_ARRAY_SIZE)]
    ionoutc = ionoutc_t()
    g0 = gpstime_t()
    g = gpstime_t()
    


    f = open(fname,'r')
    # Clear valid flag
    for ieph in range(EPHEM_ARRAY_SIZE):
        for sv in range(MAX_SAT):
            eph[ieph][sv].vflg = 0
    
    while True:
        r = f.readline()
        #print(r)

        if r.find('END OF HEADER') > 0:
            ionoutc.vflg = True
            break
        elif r.find('ION ALPHA') > 0:

            ionoutc.alpha0 = float(r[2:14].replace('D','E'))
            ionoutc.alpha1 = float(r[14:26].replace('D','E'))
            ionoutc.alpha2 = float(r[26:38].replace('D','E'))
            ionoutc.alpha3 = float(r[38:50].replace('D','E'))

        elif r.find('ION BETA') > 0:

            ionoutc.beta0 = float(r[2:14].replace('D','E'))
            ionoutc.beta1 = float(r[14:26].replace('D','E'))
            ionoutc.beta2 = float(r[26:38].replace('D','E'))
            ionoutc.beta3 = float(r[38:50].replace('D','E'))
        
        elif r.find('DELTA-UTC') > 0:

            ionoutc.A0 = float(r[3:22].replace('D','E'))
            ionoutc.A1 = float(r[22:41].replace('D','E'))
            ionoutc.tot = int(r[41:50])
            ionoutc.wnt = int(r[50:59])

            
        elif r.find('LEAP SECONDS') > 0: 
            ionoutc.dtls = int(r[0:6]) 

           # print(ionoutc.dtls)  
            
    ionoutc.vflg = True
    # Read ephemeris blocks
    g0.week = -1
    ieph = 0

    while True:
        r = f.readline()
        if r == '':
            break
        #PRN
        sv = int(r[0:2])-1
        #EPOCH
        t = datetime_t()
        t.y = int(r[3:5])+2000
        t.m = int(r[6:8])
        t.d = int(r[9:11])
        t.hh = int(r[12:14])
        t.mm = int(r[15:17])
        t.sec = float(r[18:22])
        #print(t)

        g = date2gps(t)

        if g0.week == -1:
            g0 = g
        
        dt = subGpsTime(g, g0)

        if dt > SECONDS_IN_HOUR:
            g0 = g
            ieph +=1
            if ieph >= EPHEM_ARRAY_SIZE:
                break
        # Date and time
        eph[ieph][sv].t = t
        #print(ieph,sv,t)
        eph[ieph][sv].toc = g
        eph[ieph][sv].af0 = float(r[22:22+19].replace('D','E'))
        eph[ieph][sv].af1 = float(r[41:41+19].replace('D','E'))
        eph[ieph][sv].af2 = float(r[60:60+19].replace('D','E'))

        #print(eph[ieph][sv].af0,eph[ieph][sv].af1,eph[ieph][sv].af2)
        #// BROADCAST ORBIT 1
        r = f.readline()
        eph[ieph][sv].iode = float(r[3:3+19].replace('D','E'))
        eph[ieph][sv].crs = float(r[22:22+19].replace('D','E'))
        eph[ieph][sv].deltan = float(r[41:41+19].replace('D','E'))
        eph[ieph][sv].m0 = float(r[60:60+19].replace('D','E'))
        #print('[r1]',r)
        #print('[r1]',eph[ieph][sv].iode,eph[ieph][sv].crs,eph[ieph][sv].deltan,eph[ieph][sv].m0)

        #// BROADCAST ORBIT 2
        r = f.readline()
        eph[ieph][sv].cuc = float(r[3:3+19].replace('D','E'))
        eph[ieph][sv].ecc = float(r[22:22+19].replace('D','E'))
        eph[ieph][sv].cus = float(r[41:41+19].replace('D','E'))
        eph[ieph][sv].sqrta = float(r[60:60+19].replace('D','E'))
        #print('[r2]',r)
        #print('[r2]',eph[ieph][sv].cuc,eph[ieph][sv].ecc,eph[ieph][sv].cus,eph[ieph][sv].sqrta)

        #// BROADCAST ORBIT 3
        r = f.readline()
        eph[ieph][sv].toe.sec = float(r[3:3+19].replace('D','E'))
        eph[ieph][sv].cic = float(r[22:22+19].replace('D','E'))
        eph[ieph][sv].omg0 = float(r[41:41+19].replace('D','E'))
        eph[ieph][sv].cis = float(r[60:60+19].replace('D','E'))
        #print('[r3]',r)
        #print('[r3]',eph[ieph][sv].toe.sec,eph[ieph][sv].cic,eph[ieph][sv].omg0,eph[ieph][sv].cis)

        #// BROADCAST ORBIT 4
        r = f.readline()
        eph[ieph][sv].inc0 = float(r[3:3+19].replace('D','E'))
        eph[ieph][sv].crc = float(r[22:22+19].replace('D','E'))
        eph[ieph][sv].aop = float(r[41:41+19].replace('D','E'))
        eph[ieph][sv].omgdot = float(r[60:60+19].replace('D','E'))
        #print('[r4]',r)
        #print('[r4]',eph[ieph][sv].inc0,eph[ieph][sv].crc,eph[ieph][sv].aop,eph[ieph][sv].omgdot)

        #// BROADCAST ORBIT 5
        r = f.readline()
        eph[ieph][sv].idot = float(r[3:3+19].replace('D','E'))
        eph[ieph][sv].codeL2 = float(r[22:22+19].replace('D','E'))
        eph[ieph][sv].toe.week = float(r[41:41+19].replace('D','E'))
        
        #print('[r5]',r)
        #print('[r5]',eph[ieph][sv].idot,eph[ieph][sv].codeL2,eph[ieph][sv].toe.week)

        #// BROADCAST ORBIT 6
        r = f.readline()
        eph[ieph][sv].svhlth = int(float(r[22:22+19].replace('D','E')))
        if eph[ieph][sv].svhlth>0 and eph[ieph][sv].svhlth<32:
            eph[ieph][sv].svhlth += 32 # // Set MSB to 1

        eph[ieph][sv].tgd = float(r[41:41+19].replace('D','E'))
        eph[ieph][sv].iodc = float(r[60:60+19].replace('D','E'))
        
        #print('[r6]',r)
        #print('[r6]',eph[ieph][sv].svhlth,eph[ieph][sv].tgd,eph[ieph][sv].iodc)

        # Set valid flag
        eph[ieph][sv].vflg = 1


        #// Update the working variables
        eph[ieph][sv].A = eph[ieph][sv].sqrta * eph[ieph][sv].sqrta
        eph[ieph][sv].n = np.sqrt(GM_EARTH/(eph[ieph][sv].A*eph[ieph][sv].A*eph[ieph][sv].A)) + eph[ieph][sv].deltan
        eph[ieph][sv].sq1e2 = np.sqrt(1.0 - eph[ieph][sv].ecc*eph[ieph][sv].ecc)
        eph[ieph][sv].omgkdot = eph[ieph][sv].omgdot - OMEGA_EARTH
        r = f.readline()

    f.close()

    if g0.week >= 0:
        ieph +=1
    
    return ieph, eph, ionoutc

if __name__ == "__main__":
    USER_MOTION_SIZE = 3000
    verb = True
    gmin = gpstime_t()
    gmax = gpstime_t()
    t0 = datetime_t()
    tmin = datetime_t()
    tmax = datetime_t()
    g0 = gpstime_t()
    g0.week = -1
    timeoverwrite = False # // Overwrite the TOC and TOE in the RINEX file
    iduration = USER_MOTION_SIZE
    numd = iduration

    ########################################################
    #   Receiver position
    ########################################################

    llh = np.zeros(3)
    llh[0] = 30.286502 / R2D
    llh[1] = 120.032669 / R2D
    llh[2] = 100

    print("Using static location mode.\n")
    numd = iduration
    xyz = llh2xyz(llh)
    print("xyz = ",xyz[0],xyz[1],xyz[2],'\n')
    print("llh = ",llh[0]*R2D,llh[1]*R2D,llh[2],'\n')

    ####################
    ## Read ephemeris ##
    ####################

    args.file = "brdc0730.24n"
    print('\n--- Calculate satellite position ---\nN file:',args.file)

    neph, eph, ionoutc = readRinexNavAll(args.file)

    if neph == 0:
        print("ERROR: No ephemeris available.\n")
        exit
    elif neph == -1:
        print("ERROR: ephemeris file not found.\n")
        exit
    
    if verb == True and ionoutc.vflg == True:
        print("ionoutc.alpha0, ionoutc.alpha1, ionoutc.alpha2, ionoutc.alpha3\n"
              ,ionoutc.alpha0, ionoutc.alpha1, ionoutc.alpha2, ionoutc.alpha3 )
        print("ionoutc.beta0, ionoutc.beta1, ionoutc.beta2, ionoutc.beta3\n",
              ionoutc.beta0, ionoutc.beta1, ionoutc.beta2, ionoutc.beta3)
        print("ionoutc.A0, ionoutc.A1, ionoutc.tot, ionoutc.wnt\n",
               ionoutc.A0, ionoutc.A1, ionoutc.tot, ionoutc.wnt)
        print("ionoutc.dtls\n",ionoutc.dtls)

    for sv in range(MAX_SAT):
        if eph[0][sv].vflg == 1:
            gmin = eph[0][sv].toc
            tmin = eph[0][sv].t
            break

    gmax.sec = 0
    gmax.week = 0
    tmax.sec = 0
    tmax.mm = 0
    tmax.hh = 0
    tmax.d = 0
    tmax.m = 0
    tmax.y = 0

    for sv in range(MAX_SAT):
        if eph[neph-1][sv].vflg ==1:
            gmax = eph[neph-1][sv].toc
            tmax = eph[neph-1][sv].t

            break

    if g0.week >= 0:
        if timeoverwrite == True:
            gtmp = gpstime_t()
    else:
        g0 = gmin
        t0 = tmin

    print("Start time = ",t0.y,'/', t0.m,'/', t0.d,' ', t0.hh,':', t0.mm,':',
           t0.sec,'(', g0.week,':', g0.sec,')\n')
    print("Duration = ",numd/10.0,'[sec]\n')

    #Select the current set of ephemerides
    ieph = -1

    for i in range(neph):
        for sv in range(MAX_SAT):
            if eph[i][sv].vflg == 1:
                dt = subGpsTime(g0,eph[i][sv].toc)
                if dt >= -SECONDS_IN_HOUR and dt < SECONDS_IN_HOUR:
                    ieph = i
                    break
        if ieph >= 0:
            break
    if ieph == -1:
        print("ERROR: No current set of ephemerides has been found.\n")
        exit()
    ########################################################
    #   Baseband signal buffer and output file
    ########################################################
    # Buffer size
    samp_freq = 2.6e6	
    iq_buff_size = int(samp_freq/10.0)# samples per 0.1sec

    delt = 1.0/samp_freq


    fout = open('gpssim.bin','wb')

    ########################################################
    #   Initialize channels
    ########################################################

    

    allocatedSat = np.full(MAX_SAT,-1)
    #Initial reception time
    grx = incGpsTime(g0, 0.0)

    # Allocate visible satellites
    elvmask = 0.0
    chan = allocateChannel(eph[ieph], ionoutc, grx, xyz, elvmask)

    for i in range(MAX_CHAN):
        if chan[i].prn > 0:
            print("{:2d}, {:6.1f}, {:5.1f}, {:11.1f}, {:5.1f}".format(chan[i].prn, chan[i].azel[0]*R2D, chan[i].azel[1]*R2D, chan[i].rho0.d, chan[i].rho0.iono_delay))



   # iq_buff = calloc(2*iq_buff_size, 2)


   # eph = [Eph() for x,y in zip(MAX_SAT, EPHEM_ARRAY_SIZE)]

    
   # data = gr.load(args.file)
   # df = data.to_dataframe()
    
    """ g0 = gpstime_t()
    g0.sec = 0
    
    grx = incGpsTime(g0, 0.0)

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
    plt.show() """

