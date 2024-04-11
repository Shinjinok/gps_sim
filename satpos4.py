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
import struct
import time
import rinex_comman as rc
from multiprocessing import Process, shared_memory


CA_SEQ_LEN = 1023
N_DWRD_SBF = 10
N_SBF = 5
N_DWRD = (N_SBF+1)*N_DWRD_SBF
R2D = 57.2957795131


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


USER_MOTION_SIZE = 3000
verb = True
gmin = rc.gpstime_t()
gmax = rc.gpstime_t()
t0 = rc.datetime_t()
tmin = rc.datetime_t()
tmax = rc.datetime_t()
g0 = rc.gpstime_t()
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
xyz = rc.llh2xyz(llh)
print("xyz = ",xyz[0],xyz[1],xyz[2],'\n')
print("llh = ",llh[0]*R2D,llh[1]*R2D,llh[2],'\n')

####################
## Read ephemeris ##
####################

file = "brdc0730.24n"
print('\n--- Calculate satellite position ---\nN file:',file)

neph, eph, ionoutc = rc.readRinexNavAll(file)

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
        gtmp = rc.gpstime_t()
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
            dt = rc.subGpsTime(g0,eph[i][sv].toc)
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
iq_buff_size = int(np.floor(samp_freq/10.0))# samples per 0.1sec

delt = 1.0/samp_freq


fout = open('gpssim.bin','wb')

########################################################
#   Initialize channels
########################################################




#Initial reception time
grx = rc.incGpsTime(g0, 0.0)

# Allocate visible satellites
elvmask = 0.0
chan = rc.allocateChannel(eph[ieph], ionoutc, grx, xyz, elvmask)
CHAN = 0
for i in range(MAX_CHAN):
    if chan[i].prn > 0:
        CHAN += 1
        print("{:2d}, {:6.1f}, {:5.1f}, {:11.1f}, {:5.1f}".format(chan[i].prn, chan[i].azel[0]*R2D, chan[i].azel[1]*R2D, chan[i].rho0.d, chan[i].rho0.iono_delay))

########################################################
#   Receiver antenna gain pattern
########################################################
ant_pat = np.zeros(37)
ant_pat_db = [0.00,  0.00,  0.22,  0.44,  0.67,  1.11,  1.56,  2.00,  2.44,  2.89,  3.56,  4.22,
            4.89,  5.56,  6.22,  6.89,  7.56,  8.22,  8.89,  9.78, 10.67, 11.56, 12.44, 13.33,
            14.44, 15.56, 16.67, 17.78, 18.89, 20.00, 21.33, 22.67, 24.00, 25.56, 27.33, 29.33,
            31.56]
for i in range(37):
    ant_pat[i] = np.power(10.0, -ant_pat_db[i]/20.0)

########################################################
#   Generate baseband signals
########################################################

# Update receiver time
grx = rc.incGpsTime(grx, 0.1)


gain = np.zeros(CHAN)
#shm_b = shared_memory.SharedMemory(rc.shm_a.name)
for iumd in range(numd):
    
    start = time.clock_gettime(time.CLOCK_REALTIME)
    for i in range(CHAN):
        if chan[i].prn > 0:
            #Refresh code phase and data bit counters
            #rho = range_t()
            sv = chan[i].prn - 1
            # Current pseudorange
            rho =  rc.computeRange(eph[ieph][sv], ionoutc, grx, xyz)
            chan[i].azel = rho.azel
            
            # Update code phase and data bit counters
            chan[i] = rc.computeCodePhase(rho,chan[i], 0.1)
            chan[i].carr_phasestep = int(np.round(512.0 * 65536.0 * chan[i].f_carr * delt))
            # Path loss
            path_loss = 20200000.0/rho.d
            # Receiver antenna gain
            ibs = (int)((90.0-rho.azel[1]*R2D)/5.0) #covert elevation to boresight
            ant_gain = ant_pat[ibs]
            # Signal gain
            gain[i] = (int)(path_loss * ant_gain * 128.0) # scaled by 2^7
    
    
    procs = []

    for index in range(CHAN):
        proc  = Process(target=rc.cal_acc, args=(chan[index],index))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    ip_2 = rc.a[:CHAN][:] * rc.b[:CHAN][:]
    ip_ = ip_2 * np.reshape(gain,(CHAN,1))
    ip  =  ip_ *rc.c[:CHAN][:]
    qp =  ip_ *rc.s[:CHAN][:]
    sum_ip = sum(ip)
    sum_qp = sum(qp)

    for isamp in range(iq_buff_size):
        i_acc = sum_ip[isamp]
        q_acc = sum_qp[isamp]
        # Scaled by 2^7
        i_acc = int(i_acc + 64) >> 7
        q_acc = int(q_acc + 64) >> 7
        #print(i_acc,q_acc) 
        bout = struct.pack('<hh',i_acc,q_acc)
        fout.write(bout)
        #
        # Update navigation message and channel allocation every 30 seconds
        #

    igrx = (int)(grx.sec*10.0)
    if igrx%300==0: # Every 30 seconds
        
        # Update navigation message
        for i in range(MAX_CHAN):
            if (chan[i].prn>0):
                chan[i] = rc.generateNavMsg(grx, chan[i], 0)
        # Refresh ephemeris and subframes
        # Quick and dirty fix. Need more elegant way.
        for sv in range(MAX_SAT):
            if (eph[ieph+1][sv].vflg==1):
                dt = rc.subGpsTime(eph[ieph+1][sv].toc, grx)
                if (dt<SECONDS_IN_HOUR):
                    ieph += 1
                    for i in range(MAX_CHAN):
                        # Generate new subframes if allocated
                        if (chan[i].prn!=0): 
                            chan[i].sbf = rc.eph2sbf(eph[ieph][chan[i].prn-1], ionoutc)
                break
        # Update channel allocation
        chan = rc.allocateChannel(eph[ieph], ionoutc, grx, xyz, elvmask)

            # Show details about simulated channels
        verb = True
        if (verb==True):
            print("\n")
            CHAN = 0
            for i in range(MAX_CHAN):
                if chan[i].prn > 0:
                    CHAN += 1
                    print("{:2d}, {:6.1f}, {:5.1f}, {:11.1f}, {:5.1f}".format(
                        chan[i].prn, chan[i].azel[0]*R2D, chan[i].azel[1]*R2D, chan[i].rho0.d, chan[i].rho0.iono_delay))
            gain = np.zeros(CHAN)
    # Update receiver time
    grx = rc.incGpsTime(grx, 0.1)
    # Update time counter
    print("\rTime into run = {:4.1f} takes {:3.3f}sec".format(rc.subGpsTime(grx, g0),time.clock_gettime(time.CLOCK_REALTIME)-start))
    
print("____Done______")
fout.close()
