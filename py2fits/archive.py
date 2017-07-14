'''
Jeffrey Hazboun 2017

based on archive class in PyPulse by Michael Lam

PSRFITS specification: www.atnf.csiro.au/people/pulsar/index.html?n=Main.Psrfits

EXAMPLE USAGE:
ar = Archive(filename)
ar.tscrunch()
ar.pavplot()

TODO:

Check POL_TYPE in the above to figure out how to pscrunch

Add emulate_psrchive mode?

Allow for chopping of subints, frequencies, etc?

Flip order of arguments in scrunching?

Check if time() still works.
'''

import numpy as np
import numpy.ma as ma
import gc as g
import matplotlib.pyplot as plt
import time
import py2fits.utils as u
#import pypulse.singlepulse as SP
import py2fits.par as par
Par = par.Par
#import pypulse.calibrator as calib
#Calibrator = calib.Calibrator
import decimal as d
Decimal = d.Decimal
from importlib import import_module
import inspect
import tempfile
import os
try:
    import astropy.io.fits as pyfits
except:
    import pyfits
import astropy.coordinates as coordinates
import astropy.units as units
import sys
if sys.version_info.major == 2:
    fmap = map
elif sys.version_info.major == 3:
    fmap = lambda x,*args: list(map(x,*args))
    xrange = range

PSR = "PSR"
CAL = "CAL"
FON = "FON"
FOF = "FOF"
PCM = "PCM"
SEARCH = "SEARCH"





class Archive:
    def __init__(self,filename,prepare=False,lowmem=False,verbose=True,weight=False,center_pulse=False,baseline_removal=False,wcfreq=False,thread=False,cuda=False):
        ## Parse filename here
        self.pypulse_history = []
        #self.record(inspect.currentframe())
        self.filename = str(filename) #fix unicode issue
        self.prepare = prepare
        self.lowmem = lowmem
        self.verbose = verbose
        self.center_pulse = center_pulse
        self.baseline_removal = baseline_removal
        self.wcfreq = wcfreq
        self.thread = thread
        self.cuda = cuda

        if verbose:
            print("Loading: %s" % self.filename)
            t0=time.time()

        self.load(self.filename,prepare=prepare,center_pulse=center_pulse,baseline_removal=baseline_removal,weight=weight,wcfreq=wcfreq)
        if not self.lowmem:
            self.data_orig = np.copy(self.data)
            self.weights_orig = np.copy(self.weights)
        if verbose:
            t1=time.time()
            print("Load time: %0.2f s" % (t1-t0))

        #self.reset(False) #put prepare into here?, copying to arch is done here

        #if prepare:
        #    self.pscrunch()

        #if verbose and prepare:
        #    t2=time.time()
        #    print("Prep time: %0.2f s" % (t2-t1))

    def __repr__(self):
        return "Archive(%r,prepare=%r,lowmem=%r,verbose=%r)" % (self.filename,self.prepare,self.lowmem,self.verbose)
    def __str__(self):
        return self.filename



    def load(self,filename,prepare=False,center_pulse=False,baseline_removal=False,weight=False,wcfreq=False):
        """
        Loads a PSRFITS file
        http://www.atnf.csiro.au/people/pulsar/index.html?n=PsrfitsDocumentation.Txt
        """
        if filename is None: #Needed?
            filename = self.filename
        try:
            if self.lowmem:
                hdulist = pyfits.open(filename,ignore_missing_end=True,memmap=True) #Probably don't need...
            else:
                hdulist = pyfits.open(filename,ignore_missing_end=True)
        except IOError:
            print("Filename not found")
            raise SystemExit
        self.header = hdulist[0].header
        self.keys = fmap(lambda x: x.name,hdulist)
        tablenames = self.keys[:] #temporary list for checking other tables

        if 'HISTORY' in self.keys:
            tablenames.remove('HISTORY')
            self.history = History(hdulist['HISTORY'])
            self.nsubint = self.history.getLatest("NSUB")
            self.npol = self.history.getLatest("NPOL")
            self.nchan = self.history.getLatest("NCHAN")
            self.nbin = self.history.getLatest("NBIN")
            self.nsblk = self.history.getLatest("NSBLK")
        else:
            self.history = None
            self.nsubint = hdulist['SUBINT'].header['NAXIS2']
            self.nbin,self.nchan,self.npol,self.nsblk = fmap(int,hdulist['SUBINT'].columns[-1].dim[1:-1].split(","))

        nsubint = self.nsubint
        npol = self.npol
        nchan = self.nchan
        nbin = self.nbin
        nsblk = self.nsblk

        if 'PSRPARAM' in self.keys:
            tablenames.remove('PSRPARAM')
            self.paramheaderlist = hdulist['PSRPARAM'].header.keys()
            self.paramheader = dict()
            for key in self.paramheaderlist:
                self.paramheader[key] = hdulist['PSRPARAM'].header[key]
            self.params = Par(fmap(lambda x: x[0],hdulist['PSRPARAM'].data),numwrap=float)
        elif 'PSREPHEM' in self.keys:
            tablenames.remove('PSREPHEM')
            paramkeys = fmap(lambda x: x.name,hdulist['PSREPHEM'].columns)
            paramvals = hdulist['PSREPHEM'].data[0]
            paramstrs = fmap(lambda x,y: "%s %s"%(x,y),paramkeys,paramvals)
            #self.paramheaderlist = hdulist['PSREPHEM'].header.keys()
            #self.paramheaderdict = dict()
            #for key in self.paramheaderlist:
            #    self.paramheaderdict[key] = hdulist['PSREPHEM'].header[key]
            self.params = Par(paramstrs,numwrap=float)
        else:
            self.params = None



        if 'POLYCO' in self.keys:
            tablenames.remove('POLYCO')
            self.polyco = Polyco(hdulist['POLYCO'],MJD=self.getMJD(full=True))
        else:
            self.polyco = None


        tablenames.remove('PRIMARY')
        if 'SUBINT' in tablenames:
            tablenames.remove('SUBINT')

        isFluxcal = False
        if 'FLUX_CAL' in tablenames: #need to account for is this appropriate?
            tablenames.remove('FLUX_CAL')
            isFluxcal = True
        self.tables = list()
        for tablename in tablenames: #remaining table names to store
            self.tables.append(hdulist[tablename].copy())



        #if self.header['OBS_MODE'] == 'PCM':
        if isFluxcal:

            raise SystemExit





        self.subintinfo = dict()
        self.subintinfolist = fmap(lambda x: x.name, hdulist['SUBINT'].columns[:-5])
        for i,column in enumerate(hdulist['SUBINT'].columns[:-5]):
            self.subintinfo[column.name] = (column.format,column.unit,hdulist['SUBINT'].data[column.name])
        self.subintheader = dict()
        self.subintheaderlist = hdulist['SUBINT'].header.keys()#for ordering
        for i,key in enumerate(hdulist['SUBINT'].header):
            self.subintheader[key] = hdulist['SUBINT'].header[key]

        DATA = hdulist['SUBINT'].data['DATA']
        #if np.ndim(DATA)==5:
        #    DATA = DATA[:,0,:,:,:] #remove the nsblk column
        DATA = np.ascontiguousarray(DATA)

        #Definitions in Base/Formats/PSRFITS/ProfileColumn.C
        DAT_FREQ = hdulist['SUBINT'].data['DAT_FREQ']
        DAT_WTS = np.ascontiguousarray(hdulist['SUBINT'].data['DAT_WTS'])
        if not weight:
            DAT_WTS = np.ones(np.shape(DAT_WTS))
        DAT_SCL = np.ascontiguousarray(hdulist['SUBINT'].data['DAT_SCL'])
        DAT_OFFS = np.ascontiguousarray(hdulist['SUBINT'].data['DAT_OFFS'])# + 0.5 #testing
        self.DAT_SCL = DAT_SCL #testing



        #print DAT_WTS,np.max(DAT_WTS),np.min(DAT_WTS)
        if np.size(DAT_WTS) == 1:
            #DAT_WTS[0] = 1.0
            #DAT_WTS[0,0] = 1.0
            self.weights = np.ones((1,1))
        else:
            #DAT_WTS /= np.max(DAT_WTS) #close???
            #DAT_WTS *= 50
            self.weights = DAT_WTS


        if self.lowmem: #Replace the data arrays with memmaps to reduce memory load



            SHAPE = np.shape(DATA)
            tfDATA = tempfile.NamedTemporaryFile()
            fp = np.memmap(tfDATA.name,dtype=np.int16,mode='w+',shape=SHAPE)
            fp[:] = DATA[:]
            del fp
            DATA = np.memmap(tfDATA.name,dtype=np.int16,mode='r',shape=SHAPE)

            '''
            SHAPE = np.shape(DAT_WTS)
            tfDAT_WTS = tempfile.NamedTemporaryFile()
            fp = np.memmap(tfDAT_WTS.name,dtype=np.float32,mode='w+',shape=SHAPE)
            fp[:] = DAT_WTS[:]
            del fp
            DAT_WTS = np.memmap(tfDAT_WTS.name,dtype=np.float32,mode='r',shape=SHAPE)

            SHAPE = np.shape(DAT_SCL)
            tfDAT_SCL = tempfile.NamedTemporaryFile()
            fp = np.memmap(tfDAT_SCL.name,dtype=np.float32,mode='w+',shape=SHAPE)
            fp[:] = DAT_SCL[:]
            del fp
            DAT_SCL = np.memmap(tfDAT_SCL.name,dtype=np.float32,mode='r',shape=SHAPE)

            SHAPE = np.shape(DAT_OFFS)
            tfDAT_OFFS = tempfile.NamedTemporaryFile()
            fp = np.memmap(tfDAT_OFFS.name,dtype=np.float32,mode='w+',shape=SHAPE)
            fp[:] = DAT_OFFS[:]
            del fp
            DAT_OFFS = np.memmap(tfDAT_OFFS.name,dtype=np.float32,mode='r',shape=SHAPE)
            '''

            #tf = tempfile.NamedTemporaryFile()
            self.data = DATA#np.memmap(tf.name,dtype=np.float32,mode='w+',shape=(nsubint,nsblk,npol,nchan,nbin))

        else:
            self.data = DATA#np.zeros((nsubint,nsblk,npol,nchan,nbin))

        #self.data = np.zeros((nsubint,npol,nchan,nbin))
        #data = np.zeros((nsubint,npol,nchan,nbin))

        I = range(nsubint)
        J = range(npol)
        K = range(nchan)


        self.freq = DAT_FREQ

        bw = self.getBandwidth()


        # All time-tagging info
        self.durations = self.getSubintinfo('TSUBINT')
        self.subint_starts = np.array(fmap(Decimal,self.getSubintinfo('OFFS_SUB')),dtype=np.dtype(Decimal))#+self.getTbin(numwrap=Decimal)*Decimal(nbin/2.0)#+self.getMJD(full=False,numwrap=Decimal) #converts center-of-bin times to start-of-bin times, in seconds, does not include the integer MJD part. This means that a template sitting in the center will have zero extra time
        self.channel_delays = np.zeros(nchan,dtype=np.dtype(Decimal)) #used to keep track of frequency-dependent channel delays, in time units.

        hdulist.close()
        return


    def save(self,filename):
        """Save the file to a new FITS file"""

        primaryhdu = pyfits.PrimaryHDU(header=self.header) #need to make alterations to header
        hdulist = pyfits.HDUList(primaryhdu)

        if self.history is not None:
            cols = []
            for name in self.history.namelist:
                fmt,unit,array = self.history.dictionary[name]
                #print name,fmt,unit,array
                col = pyfits.Column(name=name,format=fmt,unit=unit,array=array)
                cols.append(col)
            historyhdr = pyfits.Header()
            for key in self.history.headerlist:
                historyhdr[key] = self.history.header[key]
            historyhdu = pyfits.BinTableHDU.from_columns(cols,name='HISTORY',header=historyhdr)
            hdulist.append(historyhdu)
            # Need to add in PyPulse changes into a HISTORY
        #else: #else start a HISTORY table


        if self.params is not None:
            cols = [pyfits.Column(name='PSRPARAM',format='128A',array=self.params.filename)] #PARAM and not PSRPARAM?
            paramhdr = pyfits.Header()
            for key in self.paramheaderlist:
                paramhdr[key] = self.paramheader[key]
            paramhdu = pyfits.BinTableHDU.from_columns(cols,name='PSRPARAM')
            hdulist.append(paramhdu)
            # Need to include mode for PSREPHEM




        if self.polyco is not None:
            cols = []
            for name in self.polyco.namelist:
                fmt,unit,array = self.polyco.dictionary[name]
                #print name,fmt,unit,array
                col = pyfits.Column(name=name,format=fmt,unit=unit,array=array)
                cols.append(col)
            polycohdr = pyfits.Header()
            for key in self.polyco.headerlist:
                polycohdr[key] = self.polyco.header[key]
            polycohdu = pyfits.BinTableHDU.from_columns(cols,name='POLYCO',header=polycohdr)
            hdulist.append(polycohdu)




        if len(self.tables) > 0:
            for table in self.tables:
                hdulist.append(table)

        cols = []
        for name in self.subintinfolist:
            fmt,unit,array = self.subintinfo[name]
            col = pyfits.Column(name=name,format=fmt,unit=unit,array=array)
            cols.append(col)
            # finish writing out SUBINT!

        cols.append(pyfits.Column(name='DAT_FREQ',format='%iE'%np.shape(self.freq)[1],unit='MHz',array=self.freq)) #correct size? check units?
        cols.append(pyfits.Column(name='DAT_WTS',format='%iE'%np.shape(self.weights)[1],array=self.weights)) #call getWeights()

        nsubint,nsblk,npol,nchan,nbin = self.data.shape

        DAT_OFFS = np.zeros((nsubint,npol*nchan),dtype=np.float32)
        DAT_SCL = np.zeros((nsubint,npol*nchan),dtype=np.float32)
        DATA = self.data #getData(squeeze=False,weight=False)
        saveDATA = self.data #np.zeros(self.shape(squeeze=False),dtype=np.int16)
        # Following Base/Formats/PSRFITS/unload_DigitiserCounts.C
        #for i in xrange(nsubint):
        #    for j in xrange(npol):
        #        jnchan = j*nchan
        #        for k in xrange(nchan):
        #            MIN = np.min(DATA[i,j,k,:])
        #            MAX = np.max(DATA[i,j,k,:])
        #            RANGE = MAX - MIN
        #            if MAX == 0 and MIN == 0:
        #                DAT_SCL[i,jnchan+k] = 1.0
        #            else:
        #                DAT_OFFS[i,jnchan+k] = 0.5*(MIN+MAX)
        #                DAT_SCL[i,jnchan+k] = (MAX-MIN)/32766.0 #this is slightly off the original value? Results in slight change of data
        #
        #            saveDATA[i,j,k,:] = np.floor((DATA[i,j,k,:] - DAT_OFFS[i,jnchan+k])/DAT_SCL[i,jnchan+k] + 0.5) #why +0.5?

        cols.append(pyfits.Column(name='DAT_OFFS',format='%iE'%np.size(DAT_OFFS[0]),array=DAT_OFFS))
        cols.append(pyfits.Column(name='DAT_SCL',format='%iE'%np.size(DAT_SCL[0]),array=DAT_SCL))
        cols.append(pyfits.Column(name='DATA',format='%iI'%np.size(saveDATA[0]),array=saveDATA,unit='Jy',dim='(%s,%s,%s,%s,%s)'%(nbin,nsblk,npol,nchan,nbin))) #replace the unit here

        subinthdr = pyfits.Header()
        for key in self.subintheaderlist:
            subinthdr[key] = self.subintheader[key]
        subinthdu = pyfits.BinTableHDU.from_columns(cols,name='SUBINT',header=subinthdr)
        hdulist.append(subinthdu)



        hdulist.writeto(filename,overwrite=True)#clobber=True?



    def unload(self,filename):
        return self.save(filename)

    def gc(self):
        """Manually clear the data cube for python garbage collection"""
        if self.verbose:
            t0=time.time()
        self.data = None
        self.data_orig = None
        self.weights = None
        self.weights_orig = None
        if self.verbose:
            t1=time.time()
            print("Unload time: %0.2f s" % (t1-t0))
        g.collect()


    def shape(self,squeeze=True):
        """Return the current shape of the data array"""
        return np.shape(self.getData(squeeze=squeeze))
    def reset(self,prepare=True):
        """Replace the arch with the original clone"""
        if self.lowmem:
            self.load(self.filename,prepare=prepare)
        else:
            self.data = np.copy(self.data_orig)
            self.weights = np.copy(self.weights_orig)
        self.durations = self.getSubintinfo('TSUBINT')
        #if prepare:
        #    self.scrunch()


    def setData(self,newdata):
        """Sets the data, very dangerous!"""
        self.record(inspect.currentframe())
        if np.shape(newdata) == np.shape(self.data):
            self.data = np.copy(newdata)
    def getWeights(self,squeeze=False):
        """ Return copy of weights array """
        weights = self.weights
        if squeeze:
            weights = weights.squeeze()
        return np.copy(weights)
    def setWeights(self,val,t=None,f=None):
        """
        Set weights to a certain value
        Can be used for RFI routines
        """
        self.record(inspect.currentframe())
        if t is None and f is None:
            self.weights[:,:] = val
        elif t is None:
            self.weights[:,f] = val
        elif f is None:
            self.weights[t,f] = val
        else:
            self.weights[t,f] = val



    def saveData(self,filename=None,ext='npy',ascii=False):
        """Save the data array to a different format"""
        if filename is None:
            filename = self.filename
            filename = ".".join(filename.split(".")[:-1])+"."+ext
        if self.verbose:
            print("Saving: %s" % filename)
        if ascii:
            shape = self.shape(squeeze=False)
            nsubint = self.getNsubint()
            npol = self.getNpol()
            nchan = self.getNchan()
            nbin = self.getNbin()
            output = ""
            if shape[0] == 1 and shape[1] == 1 and shape[2] == 1:
                np.savetxt(filename,self.getData())
                return
            elif ((shape[0] == 1 and shape[1] == 1) or
                  (shape[0] == 1 and shape[2] == 1) or
                  (shape[1] == 1 and shape[2] == 1)):
                np.savetxt(filename,self.getData())
                return
            for i in xrange(nsubint):
                for j in xrange(npol):
                    for k in xrange(nchan):
                        for l in xrange(nbin):
                            output += "%i %i %i %i %.18e\n" % (i,j,k,l,self.data[i,j,k,l])

            FILE = open(filename,'w')
            FILE.write(output)
            FILE.close()
        else:
            np.save(filename,self.getData())
        return



    def outputPulses(self,filename):
        """ Write out a .npy file"""
        np.save(filename,self.getData())
        return



    ### ==============================
    ### Simple get functions
    ### ==============================


    def getNsubint(self):
        """Returns number of subintegrations"""
        return self.shape(squeeze=False)[0]
    def getNsblk(self):
        """Returns number of NSBLK"""
        return self.data.shape[1]
    def getNpol(self):
        """Returns number of polarizations"""
        return self.shape(squeeze=False)[1]
    def getNchan(self):
        """Returns number of channels"""
        return self.shape(squeeze=False)[2]
    def getNbin(self):
        """Returns number of phase bins"""
        return self.shape(squeeze=False)[3]
    def getValue(self,value):
        """Looks for a key in one of the headers and returns"""
        if value in self.header.keys():
            return self.header[value]
        if value in self.subintinfo.keys():
            return self.subintinfo[value][-1]
        if self.params is None:
            return None
        return self.params.get(value) #will return None if non-existent
    def getSubintinfo(self,value):
        """Returns value from subintinfo"""
        if value in self.subintinfo.keys():
            return self.subintinfo[value][-1]
        return None
    def getName(self):
        """Returns pulsar name"""
        return self.header['SRC_NAME']
    def getMJD(self,full=False,numwrap=float):
        """Returns MJD of observation"""
        if full:
            return numwrap(self.header['STT_IMJD'])+(numwrap(self.header['STT_SMJD'])+numwrap(self.header['STT_OFFS']))/numwrap(86400)
        return numwrap(self.header['STT_IMJD'])+numwrap(self.header['STT_OFFS'])
    def getTbin(self,numwrap=float):
        """Returns the time per bin"""
        return numwrap(self.getPeriod()) / numwrap(self.getNbin())
    def getDM(self):
        """Returns the data header DM"""
        return self.subintheader['DM']
        #if self.params is None:
        #    return
        #return self.params.getDM()
    def getRM(self):
        """Returns the data header RM"""
        return self.subintheader['RM']
    def getCoords(self,string=False,parse=False):
        """Returns the coordinate info in the header"""
        if string:
            RA = self.header['RA']
            dec = self.header['DEC']
            return RA,dec
        elif parse:
            RA = tuple(map(float,self.header['RA'].split(":")))
            dec = tuple(map(float,self.header['DEC'].split(":")))
            return RA,dec
        return coordinates.SkyCoord("%s %s"%(self.header['RA'],self.header['DEC']),unit=(units.hourangle,units.degree))
    getPulsarCoords = getCoords

    def getTelescopeCoords(self):
        """Returns the telescope coordinates"""
        return self.header['ANT_X'],self.header['ANT_Y'],self.header['ANT_Z']
    def getBandwidth(self,header=False):
        """Returns the observation bandwidth"""
        if header:
            return self.header['OBSBW']
        else:
            return self.subintheader['CHAN_BW']*self.subintheader['NCHAN']
    def getDuration(self):
        """Returns the observation duratrion"""
        #return np.sum(self.subintinfo['TSUBINT']) #This is constant.
        return np.sum(self.getSubintinfo('TSUBINT')) #This is constant.
    def getDurations(self):
        """Returns the subintegration durations"""
        return self.durations
    def getCenterFrequency(self,weighted=False):
        """Returns the center frequency"""
        if weighted:
            return np.sum(self.freq*self.weights)/np.sum(self.weights)
        if "HISTORY" in self.keys:
            return self.history.getLatest('CTR_FREQ')
        else:
            return self.header['OBSFREQ'] #perhaps do an unweighted version from DAT_FREQ?
    def getTelescope(self):
        """Returns the telescope name"""
        return self.header['TELESCOP']
    def getFrontend(self):
        """Returns the frontend name"""
        return self.header['FRONTEND']
    def getBackend(self):
        """Returns the backend name"""
        return self.header['BACKEND']
    def getSN(self):
        """Returns the average pulse S/N"""
        return self.spavg.getSN()

    def isCalibrator(self):
        if self.header['OBS_MODE'] == CAL or self.header['OBS_MODE'] == FON or self.header['OBS_MODE'] == FOF:
            return True
        return False


    #def varargtest(self,*args):
    #    self.record(inspect.currentframe())
    def record(self,frame):
        args, varargs, keywords, values = inspect.getargvalues(frame)
        funcname = frame.f_code.co_name
        string = "%s("%funcname
        for arg in args[1:]:
            if type(values[arg]) == str:
                string += "%s=\"%s\","%(arg,values[arg])
            else:
                string += "%s=%s,"%(arg,values[arg])
        if varargs is not None:  # Var args typically not implemented in PyPulse
            argdict = values[varargs]
            string += "%s,"%(str(argdict)[1:-1].replace(", ",","))
        if keywords is not None:
            kwargdict = values[keywords]
            for kwarg in kwargdict:
                string += "%s=%s,"%(kwarg,kwargdict[kwarg])
        if string[-1] == "(":
            string += ")"
        else:
            string = string[:-1] + ")"
        self.pypulse_history.append(string)

    def print_pypulse_history(self):
        for elem in self.pypulse_history:
            print(elem)






# Takes hdulist['HISTORY']
class History:
    def __init__(self,history):
        """Intializer"""
        self.header = dict()
        self.headerlist = history.header.keys()
        for key in self.headerlist:
            self.header[key] = history.header[key]
        self.dictionary = dict()
        self.namelist = list()
        for col in history.columns:
            self.namelist.append(col.name)
            self.dictionary[col.name] = (col.format,col.unit,list(col.array)) #make a np.array?
    def getValue(self,field,num=None):
        """Returns a dictionary array value for a given numeric entry"""
        if num is None:
            return self.dictionary[field][-1]
        else:
            try:
                return self.dictionary[field][-1][num]
            except IndexError:
                print("Entry out of range")
                return None
    def getLatest(self,field):
        """Returns the latest key value"""
        return self.getValue(field,-1)
    def printEntry(self,i):
        """Prints the i-th history entry"""
        for name in self.namelist:
            value = self.getValue(name,i)
            if value is None:
                return
            print(name,self.getValue(name,i))

# Takes hdulist['POLYCO']
# Similar to History class
class Polyco:
    def __init__(self,polyco,MJD=None):
        """Initializer"""
        self.MJD = MJD
        self.header = dict()
        self.headerlist = polyco.header.keys()
        for key in self.headerlist:
            self.header[key] = polyco.header[key]
        self.dictionary = dict()
        self.namelist = list()
        for col in polyco.columns:
            self.namelist.append(col.name)
            self.dictionary[col.name] = (col.format,col.unit,list(col.array)) #make a np.array?
    def getValue(self,field,num=None):
        """Returns a dictionary array value for a given numeric entry"""
        if num is None:
            return self.dictionary[field][-1]
        else:
            return self.dictionary[field][-1][num]
    def getLatest(self,field):
        """Returns the latest key value"""
        return self.getValue(field,-1)
    def calculate(self,MJD=None):
        if self.MJD is None and MJD is None:
            pass
        elif MJD is None:
            MJD = self.MJD

        #NSITE = self.getValue('NSITE',num=0)
        REF_FREQ = self.getValue('REF_FREQ',num=0)
        #PRED_PHS = self.getValue('PRED_PHS',num=0)
        REF_MJD = self.getValue('REF_MJD',num=0)
        REF_PHS = self.getValue('REF_PHS',num=0)
        REF_F0 = self.getValue('REF_F0',num=0)
        COEFF = self.getValue('COEFF',num=0)
        #print "POLYCO",REF_FREQ,REF_MJD,REF_PHS,REF_F0,COEFF
        #http://tempo.sourceforge.net/ref_man_sections/tz-polyco.txt
        DT = (MJD-REF_MJD)*1440.0
        #print "DT",DT,MJD,REF_MJD
        PHASE = REF_PHS + DT*60*REF_F0
        #print "PHASE",PHASE
        FREQ = 0.0
        for i,c in enumerate(COEFF):
            PHASE += c*np.power(DT,i)
            if i==0:
                continue
            FREQ += c*i*np.power(DT,i-1)
        FREQ = REF_F0 + FREQ/60.0
        return PHASE,FREQ
    def calculatePeriod(self,MJD=None):
        PHASE,FREQ = self.calculate(MJD=MJD)
        return 1.0/FREQ
