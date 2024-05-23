#!/usr/bin/env python
# coding: utf-8

# Fit One Spectrum and tune redshift
# ----------------------------------
# 
# First let's import the packages we will need.

# 
# Analyse outputs from GELATO fit results on a signe spectrum
# 
# - author : Sylvie Dagoret-Campagne
# - creation date : 2024-03-25
# - update : 2024-05-23
# 
# 
# - Kernel at CCIN2P3 : ``conda_desc_py310_pcigale``
# - Kernel on my laptop : ``pcigale``

# # Create dir
#   ``ResultsFitInNb/``

# In[1]:


import numpy as np

# Import packages
import gelato

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl

mpl.rcParams['font.size'] = 25
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from astropy import modeling

# For loading in data
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.table import Table
from matplotlib import pyplot  # For plotting
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# In[2]:


# define a model for a line
g_init = models.Gaussian1D(amplitude=1, mean=0, stddev=1)
# initialize a linear fitter
fit_g = fitting.LevMarLSQFitter()


# In[3]:


import gelato.ConstructParams as CP
import gelato.CustomModels as CM
import gelato.Plotting as P
import gelato.SpectrumClass as SC

# GELATO
import gelato.Utility as U
from gelato.Constants import C

#from gelato.Plotting import  Plot, PlotFig,subplotplot
#from gelato.Plotting import subplotplot
from gelato.Plotting import logbarrier
from scipy.optimize import minimize

from fors2pcigale.fors2starlightio import Fors2DataAcess

# In[4]:



#import gelato.ConstructParams as CP





# ## Test astropy gaussian fit

# In[5]:


m = modeling.models.Gaussian1D(amplitude=10, mean=30, stddev=5)
x = np.linspace(0, 100, 2000)
data = m(x)
data = data + np.sqrt(data) * np.random.random(x.size) - 0.5
data -= data.min()
plt.plot(x, data)


# In[6]:


fitter = modeling.fitting.LevMarLSQFitter()
model = modeling.models.Gaussian1D()   # depending on the data you need to give some initial values
fitted_model = fit_g(model, x, data)


# In[7]:


fitted_model.mean.value


# ### redefined functions of GELATO to View in notebook

# In[8]:


# Get pulls
def Getpulls(spectrum,args,f):
    
    # Unpack
    wav,flux,isig = args

        
    # Residual Axis
    
    pulls = (flux - f)*isig
    p_m = np.mean(pulls)
    p_s = np.std(pulls)

    return p_m,p_s


# In[9]:


# Add Linelabels
def Mysubplotplot(pt,fax,rax,hax,spectrum,args,f):
    
    # Unpack
    wav,flux,isig = args

    # Plot data
    fax.step(wav,flux,'gray',where='mid')

    # Plot model(s)
    # for p in parameters:
    #     fax.step(wav,model.evaluate(p,*args),'r',where='mid',alpha=0.5)
    fax.step(wav,f,'r',where='mid')
   
    
    # Base Y axis on flux
    ymin = np.max([0,flux.min()])
    dy = flux.max() - ymin
    ylim = [ymin,ymin+1.3*dy] # Increase Axis size by 20%
    text_height = ymin+1.2*dy 

    # Get Line Names/Positions
    linelocs = []
    linelabels = []
    for group in spectrum.p['EmissionGroups']:
        for species in group['Species']:
            if species['Flag'] >= 0:
                for line in species['Lines']:
                    x = line['Wavelength']*(1+spectrum.z)
                    if ((x < wav[-1]) and (x > wav[0])):
                        linelocs.append(x)
                        linelabels.append(species['Name'])

    # If we have lines to plot
    if len(linelocs) > 0:

        # Reorder line positions
        inds = np.argsort(linelocs)
        linelocs = np.array(linelocs)[inds]
        linelabels = np.array(linelabels)[inds]

        # Log barrier constraints
        norm = 60 if pt == 0 else 20 # Magic numbers for line spacing
        x0 = np.linspace(wav.min(),wav.max(),len(inds)+2)[1:-1] # Initial guess
        linelabellocs = minimize(logbarrier,x0,args=(wav,linelocs,norm),method='Nelder-Mead',options={'adaptive':True,'maxiter':len(inds)*750}).x

        # Plot names
        for lineloc,linelabel,linelabelloc in zip(linelocs,linelabels,linelabellocs):
            # Text
            fax.text(linelabelloc,text_height,linelabel,rotation=90,fontsize=16,fontweight="bold",color="b",ha='center',va='center')
            # Plot Lines
            #fax.plot([lineloc,lineloc,linelabelloc,linelabelloc],[ymin+dy*1.01,ymin+dy*1.055,ymin+dy*1.075,ymin+dy*1.12],ls='-',c='gray',lw=0.25)
            fax.plot([lineloc,lineloc,linelabelloc,linelabelloc],[ymin+dy*1.01,ymin+dy*1.055,ymin+dy*1.075,ymin+dy*1.12],ls='-',c='blue',lw=1.0)

    # Axis labels and limits
    fax.set(ylabel=r'$F_\lambda$ ['+spectrum.p['FlamUnits']+']',ylim=ylim)
    fax.set(yticks=[t for t in fax.get_yticks() if (t > ymin+0.05*dy) and (t < ylim[-1])],xlim=[wav.min(),wav.max()],xticklabels=[])
    fax.grid(color="g")
        
    # Residual Axis
    
    pulls = (flux - f)*isig
    p_m = np.mean(pulls)
    p_s = np.std(pulls)
    p_l = f"pulls = {p_m:.1f} +/- {p_s:.1f}"
    
    
    histarray,bin_edges = np.histogram(pulls, bins=2000, range=(-10,10), density=True)
    g = fit_g(g_init,bin_edges[1:],histarray)
    m_f = g.mean.value        
    s_f = g.stddev.value  
    fit_label = f"normed residuals gaussian fit : {m_f:.2f} +/- {s_f:.2f}"
    textstr = '\n'.join((
    r'normed residuals gaussian fit : ',
    r'$\mu=%.2f$' % (m_f, ),
    r'$\sigma=%.2f$' % (s_f, )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)


    
    rax.step(wav,pulls,'gray',where='mid')
    ymax = np.max(np.abs(rax.get_ylim()))
    rax.set(xlim=[wav.min(),wav.max()],xlabel=r'Observed Wavelength [\AA]',ylim=[-ymax,ymax])
    rax.set_ylabel('Deviation',fontsize=15)
    rax.grid(color="g")
    rax.text(0.01, 0.85,fit_label , transform=rax.transAxes, fontsize=12,verticalalignment='top', bbox=props)
    
    if pt == 0:
        inset_rax = inset_axes(rax,
            width="20%", # width = 30% of parent_bbox
            height="80%", # height : 1 inch
            loc=4)
        inset_rax.hist(pulls,bins=30,range=(-5,5),density=True, histtype='step', facecolor='g',lw=2,color="g",label=p_l)
        inset_rax.set_xlabel("pulls")
        inset_rax.legend(loc="upper right",prop={'size': 16})
        
    
    elif pt>0 and hax!= None:
        pulls = (flux - f)*isig
        p_m = np.mean(pulls)
        p_s = np.std(pulls)
        p_l = f"pulls = {p_m:.1f} +/- {p_s:.1f}"
        hax.hist(pulls,bins=30,range=(-5,5),density=True, histtype='step', facecolor='g',lw=2,color="g",label=p_l)
        hax.set_xlabel("pulls")
        hax.legend(loc="upper right",prop={'size': 16})
    return ymin


# In[10]:


# Overwrite functions from GELATO to view results in a notebook

from matplotlib import pyplot, rcParams


def MyPlotFig(spectrum,model,parameters,fpath,plottype=0):

    # Calculate Medians
    medians = np.median(parameters,0)

    # Make figure name
    #figname = U.fileName(path.split(fpath)[-1])+'-'  OLD
    figname= U.fileName(fullfilename.split("/")[-1].split(".")[0])+'-'  #SDC
    
    if plottype == 0:
        figname += 'spec'
    elif plottype == 1:
        figname += 'fit'
    elif plottype == 2:
        figname += 'comp'

  
    # Get transform for secondary axis
    transform = (lambda obs: obs / (1 + spectrum.z), lambda rest: rest * (1 + spectrum.z))

    # Define the Dicticionnary of pulls to pass to the calling function
    all_pulls = {}    
    
    if plottype == 0:
        
        all_pull_regions = {}
        
        
        # Make figure
        fig = pyplot.figure(figsize=(15,10))
        gs = fig.add_gridspec(ncols=1,nrows=2,height_ratios=[4,1],hspace=0)

        # Get Spectrum
        wav     = spectrum.wav
        flux    = spectrum.flux
        isig    = spectrum.isig

        # Model prediction
        args = wav,flux,isig
        f = model.evaluate(medians,*args)

        # Add axes flux axis and residuals axis
        fax,rax = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0])

        # Plot Power Law
        if 'PowerLaw_Index' in model.get_names():
            continuum = CM.CompoundModel(model.models[1:2]).evaluate(medians[model.models[0].nparams:],*args)
            fax.step(wav,continuum,'k',ls='--',where='mid')

        # Plot model(s)
        #for p in parameters:
        #     fax.step(wav,model.evaluate(p,*args),'r',where='mid',alpha=0.5)
        fax.step(wav,f,'r',where='mid')

        # Subplot plotting
        hax=None
        Mysubplotplot(plottype,fax,rax,hax,spectrum,args,f)
        p_m,p_s=Getpulls(spectrum,args,f)
        #print(">>>> pulls calculation , plottype = ",plottype,"pulls",p_m,p_s)
        
        # index 0 for the full region
        all_pull_regions[0] = (p_m,p_s)
        all_pulls[plottype] = all_pull_regions

        # Add secondary axis
        rax.secondary_xaxis('top', functions=transform).set(xticklabels=[])
        fax.secondary_xaxis('top', functions=transform).set_xlabel('Rest Wavelength [\AA]',labelpad=10)
        rax.axhline(0,color="b")
       
        
        
    elif plottype > 0:
        
        # group pulls in different regions
        all_pull_regions = {}

        # Make figure
        ncols   = len(spectrum.regions)
        fig = pyplot.figure(figsize = (5*ncols,10))
        gs = fig.add_gridspec(ncols=ncols,nrows=3,height_ratios=[3,1,2],hspace=0.2)

        # Continuum and Model
        args = spectrum.wav,spectrum.flux,spectrum.isig
        if 'PowerLaw_Index' in model.get_names():
            continuum = CM.CompoundModel(model.models[0:2]).evaluate(medians,*args)
        else: 
            continuum = CM.CompoundModel(model.models[0:1]).evaluate(medians,*args)
        f = model.evaluate(medians,*args)
        
        # Iterate over regions
       
        for i,region in enumerate(spectrum.regions):

            # Get Spectrum
            good    = np.logical_and(spectrum.wav < region[1],spectrum.wav > region[0])
            wav     = spectrum.wav[good]
            flux    = spectrum.flux[good]
            isig    = spectrum.isig[good]
            args    = wav,flux,isig

            # Add Axes
            fax,rax,hax = fig.add_subplot(gs[0,i]),fig.add_subplot(gs[1,i]),fig.add_subplot(gs[2,i])

            # Subplot plotting
            ymin = Mysubplotplot(plottype,fax,rax,hax,spectrum,args,f[good])
            p_m,p_s=Getpulls(spectrum,args,f[good])
            
            # local region are indexed from index 1
            all_pull_regions[i+1] = (p_m,p_s)       
            #print(">>>> pulls calculation , plottype = ",plottype,"region",i+1,"pulls",p_m,p_s)

            
            # Plot Continuum
            if plottype == 1:
                fax.step(wav,continuum[good],ls='-',c='k',where='mid')
            # Plot components
            elif plottype == 2:
                init = 1
                if 'PowerLaw_Index' in model.get_names():
                    init = 2
                for j in range(init,len(model.models)):
                    m = model.models[j]
                    idx = model.indices[j]
                    cm = CM.CompoundModel([m]).evaluate(medians[idx:idx+m.nparams],*(wav,flux,isig))
                    fax.step(wav,ymin+cm,'--',c='gray')
                    # for p in parameters:
                    #     cm = CM.CompoundModel([m]).evaluate(p[idx:idx+m.nparams],*(wav,flux,isig))
                    #     fax.step(wav,ymin+cm,'--',c='gray',alpha=0.5)
                    
            # add those pulls for that region
            all_pulls[plottype] = all_pull_regions

            # Add secondary axis
            rax.secondary_xaxis('top', functions=transform).set(xticklabels=[])
            fax.secondary_xaxis('top', functions=transform).set_xlabel('Rest Wavelength [\AA]',labelpad=10)
            rax.axhline(0,color="b")
            
    # Add title and save figure
    fig.suptitle(figname.replace('_','\_')+', $z='+str(np.round(spectrum.z,3))+'$',y=1.0)
    fig.tight_layout()
    #fig.savefig(path.join(spectrum.p['OutFolder'],figname+'.pdf'))
    plt.show()
    #pyplot.close(fig)
    return  all_pulls


# In[11]:


# Plot all Figures
def MyPlot(spectrum,model,parameters,fpath):
    # loop on plot types
    for i in range(3): 
        all_pulls = MyPlotFig(spectrum,model,parameters,fpath,plottype=i)
        #print("MyPlot::",i,all_pulls)
        if i==0:
            the_dict = all_pulls
        else:
            the_dict.update(all_pulls)
    #print("MyPlot::the_dict ==> ",the_dict)
    return the_dict


# In[12]:


# Plot from results
def myplotfromresults(params,fpath,z):

#    if params["Verbose"]:
#        print("Presenting GELATO:",path.split(fpath)[-1])

    ## Load in Spectrum ##
  
    spectrum = SC.Spectrum(fpath,z,params)
    
    #------------------------
    #print("spectrum",spectrum)
    #--------------------------
  
    # Get just the final bit of the path
    #fpath = path.split(fpath)[-1] #SDC remove this split
    #print(fpath)
    
    ## Load Results ##
    #fname = path.join(params['OutFolder'],U.fileName(fpath))+'-results.fits'
    fname = fpath
    parameters = fits.getdata(fname,'PARAMS')
    pnames =  [n for n in parameters.columns.names if not (('EW' in n) or ('RAmp' in n) or ('PowerLaw_Scale' == n))][:-1]
    ps = np.array([parameters[n] for n in pnames]).T
    
    #-------------
    # print(pnames)
    # print("ps",ps)
    #-------------

    ## Create model ##
    # Add continuum
    ssp_names = [n[4:] for n in pnames if (('SSP_' in n) and (n != 'SSP_Redshift'))]
    
    #----------------------------
    #print("(ssp_names",ssp_names)
    #----------------------------
    
    
    models = [CM.SSPContinuumFree(spectrum,ssp_names = ssp_names)]
    if 'PowerLaw_Index' in pnames:
        models.append(CM.PowerLawContinuum(spectrum))
        models[-1].starting()
    
    #-----------------------------------------
    #print("spectrum.regions",spectrum.regions)
    #-----------------------------------------

    #if spectrum.regions != []:
    if len(spectrum.regions) != 0:

        # Add spectral lines
        ind = sum([m.nparams for m in models]) # index where emission lines begin
        for i in range(ind,ps.shape[1],3):
            center = float(pnames[i].split('_')[-2])
            models.append(CM.SpectralFeature(center,spectrum))

        # Final model
        model = CM.CompoundModel(models)
        #print("final model",model)

        # Plot
        the_pulls = MyPlot(spectrum,model,ps,fpath)
        #print("myplotfromresults::MyPlot:: the_pulls ===>", the_pulls)

    else:

        # Final Model
        model = CM.CompoundModel(models)
       

        # Plot
        the_pulls = MyPlotFig(spectrum,model,ps,fpath)
        #print("myplotfromresults:: MyPlotFig:: the_pulls ===>", the_pulls)

    if params["Verbose"]:
        print("GELATO presented:",fpath)
        
    return the_pulls


# In[13]:


def MySimplePlotSpectrum(fullfilename ,redshift,title):
    """
    MySimplePlotSpectrum : plot spectrum as it is , witout model
    """
    # Let's load the spectrum
    #path_spec = 'Spectra/spec-0280-51612-0117.fits'
    path_spec = fullfilename 
    spectrum = Table.read(path_spec)

    # Start with inverse variance
    ivar = spectrum['ivar']
    good = ivar > 0 # GELATO only looks at points with nonzero weights

    # Finally, let's load in the data
    wavl = 10**spectrum['loglam'][good]
    flux = spectrum['flux'][good]
    ivar = ivar[good]
    args = (wavl,flux,ivar) # These will be useful later
        
    # Create figure
    fig, ax = pyplot.subplots(figsize=(15,7))

    # Plot Spectrum
    sig = 3/np.sqrt(ivar) # 3 Sigma boundary
    ax.fill_between(wavl,flux-sig,flux+sig,color='gray')
    ax.step(wavl,flux,where='mid',c='k',lw=0.5)

    # Axis limits
    ax.set(xlim=[wavl.min(),wavl.max()],ylim=[0,flux.max()])

    # Axis labels
    ax.set(xlabel=r'Obs. Wavelength [\AA]',ylabel=r'$F_\lambda$')

    ax.set_title(title)
    # Show figure
    pyplot.show()
    


# In[14]:


def MySimplePlotSpectrumWithFittedModel(fullfilename ,redshift,title):
    """
    My simple plot spectrum with fitted model
    """
    # Let's load the spectrum
    #path_spec = 'Spectra/spec-0280-51612-0117.fits'
    path_spec = fullfilename 
    spectrum = Table.read(path_spec)
    
    #load fit results    
    results = fits.open(path_spec) 
    summary = Table(results['SUMMARY'].data)


    # Start with inverse variance
    ivar = spectrum['ivar']
    good = ivar > 0 # GELATO only looks at points with nonzero weights

    # Finally, let's load in the data
    wavl = 10**spectrum['loglam'][good]
    flux = spectrum['flux'][good]
    ivar = ivar[good]
    args = (wavl,flux,ivar) # These will be useful later
        
    # Plot Spectrum
    sig = 3/np.sqrt(ivar) # 3 Sigma boundary
    
        
    # Create figure
    fig, ax = pyplot.subplots(figsize=(15,7))

    
    # Plot Spectrum
    ax.fill_between(wavl,flux-sig,flux+sig,color='gray',alpha=0.2)
    ax.step(wavl,flux,where='mid',c='k',lw=0.75,label='Data')
    ax.step(10**summary['loglam'],summary['MODEL'],where='mid',c='r',label='Total Model')
    ax.step(10**summary['loglam'],summary['SSP'],where='mid',c='g',label='SSP Cont.')
    #ax.step(10**summary['loglam'],summary['PL'],where='mid',c='b',label='Power-Law Cont.')
    ax.step(10**summary['loglam'],summary['LINE'],where='mid',c='orange',lw=3,label='Emission Lines')
    ax.legend(loc="upper left")

    # Axis limits
    ax.set(xlim=[wavl.min(),wavl.max()],ylim=[0,flux.max()*1.5])

    # Axis labels
    ax.set(xlabel=r'Obs. Wavelength [\AA]',ylabel=r'$F_\lambda$')

    ax.set_title(title)
    # Show figure
    pyplot.show()
    


# In[15]:


def build_pulls_table(spectrum,the_pulls):
    """
    arguments:
     gelato spectrum
     pulls from fits dictionnary
    return
     table of pulls as pandas dataframe
    """
    
    df = pd.DataFrame(columns=['regionnum','wlmin','wlmax','pulls_mean','pulls_sigma'])
    
    wl = spectrum.wav
    
    
    if len(the_pulls.keys())>0:
        the_key_plottype = 0
        if the_key_plottype in the_pulls.keys():
            the_pulls_region = the_pulls[the_key_plottype]  
            the_key_region = 0
            df.loc[the_key_region,'wlmin'] = wl[0]
            df.loc[the_key_region,'wlmax'] = wl[-1]
            df.loc[the_key_region,'regionnum'] = the_key_region
            df.loc[the_key_region,'pulls_mean'] = the_pulls_region[the_key_region][0]
            df.loc[the_key_region,'pulls_sigma'] = the_pulls_region[the_key_region][1]
            
        the_key_plottype = 1
        if the_key_plottype in the_pulls.keys():
            the_pulls_region = the_pulls[the_key_plottype]  
            for the_key_region in the_pulls_region.keys():
                index_region_inspectrum  = the_key_region-1
                df.loc[the_key_region,'wlmin'] = spectrum.regions[index_region_inspectrum][0]
                df.loc[the_key_region,'wlmax'] = spectrum.regions[index_region_inspectrum][1]
                df.loc[the_key_region,'regionnum'] = the_key_region
                df.loc[the_key_region,'pulls_mean'] = the_pulls_region[the_key_region][0]
                df.loc[the_key_region,'pulls_sigma'] = the_pulls_region[the_key_region][1]   
                
    return df
 


# In[16]:


def DecodeParamsFitEmissionLines(t):
    """
    Decode FitParams Emission Lines
    
    input:
      Astropy result of FitResults
    output
      Emission lines fitted  
    """
    
    
    df = pd.DataFrame(columns=['group','line','wl','flux','flux_err','SNR'])
    
    index=0
    for col in params_fit.colnames:
        col_split=col.split("_")
        group_tag = col_split[0]
        if (group_tag != "SSP") and ("Flux" in col):
            emission_line =  col_split[1]
            if group_tag != "Outflow":
                wavelength = float(col_split[2])
            else:
                wavelength = float(col_split[3])
            flux = params_fit[col].mean()
            flux_err = params_fit[col].std()
            snr = flux/flux_err
            df.loc[index] = [group_tag,emission_line,wavelength,flux,flux_err,snr]
            index+=1
       
    return df
        


# ## Fors2 Interface

# In[17]:


fors2 = Fors2DataAcess()


# ## Gelato Parameters

# In[18]:


# Path to the parameters file
path_params = './ExampleParametersFitInNb.json'

# Create Parameters dictionary
params_gel = gelato.ConstructParams.construct(path_params)

# Set to not multiprocessing
params_gel['NProcess'] = 1


# In[19]:


params_gel['EmissionGroups']


# In[20]:


for group in params_gel['EmissionGroups']:
    info_group = "Group : "+ group["Name"]
    print(info_group)
    all_species = group['Species']
    for the_species in all_species:
        #info_species = "\t  Species : " + the_species['Name'] + " FlagGroup :: " +  the_species['FlagGroups'][0] + "Nlines = " + str(len( the_species['Lines'] )) 
        info_species = "\t  Species : " + the_species['Name']  + " , Nlines = " + str(len( the_species['Lines'] )) 
        print(info_species)


# ## Table with Spectra name and Redshifts

# In[21]:


df = pd.read_csv("object_filelist_v0.csv",index_col=0)


# ## Input files before the fit, sorting and index

# In[22]:


path = "./spec_forgelato/v0"


# In[23]:


list_all_files = os.listdir(path)


# In[24]:


idx_selected_files = []
list_selected_files = []
for file in list_all_files:
    res = re.findall("^specgelato_SPEC.*[.]fits$",file)
    if len(res):
        list_selected_files.append(file)
        num = int(re.findall("specgelato_SPEC(.*)[.]fits$",file)[0])   
        idx_selected_files.append(num)


# In[25]:


idx_selected_files = np.array(idx_selected_files)
list_selected_files = np.array(list_selected_files)
idx_sorted_files = np.argsort(idx_selected_files)
list_sorted_files = list_selected_files[idx_sorted_files]


# In[26]:


NSPEC = len(list_sorted_files)


# ## Choose One file

# In[27]:


index = 4
shortfilename = list_sorted_files[index]
fullfilename = os.path.join(path,shortfilename) 
path_spec = fullfilename
tag_spec = re.findall(".*_(SPEC.*).fits$", shortfilename)


# In[28]:


shortfilename.split('.')[0]


# In[29]:


tag_spec


# ### Define output filenames

# #### Fit results

# In[30]:


output_filename = shortfilename.split('.')[0] + "-results.fits"


# In[31]:


output_filename


# #### pulls and emission-line results

# In[32]:


output_filename_pulls = shortfilename.split('.')[0] + "-pulls-results.csv"
output_filename_emissionlines = shortfilename.split('.')[0] + "-emissionlines-results.csv"


# #### Find the redshift

# In[33]:


if len(tag_spec)>0:
    tag_spec = tag_spec[0]
    all_inputspecfilenames = df.Path.values
    for idx_tag,filen in enumerate(all_inputspecfilenames):
        if tag_spec in filen:
                break

    df_row = df.iloc[idx_tag]
    redshift = df_row["z"]


# In[34]:


spec_name_sel = tag_spec


# In[35]:


spec_name_sel


# In[36]:


# acess to the image array
img = fors2.get_specimg(spec_name_sel)

# get the image filename and path
spec_sec_fileimg = fors2.get_specimgfile(spec_name_sel)


# In[37]:


fig = plt.figure(constrained_layout=True,figsize=(12,6))
plt.imshow(img)
ax = plt.gca()
# Hide X and Y axes label marks
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])
plt.show()


# In[38]:


print(df_row,redshift)


# In[39]:


title = f"{index}) {output_filename}, z={redshift:.3f}" 


# #### Get the spectrum

# In[40]:


spectrum = Table.read(path_spec)

# Start with inverse variance
ivar = spectrum['ivar']
good = ivar > 0 # GELATO only looks at points with nonzero weights

# Finally, let's load in the data
wavl = 10**spectrum['loglam'][good]
flux = spectrum['flux'][good]
ivar = ivar[good]
args = (wavl,flux,ivar) # These will be useful later


# In[41]:


spectrum[:5]


# Let's go ahead and plot our spectrum to get an idea of what we're dealing with.

# In[42]:


# Create figure
fig, ax = pyplot.subplots(figsize=(15,7))

# Plot Spectrum
sig = 3/np.sqrt(ivar) # 3 Sigma boundary
ax.fill_between(wavl,flux-sig,flux+sig,color='gray')
ax.step(wavl,flux,where='mid',c='k',lw=0.5)

# Axis limits
ax.set(xlim=[wavl.min(),wavl.max()],ylim=[0,flux.max()])

# Axis labels
ax.set(xlabel=r'Obs. Wavelength [\AA]',ylabel=r'$F_\lambda$')
ax.set_title(title)
# Show figure
pyplot.show()


# The main gelato function takes three inputs.
# * The path to the parameters file or the parameters dictionary.
# * The path to the spectrum.
# * The redshift of the spectrum.
# 
# We already have the last two, and we need to take a little precaution with the first.
# The main gelato function will only return the final model if the code is being run without multiprocessing (as the return statement can break Python multiprocessing). So we can either change the Parameters JSON file, or edit the parameters dictionary. 

# ## Output for results

# In[43]:


output_path = params_gel['OutFolder']


# In[44]:


print(f"output_path defined in json file : {output_path}")


# In[45]:


output_path_fullfilename = os.path.join(output_path,output_filename)
output_path_fullfilename_pulls = os.path.join(output_path,output_filename_pulls)
output_path_fullfilename_emissionlines = os.path.join(output_path,output_filename_emissionlines)


# ## Run Gelato Fit

# We are now ready to run GELATO. Note, before you do this, ensure the results directory exists, either by running the Example from the README file or creating it. It will return the final callable model, however it won't be used in this notebook. 

# In[46]:


model = gelato.gelato(params_gel,path_spec,redshift)


# ## Results of Gelato Fit

# The results have been saved to the "Results/" Directory. Let's go ahead and load them in. We will print all extensions on the folder.

# ### 1) results

# In[47]:


# Load in results
results = fits.open(output_path_fullfilename)

# Print FITS extensions
results.info()


# We have two FITS extensions, SUMMARY and PARAMS. They are described in more detail in the README File but let's play around with them directly. Let's go ahead and take a look inside the SUMMARY extension. As we can see, it is a binary FITS Table.

# ### Summary of fitted model

# In[48]:


summary = Table(results['SUMMARY'].data)
summary


# In this table, we have the original spectrum along with the various model components, we can go ahead and plot them.

# In[49]:


df = summary.to_pandas()


# In[50]:


# Create figure
fig, ax = pyplot.subplots(figsize=(15,3))

# Plot Spectrum
ax.step(10**summary['loglam'],summary['LINE'],where='mid',c='y',label='Emission Lines')
ax.legend()

# Axis limits
ax.set(xlim=[wavl.min(),wavl.max()],ylim=[0,flux.max()])
# Axis labels
ax.set(xlabel=r'Obs. Wavelength [\AA]',ylabel=r'$F_\lambda$')
ax.set_title(title)
ax.grid()
# Show figure
pyplot.show()


# In[51]:


# Create figure
fig, ax = pyplot.subplots(figsize=(15,7))

# Plot Spectrum
ax.fill_between(wavl,flux-sig,flux+sig,color='gray')
ax.step(wavl,flux,where='mid',c='k',lw=0.5,label='Data')
ax.step(10**summary['loglam'],summary['MODEL'],where='mid',c='r',label='Total Model')
ax.step(10**summary['loglam'],summary['SSP'],where='mid',c='g',label='SSP Cont.')
#ax.step(10**summary['loglam'],summary['PL'],where='mid',c='b',label='Power-Law Cont.')
ax.step(10**summary['loglam'],summary['LINE'],where='mid',c='y',label='Emission Lines')
ax.legend()

# Axis limits
ax.set(xlim=[wavl.min(),wavl.max()],ylim=[0,flux.max()])

# Axis labels
ax.set(xlabel=r'Obs. Wavelength [\AA]',ylabel=r'$F_\lambda$')
ax.set_title(title)
# Show figure
pyplot.show()


# Looks great! You can see an example of the GELATO generated plots in the results folder, but this will let you incorporate GELATO fits easily into your own work. Let's go ahead and take a look at the PARAMS extension. This is a much larger table! It's made up of the parameters from each bootstrap iteration. 

# ### A very simple view of fit results 

# In[52]:


MySimplePlotSpectrumWithFittedModel(output_path_fullfilename,redshift,title)


# ### 2) Access to the fitted spectrum object directly

# In[53]:


spectrum = SC.Spectrum(output_path_fullfilename,redshift,params_gel)


# In[54]:


spectrum.p["EmissionGroups"]


# In[55]:


len(spectrum.p["EmissionGroups"])


# ### Complete plot result

# In[56]:


the_pulls = myplotfromresults(params_gel,output_path_fullfilename, redshift)


# ### Pulls

# In[57]:


pulls_table = build_pulls_table(spectrum,the_pulls)
pulls_table 


# In[58]:


pulls_table.to_csv(output_path_fullfilename_pulls) 


# ## Fitted parameters

# In[59]:


# Open Parameters extension
params_fit = Table(results['PARAMS'].data)
#print(params_fit)


# ### Redshift

# In[60]:


params_fit["SSP_Redshift"].mean()/C


# In[61]:


params_fit["SSP_Redshift"].std()/C


# ### Reduced chi2

# In[62]:


params_fit["rChi2"].mean()


# ### Emission-lines

# In[63]:


emissionlines_table = DecodeParamsFitEmissionLines(params_fit)
emissionlines_table


# In[64]:


emissionlines_table.to_csv(output_path_fullfilename_emissionlines)

