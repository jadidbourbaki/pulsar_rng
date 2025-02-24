import numpy as np
import os, glob, json
import optparse

import enterprise
from enterprise.pulsar import Pulsar

from enterprise.signals.parameter import function
from enterprise.signals.deterministic_signals import Deterministic
from enterprise.signals import parameter, selections
from enterprise.signals.selections import Selection

from enterprise_extensions import models, sampler, hypermodel

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

parser = optparse.OptionParser()
parser.add_option('--datadir', action='store', dest='datadir', default='./', type='string')
parser.add_option('--outdir', action='store', dest='outdir', default='./report/gwb_run/', type='string')
parser.add_option('--noisedir', action='store', dest='noisedir', default='./noisefiles/', type='string')
parser.add_option('--orf1', action='store', dest='orf1', default='crn', type='string')
parser.add_option('--orf2', action='store', dest='orf2', default='crn', type='string')
parser.add_option('--common_psd', action='store', dest='common_psd', default='powerlaw', type='string')
parser.add_option('--common_components', action='store', dest='common_components', default=30, type='int')
parser.add_option('--red_components', action='store', dest='red_components', default=0, type='int')
parser.add_option('--dm_components', action='store', dest='dm_components', default=0, type='int')
parser.add_option('--chrom_components', action='store', dest='chrom_components', default=0, type='int')
parser.add_option('--num_dmdips', action='store', dest='num_dmdips', default=2, type='int')
parser.add_option('--bayesephem', action='store_true', dest='bayesephem', default=False)
parser.add_option('--common_sin1', action='store_true', dest='sin_wave1', default=False)
parser.add_option('--common_sin2', action='store_true', dest='sin_wave2', default=False)
parser.add_option('--lnweight1', action='store', dest='weight1', default=0, type='float')
parser.add_option('--lnweight2', action='store', dest='weight2', default=0, type='float')
parser.add_option('--resume', action='store_true', dest='resume', default=False)
parser.add_option('--emp', action='store', dest='emp', default=None, type='string')
parser.add_option('--number', action='store', dest='num', default=1e7, type='float')
parser.add_option('--thin', action='store', dest='thin', default=100, type='int')
(options,args) = parser.parse_args()

# load par and tim files
datadir = options.datadir
outdir = options.outdir
noisedir = options.noisedir

parfiles = sorted(glob.glob(datadir + '/J*/*.par'))
timfiles = sorted(glob.glob(datadir + '/J*/*_all.tim'))
noisefiles = sorted(glob.glob(noisedir + '/*.json'))

# filter to one set of par+tim+noisefile per pulsar
PsrList = np.loadtxt(datadir + 'scripts/psrlist.txt',dtype=str)

parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in PsrList]
timfiles = [x for x in timfiles if x.split('/')[-1].split('_')[0] in PsrList]
noisefiles = [x for x in noisefiles if x.split('/')[-1].split('_')[0] in PsrList]

psrs = []
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t, ephem='DE440')
    psrs.append(psr)
    
# set reference time for the sin wave to the earliest TOA in the data set
dataset_tmin = np.min([p.toas.min() for p in psrs])
dataset_tmax = np.max([p.toas.max() for p in psrs])

# sin wave model
@function
def sine_wave(toas, flags, A = -9, f = -9, phase = 0.0):
    return 10 ** A * np.sin(2 * np.pi * (10 ** f) * (toas - dataset_tmin) + phase)
    
def sine_signal(A, f, phase, selection=Selection(selections.no_selection), name = ""):
    return Deterministic(sine_wave(A = A, f = f, phase = phase), selection = selection, name = name)

m0 = sine_signal(A = parameter.Uniform(-9, -4)('common_sin_log10_A'),
                 f = parameter.Uniform(-9, -7.7)('common_sin_log10_f'),
                 phase = parameter.Uniform(0, 2 * np.pi)('common_sin_phase'), name='common_sin')
if options.sin_wave1:
    m1 = m0
else:
    m1 = None
if options.sin_wave2:
    m2 = m0
else:
    m2 = None

# load noise models and files
params = {}
for nf in noisefiles:
    with open(nf, 'r') as fin:
        params.update(json.load(fin))

if not options.red_components:
    try:
        red_dict = {}
        with open(noisedir + '/red_dict.json','r') as rd:
            red_dict.update(json.load(rd))
    except:
        raise UserWarning('Custom pulsar red noise frequency components not set.')
else:
    red_dict = options.red_components

if not options.dm_components:
    try:
        dm_dict = {}
        with open(noisedir + '/dm_dict.json','r') as dd:
            dm_dict.update(json.load(dd))
    except:
        raise UserWarning('Custom pulsar DM noise frequency components not set.')
else:
    dm_dict = options.dm_components

if not options.chrom_components:
    try:
        chrom_dict = {}
        with open(noisedir + '/chrom_dict.json','r') as cd:
            chrom_dict.update(json.load(cd))
    except:
        raise UserWarning('Custom pulsar scattering noise frequency components not set.')
else:
    chrom_dict = options.chrom_components

# setup model
pta = dict.fromkeys(np.arange(0, 2))

pta[0] = models.model_general(psrs, noisedict=params, orf=options.orf1,
                              common_psd=options.common_psd,
                              common_components=options.common_components,
                              bayesephem=options.bayesephem,
                              sat_orb_elements=True, tnequad=True,
                              tm_svd=True, tm_marg=True,
                              red_var=True, red_components=red_dict,
                              dm_var=True, dm_components=dm_dict,
                              dm_chrom=True, chrom_components=chrom_dict,
                              dmchrom_kernel='diag', tndm=True,
                              num_dmdips=options.num_dmdips,
                              dmpsr_list=['J1713+0747'], dm_expdip_idx=[1,4],
                              dm_expdip_tmin=[57490,54650],
                              dm_expdip_tmax=[57530,54850], extra_sigs=m1)
                              
pta[1] = models.model_general(psrs, noisedict=params, orf=options.orf2,
                              common_psd=options.common_psd,
                              common_components=options.common_components,
                              bayesephem=options.bayesephem,
                              sat_orb_elements=True, tnequad=True,
                              tm_svd=True, tm_marg=True,
                              red_var=True, red_components=red_dict,
                              dm_var=True, dm_components=dm_dict,
                              dm_chrom=True, chrom_components=chrom_dict,
                              dmchrom_kernel='diag', tndm=True,
                              num_dmdips=options.num_dmdips,
                              dmpsr_list=['J1713+0747'], dm_expdip_idx=[1,4],
                              dm_expdip_tmin=[57490,54650],
                              dm_expdip_tmax=[57530,54850], extra_sigs=m2)

# get initial sample and run sampler
super_model = hypermodel.HyperModel(pta,log_weights=[options.weight1,options.weight2])
sp = super_model.setup_sampler(resume=options.resume, outdir=outdir, empirical_distr=options.emp)
x0 = super_model.initial_sample()
sp.sample(x0, int(options.num), SCAMweight=30, AMweight=10, DEweight=50, thin=options.thin)
