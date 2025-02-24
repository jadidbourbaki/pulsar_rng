# Random Number Generation using Pulsars

We get Pulsar Timing data from [Data Release 2](https://epta.pages.in2p3.fr/epta-dr2/) of the European Pulsar Timing Array (their [repository](https://zenodo.org/records/8164425)). This data is released under Creative Commons Attribution 4.0 International and is included in this repository as part of the `external/` directory.

## Setup

Dependencies: 
- pint-pulsar (python3 package)
- [ent](https://www.fourmilab.ch/random/)
- The NIST [statistical test suite](https://csrc.nist.gov/projects/random-bit-generation/documentation-and-software)

MacOS commands for dependencies:
    - `brew install ent`

We use a Python virtual environment:

Create the environment:
```
python3 -m venv .venv
```

Activate the environment:

```
source .venv/bin/activate
```

Install python dependencies:

```
pip install -r requirements.txt
```

Side note: run all code from within activated the virtual environment

Deactivate the environment:

```
deactivate
```

## Usage

```

pulsar.py -h
RuntimeWarning: This platform does not support extended precision floating-point, and PINT will run at reduced precision.
Detected 25 pulsars in EPTA dataset.
Detected 37 pulsars in NANOGrav dataset.
usage: pulsar.py [-h] [-l] [-i INDEX] [-n NAME] [-d {NANOGrav,EPTA}] [-q {threshold,gray_coding,sha512}]
                 [-dm {none,xor,von_neumann,shake256}] [-pc PLOT_COLOR] [-e] [-me] [-v] [-pn]
                 {plot,list,rng}

Generate random bits from pulsar timing residuals.

positional arguments:
  {plot,list,rng}       Command to execute

options:
  -h, --help            show this help message and exit
  -l, --list            List all pulsars in dataset
  -i, --index INDEX     Index of the pulsar.
  -n, --name NAME       Name of the Pulsar
  -d, --dataset {NANOGrav,EPTA}
                        Dataset to use
  -q, --quantifier {threshold,gray_coding,sha512}
                        Quantifier to use
  -dm, --debiasing-method {none,xor,von_neumann,shake256}
                        Debiasing method to use
  -pc, --plot-color PLOT_COLOR
                        Matplotlib plot color (for the plot command)
  -e, --ent             Run the ent tool at the end
  -me, --min-entropy    Calculate min entropy
  -v, --verbose         Enable verbose output
  -pn, --plot-normalized
                        Plot the normalized values

```