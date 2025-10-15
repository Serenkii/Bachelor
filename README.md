# My Bachelor's thesis

This repository contains all the Python code I produced in connection to my Bachelor's thesis.
If you use any of the source, or get inspiration from it, I would greatly appreciate to be referenced, 
as explained in [Cite](#cite).

I plan to upload my thesis here too, as soon as grading is finished.

## My thesis
My thesis is based on atomistic spin dynamics simulations of a simple toy model of an emerging class of magnetism, 
coined 'Altermagnetism'. I studied equilibrium properties and the spin Seebeck effect under the influence of an 
external magnetic field. Additionally, we provide simulational proof of the novel spin Nernst effect.
Lastly, we consider the influence of the relativistic Dzyaloshinskii-Moriya interaction, although our implementation 
breaks the altermagnetic symmetry.

### Abstract
Altermagnets (AMs) represent an emerging class of magnetic materials that combine
properties of ferromagnets (FMs) and antiferromagnets (AFMs), offering significant
potential for spintronics applications. This thesis investigates the fundamental characteristics
of spin dynamics in a simplified altermagnetic system through computer
simulations. The study employs a two-dimensional toy model on a checkerboard
lattice featuring anisotropic intrasublattice exchange interactions that fulfil the required
altermagnetic symmetry criteria. Spin dynamics are modelled using atomic
spin dynamics simulations based on the stochastic Landau-Lifshitz-Gilbert (LLG)
equation.
The work establishes a clear link between the anisotropic exchange interactions and
the direction-dependent splitting of the magnon dispersion relation. The introduction
of an external magnetic field along the easy axis is shown to shift the magnon
bands, alter magnon populations, and induce a net magnetization. This interplay is
used to explain the microscopic origins of the spin Seebeck effect (SSE), with analysis
extended to magnon propagation lengths and the calculation of longitudinal spin
currents. Furthermore, the study provides evidence for the magnonic spin Nernst effect
(SNE) by identifying a transverse spin accumulation at the material’s edges and
calculating a pure transverse spin current for specific crystallographic directions. Finally,
the influence of the relativistic Dzyaloshinskii-Moriya interaction (DMI) on the
system’s equilibrium state and the SSE is examined, with the caveat that the presented
implementation breaks the altermagnetic symmetry.
This research provides valuable microscopic insights into spin excitations and
thermal transport phenomena in altermagnets, contributing to the foundational understanding
of this novel magnetic phase.

## Data Analysis and Plotting
I will very briefly give an overview over this repository. As Bachelor's theses are written under a lot of time 
pressure, this Python project is not at all 'clean' or anything.

The file ```main.py``` contains a lot of quick-and-dirty code, which was used to quickly analyze different aspects and 
quickly visualize data for specific needs. Specific states of this research are structured with methods.

The folder ```src/``` contains a bunch of files. Many of them are used in connection with extracting data, or e.g. 
performing calculations on it. A lot of them are deprecated or were only used once. Some of them contain ideas, which 
were scrapped.
Important are the files ```src/mag_util.py``` and ```src/spinconf_util.py```, which were used to work with the data
gained from spatial profiles and complete configurations.

The file ```main_thesis.py``` is the main file, i.e. entry point of the analysis of data and generation of plots, which 
was then used in the final product: my thesis

The folder ```thesis/subsections/``` contains files responsible for the analysis and plotting for different subsections of my thesis.

The file ```thesis/theoretical_figures.py``` was used to generate plots to facilitate the understanding of concepts, I 
present in the Theory and Methods section of my thesis. 

## Cite

If you use any of this published code, or gain inspiration from it, I would appreciate you citing this repository, 
and especially my Bachelor's thesis. Cite this repository according to the [CITATION.cff](CITATION.cff). If you open 
this in the browser on [GitHub](https://github.com/Serenkii/Bachelor), you should be able to see a button to cite this 
repository.

You can use the following BibTeX-entry to cite my Bachelor's thesis:
```latex
@thesis{MarianDuelli
  author={Marian Duelli},
  title={Computer simulation of altermagnets with relativistic corrections},
  school={University of Konstanz},
  year={2025},
  type={Bachelor's thesis},
  address={Konstanz, Germany},
  month={10},
  url={}
}
```
Thank you.

# License
[GNU General Public License 3](https://www.gnu.org/licenses/gpl-3.0.html)

