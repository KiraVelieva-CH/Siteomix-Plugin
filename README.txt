# Siteomix-Plugin

## Acknowledgements

The cavity detection module in this repository is derived from the CavitOmiX PyMOL plugin, developed by Innophore GmbH, 2022. 

Original source: https://innophore.com/software/cavitomix/

We acknowledge and thank the authors (Georg Steinkellner, Christian C. Gruber, Karl Gruber, and the Innophore Team) for making their code available. See the `assets/about.txt` file for the original license notice and citation request.

###

Siteomix is a PyMOL plugin designed to calculate binding sites in the form of point clouds and subsequently identify the structural similarity of binding sites by aligning the clouds and calculating the total overlap.
If you use our plugin in your research, please cite us: https://doi.org/10.1007/s10822-026-00913-3
The "Examples of binding site studies using Siteomix - KRAS, p53, kinases, GPCRs" folder contains other examples of screening the molecules indicated in the titles. 
Attached are PDF files with the molecules and calculated binding sites, screenshots of the plugin log after calculating the similarity coefficient, and a document with all the results.

#### License

The original source code of the Siteomix-Plugin (files created by the authors) is licensed under the GNU General Public License v3.0 (see the `LICENSE` file).

Please note that the following four files: `radii.py`, `pdb_structure.py`, `ligsite.py`, and `cavfind.py`, are part of the CavitOmiX PyMOL plugin (Innophore GmbH, 2022). These files are not covered by the GPLv3 license of this repository. They remain under the original permission notice provided in the `assets/about.txt` file, as only the copyright holder (Innophore GmbH) can place them under a different license.
