# Attribution Notice

This project is a standalone implementation and derivative work based on the architectural concepts of AEV-PLIG (developed by Ísak Valsson, OPIG). While the high-level logic and graph-generation protocols are derived from the original research, the codebase in this repository has been rewritten from the ground up by William Ogilvie to implement enhanced chemical resolution for radial AEVs, modernized training workflows and key architecutral changes in the data pipline, as well as numerous other minor changes listed below [Modifications](#modifications).

Ísak Valsson's paper is published in Nature's *Communications Chemistry* at [Narrowing the gap between machine learning scoring functions and free energy perturbation using augmented data](https://doi.org/10.1038/s42004-025-01428-y).

AEV-PLIG was first published in [How to make machine learning scoring functions competitive with FEP](https://chemrxiv.org/engage/chemrxiv/article-details/6675a38d5101a2ffa8274f62), and received the [people's poster prize at the 7th AI in Chemistry Symposium](https://www.stats.ox.ac.uk/news/isak-valsson-wins-poster-prize).

## Original Software: AEV-PLIG
* **Copyright:** (c) 2024, Ísak Valsson, Oxford Protein Informatics Group
* **License:** 3-Clause BSD 
* **Citation:** Valsson, Í., Warren, M.T., Deane, C.M. et al. Narrowing the gap between machine learning scoring functions and free energy perturbation using augmented data. Commun Chem 8, 41 (2025). https://doi.org/10.1038/s42004-025-01428-y

## Modifications
* **By:** William Ogilvie
* **Copyright:** (c) 2026, William Ogilvie, Oxford Protein Informatics Group
* **License:** 3-Clause BSD
* **Nature of changes:** torchani vs torchani_mod, changing to the bins of AEVs, added "other" for accepted elements on graph, AdamW with non zero weight decay instead of Adam, adding functionality to do both true early stopping but also to do early stopping on both MSE and Kendalls tau as well rather than just pearsons. Added scaling to AEVs to help with sparsity, AEV-PLIG appears to only scale targets pK. Alloed RDKit to process Du atoms as wildcards *. Added Dative, Hydrogen, Various, Ionic and Zero bonds to the allowed bonds list, in PDBbind for example 2foy has a Dative bond which gets dropped in AEV-PLIG, also added a catch all case similar to atom types. Note the refactoring of the allowable elements into the config allows for a much more expressive list than present in AEV-PLIG. Other architecture changes made to improve memory usage and code readability, not outlined here for brevity. 

## TorchANI

This project makes use of TorchANI among other third party libraries. Inside src/tagcn_bind there is a modified fork of TorchANI 2.4.0, originally created by Ísak Valsson for AEV-PLIG, with some minor modifications in this repository. 

TorchANI was developed and is currently maintained (as of 10/02/2026) by the Roitberg group, and is available for use under an MIT license. We include the original license inside the torchani_mod directory, as well as a copy inside this NOTICE.md file. 

Pickering, I., Xue, J., Huddleston, K., Terrel, N. & Roitberg, A. E. TorchANI 2.0: An Extensible, High-Performance Library for the Design, Training, and Use of NN-IPs. J. Chem. Inf. Model. 65, 11656–11671 (2025). https://doi.org/10.1021/acs.jcim.5c01853

Gao, X., Ramezanghorbani, F., Isayev, O., Smith, J. S. & Roitberg, A. E. TorchANI: A Free and Open Source PyTorch-Based Deep Learning Implementation of the ANI Neural Network Potentials. J. Chem. Inf. Model. 60, 3408–3415 (2020). https://doi.org/10.1021/acs.jcim.0c00451


## 3-Clause BSD of AEV PLIG

The 3-Clause BSD in AEV-PLIG is as follows:

3-Clause BSD License

	Copyright (c) Ísak Valsson, Oxford Protein Informatics Group, 2024. All rights reserved

	Redistribution and use in source and binary forms, with or without modification,
	are permitted provided that the following conditions are met:

	 - Redistributions of source code must retain the above copyright notice,
	   this list of conditions and the following disclaimer.

	 - Redistributions in binary form must reproduce the above copyright notice,
	   this list of conditions and the following disclaimer in the documentation
	   and/or other materials provided with the distribution.

	 - Neither the name of the copyright holder nor the names of its contributors may
	   be used to endorse or promote products derived from this software without
	   specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
	ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.Ísak Valsson

## MIT License of TorchANI

Copyright 2018- Xiang Gao and other ANI developers

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.