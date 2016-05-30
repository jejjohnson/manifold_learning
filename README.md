### Manifold Learning Repository

My primary research for my masters thesis is investigating Manifold Alignment of different datasets (There will be a repo on this shortly.) This repository holds my manifold learning schemes such as Laplacian Eigenmaps, Schroedinger Eigenmaps and Locality Preserving Projections. I started off using MATLAB but slowly migrated to Python because of the freedom aspect. That being said, I went through and reconstructed all of these algorithms and created meaningful classes and functions to make my research easier. Some code is pretty decent (meaning it works well) so I am about ready to share it with the world. There are some packages that exist in the Python world (first and foremost is Scikit-Learn) but unfortunately there were a few extra things I needed such as:

1. I needed some of the codes at a larger scale. This includes the K-Nearest Neighbor functions and the eigenvalue decomposition functions.
2. I wanted a bit more freedom to control some aspects. I've really been trying to test out different methodologies for constructing the necessary graphs so my methods have quite a few parameters in them.


---

TODO:
* Finish this readme with some more coherent background.
* Include some plots in the README for motivation
* Include more references
* Upload Code Examples
* List necessary python packages


---

##### Python Packages Used (Incomplete)

* sklearn
* numpy
* scipy
* spectralpy
* pandas
* annoy
* pyamg

---

### Inspiration

The main reason was to open source my code just in case anyone wants to use it. I found it a bit painful when I went to other scientists webpage where they had assorted zip files (if they even had any code at all).

There were some packages that inspired me to do my own little digging:
* sckit-learn - [manifold learning][2] (Python)
* megaman: Scalable [manifold learning][3] (Python)
* Tapkee: an efficient [dimension reduction][4] library (C++)
* MATLAB Toolbox for [Dimensionality Reduction][5] (MATLAB)
* Spectral Python - module for processing [hyperspectral images][6] (Python)

Unfortunately, most of these toolboxes were either did not scale well to my large dataset, had bugs for special cases that I was dealing with, and didn't really allow for clean customization with my level of coding. So I decided to start from scratch to a certain extent. I created classes for each manifold learning method (Locality Preserving Projections, Schroedinger Eigenmaps) with separate classes for each large computation portion (e.g. k-nn search, eigenvalue decomposition,etc). I like the setup of meganman where they utilize a class to control the input variables and they have distinct classes for each manifold learning algorithm which inherit those properties. This is a good idea because my classes have too many variables so I would like to cut them down. I'll put that as a TODO.

---

### Relevant Literature

For papers concerning the theory, construction and use of the Schroedinger Eigenmaps or other relevant literature, please see this [Readme][1].

[1]: https://github.com/jejjohnson/manifold_learning/blob/master/relevant_papers/README.md
[2]: http://scikit-learn.org/stable/modules/manifold.html
[3]: http://mmp2.github.io/megaman/
[4]: http://tapkee.lisitsyn.me/
[5]: https://lvdmaaten.github.io/drtoolbox/
[6]: http://www.spectralpython.net/
