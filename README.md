### Manifold Learning Repository

My primary research for my masters thesis is investigating Manifold Alignment of different datasets. This repository holds my manifold alignment schemes such as Laplacian Eigenmaps, Schroedinger Eigenmaps and Locality Preserving Projections. I started off using MATLAB but slowly migrated to Python because of the freedom aspect. That being said, I went through and reconstructed all of these algorithms and created meaningful classes and functions to make my research easier. Some code is pretty decent (meaning it works well) so I am about ready to share it with the world. There are some packages that exist in the Python world (first and foremost is Scikit-Learn) but unfortunately there were a few extra things I needed.

1. I needed some of the codes at a larger scale. This includes the K-Nearest Neighbor functions and the eigenvalue decomposition functions.
2. I wanted a bit more freedom to control some aspects. I've really been trying to test out different methodologies for constructing the necessary graphs so my methods have quite a few parameters in them.


---

TODO:
* Finish this readme with some more coherent background.
* Include references
* Upload Code
* List necessary python packages


---

##### Packages Used

* sklearn
* numpy
* scipy
* pandas
* annoy
* pyamg
