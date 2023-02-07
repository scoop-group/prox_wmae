This repository contains code corresponding to the paper 

  Baumgärtner, Herzog, Schmidt, Weiß
	The Proximal Map of the Weighted Mean Absolute Error
  arxiv: [2209.13545](https://arxiv.org/abs/2209.13545)

* The file `prox_wmae.py` contains a vectorized implementation of Algorithm 1 in the paper. 
* The file `demo_prox_wmae.py` demonstrates a simple call of this algorithm. 
* The file `demo_cameraman.py` generates the image denoising example of the noisy cameraman image, as presented in Section 4. 
* The file `demo_deflection.py` computes the deflection energy minimization, as presented in Section 5. 
The results can be rendered using `pvpython`, which is done in `paraview_render_results.py`. 

In order to resolve the dependencies, it is advised to run the code inside a Docker container, following the included `Dockerfile`: 

```bash
docker build -t prox_wmae_code .
docker run -ti -v $(pwd):/home/fenics/shared prox_wmae_code 'python3 generate_all.py'
```
