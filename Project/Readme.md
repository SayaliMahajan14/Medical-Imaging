# MRI Reconstruction using MoDL

This project reproduces the pretrained **MoDL (Model-Based Deep Learning)** model for MRI reconstruction and extends it by evaluating the effect of different measurement noise levels on reconstruction quality.

## Project Objective

- Reproduce the pretrained MoDL model.
- Test the model on the provided demo MRI image.
- Evaluate reconstruction quality under different noise levels.
- Compare reconstruction performance using PSNR.

The measurement noise level was varied:

- Noise = 0.00
- Noise = 0.01
- Noise = 0.03
- Noise = 0.05

The reconstruction quality was evaluated using PSNR.

## Files Used

- `tstDemo.py` – Original demo provided in the repository.
- `noise_experiment.py` – Extension for testing different noise levels.
- `supportingFunctions.py` – Utility functions for data loading and MRI reconstruction.
- `demoImage.hdf5` – Demo MRI dataset.
- `savedModels/` – Pretrained MoDL model.

## Results

As the measurement noise increased:

- Reconstruction quality decreased.
- PSNR values became lower.
- More reconstruction artifacts appeared.

This shows that higher measurement noise makes MRI reconstruction more challenging.

## Reference
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6760673/#S2
- Original MoDL Repository: https://github.com/hkaggarwal/modl
