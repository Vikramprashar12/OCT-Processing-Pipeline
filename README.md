# GPU-Accelerated OCT Signal Processing Pipeline

A real-time signal processing pipeline that converts raw Optical Coherence Tomography (OCT) data into diagnostic-ready retinal images, built to run on consumer GPU hardware instead of costly FPGA systems.

Capstone Design Project, COE70A/B, Department of Electrical, Computer and Biomedical Engineering, Toronto Metropolitan University (Fall 2024 - Winter 2025).

**Team:** Vikram Prashar, Avinandan Kundu, Gianmarco Arena, Roland Ogunleye

## Project Summary

Optical Coherence Tomography is a non-invasive imaging technique used in ophthalmology, cardiology, and oncology to visualize tissue structure at micrometer resolution. The catch is that OCT hardware captures raw interferometric data, not images. Turning that raw data into a usable scan requires a chain of signal processing steps (DC subtraction, interpolation, dispersion compensation, and a Fourier transform) run fast enough for real-time use. Most systems handle this with FPGAs, which are fast but expensive to develop for and hard to update.

This project builds that same processing chain on a GPU instead, using Python, CuPy, and CUDA. The goal was to find out whether commodity graphics hardware can hit real-time performance for OCT imaging without the cost and inflexibility of custom FPGA design.

**What the pipeline does:**
- Loads raw spectral-domain OCT volumes (built from the public FARSIU-2014 retinal OCT dataset)
- Runs DC subtraction to remove background noise
- Applies bicubic interpolation along the depth axis to normalize sampling
- Applies dispersion compensation to correct for optical phase distortion
- Runs a GPU-accelerated FFT to convert the data into a viewable image
- Renders the result through an interactive Matplotlib slice viewer

**Results:** on an NVIDIA RTX 4070 Laptop GPU, the pipeline processed a 512x1000x100 volumetric OCT dataset in under 50 milliseconds per file, roughly 10x faster than the equivalent CPU implementation (measured at 6.5-57 seconds per file depending on hardware). The team also benchmarked the pipeline on cloud GPUs (Kaggle, dual T4) and tested tradeoffs across thread count, chunk size, and memory limits to find a stable, scalable configuration.

## Why This Project Matters

Real-time OCT imaging has direct clinical value: faster processing means diagnoses in time-sensitive settings and imaging hardware that doesn't require a dedicated FPGA budget to build. This project is a working proof of concept that a GPU-based, open-source pipeline can match that performance bar using widely available tools, and it's built to be extended (e.g. with CNN-based segmentation) rather than treated as a one-off script.

## Technical Highlights

- **Stack:** Python, CuPy (GPU-accelerated NumPy/SciPy equivalent), CUDA, NumPy, SciPy, Matplotlib
- **Core algorithms implemented from scratch:** DC subtraction, bicubic A-line interpolation, quadratic-phase dispersion compensation, 2D FFT/IFFT for spectral-to-spatial domain conversion
- **Performance engineering:** chunked batch processing to manage GPU memory limits, thread pool concurrency tuning (workers and chunk size swept from 2-15), benchmarking across three GPU platforms (RTX 4070, RTX 3060, Kaggle dual T4) and one CPU baseline (i7-10750H)
- **Validation:** visual inspection of each pipeline stage against known OCT structural features, cross-checked against literature-reported FPGA and CPU benchmarks
- **Dual pipeline design:** a speed-optimized production pipeline (this repo) and a second quality-focused variant using non-local-means denoising, built to explore the speed/quality tradeoff space

## Repository Structure

```
pipeline.py       # Main GPU-accelerated processing pipeline
visualize.py       # Interactive Matplotlib viewer for processed OCT volumes
oct_data_loader.py  # Data loading utilities
in/                 # Input .mat files (raw spatial-domain OCT data)
out/                # Processed output files
```

## Running It

```bash
pip install cupy numpy scipy matplotlib tqdm psutil rich
```

1. Place raw `.mat` OCT files in `/in`
2. Run `python pipeline.py` to process the dataset (CUDA-capable GPU required)
3. Run `python visualize.py` to interactively view the processed slices

## Documentation

The full engineering design report, covering the theory, alternative designs considered (CPU vs. GPU vs. FPGA), testing methodology, and complete performance data, is included in the parent project folder as `COE 70B - Final EDP Report - Winter - RP02.docx.pdf`.

## References

Built on the FARSIU-2014 Retinal OCT dataset (Duke University / Kaggle) and informed by published research on GPU- and FPGA-accelerated OCT processing, cited in full in the final report.
