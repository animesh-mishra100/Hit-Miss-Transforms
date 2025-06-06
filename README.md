# 🧬 Cancer Cell Segmentation using Hourglass-Shaped Hit-or-Miss Transforms

> ⚡ A high-performance parallel computing approach for segmenting HER2/neu-stained breast cancer images using **Hourglass-Shaped Hit-or-Miss Transforms** with MPI and CUDA.

📍 **Presented at:**  
**VSPICE-2025** – International Conference on VLSI, Signal Processing, Power Electronics, IoT, Communication and Embedded Systems

---

## 🎯 Objective

- ✅ Accurately segment HER2/neu membrane regions in histopathology images.
- 🚀 Speed up image processing using **parallel computation** (MPI & CUDA).
- 🎯 Improve objectivity and reliability in breast cancer diagnosis.

---

## 🧪 Dataset

- **BCI Dataset** (Liu et al., 2022)  
- 9700+ annotated IHC-stained breast cancer images  
- HER2 scoring from 0 to 3+ with membrane boundary annotations

---

## ⚙️ Methodology

| Approach     | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| 🧠 Sequential | CLAHE + Gaussian filtering → HMT across orientations → Morph filtering     |
| 🔗 MPI        | Master–Worker model to parallelize image slices across multiple processors |
| 🎮 CUDA       | GPU kernels for pixel-level HMT parallelism with shared memory optimization |

---

## 📈 Results (Performance)

| Image Size     | Method      | Time         | Speedup    |
|----------------|-------------|--------------|------------|
| 412×532        | MPI (8 procs) | 0.0061s      | 16.46×     |
| 1024×1024      | MPI (8 procs) | 0.0417s      | 39.74×     |
| 412×532        | CUDA          | **0.56ms**   | **178×**   |
| 1024×1024      | CUDA          | **3.41ms**   | **485×**   |

---

## 🏥 Real-World Application

This approach can be integrated into **clinical decision support systems** for:
- Rapid HER2/neu membrane scoring
- Computer-aided breast cancer diagnosis
- Pathology automation pipelines

---

## 👥 Authors

- **Aniketh Hebbar**  
- **Animesh Mishra**  
- **N. Gopalakrishna Kini**  
- **Jyothi Upadhya K**  
<sub>Department of Computer Science & Engineering, Manipal Institute of Technology</sub>

---

## 📝 Note

To run the code:
- Ensure **CUDA Toolkit** and **OpenMPI** are installed
- Use appropriate compilers (`nvcc`, `mpic++`)
- Dataset used is publicly available for academic research
