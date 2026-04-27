# Benchmarking Classical Single-Channel Speech Enhancement Algorithms on the NOIZEUS Corpus

Author: Cedric Jovenet  
Affiliation: Department of Electrical and Computer Engineering, Illinois Institute of Technology  
Course context: ECE 597 Research Project  
Date: April 2026

## Abstract

Classical speech enhancement algorithms remain valuable baseline methods because they are interpretable, computationally lightweight, and strongly grounded in the literature. However, modern reports often emphasize deep-learning systems and cite classical methods only briefly, which reduces the visibility of their practical strengths and their continuing value as benchmark references. This paper presents a controlled benchmark of five classical single-channel speech enhancement algorithms implemented in a unified research dashboard: spectral subtraction, over-subtraction, multi-band spectral subtraction, Wiener filtering, and subspace enhancement. The benchmark is performed on the NOIZEUS corpus, a standard noisy-speech dataset with matched clean references and fixed signal-to-noise-ratio conditions. The present paper reports a reproducible benchmark slice based on the babble-noise subset of NOIZEUS across all available signal-to-noise-ratio levels, corresponding to 120 noisy utterances. The evaluation uses signal-to-noise ratio improvement, narrowband Perceptual Evaluation of Speech Quality (PESQ), and Short-Time Objective Intelligibility (STOI). Under the current untuned baseline settings, classical spectral subtraction provides the strongest mean improvement in signal-to-noise ratio, while the subspace method yields the highest average PESQ and STOI but also a strongly negative energy-based signal-to-noise-ratio change. This disagreement between metrics highlights the importance of reporting multiple objective measures rather than relying on a single summary score. The paper argues that classical benchmarks remain scientifically useful not only as historical references, but also as transparent, reproducible baselines for student-led speech enhancement research.

## Keywords

Speech enhancement, classical benchmark, NOIZEUS, spectral subtraction, Wiener filtering, subspace enhancement, PESQ, STOI

## 1. Introduction

Speech enhancement seeks to improve the perceptual quality and intelligibility of speech degraded by background noise. Although the literature now includes a large number of deep-learning approaches, classical algorithms remain important for three reasons. First, they provide interpretable baselines. Second, they can be reproduced with limited computational resources. Third, they often reveal methodological issues clearly because their assumptions are explicit [1], [2], [3].

For a master's research project focused on rigorous benchmarking, classical algorithms are therefore a strong starting point. They allow the experimenter to study how noise estimation, gain design, and signal reconstruction affect objective metrics without the additional confounds introduced by large-scale training pipelines. They also make it easier to document the complete system in a way that is understandable to instructors and reviewers.

This paper benchmarks five classical algorithms within a unified local software environment. The goal is not to claim state-of-the-art performance. The goal is to build a credible and reproducible reference study around transparent baseline methods. The benchmark uses the NOIZEUS corpus [4], which is well suited to this purpose because it provides matched clean and noisy speech with controlled signal-to-noise-ratio conditions.

## 2. Related Work and Rationale for Classical Baselines

Spectral subtraction is one of the most widely taught classical enhancement methods and remains a natural baseline because it explicitly estimates the noise spectrum and subtracts it from the noisy magnitude spectrum [1]. Over-subtraction variants attempt to reduce musical-noise artifacts or residual noise by applying stronger suppression, usually at the price of greater speech distortion [5]. Multi-band extensions apply different suppression strengths across frequency regions and are motivated by the fact that speech and noise interact differently across bands [6].

Wiener filtering is another foundational baseline. It can be interpreted as a frequency-dependent gain derived from relative signal and noise power, and it is directly connected to minimum mean-square error reasoning [2], [3]. Subspace approaches instead model speech and noise through matrix structure and attempt to reconstruct the signal from an estimated speech subspace [7]. These methods are especially interesting in a benchmark because they sometimes improve perceptual criteria even when energy-based metrics are less favorable.

The benchmark also draws on accepted evaluation practice from speech enhancement literature. PESQ remains a common objective quality measure, while STOI is widely used to approximate intelligibility [8], [9]. Because NOIZEUS is an 8 kHz corpus [4], the correct methodological choice is narrowband PESQ rather than wideband PESQ. This point matters because metric definitions and sample-rate assumptions directly affect the legitimacy of any comparison.

## 3. Benchmark Objective and Scope

The benchmark objective is to compare implemented classical speech enhancement algorithms under a shared single-channel protocol. The study deliberately excludes other tasks such as active noise control because task mixing weakens comparability. The benchmark scope in this paper is as follows:

- task: single-channel speech enhancement,
- corpus: NOIZEUS,
- current reported slice: babble-noise conditions across 0 dB, 5 dB, 10 dB, and 15 dB,
- utterances: 30 sentences per signal-to-noise-ratio condition,
- total noisy utterances in this paper: 120,
- metrics: signal-to-noise ratio improvement, narrowband PESQ, and STOI.

This scope is intentionally narrow. A clear benchmark slice is preferable to a broad but weakly controlled study. The accompanying software can execute broader sweeps across the full NOIZEUS corpus, but the current written report focuses on one representative and fully reproducible subset.

## 4. Algorithms Under Study

### 4.1 Spectral subtraction (classic)

The classic spectral subtraction baseline estimates a stationary noise magnitude spectrum from early frames and subtracts it from the noisy spectrum [1]. It is simple, interpretable, and often serves as a first benchmark reference in speech enhancement studies.

### 4.2 Spectral subtraction with over-subtraction

The over-subtraction variant uses a larger subtraction factor and a spectral floor. In principle, this can reduce residual noise more aggressively, but it can also suppress speech components when the noise estimate is imperfect [5].

### 4.3 Multi-band spectral subtraction

The multi-band baseline partitions the spectrum and applies band-dependent suppression. In the present implementation, a methodological correction was necessary during development: a uniform-band version of the method behaved almost identically to over-subtraction. The final implementation uses band-specific scaling so that the method is a genuine distinct baseline rather than a duplicate under a different name.

### 4.4 Wiener filtering

The Wiener baseline estimates speech and noise power and derives a gain that attenuates bins with poor local signal-to-noise ratio [2]. It is a strong classical reference because it captures the core logic behind many later statistical enhancement methods.

### 4.5 Subspace enhancement

The subspace baseline uses singular value decomposition to project the observed signal onto a lower-dimensional structure assumed to be more speech-dominated [7]. In a benchmark, this family is useful because it may behave differently from purely spectral gains and may expose disagreements between objective metrics.

## 5. Dataset and Protocol

NOIZEUS was developed to support comparison of speech enhancement algorithms across research groups [4]. The corpus contains 30 IEEE sentences produced by three male and three female speakers. The clean utterances were downsampled to 8 kHz and corrupted with eight real-world noises obtained from the AURORA framework, including babble, car, exhibition hall, restaurant, street, airport, train station, and train. Noise was added at 0 dB, 5 dB, 10 dB, and 15 dB.

The present benchmark paper reports the babble-noise slice only. This means that each of the 30 clean sentences is evaluated at four signal-to-noise-ratio levels, yielding 120 noisy test utterances. All algorithms are run on the same input signals. No method-specific retuning is performed across utterances or signal-to-noise-ratio conditions. This decision prioritizes fairness and reproducibility over task-specific optimization.

### 5.1 Metric definitions

The first metric is signal-to-noise ratio improvement:

$$
\Delta \mathrm{SNR} = \mathrm{SNR}(s, \hat{s} - s) - \mathrm{SNR}(s, x - s),
$$

where $s$ is the clean reference, $x$ is the noisy input, and $\hat{s}$ is the enhanced signal.

The second metric is PESQ [8]. Because NOIZEUS is an 8 kHz corpus, PESQ is computed in narrowband mode rather than by resampling to 16 kHz. The third metric is STOI [9], which provides an objective estimate more closely related to intelligibility than to perceptual quality.

### 5.2 Reproducibility path

The benchmark can be reproduced through the offline runner included in the project:

```bash
python -m src.evaluation.noizeus_benchmark --noise babble --output-dir artifacts
```

The current run writes raw rows to `artifacts/noizeus_raw_babble.csv` and summary rows to `artifacts/noizeus_summary_babble.csv`.

## 6. Experimental Environment

All methods were executed in the same local Python environment used by the interactive dashboard. The implementation relies on NumPy, Librosa, the `pesq` package, and PySTOI. The benchmark code and the visualization code share the same algorithm registry, which reduces the risk that the reported benchmark differs from the demonstrated implementation.

This is an important design choice. In many student projects, the demo code, report code, and experimental code drift apart over time. In the present project, the benchmark path is explicitly tied to the same registered implementations exposed through the application. This improves traceability.

## 7. Results

Table 1 reports the aggregated results for the babble-noise slice across all four signal-to-noise-ratio levels. Values are reported as mean $\pm$ standard deviation over 120 noisy utterances.

### Table 1. Objective results on the babble slice of NOIZEUS

| Algorithm | Signal-to-noise ratio improvement (dB) | PESQ | STOI |
| --- | ---: | ---: | ---: |
| Spectral Subtraction (Classic) | 0.243 ± 3.855 | 1.830 ± 0.275 | 0.776 ± 0.103 |
| Spectral Subtraction (Over-subtraction) | -0.397 ± 4.254 | 1.754 ± 0.237 | 0.762 ± 0.101 |
| Multi-band Spectral Subtraction | -0.234 ± 4.115 | 1.760 ± 0.233 | 0.760 ± 0.102 |
| Wiener Filtering | 0.076 ± 3.800 | 1.679 ± 0.191 | 0.754 ± 0.096 |
| Subspace Enhancement (SVD) | -11.544 ± 5.121 | 1.895 ± 0.390 | 0.800 ± 0.117 |

### 7.1 Main observations

Three observations stand out.

First, classic spectral subtraction gives the strongest average signal-to-noise-ratio improvement among the tested methods. It also delivers strong PESQ and STOI values relative to the other spectral-gain baselines. This makes it a good default classical reference in the current project.

Second, Wiener filtering is conservative. Its average signal-to-noise-ratio improvement is positive but small, and its PESQ and STOI values are lower than those of classic spectral subtraction under the current untuned settings. This does not invalidate Wiener filtering as a classical baseline, but it suggests that the present implementation and parameter choices are not yet optimized for the babble slice.

Third, the subspace baseline is the most surprising method. It obtains the highest mean PESQ and STOI in the current table, yet it produces a strongly negative mean signal-to-noise-ratio improvement. This means that the method is not preserving the clean-reference energy relationship in a way that benefits the energy-based metric, even though the perceptual and intelligibility metrics appear more favorable. This disagreement is scientifically interesting and demonstrates why a benchmark should not rely on one metric alone.

### 7.2 Interpretation of the spectral family

The over-subtraction and multi-band spectral subtraction methods do not outperform the classic spectral subtraction baseline in the current benchmark. Their mean signal-to-noise-ratio improvements are negative, and their PESQ and STOI values remain below the classic method. This suggests that stronger or more frequency-shaped suppression is not automatically beneficial under the present parameter choices. It also highlights a practical lesson for benchmarking: adding algorithmic complexity does not guarantee a better baseline.

## 8. Discussion

The benchmark supports four broader conclusions.

### 8.1 Classical baselines still matter

Even in an era dominated by deep-learning results, a well-implemented classical baseline study remains useful. The current experiment is easy to reproduce, its assumptions are readable, and its failure modes are visible. That makes it a strong educational and methodological foundation.

### 8.2 Metric disagreement is not an edge case

The subspace result shows that objective metrics can disagree substantially. A method can degrade the energy-based signal-to-noise-ratio calculation while still improving perceptual-quality or intelligibility proxies. That does not mean one metric is wrong and another is right. It means the benchmark must report several measures and interpret them carefully.

### 8.3 Fairness requires protocol discipline

Small implementation decisions matter. The correction from wideband PESQ to narrowband PESQ for NOIZEUS is not a cosmetic change. It directly affects the comparability of the results. Likewise, distinguishing multi-band suppression from a duplicated over-subtraction baseline is necessary if the benchmark is to be taken seriously.

### 8.4 Untuned baselines can still be informative

The present study does not claim that each method has been exhaustively optimized. The benchmark is deliberately based on a common implementation environment with shared defaults. This is acceptable because the main goal is not to report record-breaking scores. The goal is to build a coherent and extensible reference benchmark.

## 9. Threats to Validity

Several validity threats should be stated clearly.

First, the benchmark reported in this paper is a corpus slice, not the full NOIZEUS study. The protocol is designed for extension to other noises, but the current reported numbers are limited to babble conditions.

Second, only objective measures are used. No subjective listening test is included in the present paper, although the NOIZEUS literature strongly emphasizes the value of subjective comparison [4].

Third, the implementations are project baselines, not exact reproductions of the parameter settings from every cited paper. That is acceptable for a project benchmark, but it prevents strong claims against published best-case results.

Fourth, the subspace implementation likely deserves further investigation because its signal-to-noise-ratio behavior is unusually poor relative to its PESQ and STOI values. This could indicate a useful perceptual tradeoff, or it could reflect an implementation weakness.

## 10. Conclusion

This paper presented a controlled benchmark of five classical single-channel speech enhancement methods on the babble subset of the NOIZEUS corpus. Under the current project settings, classic spectral subtraction emerges as the strongest overall baseline when signal-to-noise-ratio improvement is treated as the primary criterion, while the subspace method yields the highest average PESQ and STOI at the cost of strongly negative energy-based signal-to-noise-ratio change. The benchmark therefore shows both the value of classical baselines and the importance of multi-metric analysis. More broadly, the study demonstrates that a carefully scoped benchmark can remain scientifically useful even without deep-learning models, provided that the methodology is explicit, the software is reproducible, and the claims are kept within the true scope of the data.

## References

[1] S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction," IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 27, no. 2, pp. 113-120, 1979.

[2] J. S. Lim and A. V. Oppenheim, "Enhancement and bandwidth compression of noisy speech," Proceedings of the IEEE, vol. 67, no. 12, pp. 1586-1604, 1979.

[3] P. C. Loizou, Speech Enhancement: Theory and Practice, 2nd ed. Boca Raton, FL, USA: CRC Press, 2013.

[4] Y. Hu and P. C. Loizou, "Subjective evaluation and comparison of speech enhancement algorithms," Speech Communication, vol. 49, no. 7-8, pp. 588-601, 2007.

[5] N. Virag, "Single channel speech enhancement based on masking properties of the human auditory system," IEEE Transactions on Speech and Audio Processing, vol. 7, no. 2, pp. 126-137, 1999.

[6] Y. Kamath and P. Loizou, "A multi-band spectral subtraction method for enhancing speech corrupted by colored noise," in Proceedings of ICASSP, 2002, pp. IV-4164-IV-4167.

[7] Y. Ephraim and H. L. Van Trees, "A signal subspace approach for speech enhancement," IEEE Transactions on Speech and Audio Processing, vol. 3, no. 4, pp. 251-266, 1995.

[8] ITU-T Recommendation P.862, Perceptual Evaluation of Speech Quality (PESQ): An Objective Method for End-to-End Speech Quality Assessment of Narrow-Band Telephone Networks and Speech Codecs, 2001.

[9] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, "An algorithm for intelligibility prediction of time-frequency weighted noisy speech," IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, no. 7, pp. 2125-2136, 2011.