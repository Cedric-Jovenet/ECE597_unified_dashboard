# A Unified Interactive Dashboard for Classical Speech Enhancement Research, Benchmarking, and Demonstration

Author: Cedric Jovenet  
Affiliation: Department of Electrical and Computer Engineering, Illinois Institute of Technology  
Course context: ECE 597 Research Project  
Date: April 2026

## Abstract

Speech enhancement is a mature research area with a deep classical literature, but the practical presentation of that literature remains fragmented. Definitions are often separated from code, benchmark descriptions are separated from demonstrations, and comparable baseline results are rarely exposed through a single student-friendly interface. This fragmentation is a real obstacle in a graduate research setting because it slows down literature review, weakens methodological clarity, and makes demonstrations less convincing than they should be. This paper presents a unified interactive dashboard designed for classical single-channel speech enhancement research. The system integrates algorithm descriptions, runnable implementations, dataset notes, benchmark execution, objective metrics, and literature context in one interface. The present version is intentionally centered on a narrow and defensible scope: classical speech enhancement algorithms evaluated on the NOIZEUS corpus, with VoiceBank+DEMAND retained only as a reference benchmark model for literature organization. The implemented algorithms include spectral subtraction, over-subtraction, multi-band spectral subtraction, Wiener filtering, and a subspace-based enhancement method. The application is built in Streamlit and supported by a small registry-based architecture and a reproducible offline benchmark runner. The main contribution is not a new denoising algorithm, but a structured research tool that improves clarity, comparability, and reusability for a master's-level research workflow. The resulting system supports classroom demonstration, reproducible benchmarking, and technical writing in a more coherent way than scattered notes, standalone scripts, and slide decks.

## Keywords

Speech enhancement, classical algorithms, NOIZEUS, benchmark dashboard, research software, interactive demonstration, reproducibility

## 1. Introduction

Speech enhancement research has produced a large body of methods, datasets, and evaluation practices over several decades. Classical approaches such as spectral subtraction, Wiener filtering, and subspace methods remain important because they are interpretable, computationally accessible, and still useful as reference baselines [1], [2], [3], [4]. However, students and new researchers usually encounter this material in a fragmented form. Dataset descriptions appear on separate web pages, algorithm explanations are distributed across books and papers, source code is inconsistent across repositories, and benchmark claims are often presented without an accompanying interactive tool.

This project was motivated by that fragmentation. In the context of a master's research project, it became clear that a strong literature review and a strong demonstration should not be treated as separate outputs. The same environment should help explain what each algorithm does, show how it behaves on audio, document the benchmark protocol, and make the resulting claims traceable. A tool that cannot support those four activities together is less useful than it appears, especially when the research goal is to build a clean and defensible study around classical speech enhancement.

The system presented in this paper addresses that need. It is not proposed as a universal benchmark platform for all speech processing tasks. Instead, it is designed around one clear objective: to support a rigorous and understandable presentation of classical single-channel speech enhancement methods. The current version deliberately avoids mixing unrelated tasks such as active noise control with offline speech enhancement, because such a mixture weakens methodological clarity and makes benchmark claims harder to defend.

## 2. Motivation and Gap

The practical gap addressed by this work is not the absence of speech enhancement literature. The field is well documented. The problem is that the supporting material needed by a student researcher is usually split across several disconnected formats:

- textbook explanations for theory,
- separate papers for benchmark results,
- unofficial repositories for code,
- dataset pages for corpus details,
- presentation slides for simplified explanations,
- ad hoc notebooks or scripts for local experiments.

This separation creates three recurring problems.

First, explanations and implementations drift apart. A slide deck may describe a Wiener filter correctly while the local script uses a different noise-estimation assumption. Second, benchmark claims become difficult to audit. It is easy to quote PESQ or STOI values from prior work without recording whether the dataset, sample rate, or metric implementation actually matches the current experiment. Third, demonstrations become shallow. A demo often plays processed audio but cannot explain the algorithm family, metric definitions, or comparability limits in the same space.

The dashboard developed in this project treats those issues as a software design problem. The goal is to expose explanation, code, benchmark protocol, and results together so that the presentation layer supports the research methodology instead of obscuring it.

## 3. Design Objectives

The dashboard was designed around five objectives.

### 3.1 Scope discipline

The tool should prioritize one coherent task rather than multiple loosely related tasks. For that reason, the current interface is centered on single-channel speech enhancement, not active noise control or hardware-oriented noise cancellation.

### 3.2 Methodological clarity

Users should immediately understand what is comparable and what is not. The runnable benchmark section is therefore limited to one project benchmark on NOIZEUS, while literature-oriented datasets remain explicitly separated.

### 3.3 Interactivity

The application should support listening tests, waveform and spectrogram inspection, and metric visualization without requiring the user to switch tools.

### 3.4 Reusability

The software should support future extension through structured catalogs rather than one-off hardcoded pages. This requirement matters because the project is intended to support later algorithm additions and future reports.

### 3.5 Writing support

The tool should directly support technical writing by making datasets, metrics, protocols, and benchmark outputs easy to recover in a documented form.

## 4. System Overview

The implemented system is a Streamlit application organized around four central registries and four user-facing views.

### 4.1 Core registries

The backend uses structured catalogs for:

- algorithms,
- datasets,
- benchmark definitions,
- knowledge topics and literature context.

This registry-based design has two advantages. It keeps the user interface concise, and it ensures that additions to the project remain explicit. New algorithms, datasets, or literature entries are not scattered through the interface logic; they are described in one place and consumed consistently by the frontend.

### 4.2 User-facing views

The main application is divided into the following pages.

#### Experiment Workspace

This page is intended for qualitative inspection. A user selects an algorithm family, an implemented method, and a dataset condition, then runs the method on a single utterance. The output includes before/after audio playback, objective metrics when a clean reference is available, waveform plots, spectrogram plots, and additional statistics. The workspace is explicitly described as a single-sample analysis space rather than a benchmark engine.

#### Dataset Catalog

The dataset page provides descriptive information independently from the experiment flow. In the current version, NOIZEUS is the primary runnable dataset, while VoiceBank+DEMAND is documented as a literature-facing benchmark reference. This separation improves clarity because the presence of a dataset in the interface does not automatically imply that it is part of the local benchmark protocol.

#### Benchmark Study

The benchmark page contains the core scientific workflow of the tool. It includes an overview of the project benchmark, a runnable benchmark section, and a literature-context section. The runnable section executes the implemented classical algorithms under shared NOIZEUS conditions and summarizes the outputs through means, standard deviations, distribution plots, and downloadable comma-separated files.

#### Knowledge Base

The knowledge base groups topic summaries, definitions, references, and supporting resources in one place. The intent is to prevent the application from becoming only a control panel. In a research setting, definitions and references need to live close to the demonstrations they support.

## 5. Implemented Algorithms and Metrics

The present version includes five classical baselines:

1. Spectral subtraction (classic)
2. Spectral subtraction with over-subtraction
3. Multi-band spectral subtraction
4. Wiener filtering
5. Subspace enhancement based on singular value decomposition

These methods were chosen for two reasons. First, they are representative of major classical families discussed in speech enhancement literature [1], [4]. Second, they are appropriate for a graduate-level project whose value comes from methodological clarity rather than from training a large deep network.

The objective metrics currently computed are signal-to-noise ratio improvement, Perceptual Evaluation of Speech Quality (PESQ), and Short-Time Objective Intelligibility (STOI) [6], [7]. A small but important methodological correction was made during development: NOIZEUS is an 8 kHz corpus [5], so the software now uses narrowband PESQ for NOIZEUS rather than artificially upsampling to 16 kHz and applying wideband PESQ. This change improves the credibility of benchmark claims.

Another methodological refinement was applied to the multi-band spectral subtraction baseline. An earlier implementation used uniform suppression parameters across all bands, which made it effectively collapse into the same behavior as the over-subtraction baseline. The current version applies band-dependent scaling, which gives the method a distinct identity and makes the comparison more meaningful.

## 6. Software Architecture and Reproducibility

The dashboard is intentionally lightweight. Streamlit was selected because it enables rapid creation of interactive scientific interfaces without building a separate client-server web stack. That decision is pragmatic rather than ideological. For a research tool developed within one semester, fast iteration and transparent Python integration are more valuable than a heavier frontend architecture.

At the same time, the project is not only a user interface. Reproducibility depends on separating benchmark execution from interface interaction. For that reason, the project now includes an offline benchmark runner at `src/evaluation/noizeus_benchmark.py`. The script can execute the same project benchmark from the command line and export raw and summary comma-separated files. This makes it easier to reuse results in papers and makes the project less dependent on the graphical interface.

The registry structure also supports scalability. New algorithm entries can be added by registering a label, family, purpose, benchmark group, implementation status, callable runner, and summary. New literature entries can be recorded through structured protocol fields such as sample rate, signal-to-noise-ratio conditions, metrics, preprocessing notes, and comparability status. This is a small design decision, but it has a large effect on research hygiene.

## 7. Demonstration Workflow

The tool was designed to support an oral demo as well as a written report. A credible demonstration can proceed in the following order:

1. Introduce the benchmark objective and the current algorithm families.
2. Select a NOIZEUS condition in the workspace and play the noisy reference.
3. Run one algorithm and inspect waveform, spectrogram, and metric changes.
4. Repeat with a second algorithm from another classical family.
5. Move to the benchmark page and run the same condition across all implemented methods.
6. Show the summary table and distribution plots.
7. Open the literature context section to explain what is and is not directly comparable with the local results.

This flow is useful because it ties explanation, listening, benchmarking, and literature positioning together. A common failure mode in student demonstrations is that the live demo and the written report seem to describe different projects. The present dashboard was explicitly designed to avoid that split.

## 8. Discussion

The main contribution of this work is organizational rigor. The system does not claim algorithmic novelty. Its value lies in the fact that it creates a clean research surface for classical speech enhancement. In many academic settings, that is more useful than adding one more partially documented algorithm. A narrow, credible benchmark with transparent methodology is preferable to a broad and confusing interface that mixes tasks, metrics, and unsupported comparison claims.

The project also shows that research software for literature review can be more than a static website. Once algorithms, datasets, benchmark definitions, and references are structured in one environment, the interface becomes a bridge between study, implementation, and communication. This is especially useful for master's students, who often need to defend methodology and demonstrate understanding at the same time.

## 9. Limitations and Future Work

The current system has several limits.

First, the benchmark scope is intentionally narrow. The runnable benchmark is based on NOIZEUS, and the currently written benchmark analysis is strongest for the babble-noise slice. A full multi-noise sweep is supported by the provided evaluation script, but the manuscript should distinguish clearly between completed results and extensible protocol.

Second, the implemented methods are baseline implementations rather than heavily tuned reproductions of published systems. This is acceptable for a pedagogical classical benchmark, but it should be stated explicitly.

Third, the current evaluation is objective rather than perceptual. Listening tests remain important in speech enhancement [5], and a stronger future version of the tool could support small-scale subjective evaluation forms.

Fourth, the literature context section is structured but still incomplete. Some entries deliberately avoid reporting numbers until exact protocol details are recovered from primary sources. This is a strength from a methodological point of view, but it means the literature section still requires curation work.

Reasonable next steps include adding well-documented Wiener and minimum mean-square error variants, extending the benchmark export tools, improving the subspace baseline, and adding a publication-oriented report generator that consumes the exported benchmark files.

## 10. Conclusion

This paper presented a unified dashboard for classical speech enhancement research that combines explanation, implementation, benchmark execution, and literature context in one reproducible environment. The project contributes a practical research tool rather than a new enhancement method. Its main strengths are scope discipline, benchmark clarity, interactivity, and direct usefulness for technical writing and demonstration. By centering the application on classical single-channel speech enhancement and by separating project results from literature context, the tool supports a cleaner research workflow and a more defensible presentation of benchmark claims. For a master's-level project, this combination of methodological clarity and usable software is a meaningful contribution.

## References

[1] S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction," IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 27, no. 2, pp. 113-120, 1979.

[2] J. S. Lim and A. V. Oppenheim, "Enhancement and bandwidth compression of noisy speech," Proceedings of the IEEE, vol. 67, no. 12, pp. 1586-1604, 1979.

[3] Y. Ephraim and H. L. Van Trees, "A signal subspace approach for speech enhancement," IEEE Transactions on Speech and Audio Processing, vol. 3, no. 4, pp. 251-266, 1995.

[4] P. C. Loizou, Speech Enhancement: Theory and Practice, 2nd ed. Boca Raton, FL, USA: CRC Press, 2013.

[5] Y. Hu and P. C. Loizou, "Subjective evaluation and comparison of speech enhancement algorithms," Speech Communication, vol. 49, no. 7-8, pp. 588-601, 2007.

[6] ITU-T Recommendation P.862, Perceptual Evaluation of Speech Quality (PESQ): An Objective Method for End-to-End Speech Quality Assessment of Narrow-Band Telephone Networks and Speech Codecs, 2001.

[7] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, "An algorithm for intelligibility prediction of time-frequency weighted noisy speech," IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, no. 7, pp. 2125-2136, 2011.

[8] C. Valentini-Botinhao, X. Wang, S. Takaki, and J. Yamagishi, "Investigating RNN-based speech enhancement methods for noise-robust text-to-speech," in Proceedings of SSW10, 2016.

[9] C. Valentini-Botinhao, "Noisy speech database for training speech enhancement algorithms and TTS models," University of Edinburgh, 2017.