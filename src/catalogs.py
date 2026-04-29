from src.algorithms.spectral import (
    adaptive_spectral_subtraction,
    generalized_spectral_subtraction,
    multiband_spectral_subtraction,
    power_spectral_subtraction,
    spectral_subtraction,
    spectral_subtraction_oversubtraction,
    wiener_filter_enhance,
)
from src.algorithms.statistical import (
    decision_directed_wiener_filter,
    log_mmse_enhance,
    mmse_stsa_enhance,
    om_lsa_enhance,
)
from src.algorithms.subspace import subspace_enhance
from src.algorithms.subspace import signal_subspace_enhance_evt


def get_algorithm_catalog():
    """Central registry for algorithms, grouped by research objective."""
    return {
        "Speech Enhancement": [
            {
                'label': "Spectral Subtraction (Classic)",
                'family': "Spectral Subtraction",
                'purpose': "Single-channel speech denoising",
                'benchmark_group': "speech_enhancement",
                'implemented': True,
                'runner': spectral_subtraction,
                'needs_sr': True,
                'summary': "Baseline spectral subtraction using a stationary noise estimate."
            },
            {
                'label': "Spectral Subtraction (Over-subtraction)",
                'family': "Spectral Subtraction",
                'purpose': "Single-channel speech denoising",
                'benchmark_group': "speech_enhancement",
                'implemented': True,
                'runner': spectral_subtraction_oversubtraction,
                'needs_sr': True,
                'summary': "More aggressive subtraction to suppress residual noise."
            },
            {
                'label': "Spectral Subtraction (Adaptive Over-subtraction)",
                'family': "Spectral Subtraction",
                'purpose': "Single-channel speech denoising",
                'benchmark_group': "speech_enhancement",
                'implemented': True,
                'runner': adaptive_spectral_subtraction,
                'needs_sr': True,
                'summary': "Frame-adaptive over-subtraction with stronger suppression in low local signal-to-noise conditions."
            },
            {
                'label': "Power Spectral Subtraction",
                'family': "Spectral Subtraction",
                'purpose': "Single-channel speech denoising",
                'benchmark_group': "speech_enhancement",
                'implemented': True,
                'runner': power_spectral_subtraction,
                'needs_sr': True,
                'summary': "Subtracts the estimated noise power spectrum before returning to amplitude domain."
            },
            {
                'label': "Generalized Spectral Subtraction",
                'family': "Spectral Subtraction",
                'purpose': "Single-channel speech denoising",
                'benchmark_group': "speech_enhancement",
                'implemented': True,
                'runner': generalized_spectral_subtraction,
                'needs_sr': True,
                'summary': "Generalized-domain subtraction that interpolates between magnitude and power spectral suppression."
            },
            {
                'label': "Multi-band Spectral Subtraction",
                'family': "Spectral Subtraction",
                'purpose': "Single-channel speech denoising",
                'benchmark_group': "speech_enhancement",
                'implemented': True,
                'runner': multiband_spectral_subtraction,
                'needs_sr': True,
                'summary': "Frequency-dependent subtraction to adapt suppression strength by band."
            },
            {
                'label': "Wiener Filtering",
                'family': "Wiener Filtering",
                'purpose': "Single-channel speech denoising",
                'benchmark_group': "speech_enhancement",
                'implemented': True,
                'runner': wiener_filter_enhance,
                'needs_sr': True,
                'summary': "Short-time spectral Wiener filtering using a noise estimate from the first frames."
            },
            {
                'label': "Decision-Directed Wiener Filtering",
                'family': "Wiener Filtering",
                'purpose': "Single-channel speech denoising",
                'benchmark_group': "speech_enhancement",
                'implemented': True,
                'runner': decision_directed_wiener_filter,
                'needs_sr': True,
                'summary': "Recursive Wiener filtering with a decision-directed estimate of the a priori signal-to-noise ratio."
            },
            {
                'label': "MMSE-STSA",
                'family': "Statistical Model-Based Enhancement",
                'purpose': "Single-channel speech denoising",
                'benchmark_group': "speech_enhancement",
                'implemented': True,
                'runner': mmse_stsa_enhance,
                'needs_sr': True,
                'summary': "Minimum mean-squared error short-time spectral amplitude estimator using recursive a priori signal-to-noise tracking."
            },
            {
                'label': "Log-MMSE",
                'family': "Statistical Model-Based Enhancement",
                'purpose': "Single-channel speech denoising",
                'benchmark_group': "speech_enhancement",
                'implemented': True,
                'runner': log_mmse_enhance,
                'needs_sr': True,
                'summary': "Log-spectral minimum mean-squared error estimator for perceptually smoother spectral gains."
            },
            {
                'label': "OM-LSA",
                'family': "Statistical Model-Based Enhancement",
                'purpose': "Single-channel speech denoising",
                'benchmark_group': "speech_enhancement",
                'implemented': True,
                'runner': om_lsa_enhance,
                'needs_sr': True,
                'summary': "Optimal modified log-spectral amplitude enhancement with speech-presence-aware gain flooring."
            },
            {
                'label': "Subspace Enhancement (SVD)",
                'family': "Subspace Enhancement",
                'purpose': "Single-channel speech denoising",
                'benchmark_group': "speech_enhancement",
                'implemented': True,
                'runner': subspace_enhance,
                'needs_sr': False,
                'summary': "Separates speech and noise through subspace projection."
            },
            {
                'label': "Signal Subspace Enhancement (EVT)",
                'family': "Subspace Enhancement",
                'purpose': "Single-channel speech denoising",
                'benchmark_group': "speech_enhancement",
                'implemented': True,
                'runner': signal_subspace_enhance_evt,
                'needs_sr': False,
                'summary': "Frame-wise signal subspace projection with singular-value shrinkage inspired by Ephraim and Van Trees style filtering."
            }
        ]
    }


def get_dataset_catalog(noizeus_samples):
    """Describe datasets independently from the currently selected workflow."""
    noizeus_noise_types = sorted({sample['noise_name'] for sample in noizeus_samples})
    noizeus_snr_levels = sorted({sample['snr'] for sample in noizeus_samples})

    return {
        "NOIZEUS Database": {
            'id': 'noizeus',
            'available': bool(noizeus_samples),
            'task_scope': "speech_enhancement",
            'benchmark_ready': True,
            'description': "Benchmark corpus for single-channel speech enhancement with matched clean references.",
            'reference': "http://ecs.utdallas.edu/loizou/speech/noizeus/",
            'details': [
                "30 clean IEEE sentences with aligned noisy counterparts.",
                f"Noise environments available: {', '.join(noizeus_noise_types) if noizeus_noise_types else 'None detected'}.",
                f"SNR levels available: {', '.join(noizeus_snr_levels) if noizeus_snr_levels else 'None detected'}.",
                f"Total noisy samples detected: {sum(sample['count'] for sample in noizeus_samples)}."
            ]
        },
        "VoiceBank+DEMAND": {
            'id': 'voicebank_demand',
            'available': False,
            'task_scope': "speech_enhancement",
            'benchmark_ready': True,
            'description': "Canonical single-channel speech-enhancement benchmark combining clean VoiceBank/VCTK speech with DEMAND noise.",
            'reference': "https://www.emergentmind.com/topics/voicebank-demand-dataset-6a395fe3-bd1f-41ee-919b-0986fce1d1f3",
            'details': [
                "Typical split: 28 training speakers and 2 held-out test speakers, often with an additional validation split.",
                "Canonical resampling at 16 kHz with additive mixing against DEMAND noise.",
                "Common train SNRs: 0, 5, 10, 15 dB. Common test SNRs: 2.5, 7.5, 12.5, 17.5 dB.",
                "Common metrics: PESQ, STOI, CSIG, CBAK, COVL, SSNR.",
                "Very strong as a reference benchmark for single-channel speech enhancement, but not directly interchangeable with NOIZEUS results."
            ]
        },
        "Custom Upload": {
            'id': 'custom_upload',
            'available': False,
            'task_scope': "exploration",
            'benchmark_ready': False,
            'description': "Manual audio import for ad-hoc listening tests and future experiments.",
            'reference': None,
            'details': [
                "Useful for exploratory demos or future unpublished datasets.",
                "Not benchmark-ready yet because metadata and references are missing."
            ]
        }
    }


def get_benchmark_catalog():
    """Describe project and reference benchmarks in one place."""
    return [
        {
            'id': 'project_noizeus',
            'label': "Project benchmark: single-channel speech enhancement on NOIZEUS",
            'type': "Project benchmark",
            'scope': "speech_enhancement",
            'status': "Runnable in this app",
            'datasets': ["NOIZEUS Database"],
            'goal': "Compare single-channel speech enhancement algorithms under identical speech, noise, and SNR conditions.",
            'comparison_rule': "Only compare algorithms that share the same objective and use the same clean-reference metrics.",
            'notes': [
                "This is the main benchmark of the project and the primary result section to show in demonstrations.",
                "Metrics currently supported: signal-to-noise ratio improvement, Perceptual Evaluation of Speech Quality (PESQ), and Short-Time Objective Intelligibility (STOI)."
            ],
            'study_ids': []
        },
        {
            'id': 'reference_noizeus',
            'label': "Literature context: published NOIZEUS studies",
            'type': "Literature context",
            'scope': "speech_enhancement",
            'status': "Documentation only",
            'datasets': ["NOIZEUS Database"],
            'goal': "Position your results against prior studies using the same corpus or close evaluation protocol.",
            'comparison_rule': "Keep literature baselines visible, but separate from your in-app computed results unless protocol details match.",
            'notes': [
                "Use this lane for papers, books, and external baselines.",
                "Important fields to record later: train/test split, metric implementation, signal preprocessing."
            ],
            'study_ids': [
                'loizou_book_reference',
                'noizeus_study_template'
            ]
        },
        {
            'id': 'reference_voicebank_demand',
            'label': "Literature context: VoiceBank+DEMAND benchmark",
            'type': "Literature context",
            'scope': "speech_enhancement",
            'status': "Documentation only",
            'datasets': ["VoiceBank+DEMAND"],
            'goal': "Track the canonical deep-learning speech-enhancement benchmark and position project results against a widely used reference protocol.",
            'comparison_rule': "Use this lane to compare protocol structure and literature baselines. Do not claim direct comparability with your NOIZEUS benchmark unless task, data generation, and metrics are aligned.",
            'notes': [
                "This lane is especially useful to document the standard benchmark design used by recent single-channel SE papers.",
                "Literature values in this lane must be labeled as external references, not in-app computed results."
            ],
            'study_ids': [
                'voicebank_demand_canonical_protocol',
                'voicebank_demand_result_snapshot'
            ]
        }
    ]


def get_reference_study_catalog():
    """Structured literature entries used in the reference benchmark lanes."""
    return [
        {
            'id': 'loizou_book_reference',
            'lane_id': 'reference_noizeus',
            'label': "Loizou book baseline notes",
            'citation': "P. C. Loizou, Speech Enhancement: Theory and Practice, 2nd ed., CRC Press, 2013.",
            'scope': 'speech_enhancement',
            'dataset': 'NOIZEUS Database',
            'status': 'Protocol extraction pending',
            'summary': "Reference source to align your benchmark presentation with common speech-enhancement evaluation practice.",
            'protocol': {
                'sample_rate': 'To extract from source',
                'subset': 'To extract from source',
                'snr_levels': ['To extract from source'],
                'metrics': ['PESQ', 'STOI', 'SNR improvement'],
                'preprocessing': 'To extract from source',
                'notes': [
                    'Use this entry to document the exact evaluation recipe once you recover it from the source.',
                    'Do not compare directly until the protocol fields are completed.'
                ]
            },
            'comparability': {
                'status': 'needs_review',
                'summary': 'Potentially comparable to the project benchmark, but the precise protocol still needs to be extracted.',
                'blocking_points': [
                    'Exact NOIZEUS condition coverage is not yet recorded.',
                    'Metric implementation details are not yet recorded.'
                ]
            },
            'reported_results': [],
            'notes': [
                'Good first candidate to populate once you extract exact tables from the book.',
                'Keeps literature references structured without forcing fake numeric values into the app.'
            ]
        },
        {
            'id': 'noizeus_study_template',
            'lane_id': 'reference_noizeus',
            'label': "Template for future NOIZEUS paper entries",
            'citation': "Fill with a paper that reports NOIZEUS results relevant to your benchmark.",
            'scope': 'speech_enhancement',
            'dataset': 'NOIZEUS Database',
            'status': 'Template',
            'summary': "Reusable structure for papers that should appear next to your own benchmark lane.",
            'protocol': {
                'sample_rate': 'Unknown',
                'subset': 'Unknown',
                'snr_levels': [],
                'metrics': [],
                'preprocessing': 'Unknown',
                'notes': [
                    'Record exactly what the paper used before surfacing comparison claims.',
                    'Prefer explicit fields over prose-only descriptions.'
                ]
            },
            'comparability': {
                'status': 'unreviewed',
                'summary': 'No comparison claim yet. Complete the protocol first.',
                'blocking_points': [
                    'Missing dataset subset details.',
                    'Missing metric definitions and units.'
                ]
            },
            'reported_results': [],
            'notes': [
                'Duplicate this entry pattern when you add a real paper.',
                'Once populated, add one result row per algorithm reported in the paper.'
            ]
        },
        {
            'id': 'voicebank_demand_canonical_protocol',
            'lane_id': 'reference_voicebank_demand',
            'label': "Canonical VoiceBank+DEMAND protocol",
            'citation': "Emergent Mind topic summary of the VoiceBank+DEMAND benchmark, updated 15 Nov 2025, synthesizing representative papers.",
            'scope': 'speech_enhancement',
            'dataset': 'VoiceBank+DEMAND',
            'status': 'Secondary-source protocol summary',
            'summary': "Structured snapshot of the benchmark design commonly used in recent single-channel speech-enhancement papers.",
            'protocol': {
                'sample_rate': '16 kHz',
                'subset': 'Typical canonical split: 28 training speakers, 2 held-out test speakers, optional validation speakers held out from training',
                'snr_levels': ['Train: 0 dB', 'Train: 5 dB', 'Train: 10 dB', 'Train: 15 dB', 'Test: 2.5 dB', 'Test: 7.5 dB', 'Test: 12.5 dB', 'Test: 17.5 dB'],
                'metrics': ['PESQ', 'STOI', 'CSIG', 'CBAK', 'COVL', 'SSNR'],
                'preprocessing': 'Clean speech and noise typically resampled to 16 kHz; additive mixing with DEMAND noise; TF-domain systems often use STFT with 400-sample window and 100-sample hop.',
                'notes': [
                    'This is a benchmark-structure reference for your app because it clearly separates task, split, protocol, metrics, and literature baselines.',
                    'Use source papers for exact implementation details before making strict numeric comparisons.'
                ]
            },
            'comparability': {
                'status': 'related_but_distinct',
                'summary': 'Highly relevant as a model benchmark lane for single-channel speech enhancement, but not directly interchangeable with NOIZEUS results.',
                'blocking_points': [
                    'Different dataset and noise-generation protocol from NOIZEUS.',
                    'Many papers include model-specific augmentation and preprocessing choices that affect comparability.'
                ]
            },
            'reported_results': [],
            'notes': [
                'This entry should be kept even if you later replace the secondary-source citation with primary references.',
                'It is a strong template for how to represent benchmark protocol metadata in the app.'
            ]
        },
        {
            'id': 'voicebank_demand_result_snapshot',
            'lane_id': 'reference_voicebank_demand',
            'label': "VoiceBank+DEMAND result snapshot",
            'citation': "Emergent Mind topic summary of VoiceBank+DEMAND benchmark values across representative works.",
            'scope': 'speech_enhancement',
            'dataset': 'VoiceBank+DEMAND',
            'status': 'Secondary-source results snapshot',
            'summary': "Compact external baseline snapshot to expose the scale of results commonly reported on VoiceBank+DEMAND.",
            'protocol': {
                'sample_rate': '16 kHz in the canonical setup',
                'subset': 'Representative benchmark summary across multiple works',
                'snr_levels': ['Canonical training: 0, 5, 10, 15 dB', 'Canonical testing: 2.5, 7.5, 12.5, 17.5 dB'],
                'metrics': ['PESQ', 'STOI', 'CSIG', 'COVL'],
                'preprocessing': 'Aggregated from multiple studies, not a single unified implementation.',
                'notes': [
                    'These are literature-facing reference values, not results computed in this app.',
                    'Ranges and approximate values should be replaced by paper-specific entries when you curate the benchmark in depth.'
                ]
            },
            'comparability': {
                'status': 'reference_only',
                'summary': 'Useful to communicate the performance scale of the benchmark, but too aggregated for strict model-to-model claims.',
                'blocking_points': [
                    'Values are summarized from multiple works rather than one controlled reproduction.',
                    'Some entries are approximate or reported as ranges.'
                ]
            },
            'reported_results': [
                {
                    'Algorithm': 'No enhancement',
                    'PESQ': '1.97',
                    'STOI': 'Not listed in summary',
                    'CSIG': 'Not listed in summary',
                    'COVL': 'Not listed in summary',
                    'Source note': 'Representative baseline from the summary page'
                },
                {
                    'Algorithm': 'Wiener filter',
                    'PESQ': '2.22',
                    'STOI': 'Not listed in summary',
                    'CSIG': 'Not listed in summary',
                    'COVL': 'Not listed in summary',
                    'Source note': 'Representative classical baseline from the summary page'
                },
                {
                    'Algorithm': 'SEGAN',
                    'PESQ': '2.16',
                    'STOI': 'Not listed in summary',
                    'CSIG': 'Not listed in summary',
                    'COVL': 'Not listed in summary',
                    'Source note': 'Representative early GAN baseline from the summary page'
                },
                {
                    'Algorithm': 'Conv-TasNet',
                    'PESQ': '2.53',
                    'STOI': 'Not listed in summary',
                    'CSIG': 'Not listed in summary',
                    'COVL': 'Not listed in summary',
                    'Source note': 'Representative discriminative DNN baseline from the summary page'
                },
                {
                    'Algorithm': 'Demucs',
                    'PESQ': '2.93-3.07',
                    'STOI': 'Not listed in summary',
                    'CSIG': 'Not listed in summary',
                    'COVL': 'Not listed in summary',
                    'Source note': 'Range reported in the summary page'
                },
                {
                    'Algorithm': 'Recent state-of-the-art snapshot',
                    'PESQ': '3.33-3.50',
                    'STOI': '>0.96',
                    'CSIG': '>4.7',
                    'COVL': '>4.1',
                    'Source note': 'Aggregated recent-model range from the summary page'
                }
            ],
            'notes': [
                'Good for orienting the user in the benchmark landscape.',
                'For rigorous comparison, split this snapshot into paper-specific studies later.'
            ]
        }
    ]


def get_knowledge_topics():
    """Organize concepts, references, notes, and resources by topic."""
    return {
        "Perceptual Evaluation of Speech Quality (PESQ)": {
            'summary': "Objective speech-quality score used in enhancement benchmarks.",
            'definition': "PESQ (Perceptual Evaluation of Speech Quality) estimates perceived speech quality, typically on a scale between 1.0 and 4.5.",
            'references': [
                "ITU-T P.862: https://www.itu.int/rec/T-REC-P.862/en",
                "Python implementation: https://github.com/ludlows/PESQ/tree/v0.0.4"
            ],
            'notes': "In this project, PESQ is computed with the pesq Python package. The app uses narrowband PESQ for 8 kHz material such as NOIZEUS and wideband PESQ for 16 kHz material, so benchmark claims stay consistent with the evaluation protocol.",
            'resources': []
        },
        "Short-Time Objective Intelligibility (STOI)": {
            'summary': "Objective intelligibility score for processed speech.",
            'definition': "STOI (Short-Time Objective Intelligibility) compares clean and processed signals and returns a score between 0 and 1.",
            'references': [
                "Taal et al., ICASSP 2010: https://doi.org/10.1109/ICASSP.2010.5495701",
                "Taal et al., IEEE TASLP 2011: https://doi.org/10.1109/TASL.2011.2114881",
                "Jensen and Taal, IEEE/ACM TASLP 2016: https://doi.org/10.1109/TASLP.2016.2585878"
            ],
            'notes': "STOI complements PESQ because it focuses more directly on intelligibility than perceptual quality.",
            'resources': []
        },
        "Signal-to-noise ratio (SNR)": {
            'summary': "Core signal-to-noise metric used to quantify improvement.",
            'definition': "SNR (Signal-to-Noise Ratio) measures the ratio between clean-signal power and noise power, usually in dB.",
            'references': [
                "Background article: https://en.wikipedia.org/wiki/Signal-to-noise_ratio"
            ],
            'notes': r"Equation used in the app: $\mathrm{SNR}_{\mathrm{dB}} = 10\log_{10}\left(\frac{P_{\mathrm{signal}}}{P_{\mathrm{noise}}}\right)$. For aligned noisy speech $x[n]$ and clean speech $s[n]$, noise is estimated as $x[n] - s[n]$.",
            'resources': []
        },
        "NOIZEUS": {
            'summary': "Primary dataset used for the current speech-enhancement benchmark.",
            'definition': "NOIZEUS is a speech corpus with noisy recordings across several real-world noise types and SNR conditions, with clean references.",
            'references': [
                "Official dataset page: https://ecs.utdallas.edu/loizou/speech/noizeus/",
                "Y. Hu and P. C. Loizou, \"Subjective evaluation and comparison of speech enhancement algorithms,\" Speech Communication, vol. 49, no. 7-8, pp. 588-601, 2007."
            ],
            'notes': "NOIZEUS fits the current project benchmark well because it provides matched clean references for objective scoring.",
            'resources': []
        },
        "Speech Enhancement: Theory and Practice": {
            'summary': "Core textbook reference used to frame the classical enhancement families implemented in the project.",
            'definition': "Loizou's textbook is a standard reference for spectral subtraction, Wiener filtering, MMSE estimators, subspace methods, metrics, and benchmark practice in speech enhancement.",
            'references': [
                "P. C. Loizou, Speech Enhancement: Theory and Practice, 2nd ed., CRC Press, 2013."
            ],
            'notes': "This book is the main project-wide theoretical reference when describing the classical algorithm families and evaluation methodology.",
            'resources': []
        },
        "VoiceBank+DEMAND": {
            'summary': "Canonical reference benchmark for modern single-channel speech enhancement.",
            'definition': "VoiceBank+DEMAND combines clean VoiceBank/VCTK speech with DEMAND noise under a standardized additive-mixing protocol and fixed benchmark splits.",
            'references': [
                "Emergent Mind benchmark summary: https://www.emergentmind.com/topics/voicebank-demand-dataset-6a395fe3-bd1f-41ee-919b-0986fce1d1f3"
            ],
            'notes': "This benchmark is a strong model for organizing your benchmark section because it makes task, split, protocol, metrics, and reported baselines explicit. It is highly relevant for your single-channel speech-enhancement priority, but it should stay in a separate lane from NOIZEUS.",
            'resources': []
        },
        "Spectral Subtraction": {
            'summary': "Classic baseline family for speech enhancement.",
            'definition': "Spectral subtraction estimates noise from non-speech segments and subtracts it from the noisy magnitude spectrum.",
            'references': [
                "Boll, 1979: https://ieeexplore.ieee.org/document/1163209",
                "Berouti et al., 1979: https://ieeexplore.ieee.org/document/1163208",
                "Kamath and Loizou, 2002: https://ieeexplore.ieee.org/document/1004105",
                "Loizou, Speech Enhancement: Theory and Practice, 2nd ed."
            ],
            'notes': "Useful as a baseline family in your project benchmark because it is simple, interpretable, and common in the literature. The project now includes classic, adaptive over-subtraction, power-domain, generalized-domain, and multi-band variants.",
            'resources': [
                "Presentation PDF: spectral_subtraction_presentation"
            ]
        },
        "Wiener Filtering": {
            'summary': "Classical minimum-mean-square-error-inspired baseline for speech enhancement.",
            'definition': "Wiener filtering estimates a frequency-dependent gain from the relative speech and noise power and attenuates frequency bins where the estimated noise dominates.",
            'references': [
                "Loizou, Speech Enhancement: Theory and Practice, 2nd ed.",
                "Lim and Oppenheim, IEEE ASSP 1979: https://ieeexplore.ieee.org/document/1163203",
                "Scalart and Vieira Filho, 1996: https://ieeexplore.ieee.org/document/502009"
            ],
            'notes': "Wiener filtering is a strong classical baseline for this project because it is widely cited, conceptually clean, and close to many later statistical enhancement methods. The dashboard now includes both a direct short-time Wiener baseline and a decision-directed recursive variant.",
            'resources': []
        },
        "Statistical Model-Based Enhancement": {
            'summary': "Classical enhancement family based on minimum mean-squared error spectral estimation.",
            'definition': "Statistical model-based enhancement methods estimate spectral amplitudes or log-spectral amplitudes from noisy observations under explicit probability and signal-to-noise-ratio models.",
            'references': [
                "Ephraim and Malah, 1984: https://ieeexplore.ieee.org/document/1164317",
                "Ephraim and Malah, 1985: https://ieeexplore.ieee.org/document/1164550",
                "Cohen and Berdugo, 2001: https://ieeexplore.ieee.org/document/941023"
            ],
            'notes': "This family covers MMSE-STSA, Log-MMSE, and OM-LSA. These methods are important because they connect classical Wiener logic, recursive a priori signal-to-noise estimation, and perceptual gain design.",
            'resources': []
        },
        "Noise Estimation": {
            'summary': "Shared backend family used to track time-varying noise power spectra.",
            'definition': "Noise estimation algorithms update a noise power spectral density over time so enhancement gains can adapt beyond a fixed initial silence model.",
            'references': [
                "Martin, 2001: https://ieeexplore.ieee.org/document/929863",
                "Cohen, 2002: https://ieeexplore.ieee.org/document/1045580",
                "Cohen, 2003: https://ieeexplore.ieee.org/document/1237439"
            ],
            'notes': "The project now includes stationary, minimum-statistics, minima-controlled recursive averaging, and improved minima-controlled recursive averaging noise estimators as reusable modules for the Wiener and statistical families.",
            'resources': []
        },
        "Subspace Enhancement": {
            'summary': "Model-based denoising family using signal and noise subspaces.",
            'definition': "Subspace enhancement relies on matrix decompositions to reconstruct speech from an estimated signal subspace while suppressing noise.",
            'references': [
                "Ephraim and Van Trees, 1995: https://ieeexplore.ieee.org/document/410222",
                "Huang and Huang, 2001: https://ieeexplore.ieee.org/document/969612",
                "Loizou, Speech Enhancement: Theory and Practice, 2nd ed."
            ],
            'notes': "This family is benchmark-compatible with spectral subtraction because the end task is still single-channel speech enhancement. The project now includes both a simple singular-value truncation baseline and a stronger signal-subspace projection variant.",
            'resources': []
        }
    }