[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
<p><a href=""> <img src="https://img.shields.io/github/stars/koo-ec/Awesome-LLM-Explainability?style=flat-square&logo=github" alt="GitHub stars"></a>
<a href=""> <img src="https://img.shields.io/github/forks/koo-ec/Awesome-LLM-Explainability?style=flat-square&logo=github" alt="GitHub forks"></a>
<a href=""> <img src="https://img.shields.io/github/issues/koo-ec/Awesome-LLM-Explainability?style=flat-square&logo=github" alt="GitHub issues"></a>
<a href=""> <img src="https://img.shields.io/github/last-commit/koo-ec/Awesome-LLM-Explainability?style=flat-square&logo=github" alt="GitHub Last commit"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://standardjs.com"><img src="https://img.shields.io/badge/code_style-standard-brightgreen.svg" alt="Standard - \Python Style Guide"></a></p> 



# Awesome-LLM-Explainability
<p align="justify">A curated list of explainability-related papers, articles, and resources focused on Large Language Models (LLMs). This repository aims to provide researchers, practitioners, and enthusiasts with insights into the explainability implications, challenges, and advancements surrounding these powerful models.</p>

## 🚧 This repository is under construction (with daily updates) 🚧

## Table of Contents
- [Awesome LLM-Explainability](#️awesome-llm-xai)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#intro)
  - [Webinars](#Webinars)
  - [Articles](#papers)
    - [📑Papers](#papers)
    - [📖Articles, and Presentations](#articles)
    - [Other](#other)
  - [Datasets \& Benchmark](#datasets--benchmark)
    - [📑Papers](#papers-4)
    - [📖Tutorials, Articles, Presentations and Talks](#tutorials-articles-presentations-and-talks-5)
    - [📚Resource](#resource)
    - [Other](#other-5)
  - [Contributors](#contributors)

<a id ="intro"></a>
# Introduction
<p align="justify">We've curated a collection of the latest 📈, most comprehensive 📚, and most valuable 💡 resources on large language model explainability (LLM Explainability)). But we don't stop there; included are also relevant talks, tutorials, conferences, news, and articles. Our repository is constantly updated to ensure you have the most current information at your fingertips.</p>

<a id="webinars"></a>
# Webinars

## Recorded Videos
* [LLM Explainability or Controllability Improvements with Tensor Networks](https://www.youtube.com/watch?v=dj9O9w16VzQ), ChemicalQDevice, March 28.
* [AI Explained: Inference, Guardrails, and Observability for LLMs](https://www.fiddler.ai/webinars/ai-explained-inference-guardrails-observability-for-llms)

## Events
* [LLM Explainability, Mitigating Hallucinations & Ensuring Ethical Practices](https://www.eventbrite.de/e/llm-explainability-mitigating-hallucinations-ensuring-ethical-practices-tickets-856248691887), April 2nd, 5:30 - 9pm CEST, Berlin.

<a id="papers"></a>
# Papers

## LLM Explainability Evaluation
| Date | Institute        | Publication | Paper Title | Code |
|------|------------------|-------------|-------------|------|
| 2023 | Tsinghua University | [Arxiv](https://arxiv.org/pdf/2311.18702v1.pdf) | Scaling LLM-as-Critic for Effective and Explainable Evaluation of Large Language Model Generation | [GitHub](https://github.com/THUDM/AlignBench/blob/master/README-en.md)|
| 2023 | UC Brekley | [NIPS23](https://arxiv.org/pdf/2306.05685v4.pdf) | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | [GitHub](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) |

## Neural Network Analysis
| Date | Institute        | Publication | Paper Title |
|------|------------------|-------------|-------------|
| 2023 | MIT/Harvard      | [Arxiv](https://arxiv.org/pdf/2305.01610.pdf) | Finding Neurons in a Haystack: Case Studies with Sparse Probing |
| 2023 | UoTexas/DeepMind | [Arxiv](https://arxiv.org/pdf/2310.04625.pdf) | Copy Suppression: Comprehensively Understanding an Attention Head |
| 2023 | UCL              | [Arxiv](https://arxiv.org/pdf/2304.14997.pdf)     | Towards Automated Circuit Discovery for Mechanistic Interpretability |
| 2023 | OpenAI           | [OpenAI Publication](https://openai.com/research/language-models-can-explain-neurons-in-language-models) | Language models can explain neurons in language models |
| 2023 | MIT              | [NIPS23](https://openreview.net/forum?id=RSGmZ7HZaA)     | Toward a Mechanistic Understanding of Stepwise Inference in Transformers: A Synthetic Graph Navigation Model |
| 2023 | Cambridge | [Arxiv](https://arxiv.org/pdf/2312.09230.pdf) | Successor Heads: Recurring, Interpretable Attention Heads In The Wild |
| 2023 | Meta | [Arxiv](https://arxiv.org/pdf/2309.04827.pdf) | Neurons in Large Language Models: Dead, N-gram, Positional |
| 2023 | Redwood/UC Berkeley | [Arxiv](https://arxiv.org/pdf/2211.00593.pdf) | Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small |
| 2023 | Microsoft | [Arxiv](https://arxiv.org/pdf/2305.09863.pdf) | Explaining black box text modules in natural language with language models |
| 2023 | ApartR/Oxford | [ICLR23](https://openreview.net/forum?id=ZB6bK6MTYq) | N2G: A Scalable Approach for Quantifying Interpretable Neuron Representations in Large Language Models |
| 2023 | --- | [Blog](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) | Interpreting GPT: the Logit Lens |

## Algorithmic Approaches
| Date       | Institute | Publication | Paper Title |
|------------|-----------|-------------|-------------|
| YYYY-MM-DD | Institute | Journal     | Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias |
| YYYY-MM-DD | Institute | Journal     | Discovering Latent Knowledge in Language Models Without Supervision |
| YYYY-MM-DD | Institute | Journal     | Towards Monosemanticity: Decomposing Language Models With Dictionary Learning |
| YYYY-MM-DD | Institute | Journal     | Spine: Sparse interpretable neural embeddings |
| YYYY-MM-DD | Institute | Journal     | Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors |
| YYYY-MM-DD | Institute | Journal     | Sparse Autoencoders Find Highly Interpretable Features in Language Models |
| YYYY-MM-DD | Institute | Journal     | Attribution Patching: Activation Patching At Industrial Scale |
| YYYY-MM-DD | Institute | Journal     | Causal Scrubbing: a method for rigorously testing interpretability hypotheses [Redwood Research] |

## Representation Analysis
| Date       | Institute | Publication | Paper Title |
|------------|-----------|-------------|-------------|
| YYYY-MM-DD | Institute | Journal     | Linear Representations of Sentiment in Large Language Models |
| YYYY-MM-DD | Institute | Journal     | Emergent Linear Representations in World Models of Self-Supervised Sequence Models |
| 2023 | MIT/Standford/Oxford | [Arxiv](https://arxiv.org/pdf/2310.07837.pdf) | Measuring Feature Sparsity in Language Models |
| YYYY-MM-DD | Institute | Journal     | Polysemanticity and capacity in neural networks |
| 2019 | Google/Cambridge | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2019/file/159c1ffe5b61b41b3c4d8f4c2150f6c4-Paper.pdf) | Visualizing and measuring the geometry of BERT |
| YYYY-MM-DD | Institute | Journal     | The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets |
|2021 |  | Conference  | Attention is not all you need: pure attention loses rank doubly exponentially with depth|
|2019 |  | [arXiv](https://arxiv.org/pdf/1907.07355.pdf)  | Probing neural network comprehension of natural language arguments|
|2024 |  | [arXiv](https://arxiv.org/pdf/2406.05644.pdf)| How Alignment and Jailbreak Work: Explain LLM Safety through Intermediate Hidden States|
|2024 |  | [arXiv](https://arxiv.org/pdf/2409.04808.pdf) | HULLMI: Human vs LLM identification with explainability|
|2024 |  | [arXiv](https://arxiv.org/pdf/2406.12235.pdf) | Holmes-VAD: Towards Unbiased and Explainable Video Anomaly Detection via Multi-modal LLM|
|2024 |  | [arXiv](https://arxiv.org/pdf/2408.13006v1.pdf) | Systematic Evaluation of LLM-as-a-Judge in LLM Alignment Tasks: Explainable Metrics and Diverse Prompt Templates |
|2023 |  | [arXiv](https://arxiv.org/pdf/2311.18702v2.pdf) |  CritiqueLLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Model Generation |






## Bias and Robustness Studies
| Date       | Institute | Publication | Paper Title |
|------------|-----------|-------------|-------------|
| YYYY-MM-DD | Institute | Journal     | Large Language Models Are Not Robust Multiple Choice Selectors |
| YYYY-MM-DD | Institute | Journal     | The Devil is in the Neurons: Interpreting and Mitigating Social Biases in Language Models |
| YYYY-MM-DD | Institute | Journal     | ChainPoll: A High Efficacy Method for LLM Hallucination Detection |
| 2023 | PrincetonU | [Online Presentation](https://www.cs.princeton.edu/~arvindn/talks/evaluating_llms_minefield/) | Evaluating LLMs is a minefield |

## Interpretability Frameworks
| Date       | Institute | Publication | Paper Title |
|------------|-----------|-------------|-------------|
| YYYY-MM-DD | Institute | Journal     | Let's Verify Step by Step |
| YYYY-MM-DD | Institute | Journal     | Interpretability Illusions in the Generalization of Simplified Models |
| YYYY-MM-DD | Institute | Journal     | Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling |
| 2024 | Polytechnique Montreal | [Arxiv](https://arxiv.org/pdf/2401.07927.pdf)     | Can Large Language Models Explain Themselves? |
| YYYY-MM-DD | Institute | Journal     | A Mechanistic Interpretability Analysis of Grokking |
| YYYY-MM-DD | Institute | Journal     | 200 Concrete Open Problems in Mechanistic Interpretability |
| YYYY-MM-DD | Institute | Journal     | Interpretability at Scale: Identifying Causal Mechanisms in Alpaca |
| YYYY-MM-DD | Institute | Journal     | Representation Engineering: A Top-Down Approach to AI Transparency |
| 2023 | UC Berkeley | [Nature Communication](https://www.nature.com/articles/s41467-023-43713-1) | Augmenting Interpretable Models with LLMs during Training |

## Application-Specific Studies
| Date       | Institute | Publication | Paper Title |
|------------|-----------|-------------|-------------|
| YYYY-MM-DD | Institute | Journal     | Emergent world representations: Exploring a sequence model trained on a synthetic task |
| YYYY-MM-DD | Institute | Journal     | How does GPT-2 compute greater than?: Interpreting mathematical abilities in a pre-trained language model |
| YYYY-MM-DD | Institute | Journal     | Interpreting the Inner Mechanisms of Large Language Models in Mathematical Addition |
| YYYY-MM-DD | Institute | Journal     | An Overview of Early Vision in InceptionV1 |

## Theoretical Approaches
| Date       | Institute | Publication | Paper Title |
|------------|-----------|-------------|-------------|
| YYYY-MM-DD | Institute | Journal     | A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations |
| YYYY-MM-DD | Institute | Journal     | The Quantization Model of Neural Scaling |
| YYYY-MM-DD | Institute | Journal     | Toy Models of Superposition |
| YYYY-MM-DD | Institute | Journal     | Engineering monosemanticity in toy models |
| YYYY-MM-DD | Institute | Journal     | A New Approach to Computation Reimagines Artificial Intelligence |
 
<a id="githubs"></a>
# Related GitHub Repositories:
* [Awesome LLM Interpretability ](https://github.com/JShollaj/awesome-llm-interpretability)
* [Explainability-for-Large-Language-Models: A Survey](https://github.com/hy-zhao23/Explainability-for-Large-Language-Models)

<a id="blogs"></a>
# Blogs
* Adly Templeton, et al. (May 2024), [Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model)
* Georgia Deaconu, (December 2023) [Towards LLM Explainability: Why Did My Model Produce This Output?](https://towardsdatascience.com/towards-llm-explainability-why-did-my-model-produce-this-output-8f730fc73713)

<a id="tools"></a>
# Tools
* [Gemma Scope](https://ai.google.dev/gemma/docs/gemma_scope)
    - Gemma Scope Tutorial: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp)
* [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)

<a id = "contributors"></a>
## Contribution and Collaboration:
Please feel free to check out <a href = "https://github.com/koo-ec/Awesome-LLM-Explainability/blob/main/.github/CONTRIBUTING.md">CONTRIBUTING</a> and <a href = "https://github.com/koo-ec/Awesome-LLM-Explainability/blob/main/.github/CODE_OF_CONDUCT.md">CODE-OF-CONDUCT</a> to collaborate with us.

## Future Research Directions
* One future direction is Fairness-Explainability Evaluation for LLMs.
