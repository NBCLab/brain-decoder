# **Response to Reviewers: Manuscript NCOMMS-25-63915-T**

We thank the Reviewers for their helpful and constructive comments regarding our manuscript.

The Reviewers brought up several valuable critiques of the manuscript, which we have addressed in the revised manuscript. Below, these concerns are addressed on a point-by-point basis. Revisions to the manuscript are shown in red font.

# **Reviewer 1 Comments**

Summary：  
This paper introduces NiCLIP, a neuroimaging contrastive language-image pretraining model, aiming to predict cognitive tasks, concepts, and domains from brain activation patterns. The authors trained NiCLIP on over 23,000 neuroscientific articles, leveraging large language models (LLMs) and deep contrastive learning to establish text-to-brain associations. Key findings indicate that fine-tuned LLMs and the use of full-text articles with a curated cognitive ontology optimize NiCLIP's predictive accuracy. Evaluations, particularly with group-level activation maps from the Human Connectome Project, show NiCLIP's capability to accurately predict cognitive tasks across various domains (e.g., emotion, language, motor) and characterize the functional roles of specific brain regions. However, the model exhibits limitations with noisy subject-level activation maps. While presented as a significant advancement for hypothesis generation in neuroimaging, my assessment is that despite the extensive literature data and thorough experimental validation demonstrating the model's effectiveness in reverse-decoding brain states, the article's innovation is insufficient. The comparison with baseline methods appears to be not entirely fair, and the discussion of future prospects lacks both depth and convincing evidence.

We appreciate Reviewer \#1's comments and their recognition that NiCLIP demonstrates "extensive literature data and thorough experimental validation" for decoding brain maps. We address below the concerns regarding innovation, experimental comparisons, and application prospects.

## **Reviewer 1 Comment 1**

1\. Lack of Innovation  
The training pipeline for this paper is nearly identical to that of NeuroConText, with only minor differences in the inference stage. While NeuroConText indexes corresponding brain activity from text, this paper indexes text descriptions from a knowledge graph based on brain activation. The authors highlight the "lack of validation for reverse inference (brain imaging to text) in functional decoding" as a primary shortcoming of previous work. However, once a brain-text contrastive learning model is trained, bidirectional decoding (both brain-to-text and text-to-brain) becomes feasible. Therefore, this cannot be considered a distinctive contribution of the current work.

### **Response**

We thank the reviewer for this observation. We acknowledge that both NiCLIP and NeuroConText share a common foundation: using a CLIP-based contrastive learning framework to align brain activation maps with text in a shared latent space. We stated in the Introduction and Methods section about such similarities:

“The CLIP model architecture (Fig. 4A) adheres to the identical settings employed in the NeuroConText framework (Meudec et al., 2024).”

While the reviewer is correct that a trained CLIP model in principle enables bidirectional retrieval, we emphasize that \*\*the contribution of NiCLIP is not simply "reversing the arrow" of inference\*\*. Rather, the novelty lies in the combined integration of three specific advances that together constitute a qualitatively different system for functional decoding:

1\. Ontology-driven Bayesian decoding framework. NeuroConText operates purely data-driven: it retrieves the nearest text embedding in latent space. In contrast, NiCLIP introduces a structured Bayesian decoding pipeline (Section 5.3) where the CLIP similarity is used as the likelihood P(A|T) within Bayes' theorem, combined with literature-derived priors P(T), to compute posterior probabilities P(T|A) for tasks. This posterior is then propagated through a cognitive ontology using the noisy-OR model to predict concepts P(C|A) and domains P(D|A). This hierarchical, ontology-grounded inference is not a feature of NeuroConText and represents a fundamentally different approach to interpreting brain activation patterns.

2\. Domain-specific LLMs (BrainGPT). NiCLIP is the first to systematically evaluate and demonstrate the benefit of neuroscience-specific fine-tuned LLMs (BrainGPT-7B-v0.1 and v0.2) versus their base models (Llama-2 and Mistral) for text-to-brain association. Our results show that BrainGPT-7B-v0.2 provides superior text-to-brain associations compared to general-purpose LLMs (Table 1).

3\. Integration of the Cognitive Atlas ontology for structured predictions. Unlike NeuroConText, which outputs free-text retrievals (i.e., without guardrails), NiCLIP maps predictions to a curated cognitive vocabulary with task-to-concept and concept-to-domain mappings, enabling structured, interpretable outputs at multiple granularity levels. We demonstrated that the choice of ontology significantly impacts decoding accuracy (Table 2).

We will revise the introduction (Lines 33–34 of the current manuscript) to more clearly delineate the shared training pipeline and articulate these distinct contributions:

"While NiCLIP shares the CLIP-based contrastive training framework with NeuroConText (Meudec et al., 2024), it advances beyond text-to-brain retrieval by introducing: (1) a Bayesian decoding framework that integrates CLIP-derived likelihoods with literature priors and structured ontologies for hierarchical reverse inference, (2) the systematic evaluation of neuroscience-specific LLMs for improved text-brain alignment, and (3) the first formal validation of ontology-driven functional decoding in a contrastive learning framework."

## **Reviewer 1 Comment 2**

Experimental Issues

## **Reviewer 1 Comment 2.1**

Why was NeuroConText not included as a baseline for comparison in this study?.

### **Response**

NeuroConText and NiCLIP address fundamentally different directions of the brain-text mapping problem. NeuroConText is designed to predict brain activation maps from text (text → brain), whereas NiCLIP is designed to predict text from brain activation maps (brain → text). This asymmetry means that a direct decoding comparison is not feasible, as the two models solve different tasks.

However, a meaningful comparison is possible at the level of text-to-brain association (i.e., the quality of the learned brain-text embedding space). In Table 1, we present results comparing our best-performing setting using BrainGPT embeddings against the best-performing base model setting from NeuroConText. This comparison demonstrates that NiCLIP achieves superior text-to-brain association performance.

We will expand this discussion in the revised manuscript to clarify the complementary nature of the two approaches: NeuroConText is capable of generating brain maps from arbitrary text descriptions, while NiCLIP provides structured Bayesian decoding from brain maps to cognitive tasks, concepts, and domains grounded in a cognitive ontology.

**Section:**   
	**Page** 

On average,.

 

## **Reviewer 1 Comment 2.2**

Figures 2 and 3 only present titles for the descriptions of each brain region. Given that the model utilized full-text information during training, why are more detailed descriptions not provided?.

### **Response**

NiCLIP's decoding framework deliberately predicts from a structured vocabulary of Cognitive Atlas tasks rather than generating open-ended text. This was a purposeful design choice to set guardrails on the model's predictions and improve interpretability. While open-text prediction is an interesting direction, unconstrained text generation from brain maps is prone to hallucination, making the resulting predictions difficult to interpret and validate. By connecting predictions to a curated cognitive ontology, each decoded label carries a well-defined meaning with associated concept mappings and domain classifications.

The predictions displayed in Figures 2 and 3 are Cognitive Atlas task names because the decoding stage computes posterior probabilities for each task in the vocabulary, where the task embedding is the weighted combination of its name and definition (Eq. in Section 5.3, λ \= 0.5). While training uses full-text articles, the decoding projects task names and definitions into the shared latent space to compute similarities.

To address the reviewer's concern, in the revision we will add a supplementary table listing all predicted tasks with their full Cognitive Atlas definitions for each decoded map, allowing readers to appreciate the richer semantic content driving the predictions.

	**Section: Results**  
	**Page 26**

Together.

## **Reviewer 1 Comment 2.3**

The paper states that "the most commonly used 'association-based decoders' are not based on formal models, cannot identify underlying structures related to specific cognitive processes, and lack sensitivity to unseen brain patterns." However, I found no evidence in this manuscript to support the claim that NiCLIP can interpret unseen brain patterns.

### **Response**

We acknowledge that this claim requires clearer evidence. The statement about "unseen brain patterns" refers to NiCLIP's capacity to generalize beyond the specific brain maps present in its training data, which are modeled activation maps from PubMed coordinates. Specifically:

1\. Generalization to unseen map types. The HCP group-level statistical maps used for evaluation (Section 2.3, Figures 2 and 3\) are whole-brain t-statistic maps that are structurally different from the MKDA-modeled coordinate maps used for training. The fact that NiCLIP correctly predicts tasks from these unseen map types (e.g., Motor task predicted at 48.3%, Relational processing at 54.5%) demonstrates generalization.

2\. ROI-based decoding. The six ROI maps (Figure 3\) represent a qualitatively different input type, binary/sparse masks rather than continuous statistical maps, yet NiCLIP produces highly selective predictions (e.g., rTPJ → social cognition at 98.5%).

3\. Cross-domain generalization. NiCLIP can predict tasks it was never explicitly trained on, since the vocabulary embeddings are computed from Cognitive Atlas definitions, not from training data labels. Any task with a name and definition in the ontology can be included in the prediction vocabulary.

In the revision, we will:  
\- Add a dedicated subsection validating generalization by evaluating NiCLIP on additional datasets beyond HCP (see response to point 2(5) below).  
\- Clarify the wording to: \*"NiCLIP's self-supervised CLIP architecture learns a shared latent space from text-image pairs, enabling it to generalize to brain maps and cognitive tasks not present in the training data, including different map types (statistical maps vs. modeled coordinates) and novel vocabulary terms."\*

**Section: Introduction**  
	**Page 2**

Recent .

## **Reviewer 1 Comment 2.4**

In Figure 2, the model is shown to identify the current task (e.g., emotion) based on brain activation states. I am curious whether the model can further identify specific emotions, such as amusement or disgust.

### **Response**

This is an excellent question. The current Cognitive Atlas ontology includes some specific emotion tasks (e.g., "emotion regulation task," "emotional face recognition task") and concepts (e.g., "fear," "anger," "disgust," "happiness"), but the granularity of emotion-specific predictions depends on the vocabulary and training data coverage.

For the revision, we will:

1\. Test NiCLIP on emotion-specific contrasts. The HCP emotion processing task includes a "Faces vs. Shapes" contrast that is relatively coarse. We will additionally evaluate NiCLIP on more fine-grained emotion contrasts from other datasets (e.g., the CNP dataset, which we have already performed decoding, or the IBC dataset, which includes multiple emotion-related contrasts).

2\. Expand the Cognitive Atlas vocabulary to include specific emotion concepts and evaluate whether NiCLIP can discriminate between specific emotional states when presented with activation maps from paradigms that target individual emotions.

3\. Discuss the granularity limitation in the revised manuscript, noting that NiCLIP's ability to distinguish fine-grained cognitive states depends on (a) the specificity of the ontology vocabulary and (b) the diversity of training data covering such distinctions.

**Section: Materials and Methods**  
	**Page 8**

Following seg.

## **Reviewer 1 Comment 2.5**

Can this model be generalized across datasets during inference? For example, would it operate effectively on the NSD dataset?.

### **Response**

We agree that demonstrating cross-dataset generalization is essential, and we have now completed this analysis on two additional NeuroVault datasets using the same reduced Cognitive Atlas evaluation interface as for HCP.

1\. Individual Brain Charting (IBC) dataset: We decoded 1,608 IBC maps spanning 11 task families. In the reduced ontology benchmark, NiCLIP achieved task Recall@4 \= 15.67%, concept Recall@4 \= 10.38%, and domain Recall@2 \= 33.74% for the combined task-name+definition vocabulary. Using task names alone, NiCLIP achieved task Recall@4 \= 8.46%, concept Recall@4 \= 6.30%, and domain Recall@2 \= 40.55%. For comparison, on the task-only benchmark the association-based baselines scored 16.36% (GCLDA) and 7.34% (Neurosynth) task Recall@4.

2\. Consortium for Neuropsychiatric Phenomics (CNP) dataset: We evaluated 130 CNP maps in the reduced ontology benchmark, spanning BART, PAM retrieval, SCAP, and Stop Signal tasks. In this harder transfer setting, NiCLIP achieved task Recall@4 \= 0.77%, concept Recall@4 \= 2.56%, and domain Recall@2 \= 20.77% for the combined vocabulary. With task names alone, performance was task Recall@4 \= 0.77%, concept Recall@4 \= 2.05%, and domain Recall@2 \= 6.54%. The task-only baselines scored 1.54% (GCLDA) and 3.08% (Neurosynth) task Recall@4. Task Switching remains available in the full ontology mapping, but is not part of the reduced Menuet et al. task set and was therefore excluded from the reduced evaluation.

These cross-dataset results show that NiCLIP generalizes substantially better to IBC than to CNP. We interpret this as evidence that transfer depends strongly on ontology coverage and on how closely the external task paradigms resemble the task-evoked activation structure represented in the training corpus.

Regarding the Natural Scenes Dataset (NSD), we note that NSD is a visual perception dataset where subjects view naturalistic images. NiCLIP was trained on CBMA data from the fMRI literature, which primarily covers task-based paradigms. The NSD paradigm (passive viewing of natural scenes) is substantially different from the activation patterns in our training data. Applying NiCLIP to NSD would likely reveal the model's capacity to identify visual perception-related concepts but may underperform for scene-specific content, since the training data does not include naturalistic viewing paradigms at scale. We will add this as a discussion point and, if feasible, include a preliminary evaluation.

In the revision, we will include a new cross-dataset generalization subsection reporting these IBC and CNP results and discussing the boundaries of transfer.

**Section: Introduction**  
	**Pages 2-3**

Using

## **Reviewer 1 Comment 3**

Insufficient Credibility Regarding Future Application Prospects

## **Reviewer 1 Comment 3.1**

There is an absence of a user-friendly, one-click web interface for researchers.

### **Response**

We appreciate this practical suggestion. While building a full web interface is outside the scope of this manuscript, we are committed to making NiCLIP accessible to the community. For the revision, we will:

1\. Provide a pip-installable Python package (\`braindec\`) with a simple API that allows decoding in three lines of code:  
   \`\`\`python  
   from braindec.predict import image\_to\_labels  
   results \= image\_to\_labels("my\_brain\_map.nii.gz", ...)  
   \`\`\`

2\. Include a Jupyter notebook tutorial demonstrating a complete decoding workflow from input brain image to task/concept/domain predictions, which can run on Google Colab.

3\. Discuss plans for a web interface in the revised manuscript, noting that a web tool in Neurosynth Compose is planned for future development and will be hosted at a publicly accessible URL.

4\. Release all trained model weights and vocabulary files on the Open Science Framework (OSF), enabling any researcher to run NiCLIP predictions on their own data.

## **Reviewer 1 Comment 3.2**

The authors state that "this paper provides researchers with a powerful tool for hypothesis generation and scientific discovery." I believe the author team should empirically demonstrate this claim by exploring or validating a specific hypothesis within the paper, as I currently do not perceive the utility of NiCLIP in hypothesis generation and scientific discovery.

### **Response**

This is a fair and constructive point. In the revision, we will include a concrete example of hypothesis generation using NiCLIP. Specifically:

\*\*Proposed example: Functional characterization of the striatum.\*\*

In our ROI analysis (Figure 3), NiCLIP predicted a language specialization for the striatum, which is a notable and perhaps surprising finding. We will leverage this prediction as a hypothesis-generation example:

1\. NiCLIP prediction: The striatum shows strong association with semantic processing tasks (52.1%) and language-related concepts (52.2%), with Language (53.6%) as the dominant domain.

2\. Literature validation: We will conduct a targeted literature review and meta-analytic verification. The striatum has indeed been implicated in language functions, including lexical-semantic processing (Crosson et al., 2007; Crinion et al., 2006), bilingual language control (Abutalebi & Green, 2016), and word learning (Shohamy & Adcock, 2010). These findings corroborate NiCLIP's prediction.

3\. Demonstration value: We will present this as a complete cycle: NiCLIP generates a non-obvious prediction → the prediction is validated against existing literature → this demonstrates the model's utility for generating testable hypotheses from brain maps.

Additionally, we will discuss how NiCLIP could be used to characterize under-studied brain regions or novel parcellations where no prior functional annotation exists, directly serving the hypothesis-generation use case.

# **Reviewer 2 Comments**

The paper NCOMMS-25-63915-T entitled "NiCLIP: Neuroimaging contrastive language-image pretraining model for predicting text from brain activation images" presents a framework that leverages existing fMRI literature to learn associations between brain locations and cognitive concepts. This framework uses an ontology to define these concepts and LLMs to extract them from text. The spatial aspect of the model, which involves mapping reported MNI coordinates to brain maps, has been addressed using the PubGet framework. The remaining weak point was extracting the concepts from each individual publication in order to build the associations. The novelty here lies in using LLMs for this task. Furthermore, the concepts are organised within an ontology derived from the Cognitive Atlas. The paper places an important focus on validation, particularly decoding, where images from the Human Connectome Project dataset are decoded into task-specific and contrast-specific concepts.

I enjoyed reading the paper because it offers a significant opportunity to advance coordinate-based meta-analyses in neuroimaging and make a valuable contribution to improving this field with the use of advanced AI technologies: LLMs and contrastive learning. The effort put into validation is particularly noteworthy, as it comprises both map- and ROI-level experiments.  
I would certainly like to see it published in a high-profile venue such as Nature Communications.

However, there is room for improvement that could be addressed in a revision.

We sincerely appreciate Reviewer \#2's enthusiasm for the paper and their expert feedback. The reviewer highlights that NiCLIP offers "a significant opportunity to advance coordinate-based meta-analyses in neuroimaging" and notes the "noteworthy" validation effort. Below we address each concern in detail.

## **Reviewer 2 Comment 1**

Position with respect to the state of the art

While the paper cites a wide range of state-of-the-art contributions, it is uneven in its coverage. For example, several papers cited in the discussion (Mensch et al., 2021, 2017; Menuet et al., 2022; Varoquaux et al., 2018\) are omitted from the introduction, despite their conceptual contributions being relevant to NiClip (emphasis on ontology-based analysis and the large-scale use of the Cognitive Atlas and NeuroVault, emphasis on decoding as the most principled validation approach). In my opinion, the contribution of NiClip is clear and it should not be an issue to acknowledge that some of the core intuitions of this paper have been introduced in previous publications, since these did not contribute to any improvement in CBMA (instead, they were about IBMA). The same applies to (Oudyk et al., 2025).

Moreover, reading the methods section shows that the paper followed the recent contribution by Meudec et al. (2024) quite closely. However, the technical similarity with this prior contribution is not clearly outlined in the introduction. As an online version of the paper is available with some code (https://www.biorxiv.org/content/10.1101/2025.05.23.655707v1.full), I would expect it to perform a formal comparison between the two approaches.

### **Response**

We agree and will substantially revise the introduction to provide a more balanced and comprehensive coverage of the state of the art. Specifically:

\- Mensch et al. (2017, 2021): We will acknowledge their pioneering work on learning neural representations across multiple fMRI studies and their emphasis on supervised decoding as a principled validation approach. Their work demonstrated the value of large-scale aggregation for brain decoding, which is a conceptual foundation of NiCLIP.

\- Menuet et al. (2022): We will discuss their comprehensive decoding framework using NeuroVault statistical maps and Cognitive Atlas, acknowledging that they introduced (a) the reduced Cognitive Atlas ontology that we also employ, (b) per-term accuracy evaluation for decoding, and (c) the use of image-based meta-analysis (IBMA) for decoding. We will position CBMA vs. IBMA as complementary approaches (see response to point 2 below).

\- Varoquaux et al. (2018): We will credit their work on atlases of cognition and their emphasis on principled decoding evaluation.

\- Oudyk et al. (2025): We will incorporate their recent overview of neuroimaging meta-analyses in the introduction.

\- Meudec et al. (2024): We will add a transparent discussion of the architectural similarities with NeuroConText, acknowledging that NiCLIP follows the CLIP training framework introduced therein, while clearly delineating the novel contributions (Bayesian decoding, domain-specific LLMs, ontology integration).

The revised introduction will include a paragraph explicitly contrasting IBMA and CBMA approaches:

"Image-based meta-analysis (IBMA) approaches, which leverage whole-brain statistical maps from repositories like NeuroVault, have demonstrated strong decoding performance by preserving rich spatial information (Mensch et al., 2017, 2021; Menuet et al., 2022; Varoquaux et al., 2018). However, IBMA approaches face limitations in data coverage: despite community efforts, most neuroimaging studies do not share their statistical maps, resulting in sparse and unevenly annotated repositories (Peraza et al., 2025; Salo et al., 2023). In contrast, coordinate-based meta-analysis (CBMA) approaches benefit from the much larger coverage of databases like Neurosynth and BrainMap, which encompass \>30,000 publications. NiCLIP builds on the CLIP framework introduced by NeuroConText (Meudec et al., 2024\) but targets CBMA-based functional decoding, trading the spatial richness of statistical maps for the broader domain coverage of coordinate databases.”  
.

## **Reviewer 2 Comment 2**

The authors have chosen to frame the decoding part in a Bayesian way, which constrains the interpretation of their results and the type of experiments they can conduct to validate the model.  
Ideally, I would like to see a per-term accuracy score for as many terms as possible, as in Menuet et al. (2022). Currently, we don't even have a rough idea of how many concepts, or which ones, can be properly decoded from data or sets of reported locations.

### **Response**

We agree this is important for understanding which cognitive terms NiCLIP can reliably decode, and we have now added a per-term analysis with a within-dataset permutation null baseline.

For the reduced cross-dataset benchmark:

1\. IBC: In the combined vocabulary setting, 6/11 task terms, 17/30 concepts, and 6/9 domains were above chance. In the names-only setting, 4/11 task terms, 7/30 concepts, and 2/9 domains were above chance.

2\. CNP: In the combined vocabulary setting, 0/4 task terms and 0/5 domains were above chance, while 1/12 concepts ("recall") exceeded the permutation baseline. In the names-only setting, no task, concept, or domain terms were above chance.

3\. The above-chance IBC task terms were concentrated in social, emotional, motor, and spatial paradigms rather than being uniformly distributed across the ontology. Examples include emotion processing fMRI task paradigm, emotional localizer fMRI task paradigm, motor fMRI task paradigm, social cognition (theory of mind) fMRI task paradigm, Social localizer fMRI task paradigm, and spatial localizer fMRI task paradigm.

We will present these per-term results as a heatmap/bar chart in the revision and discuss the term-specific heterogeneity explicitly, rather than relying only on averaged benchmark scores.

## **Reviewer 2 Comment 2.1**

The authors have chosen to base their decoding validation on the HCP dataset. I have several issues with this:

\* First, I did not understand how the decoding was carried out at the task level. There are 22 maps in Collection 457\. How do we go from there to task-level maps? Also, I did not understand how the task-to-concept mapping was done in the method section. Is it simply the Cognitive Atlas, or an improved version of it? Can this be made explicit?

### **Response**

We apologize for the lack of clarity. We will revise the Methods section to explicitly describe the mapping procedure:

Collection 457 in NeuroVault contains 22 contrast maps from 7 HCP task domains. For task-level decoding, we selected one representative contrast per task domain that best captures the core cognitive process:

Emotion: Faces vs. Shapes  
Gambling: Reward vs. Baseline  
Language: Story vs. Math  
Motor: Average (all movements)  
Relational: Relational vs. Match  
Social: TOM vs. Random  
Working Memory: 2-Back vs. 0-Back

The task-to-concept mapping relies on the Cognitive Atlas. Each HCP task (e.g., "Emotion processing fMRI task paradigm") has a corresponding entry in the Cognitive Atlas with associated concepts and domains. We used the reduced Cognitive Atlas ontology (derived from Menuet et al., 2022\) for this mapping. We will make this explicit in a revised Methods subsection and provide the full mapping table as supplementary material.

**Section: Results**  
	**Page 20**

Overall, 

## **Reviewer 2 Comment 2.2**

\* The results of the HCP group-level map decoding are disappointing, whether expressed as decoding probabilities or in Fig. 2\. I believe any expert neuroscientist could easily match group maps from each HCP contrast to their label without error. The failure to do so with NiClip suggests that there is still room for improvement in the framework. In my opinion, the main issue is that the authors extract LLM latents from short task and contrast definitions that are not homogeneous with the long texts used to train the model, resulting in a significant shift in covariates between neuroimaging publications and task/contrast descriptions. This phenomenon is described e.g. in https://www.biorxiv.org/content/10.1101/2025.05.23.655707v1.full.

### **Response**

We acknowledge that the HCP group-level maps represent a relatively "easy" benchmark where an expert could likely achieve perfect accuracy by visual inspection. However, we view the HCP evaluation differently:

1\. The goal is not to outperform human experts. A human expert cannot manually classify a high volume of brain images against a comprehensive cognitive ontology; NiCLIP is designed to provide automated, scalable decoding of any brain map against a structured vocabulary, enabling functional interpretation at a scale that manual classification cannot achieve. The HCP serves as a proof-of-concept validation where ground truth is known.

2\. NiCLIP's Recall@4 of 62.86% for tasks (Table 2\) should be interpreted in context: the vocabulary contains hundreds of possible tasks, so even Recall@4 means identifying the correct task among the top 4 out of hundreds of candidates. For domains, NiCLIP achieves 90.48% Recall@2.

3\. The covariate shift issue. We agree with the reviewer that a key limitation is the mismatch between long training texts and short task descriptions used during inference. As described in the expanded NeuroConText preprint (Meudec et al., 2025), there is a distributional shift between LLM embeddings of long publication texts and short task/concept definitions. In NiCLIP, this is partially mitigated by combining the task name and definition embeddings (Eq. in Section 5.3), but we acknowledge that further work on text augmentation strategies (e.g., LLM-based expansion of short definitions to article-like formats, as proposed in Meudec et al., 2025\) could improve performance. We will add this discussion and propose augmentation as a concrete avenue for improvement.

4\. Improvement over baselines. Despite the acknowledged room for improvement, NiCLIP substantially outperforms both Neurosynth and GC-LDA baselines (\>40% improvement in Recall scores, Table 2), demonstrating that the CLIP-based framework provides meaningful advances for functional decoding.

## **Reviewer 2 Comment 2.3**

\* The results on individual HCP data are poor, but this is likely due to the same inadequate matching being exacerbated by the lower SNR and additional variability in the individual data.

### **Response**

We agree that subject-level decoding performance is a clear limitation. As discussed in Section 2.4.3, subject-level maps exhibit high variability and noise compared to group-level maps. However, we note that the observed performance (Recall@4 of 38.19% for tasks, Recall@2 of 52.01% for domains) still exceeds chance, suggesting that NiCLIP captures some individual-level signal.

In the revision, we will:  
1\. Conduct an SNR sensitivity analysis by varying group sizes (n=5, 10, 20, 50, 100, all 787 subjects) and reporting decoding accuracy as a function of effective SNR (see also response to the reviewer's related minor point).  
2\. Discuss the covariate shift issue as a likely contributing factor, as suggested by the reviewer.  
3\. Propose training data augmentation (adding subject-level maps with noise to the training set) as a concrete path to improvement.

## **Reviewer 2 Comment 2.4**

\* Finally, I would like the authors to compare their approach with the IBMA decoding of Menuet et al. (2022). It would be extremely interesting to know which decoding approach works best. I actually think that IBMA outperforms...

### **Response**

We appreciate the interest in this comparison. However, we note that a direct comparison on the HCP benchmark would not be fair. Menuet et al. (2022) trained their IBMA decoder on NeuroVault statistical maps, which include HCP-derived images in their training set. In contrast, NiCLIP was not trained on HCP data, we use HCP exclusively as a held-out evaluation benchmark. Comparing the two methods on HCP would therefore favor the IBMA decoder due to data leakage.

That said, a key advantage of NiCLIP's architecture is its scalability. Because NiCLIP is trained on coordinate-based data from \~23,865 publications, it already covers a far broader range of cognitive tasks and domains than what is available in NeuroVault. Moreover, NiCLIP's training set can be easily augmented with image-based data (statistical maps) in the future, combining the breadth of coordinate databases with the richer spatial information of full brain maps. The IBMA approach, by contrast, remains constrained by the limited and inconsistently annotated collection of maps in NeuroVault.

We will discuss this tradeoff in the revised manuscript, positioning the two approaches as complementary: IBMA provides richer per-map information when high-quality statistical maps are available, while NiCLIP offers broader coverage and a scalable framework that can incorporate both coordinate and image data.

## **Reviewer 2 Comment 3**

Additionally, I would like to see validation on datasets other than HCP. Some of the CogAt entries seem to have been designed for HCP in particular, which creates a kind of circularity. Alternative, comprehensive datasets have been shared on NeuroVault, for example.

### **Response**

This is an important methodological concern. We address the potential circularity and cross-dataset validation:

Regarding circularity: The Cognitive Atlas is a community-driven ontology developed independently of HCP. While some HCP tasks have corresponding entries in the Cognitive Atlas, the ontology was not designed specifically for HCP. The reduced Cognitive Atlas vocabulary we use (from Menuet et al., 2022\) contains tasks from diverse sources, not exclusively HCP. Furthermore, NiCLIP's training data consists of \~23,865 PubMed articles spanning the entire fMRI literature, not HCP-specific publications.

Cross-dataset validation: As noted in our response to Reviewer 1 (point 2(5)), we now include decoding results on:  
\- IBC dataset (Individual Brain Charting): 1,608 reduced-ontology maps spanning 11 task families, with task Recall@4 \= 15.67% and domain Recall@2 \= 33.74% in the combined NiCLIP setting  
\- CNP dataset (Consortium for Neuropsychiatric Phenomics): 130 reduced-ontology maps spanning BART, PAM retrieval, SCAP, and Stop Signal, with task Recall@4 \= 0.77% and domain Recall@2 \= 20.77% in the combined NiCLIP setting

We now have runnable decoding pipelines for both datasets (\`jobs/decoding\_ibc.py\`, \`jobs/decoding\_cnp.py\`) and will present these results in a new Results subsection.  
 

**Section: Materials and Methods**  
	**Page 16**

In Fig. S5, we re

## **Reviewer 2 Comment 4**

The authors introduce a reduced version of the Cognitive Atlas, but it is difficult to ascertain how it differs from the original. Providing more details on the motivations and actual differences is important. I am actually puzzled by the necessity to reduce Cognitive Atlas: what do the authors mean? What did they actually do? 

In the discussion, the authors state, 'We demonstrated that a reduced and curated representation of the Cognitive Atlas tasks, combined \[...\]  
with a more robust and comprehensive mapping of concepts, outperforms the original Cognitive Atlas ontology." However, this is not clear from their experimental results. 

### **Response**

We will substantially expand the description of the reduced Cognitive Atlas in the revised Methods. Specifically:

Motivation: The original Cognitive Atlas contains 851 tasks and 912 concepts, many with incomplete definitions, missing task-concept mappings, or inconsistent annotations. For instance, some popular tasks like "motor fMRI task paradigm" are only linked to a few concepts (e.g., only "working memory"), missing obvious associations (e.g., "movement," "motor control"). This incompleteness propagates to the concept and domain predictions.

What was done: Following Menuet et al. (2022), we used a curated subset that:  
\- Retains \~100 of the most commonly used fMRI tasks based on their prevalence in the literature.  
\- Manually enriches task-to-concept mappings to ensure that each task is associated with all relevant cognitive concepts. For example, the motor fMRI task paradigm is linked to "movement," "motor control," "motor learning," and other motor-related concepts in the reduced version.  
\- Adds manual concept-to-domain mappings for concepts missing domain annotations (16 manual mappings, listed in our code at \`braindec/cogatlas.py\`, Lines 26–43).  
\- Maintains the original concept-to-domain structure using the 10 cognitive process categories from Cognitive Atlas.

Evidence of improvement: Table 2 demonstrates that the reduced ontology consistently outperforms the original across all models and metrics. For example, with BrainGPT-7B-v0.2 (body, name+definition), Recall@4 for tasks increases from near-zero with the full Cognitive Atlas to 62.86% with the reduced version.

We will add a supplementary table comparing the full vs. reduced Cognitive Atlas (number of tasks, concepts, task-concept edges, concept-domain edges) and provide the complete mapping file as supplementary material.

**Section: Abstract**  
	**Page 1**

Finally,.

## **Reviewer 2 Comment 5**

## Related to point 2, I think the authors should provide visualisations of the geometry of the embeddings so that their structure can be checked..

### **Response**

We agree that embedding visualization would strengthen the paper. In the revision, we will add:

1\. UMAP/t-SNE visualization of the shared latent space showing both text embeddings (Cognitive Atlas tasks, colored by domain) and image embeddings (HCP maps, colored by task), demonstrating the alignment quality in the learned space.

2\. Embedding structure analysis showing:  
   \- Clustering of task embeddings by cognitive domain  
   \- Distance between HCP image embeddings and their corresponding task text embeddings  
   \- Whether semantically related tasks cluster together (e.g., motor tasks, language tasks)

3\. Comparison of embedding geometries across different LLMs (BrainGPT vs. Mistral vs. Llama), providing insight into why BrainGPT achieves better performance.

This will be presented as a new supplementary figure.

**Section: Introduction**  
	**Page 2**

The main goal of parcellating functional connectomes is to reduce high-dimensional connectivity space into a.

## **Reviewer 2 Comment 6**

## Introduction: I suggest discussing the relative merits of IBMA and CBMA more explicitly.

### **Response**

Agreed. We will add a dedicated paragraph in the Introduction comparing IBMA and CBMA for functional decoding (see revised text in response to Major Point 1 above).

## **Reviewer 2 Comment 7**

## In the abstract, the authors emphasize the difference between BrainGPT and other LLMs, but the effect is actually modest, as can be seen in Table 1\.

### **Response**

We acknowledge this. The differences are statistically modest (e.g., Recall@10: 33.56 vs. 33.36 for BrainGPT-7B-v0.2 vs. Mistral). We will revise the abstract to temper this claim:

\> "We demonstrated that domain-specific fine-tuned LLMs (e.g., BrainGPT) provide modestly improved text-to-brain associations compared to their base counterparts, with more pronounced benefits observed in downstream decoding tasks."

We note that the BrainGPT advantage becomes more pronounced in the decoding evaluation (Table 2), where the difference between BrainGPT and base LLMs is larger (e.g., Task Recall@4: 62.86% vs. 55% for BrainGPT-v0.2 vs. v0.1).

## **Reviewer 2 Comment 8**

## Why didn't the authors use the DiFuMo1024 dictionary instead of the DiFuMo512 dictionary?

### **Response**

We chose DiFuMo512 following the NeuroConText framework (Meudec et al., 2024, which also used DiFuMo512) to ensure a fair comparison and because it provides a balance between spatial resolution and model complexity. DiFuMo1024 doubles the image embedding dimension, which would increase training time and may require architectural adjustments. The NeuroConText ablation studies (Meudec et al., 2025\) showed that DiFuMo512 provided a meaningful improvement over DiFuMo256, but we are not aware of published evidence that DiFuMo1024 provides substantial further improvement for text-brain association tasks. We will add this justification to the Methods and note DiFuMo1024 as a future direction.

## **Reviewer 2 Comment 9**

## Could the authors provide more comprehensive captions for the tables in the paper? Currently, it is difficult to identify what the numbers in the tables actually mean.

### **Response**

Agreed. We will expand all table captions to include:  
\- Clear definition of each metric  
\- Description of what rows and columns represent  
\- Explanation of how to interpret the numbers (e.g., "higher is better")  
\- Reference to the specific sections where the experimental setup is described

## **Reviewer 2 Comment 10**

## Regarding Fig. 2, I don't understand how the authors end up with one map per task since tasks generally have more than one possible contrast and sometimes several contrasts of interest (e.g., MOTOR, WM).

### **Response**

See our response to Major Point 2 above regarding the task-level mapping. We selected one representative contrast per task domain. We will make this explicit in the figure caption and Methods section, and provide the full mapping in supplementary material.

## **Reviewer 2 Comment 11**

## I am quite surprised by the finding that the striatum shows a specialization for language. Image decoding results are quite underwhelming overall.

### **Response**

We share the reviewer's initial surprise, but note that the striatum's role in language has been documented in the literature. The caudate nucleus and putamen have been implicated in:  
\- Lexical-semantic processing (Crosson et al., 2007\)  
\- Bilingual language control (Abutalebi & Green, 2016; Crinion et al., 2006\)  
\- Speech production (Bohland & Guenther, 2006\)  
\- Syntactic processing (Ullman, 2004\)

The striatum meta-analytic parcellation we used (Liu et al., 2020\) encompasses both reward-related and language-related subregions, which likely contributes to this prediction. We will add this discussion to the revised Results section and use it as the hypothesis-generation example (see response to Reviewer 1, point 3(2)).

## **Reviewer 2 Comment 12**

## The statement "The current trained model should not be used to decode images with high noise, such as subject-level activation maps, as our decoding model performs poorly on this type of data" could be clarified by varying the signal-to-noise ratio (SNR) in the data and taking group maps from different sample sizes. Nevertheless, as discussed above, the main issue is probably not the noise in the images but rather the difficulty of obtaining good embeddings for contrasts or tasks text descriptions.

### **Response**

We agree this analysis would be informative. In the revision, we will:  
\- Compute group-average maps from subsets of N \= {5, 10, 20, 50, 100, 200, 787} subjects  
\- Report decoding accuracy as a function of group size  
\- Present this as a figure showing the relationship between effective SNR (or sample size) and decoding performance  
\- Discuss whether the primary limitation is SNR or the covariate shift issue

## **Reviewer 2 Comment 13**

## " If one is interested in decoding both ends of the activation distribution in an image separately, one could flip the sign of the image to force the decoder to predict the negative tail." I'm not sure I understand the use case.

### **Response**

We will clarify this use case in the revision:

"In task fMRI, statistical maps often contain both positively activated regions (task \> baseline) and negatively activated regions (baseline \> task). NiCLIP was trained primarily on positively activated coordinates. To decode the functional significance of deactivated regions, a user could invert the sign of the map so that deactivations become positive, then apply NiCLIP to characterize the cognitive processes associated with those deactivated areas. For example, decoding the inverted default mode network deactivation during a working memory task might reveal associations with resting-state or self-referential processing concepts."

## **Reviewer 2 Comment 14**

## Discussion: What does "continuous decoding" mean ?

### **Response**

We will replace "continuous decoding" with more precise language:

"NiCLIP accurately performs functional decoding on whole-brain statistical maps (dense activation maps covering the full brain volume) as well as sparse brain images, such as regions of interest."

## **Reviewer 2 Comment 15**

## In the methods section, the authors define a "likelihood" P(A\_k|T) \= softmax(Emb(T).Emb(Ak))

## This is a heuristic; besides the fact that the result is a number between 0 and 1, it is unclear how it can be interpreted as a proper distribution because it is not calibrated as a proper probability.

### **Response**

The reviewer is correct. The softmax of cosine similarities produces a distribution that sums to 1, but this is not a calibrated probability in the statistical sense. It is a scoring function normalized to produce a probability-like output. We will revise the Methods to clarify this:

"We note that P(A\_k|T) as defined by the softmax of cosine similarities in the CLIP latent space is a heuristic likelihood that quantifies the relative compatibility between activation patterns and task descriptions. While this normalized score shares properties with a probability distribution (non-negative, sums to 1), it is not calibrated as a proper statistical likelihood. Similarly, the posterior P(T|A) should be interpreted as a relative ranking score rather than a rigorously calibrated probability. This is consistent with the common use of softmax-normalized similarity scores in contrastive learning frameworks (Radford et al., 2021)."

## **Reviewer 2 Comment 16**

## The formula for the probability of a concept (C\_j ) given an activation A\_k seems to assume some independence of the P(T\_i |A\_k) probabilities, which likely does not hold. Therefore, these "probabilities" are, at best, a proxy and are not rigorous. This must be made clear.

### **Response**

The reviewer is correct that the noisy-OR model assumes conditional independence of the task probabilities given a concept, which is an approximation. We will add the following clarification:

"The noisy-OR model used to compute P(C\_j|A\_k) assumes conditional independence among tasks that measure the same concept. This assumption is a simplification, as tasks sharing a concept may have correlated activation patterns. The resulting concept and domain probabilities should therefore be interpreted as approximate scores reflecting the aggregate evidence from related tasks, rather than as rigorously derived posterior probabilities. Despite this approximation, the noisy-OR model provides a principled mechanism for propagating task-level predictions through the ontological hierarchy, and its effectiveness is empirically validated by the meaningful concept and domain predictions observed in our results."

# **Reviewer 3 Comments**

Peraza and colleagues present NiCLIP, a CLIP-based neural network model, that learns to decode cognitive terms from coordinate-based brain activation maps. The model outputs predictions based on task, concept, and domain labels from the Cognitive Atlas. This provides a qualitative advance over previous reverse-inference models (like NeuroSynth) by learning a nonlinear mapping between brain images and words, taking advantage of the rich contextual-semantic representations encoded by large language models (LLMs). This manuscript is very timely and already in pretty good shape. The methodology seems solid and I believe the results. That said, I found certain bits of the text were difficult to follow; for example, I had a hard time understanding the distinguishing features of the models, which comparisons are most important, and which elements of NiCLIP are driving improved performance. Most of the following comments are clarification questions or suggestions that I think the authors can readily address.

### **Response**

We thank Reviewer \#3 for their positive assessment ("very timely," "methodology seems solid," "cool paper\!") and their constructive suggestions for improving clarity. We address each point below.

## **Reviewer 3 Comment 1**

In my first read through the manuscript, I had some difficulty following the narrative. I found myself asking questions like “wait, what exactly are the differences between the NiCLIP model and the CLIP model they were discussing in the previous section?” and “what exactly is this model trained and tested on? and is that different from the previous model?” Is the distinguishing feature of the “CLIP” model that it’s text-to-brain, whereas the distinguishing feature of the “NiCLIP” model is that it’s brain-to-text (and also trained on CogAtlas)? Couldn’t you theoretically also decode caption-style text directly from the CLIP model trained on brain images, without the CogAtlas? A good deal of this becomes somewhat clearer upon reading the Methods (at the end), but I think readers would benefit from a little more hand-holding throughout the Results. To be clear, I don’t think this is done poorly at all even in the current version—it’s just that this whole methodology is a complex beast, and very few readers will be familiar enough with all the different components to fully “get it” on the first read. Maybe introducing each section with a question or motivation sentence would help.

### **Response**

We appreciate this feedback about readability. In the revision, we will:

1\. Add a "motivation sentence" at the beginning of each Results subsection, framing the question being addressed. For example:  
   \- Section 2.2: "Before evaluating functional decoding, we first assessed whether the CLIP framework can learn meaningful text-to-brain associations from the neuroimaging literature."  
   \- Section 2.3: "Given a trained text-to-brain CLIP model, can we perform functional decoding, predicting cognitive tasks from brain activation maps?"

2\. Provide a clear conceptual distinction early in the Results:  
   \- CLIP model: The contrastive learning framework that aligns article text embeddings with brain activation map embeddings in a shared latent space. Trained on 23,865 PubMed article-coordinate pairs. Evaluated with Recall@K and Mix\&Match on held-out articles.  
   \- NiCLIP model: The decoding framework that uses the trained CLIP model for reverse inference. Instead of retrieving articles, it computes posterior probabilities for cognitive tasks from a structured ontology (Cognitive Atlas) by comparing their name/definition embeddings to a new brain map embedding in the CLIP latent space.

3\. Add a "Couldn't you also decode caption-style text directly from the CLIP model?" paragraph explaining that yes, one could do nearest-neighbor text retrieval (as NeuroConText does), but NiCLIP's ontology-based approach provides structured, interpretable outputs at the task/concept/domain levels, which are more useful for hypothesis generation..

## **Reviewer 3 Comment 2**

Following on the bit from the previous comment about training/testing, should readers be worried about potential leakage between the PubMed Central data used for training and HCP data used for testing the models? Isn’t it possible that some of the PMC training articles are reporting coordinates derived from exactly the same HCP data you use to test the model? I assume the authors ensure these are non-overlapping somehow (or maybe I just don’t fully understand the structure of the data), but I think this could be made more explicit. Related thought: I assume the articles are effectively randomized regarding topic, so that you don’t end up holding out a large chunk of articles on a single topic for a particular test set?

### **Response**

This is an important methodological concern. We want to clarify:

1\. No coordinate-level leakage. NiCLIP is trained on article text paired with coordinates reported in those articles. Even if some training articles report HCP-derived results, the model does not see the HCP statistical maps during training — it only sees MKDA-modeled coordinate maps, which are a lossy representation of the original data.

2\. Text-level potential overlap. Some PubMed articles may describe HCP task contrasts. However, these articles would describe many different analyses and findings from HCP, not just the specific contrasts we test on. The CLIP model learns general text-brain associations, not memorized mappings from specific contrasts.

3\. Structural safeguard. The evaluation uses group-level t-statistic maps from NeuroVault (Collection 457), which are fundamentally different from the MKDA-modeled coordinate maps used in training. This structural mismatch between training (modeled coordinates) and evaluation (statistical maps) data makes direct leakage unlikely.

4\. Topic randomization. Our 23-fold cross-validation splits articles randomly, so no systematic topic exclusion occurs. Articles are distributed across folds without topic-based stratification.

We will add a paragraph in the Methods discussing potential leakage and these mitigating factors:

\> \*"We note that the PubMed training corpus may include articles analyzing HCP data. However, several factors mitigate potential leakage: (1) the model is trained on MKDA-modeled coordinate maps, which are structurally distinct from the group-level t-statistic maps used for evaluation; (2) even if an article discusses an HCP contrast, the text embedding captures the full article content, not a one-to-one mapping to a specific statistical map; and (3) cross-validation folds are constructed by randomly sampling articles without topic stratification, ensuring no systematic bias toward HCP-related content."\*  
.

**Section: Materials and Methods**  
	**Pages 6-7**

Importantly, . 

## **Reviewer 3 Comment 3**

I had some difficulty understanding how exactly the language component of the model is encoding the text. For example, in my own work with LLMs, we’re often using the time series of word-by-word embeddings to capture the meaning of text. Does a single embedding for an entire article comprising thousands of words really capture all the nuances of meaning (the “deep semantic relationships” the authors advertise in the introduction) in that article? I could understand how a whole trajectory of word-by-word embeddings could capture the narrative of an article in a fairly rich, context-sensitive way—but wouldn’t you lose a good bit of this meaning and structure in collapsing the article into a single embedding?

### **Response**

This is an astute question about representation capacity. We will add a discussion:

1\. How embeddings are computed: For long articles, we chunk the text into segments within the LLM's context window, compute an embedding for each chunk, and then average across chunks (Section 5.1.1). This mean-pooling approach does lose sequential/narrative structure but preserves the aggregate semantic content.

2\. Why this works for our application: We are not trying to capture the narrative arc of an article. Rather, we need an embedding that represents \*what cognitive topics and brain regions the article discusses\*. Mean-pooled LLM embeddings have been shown to effectively capture document-level topics and semantic content in information retrieval tasks.

3\. Empirical evidence: Our Table 1 results show that full-text (body) embeddings substantially outperform abstract-only embeddings (Recall@10: 33.56 vs. 24.01), indicating that the additional text does provide meaningful discriminative information beyond what's in the abstract.

4\. Acknowledged limitation: We agree that richer text representations (e.g., attention-weighted pooling, multi-vector embeddings) could capture more nuanced semantic content. We will note this as a future direction.  
.

**Section: Discussion**  
	**Page 34**

We recog.

## **Reviewer 3 Comment 4**

Table 2 is very dense. Can you hold the reader’s hand a bit more as to which numbers we should be comparing? For example, am I correct in understanding that GC-LDA task Recall@4 (17.14) outperforms NiCLIP task Recall@4 (10.71)? Isn’t this comparison a bit surprising?

### **Response**

We will substantially revise Table 2 for clarity:

1\. Add a more detailed caption explaining what each row, column, and number represents.  
2\. Highlight key comparisons (e.g., bold the best NiCLIP configuration, add asterisks for significant improvements over baselines).  
3\. Add a summary row showing the best NiCLIP configuration vs. baselines.

Regarding the specific comparison: If the reviewer is comparing GC-LDA task Recall@4 (17.14) with a specific NiCLIP configuration that shows 10.71, this is likely a configuration using the \*\*full (uncurated) Cognitive Atlas\*\* ontology, which performs poorly due to incomplete mappings. The best NiCLIP configuration (body \+ BrainGPT-7B-v0.2 \+ reduced CogAt \+ name+definition) achieves \*\*62.86% task Recall@4\*\*, which substantially outperforms GC-LDA's 17.14%. The key message is that ontology choice dramatically affects performance. We will add a note in the text explicitly guiding readers to the most important comparisons.

.

**Section: Results**  
	**Page 20**

Overal

## **Reviewer 3 Comment 5**

The authors mention bag-of-words methods like TF-IDF in the Introduction, suggesting that LLMs will improve on this method. This set me up to expect a comparison to TF-IDF—but then I didn’t see it directly mentioned. Is the NeuroSynth baseline model effectively using TF-IDF?

### **Response**

Yes, the Neurosynth correlation decoder effectively uses TF-IDF. Neurosynth extracts term frequencies from article abstracts using automated text mining and creates meta-analytic maps by aggregating coordinates associated with each term — this is functionally equivalent to a TF-IDF-based approach. The GC-LDA baseline also uses term counts (a simpler form of bag-of-words).

We will make this connection explicit in the Results:

"Both baseline models rely on bag-of-words text representations: the Neurosynth correlation decoder uses automated term extraction akin to TF-IDF (Yarkoni et al., 2011), while GC-LDA uses raw term counts (Rubin et al., 2017). NiCLIP's substantial improvement over these baselines demonstrates the benefit of replacing bag-of-words representations with LLM-derived contextual embeddings."

**Section: Limitations**  
	**Page 34-35**

Second, the. 

## **Reviewer 3 Comment 6**

Can you say a little bit more about the metrics (e.g., Recall@k, Mix\&Match at 2.2) when you first introduce them, without having to refer to the Methods? For example, in 2.3, you say, “In decoding, Recall@k represents…”—a nice, concise definition like this would also be useful earlier on.

### **Response**

Agreed. We will add concise metric definitions at first use in the Results:

"We assessed the CLIP model using Recall@K and Mix\&Match (see Methods for formal definitions). Recall@K measures the percentage of test samples where the true text-image match appears among the top K ranked candidates — higher values indicate better retrieval. Mix\&Match assesses whether each brain map is more similar to its true corresponding text than to other texts in the set, serving as a discriminability measure."

## **Reviewer 3 Comment 7**

On page 9, you refer to “The reduced and enhanced versions of Cognitive Atlas.” What does “enhanced” mean here? Are you referring to two different versions (1 \= reduced, 2 \= enhanced) of the CogAtlas, or do you just mean the “reduced” version is also “enhanced”?

### **Response**

We apologize for the confusing wording. "Reduced" and "enhanced" refer to the same version: a Cognitive Atlas that is \*reduced\* in the number of tasks (retaining only popular tasks) and \*enhanced\* in its task-to-concept mappings (with enriched, manually curated connections). We will revise to:

"The curated version of Cognitive Atlas, which retains the most commonly used fMRI tasks while enriching their concept mappings, consistently outperformed the original Cognitive Atlas ontology across all models."  
.

## **Reviewer 3 Comment 8**

On page 9, you say “The predictions of domains consistently showed higher recall rates than tasks and concepts across all models and configurations…” Could this difference just be due to differences in the structure of these target variables? For example, maybe domains has fewer distinct elements than tasks, so it makes for an easier decoding task? Some kind of shuffled/permuted/null baseline could provide a useful point of comparison here.

### **Response**

This is correct. Domain prediction is an easier classification problem:  
\- Domains: 10 categories  
\- Tasks: \~400 categories (reduced CogAt)  
\- Concepts: \~600 categories

The higher domain Recall@2 therefore reflects both model informativeness and the smaller candidate set. To address this directly, we now compute permutation-based null baselines within each dataset/configuration rather than relying only on structural chance.

For example, in the reduced cross-dataset benchmark:
1\. IBC combined: mean per-term hit@k was 0.201 for tasks versus a null mean of 0.101, 0.114 for concepts versus 0.055, and 0.284 for domains versus 0.221.
2\. CNP combined: mean per-term hit@k was 0.010 for tasks versus 0.0017, 0.0239 for concepts versus 0.0122, and 0.247 for domains versus 0.297.
3\. We also report normalized accuracy and the number of terms above chance. On IBC, 6/11 task terms, 17/30 concepts, and 6/9 domains were above chance in the combined setting, whereas on CNP only 1/12 concepts exceeded the null baseline.

We will add a note in the manuscript explaining the structural difference between label levels and report both raw and chance-normalized scores.

## **Reviewer 3 Comment 9**

For the impact of this paper, I think it’s important to ask: How can people actually use this model? Can you provide a little more logistical details (or a recipe) for how others might use the trained model and code for their own research? (I see there are some pointers on the GitHub repo… Are authors planning to build a Neurosynth-style website for this?)

### **Response**

We will add a "Practical Usage" subsection in the Discussion describing:

1\. Code and model availability: All code is available at https://github.com/NBCLab/brain-decoder. Trained model weights and vocabulary files will be released on OSF.

2\. Quick-start recipe:  
   \`\`\`python  
   pip install braindec  
   from braindec.predict import image\_to\_labels  
   results \= image\_to\_labels(  
       "path/to/brain\_map.nii.gz",  
       model\_path="path/to/trained\_model.pth",  
       vocabulary=vocabulary,  
       vocabulary\_emb=vocabulary\_emb,  
       vocabulary\_prior=vocabulary\_prior,  
   )  
   \`\`\`

3\. Google Colab notebook demonstrating the complete workflow.

4\. Future plans for a NeuroSynth Compose interface.

## **Reviewer 3 Comment 10**

Page 3: “Finally, we examined the extent to which NiCLIP’s capability in predicting subject-level activation maps.”—seems like there’s a word missing here.

### **Response**

Corrected to: \*"Finally, we examined the extent of NiCLIP's capability in predicting subject-level activation maps."\*

## **Reviewer 3 Comment 11**

5.1.2: “and may not always be factual”—I’m not sure what “factual” would mean in this context, but I get your point; maybe “widely agreed upon” is better?

### **Response**

Revised to: \*"As a community-based ontology, these mappings reflect the opinions of individual researchers and may not always be widely agreed upon or empirically validated."\*

## **Reviewer 3 Comment 12**

5.2: “The text encoder is characterized by a projection head and two residual heads, while the image encoder comprises three residual heads.” Is “head” the typical terminology here? I’m not super familiar with CLIP, but my brain is getting interference with the use of “head” in describing the attention heads in each layer?

### **Response**

We understand the potential confusion with attention heads. In our architecture, "head" refers to modular network blocks (fully connected layers with activation, dropout, and normalization), not attention heads. We will revise the terminology to avoid confusion:

\> \*"The text encoder consists of a projection block and two residual blocks, while the image encoder comprises three residual blocks. Each block contains a fully connected layer, GELU activation, dropout, and layer normalization. We use 'block' rather than 'head' to distinguish these architectural components from the attention heads in transformer models."\*
