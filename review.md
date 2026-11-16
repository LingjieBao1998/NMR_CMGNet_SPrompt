31-Aug-2026

Journal:  Analytical Chemistry
Manuscript ID: ac-2026-04568y
Title: "CMGNet-SPrompt: a deep learning framework for quinolone scaffold detection in cosmetics via 13C NMR spectroscopy"
Author(s): Li, Yutong; Bao, Lingjie; Hu, Jiaqi; Xiong, Tianning; Wang, Jike; Zou, Borui; Lu, Dejin; Wang, Qing; Yang, Huiying; Hou, Tingjun; Wu, Xianfu; Lu, Yong

Dear Dr. Wu:

Thank you for submitting your manuscript for publication in the American Chemical Society journal Analytical Chemistry. The reviewers have mixed views of the manuscript. While they appreciate the general goals, one questions the relevance and importance of the problem as well as the chosen solution. The reviewers also point to numerous issues related to the training vs test data sets, comparisons with other approaches, lack of analysis of the errors, among others.

Given these issues, I have therefore decided to reject the manuscript. However, if the authors believe they can address the reviewers' comments then I would be willing to consider a significantly revised manuscript for possible additional review. In that case please provide a point by point response to all reviewer comments along with the new manuscript.

Finally, please note that in terms of formatting, the main text can only have 6 figures and tables (in total), so many of these will need to be moved to Supporting Information. Also, please make sure that there are the correct number of significant figures (probably 2 or 3 at most) in the tables to reflect the proper precision.

The option to resubmit your manuscript will be available until 29-Nov-2026.  Please ensure that your manuscript fully conforms to the submission/format requirements outlined in the author guidelines: http://pubs.acs.org/page/ancham/submission/index.html.

Should you decide to resubmit, please include a highlighted/tracked changes version showing the changes made, designated as "Supporting Information for Review Only".  I do not require a highlighted version of Supporting Information for Publication (if applicable); this must be uploaded as a clean copy.

ACS Publications uses Crossref Similarity Check Powered by iThenticate to detect instances of similarity in submitted manuscripts.  In publishing only original research, ACS is committed to deterring plagiarism, including self-plagiarism.  Your manuscript may be screened for similarity to published material.

I am sorry to have to convey this negative decision to you.  Nevertheless, we appreciate your choice of Analytical Chemistry as a possible forum for the presentation of your work.

With sincere regards,

Prof. Daniel Raftery
Analytical Chemistry
Associate Editor
Email: raftery-office@anchem.acs.org
------------------------------------

Reviewer Comments to Author:
Reviewer: 1

Recommendation: Should not be published in Analytical Chemistry.

Comments:
This study presents a deep learning-based computational framework called CMGNet-SPrompt, for detecting illegal quinolone additives in cosmetics from 13C NMR spectra. The CMGNet-SPrompt is a deep learning ensemble framework that combines CMGNet with a discriminative XGBoost classifier, enabling de novo skeleton resolution directly from ¹³C NMR spectra without the need for reference standards. The model was trained on experimental data and large-scale synthetic spectra, achieving an F1 score of 0.9545 on a highly imbalanced external test set. However, in my option, there are approximately 35 to 40 types of quinolone drugs that have been approved for clinical use; there is no sufficient literature to support the existence of quinolone derivatives being abused in cosmetics. This manuscript lacks the application of real samples or actual scenarios. The composition of cosmetics is complex, and the nuclear magnetic resonance spectra are highly interfered. Mass spectrometry is the powerful tool for solving the identification problem. Moreover, a number of flaws with the manuscript remain to be addressed.

1. While the proposed CMGNet-SPrompt achieves impressive overall metrics, the manuscript lacks any targeted analysis of misclassified error cases. Given the extreme class imbalance (64 positive samples, 4438 negative samples), detailed dissection of the 5 FP and 1 FN is mandatory to strengthen the interpretability and reliability of the developed model. Firstly, for the 5 FP samples, the authors should elucidate whether the predicted structures completely lack characteristic quinolone features or whether they share partial substructural similarity. This distinction is essential to determine whether the errors stem from a genuine “hallucination” of the generative model (indicating a failure of the scaffold‑prompting mechanism) or from an ambiguity in scaffold boundary definition (indicating that the model is over‑sensitive to substructural fragments that resemble, but do not constitute, the quinolone pharmacophore). Clarifying this point will help readers understand the structural patterns most prone to false positives and define the practical applicability domain of the method. Secondly, for the sole FN case, the authors are urged to examine whether this sample is also misclassified by other models, and more importantly, whether the failure arises from poor spectral quality or insufficient informative signals (e.g., a low number of discernible ¹³C peaks or a low signal‑to‑noise ratio). If so, this would establish a lower bound on the spectral information content required for reliable scaffold detection—a finding of significant practical value for end‑users who may upload spectra of varying quality.

2. This reviewer’s biggest concern relates to the machine-learning methodology. While the authors mentioned using a validation set for hyperparameter tuning and early stopping in their manuscript, the current evaluation scheme still suffers from instability due to an insufficient number of positive samples in the test set and potential data leakage during training. The external imbalanced test set cited by the authors contains only 64 positive samples versus 4,438 negative ones, while the model architecture is relatively complex—featuring a Transformer-based encoder-decoder structure, a multi-party integration framework, and a classifier. Given this complexity, performance estimates on small datasets exhibit high variance, limiting the reliability of single-test evaluations. Moreover, the authors have only removed duplicates based on SMILES strings in the training and test sets, which is insufficient to prevent structural-level data leakage (ang K. et al. Are Learned Molecular Representations Ready For Prime Time? arXiv:1904.01561, 2019.). Even if a compound does not appear in the training set, its core scaffold or key fragments may be heavily represented there, leading to overestimation of model performance. Therefore, the model performance results generated by the current validation approach are not entirely reliable. Potential solutions include adopting a hold-out validation strategy to ensure that at least one class of scaffold (rather than just spectral data) is entirely absent from the training set and to assess the model's generalization ability on truly novel structures, or using nested cross-validation to re-evaluate model performance across multiple levels.

3. From the methods section and supplementary information, it is difficult to clearly determine the exact functions and interrelationships of each component in the "CMGNet-SPrompt" framework. Firstly, the term "structure elucidation" is frequently used in the text, which usually refers to generating a complete chemical structure from scratch. However, in this work, the authors describe that the framework consists of a generative model based on BART (CMGNet) and an XGBoost classifier. The output of XGBoost serves as a structured prior prompt for the generative model. Therefore, XGBoost performs binary classification (whether it is a quinolone scaffold), while CMGNet-SPrompt generates a complete molecule given the scaffold prompt. Describing this process as "conditional molecule generation based on scaffold prompts" or "molecular structure prediction under scaffold constraints" might be more appropriate. Additionally, if the predicted probability given by XGBoost-Cress happens to be near the threshold (e.g., around 0.5), how will the generation process of CMGNet-SPrompt handle this uncertainty? Will it generate multiple possibilities? It is recommended that the authors clearly discuss this issue and explain the behavior strategy of the model in boundary cases. Meanwhile, in Section 3.2, the authors mention that "CMGNET-SPrompt surpassed ensemble techniques by leveraging simple scaffold prompts." Here, "simple" requires a clearer definition - does it refer to architectural simplicity, parameter quantity, or reasoning complexity? Since the Transformer architecture is currently generally recognized as a complex architecture and this architecture itself is a multi-dimensional integrated architecture. It is recommended that the authors provide a clearer comparison of architectural definitions and a quantitative comparison of the computational time and resource usage of CMGNet-SPrompt with integration methods (Ensemble-Voting and Ensemble-Union) to objectively support the "simple" assertion.

4. In Section 3.5 of the paper, "Ablation Study on the Impact of Synthetic Data and Ensemble Strategy", it is emphasized that "synthetic data is essential for overcoming data scarcity in both architectures" and "narrowly distributed data artificially increases the generation probability of specific classes, resulting in high false positive rates." Although these claims are based on ablation experiments, their expressions still have issues. This ablation experiment only shows the correlation between data combination and model performance, and does not prove causality. Moreover, emotional expressions such as "failed completely" (F1 = 0.0000) and "robust scalability" are not objective enough and should be replaced with more neutral technical descriptions. Most importantly, the title "Ablation Study on the Impact of Synthetic Data and Ensemble Strategy" in Section 3.5 and the title "Ablation Study on the Impact of Synthetic Data" in Section 3.5.1 do not have a corresponding Section 3.5.2. This paragraph does not fully describe and clearly distinguish the complete ablation experimental strategy of Synthetic Data and Ensemble Strategy, and it is recommended to add.

5. Most of the text in Figures 2 and 3-A, 3-B is difficult to read. It is suggested that all the images be replaced with high-resolution (≥300 dpi) vector formats and the font and label sizes in the figures be increased.

6. The citation format of the literature is rather poor. For instance, the information in literature 17, 18, 23, 29, 32, and 35 is incomplete. The format of some literatures were not uniform (Literature 27).

7. The website mentioned in the article, "a web server to detect quinolone scaffolds from pure 13C NMR Chemistry shifts", is inaccessible, and the training process code (such as scripts for data preprocessing, training loops, and result generation) is not seen in the supplementary materials.


Additional Questions:
Significance: Moderate (not suitable for Analytical Chemistry)

Originality: High (suitable for Analytical Chemistry)

Broad Interest: Moderate (not suitable for Analytical Chemistry)

Technical Quality: Moderate (not suitable for Analytical Chemistry)

Are the conclusions adequately supported by the data?: In Part

Are the references appropriate and correct?: In Part

Is the English adequate?: Yes

Is the quality of the figures sufficient?: In Part

Are the data statistically significant with appropriate controls included?: In Part


Reviewer: 2

Recommendation: May eventually be publishable, but requires major revisions as indicated.

Comments:
This article proposes a deep learning framework CMGNet-SPrompt, which constrains the CMGNet generation model by using the predicted results of the XGBoost discriminant model as structural prior prompts. It achieves high-precision detection of quinolone antibiotic skeletons illegally added in cosmetics from 13C NMR spectra, and achieves good performance on an imbalanced external test set. An online web tool is also constructed, and the overall method design has strong innovation and practical application value; However, the current version has significant shortcomings in terms of code and data openness, fair comparison with existing methods, uncertainty analysis, discussion of failed cases, and selection criteria for some hyperparameters. Therefore, a major revision is suggested.

Major Comments：
1. As a research article with deep learning as its core method, this article does not provide any code repository, pre-trained model weights, data preprocessing scripts, or inference examples. It only provides an online web tool, which makes it almost impossible for independent researchers to reproduce the results of this article. I strongly recommend the author to publicly disclose the complete executable code, training weights, and the complete multi-source dataset used in the article (link).
2. The paper lists the performance of various comparison models such as NMR Solver and NMRMind in Table 5, but these comparisons were not conducted under completely fair conditions - different models used different training data, different input dimensions (such as NMRMind using multidimensional NMR while this paper only used 1D ¹³ C NMR), and different inference settings, lacking direct performance comparisons on a unified dataset; It is suggested that the author run the inference (or fine tuned inference) of the comparative model on the test set of this article, report the quantitative results under fair conditions, and add qualitative visual comparisons (such as the output structure comparison chart of each model for the same quinolone derivative), while clearly indicating the input dimensions and training data configuration of each comparative model.
3. The paper reports on overall performance indicators but does not analyze under what circumstances the model may fail or perform poorly, which is crucial for cosmetic safety regulatory applications - regulatory agencies need to understand the risks of false positives and false negatives in the model, as well as the applicability boundaries of the model; Suggest discussing in detail the spectral and structural characteristics of missed positive samples and misjudged negative samples, analyzing whether the model will misjudge other nitrogen-containing heterocycles (such as naphthoprim, indole, etc.) as quinolones, and exploring the performance sensitivity of the model under different signal-to-noise ratios or spectral resolutions.
4. As a research aimed at regulatory decision-making applications, quantifying the uncertainty of model predictions is crucial. However, this article does not provide any form of confidence calibration, prediction intervals, or uncertainty estimation, which makes it difficult for regulatory agencies to determine when to trust model outputs and when manual review is needed. It is suggested to discuss how to integrate uncertainty information into actual regulatory workflows, following the recent uncertainty methodology (DOI: 10.1007/s10462-023-10562-9; 10.1021/acs.analchem.5c07201），especially using Monte Carlo Dropout for uncertainty estimation, which appears to be compatible with CMGNet-Sprompt using Dropout, while properly citing the suggested references to ground the discussion in the literature.
5. Although the paper lists a large number of hyperparameters in the text (such as learning rate 1e-4, dropout 0.1, number of bins 4000, L1/L2 regularization coefficient, etc.), it does not provide the selection criteria or sensitivity analysis for these hyperparameters. Other researchers find it difficult to understand why these specific values were chosen and how to adjust them for new tasks; Suggest the authors to supplement the sensitivity analysis of hyperparameters (such as the impact curve of box number, dropout rate, etc. on F1 score), or at least clearly explain the selection method of each hyperparameter and provide the search range.
6. The paper provides the web server address in the text, but does not specify the support period and maintenance plan of the server, specific requirements for input formats, concurrent processing capabilities, privacy protection policies for uploading data, and whether API interfaces are provided for batch processing; It is suggested to add detailed usage instructions for the web server (including screenshots and example inputs/outputs) in the Supporting Information, clarify the data privacy policy, and consider providing Docker images for localized deployment to ensure long-term tool availability and data security.
7. It is suggested to add 2-3 latest review articles on AI analysis of NMR spectra in the Introduction (DOI: 10.1063/5.0277355 and 10.1016/j.pnmrs. 2022.101575) to better position this work at the forefront of the field and strengthen the background and discussion depth.
8. There is a formatting issue with the references. For example, references 18 and 24 lack information. In addition, the resolution of the images in the article needs to be improved.


Additional Questions:
Significance: Top 5% (suitable for Analytical Chemistry)

Originality: High (suitable for Analytical Chemistry)

Broad Interest: Top 5% (suitable for Analytical Chemistry)

Technical Quality: High (suitable for Analytical Chemistry)

Are the conclusions adequately supported by the data?:

Are the references appropriate and correct?:

Is the English adequate?:

Is the quality of the figures sufficient?:

Are the data statistically significant with appropriate controls included?:


If the Editor has suggested that your manuscript may benefit from language editing, you are encouraged to have it reviewed by a fluent English speaker before resubmitting. ACS Authoring Services can assist you by matching your manuscript with trained chemists who will edit your work for accurate terminology and phrasing. For pricing and information, visit https://authoringservices.acs.org/. Please note that using this (or any) service does not guarantee that your manuscript will be accepted for publication.

------------------------------------
FOR ASSISTANCE WITH YOUR MANUSCRIPT SUBMISSION PLEASE CONTACT:
ACS Publications Customer Services & Information (CSI)
Email: support@services.acs.org
Phone: 202-872-4357
Toll Free Phone: 800-227-9919 (USA/Canada only)
   

------------
PLEASE NOTE: This email message, including any attachments, contains confidential information related to peer review and is intended solely for the personal use of the recipient(s) named above. No part of this communication or any related attachments may be shared with or disclosed to any third party or organization without the explicit prior written consent of the journal Editor and ACS. If the reader of this message is not the intended recipient or is not responsible for delivering it to the intended recipient, you have received this communication in error. Please notify the sender immediately by e-mail, and delete the original message.
 
As an author or reviewer for ACS Publications, we may send you communications about related journals, topics or products and services from the American Chemical Society. Please email us at pubs-comms-unsub@acs.org if you do not want to receive these. Note, you will still receive updates about your manuscripts, reviews, or future invitations to review.   

Thank you.