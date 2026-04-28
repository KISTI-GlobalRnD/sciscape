# Consensus Taxonomy Summary

## Headline

`consensus_all` wins 101/144 cases (70.1%), while `sum_minus_emb` wins 30/144 and `sum_minus_cc` wins 13/144.
The dominant consensus-win regimes are `broad_context_noise` (50), `material_family_coherence` (37), `method_family_coherence` (22), which is consistent with a boundary-regularization interpretation.

## Winner By K

| Slice | consensus_all | sum_minus_emb | sum_minus_cc |
|---|---:|---:|---:|
| top-k=6 | 66 | 30 | 0 |
| top-k=30 | 35 | 0 | 13 |
| total | 101 | 30 | 13 |

## Taxonomy By Winner

| Taxonomy label | Count | Share | consensus_all wins | sum_minus_emb wins | sum_minus_cc wins |
|---|---:|---:|---:|---:|---:|
| broad_context_noise | 50 | 34.7% | 47 | 2 | 1 |
| material_family_coherence | 37 | 25.7% | 28 | 6 | 3 |
| method_family_coherence | 22 | 15.3% | 14 | 8 | 0 |
| over_regularized_consensus | 16 | 11.1% | 0 | 11 | 5 |
| application_umbrella_noise | 6 | 4.2% | 3 | 2 | 1 |
| single_cue_specificity | 6 | 4.2% | 2 | 1 | 3 |
| semantic_drift | 4 | 2.8% | 4 | 0 | 0 |
| coherent_refinement | 3 | 2.1% | 3 | 0 | 0 |

## Representative Winner Cases

### consensus_all
- `W2012804551` Application of Poly(ethylene glycol)-based Aqueous Biphasic Systems as Reaction and Reactive Extraction Media (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=broad_context_noise, gap=4.0)
  Winner advantage: Group A provides an excellent local neighborhood. The top-ranked papers focus specifically on the partitioning and phase diagrams of Poly(ethylene glycol)-based Aqueous Biphasic Systems (PEG-ABS), which is the exact system studied in the target paper.
  Loser failure mode: Group B is almost entirely focused on a different class of solvents, Deep Eutectic Solvents (DES), and fails to capture the target's core topic of ABS, making it a poor representation of the immediate research context.
- `W2021514999` Tunable thermomorphism and applications of ionic liquid analogues of Girard's reagents (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=material_family_coherence, gap=4.0)
  Winner advantage: Group A provides an excellent and highly coherent neighborhood focused on the target's specific topic: using ionic liquids that exhibit temperature-dependent phase behavior (thermomorphism, LCST/UCST) for homogeneous liquid-liquid extraction.
  Loser failure mode: Group B is a very broad and noisy collection of papers generally related to 'ionic liquids' but covering disparate applications like crystallization, electrolytes, and general synthesis, completely missing the target's specific context.
- `W2024795243` Chemical Absorption of Sulfur Dioxide in Room-Temperature Ionic Liquids (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=material_family_coherence, gap=4.0)
  Winner advantage: Group A provides a perfectly coherent local neighborhood where every paper is directly about the target's specific topic: SO2 absorption in ionic liquids. Group B is extremely weak; after the first paper, it diverges into papers about different gases (CO2, H2S) or the general properties of ionic liquids, completely missing the target's core research problem.
  Loser failure mode: Group A provides a perfectly coherent local neighborhood where every paper is directly about the target's specific topic: SO2 absorption in ionic liquids. Group B is extremely weak; after the first paper, it diverges into papers about different gases (CO2, H2S) or the general properties of ionic liquids, completely missing the target's core research problem.
- `W2043931076` A New Category of Liquid Salt--Liquid Ionic Phosphates (LIPs) (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=material_family_coherence, gap=4.0)
  Winner advantage: Group A provides an exceptional local neighborhood. Its top-ranked paper is a direct continuation or companion paper to the target, sharing the key term "LIPs" (Liquid Ionic Phosphates) and focusing on polyammonium salts. The subsequent papers cohere tightly around the theme of dicationic/polycationic ionic liquids, which is the central structural motif of the target paper.
  Loser failure mode: Group B is a very broad and unfocused collection of general reviews and disparate specific studies on ionic liquids, failing entirely to capture the target's specific contribution to the field.

### sum_minus_emb
- `W2001501144` Impact of Ionic Liquid Physical Properties on Lipase Activity and Stability (field_15_k06_sum_minus_emb_vs_consensus_bank_n48, k=6, label=material_family_coherence, gap=3.0)
  Winner advantage: Group A provides a much better local neighborhood. Its top-ranked papers are directly about lipase or other enzymes in ionic liquids, which is the core topic of the target paper. Group B is extremely poor, with its top-ranked papers focusing on an entirely different application of ionic liquids (metal ion extraction), demonstrating a fundamental misunderstanding of the target's research context.
  Loser failure mode: Group A provides a much better local neighborhood. Its top-ranked papers are directly about lipase or other enzymes in ionic liquids, which is the core topic of the target paper. Group B is extremely poor, with its top-ranked papers focusing on an entirely different application of ionic liquids (metal ion extraction), demonstrating a fundamental misunderstanding of the target's research context.
- `W2033302412` Combinatorial Rheology of Branched Polymer Melts (field_15_k06_sum_minus_emb_vs_consensus_bank_n48, k=6, label=method_family_coherence, gap=3.0)
  Winner advantage: Group A provides an excellent, coherent neighborhood. Its top-ranked papers focus on the same core problem as the target: developing general algorithms and computational models to predict the linear rheology of complex branched polymers. The lower-ranked papers in A provide relevant context by focusing on specific architectures like comb and star polymers, which the target paper explicitly mentions. Group B is significantly weaker, as it buries the most relevant papers at the bottom of its ranking and includes a clear topical mismatch (rank 6, polymer film surface dynamics) which is distinct from the target's focus on bulk melts.
  Loser failure mode: Group A provides an excellent, coherent neighborhood. Its top-ranked papers focus on the same core problem as the target: developing general algorithms and computational models to predict the linear rheology of complex branched polymers. The lower-ranked papers in A provide relevant context by focusing on specific architectures like comb and star polymers, which the target paper explicitly mentions. Group B is significantly weaker, as it buries the most relevant papers at the bottom of its ranking and includes a clear topical mismatch (rank 6, polymer film surface dynamics) which is distinct from the target's focus on bulk melts.
- `W1968106432` Binuclear chromium–salan complex catalyzed alternating copolymerization of epoxides and cyclic anhydrides (field_15_k06_sum_minus_emb_vs_consensus_bank_n48, k=6, label=over_regularized_consensus, gap=3.0)
  Winner advantage: Group A provides an excellent neighborhood by immediately focusing on the correct reaction (epoxide/anhydride copolymerization) with the correct catalyst family (chromium-salen complexes).
  Loser failure mode: Group B's top-ranked papers are significant mismatches, focusing on catalysts with different metals (Mg, Zn, Mn), which dilutes the target's specific context. Although Group B contains a highly relevant paper on dinuclear chromium catalysts at rank 5, its poor ranking and the noise at the top make it a much weaker neighborhood overall.
- `W2503430913` A Vygotskian sociocultural perspective on immersion education (field_12_k06_sum_minus_emb_vs_consensus_bank_n48, k=6, label=over_regularized_consensus, gap=3.0)
  Winner advantage: Group A provides an excellent, highly coherent neighborhood focused on the target's specific topic: the use of L1 in L2 learning from a sociocultural perspective. Its top-ranked papers are directly relevant, and it even includes a paper on the target's specific application context of 'immersion education'.
  Loser failure mode: Group B is much broader and more diffuse, focusing on the general sociocultural theory in second language acquisition rather than the target's specific research question, and it completely misses the immersion education context.

### sum_minus_cc
- `W2060442727` Carbon Paste Electrode Modified with Functionalized Nanoporous Silica Gel as a New Sensor for Determination of Silver Ion (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=single_cue_specificity, gap=3.0)
  Winner advantage: Group A provides an excellent and highly coherent neighborhood. Its top-ranked papers are directly relevant, focusing on either the same method (modified carbon paste electrodes) or the same target analyte (silver ion).
  Loser failure mode: Group B is much broader and noisier, with many of its top-ranked papers focusing on the determination of organic drug molecules, which is a different application area from the target's focus on metal ion sensing.
- `W2112002317` A single-site hydroxyapatite-bound zinc catalyst for highly efficient chemical fixation of carbon dioxide with epoxides (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=single_cue_specificity, gap=3.0)
  Winner advantage: Group A provides an excellent local neighborhood by correctly identifying both the target's specific chemical reaction (CO2 fixation with epoxides) and the specific catalyst family (zinc-based catalysts).
  Loser failure mode: Group B identifies the general reaction but completely misses the crucial zinc catalyst aspect, instead providing a diverse and less relevant set of catalysts (Ni, Ru, Cr, organocatalysts) and even including mismatched papers about polymerization.
- `W2042746129` Preparation and Characterization of New Room Temperature Ionic Liquids (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=material_family_coherence, gap=2.0)
  Winner advantage: Group A provides a more coherent and focused neighborhood. Its top-ranked papers are all centered on the synthesis and characterization of imidazolium-based ionic liquids, which directly matches the target's scope.
  Loser failure mode: Group B is noisier, including papers on different cation types (guanidinium and pyridinium) that are less relevant to the target's specific chemistry, thereby diluting the immediate research context.
- `W2326758683` Surface Analysis of Ionic Liquids with and without Lithium Salt Using X-ray Photoelectron Spectroscopy (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=material_family_coherence, gap=2.0)
  Winner advantage: Group B provides a much more specific and relevant neighborhood for the target paper. It includes multiple papers studying the exact same ionic liquid ([EMIM][Tf2N]) and, critically, several papers investigating the interaction of this class of ionic liquids with lithium, which is the central perturbation in the target study. Group A is plausible, as it focuses on the correct technique (XPS) and general problem (ionic liquid surfaces with dissolved salts), but its focus on platinum and palladium complexes makes it less directly relevant than Group B's focus on lithium.
  Loser failure mode: Group B provides a much more specific and relevant neighborhood for the target paper. It includes multiple papers studying the exact same ionic liquid ([EMIM][Tf2N]) and, critically, several papers investigating the interaction of this class of ionic liquids with lithium, which is the central perturbation in the target study. Group A is plausible, as it focuses on the correct technique (XPS) and general problem (ionic liquid surfaces with dissolved salts), but its focus on platinum and palladium complexes makes it less directly relevant than Group B's focus on lithium.

## Representative Taxonomy Cases

### application_umbrella_noise
- `W2604609627` Heterogeneous Catalysis for Oxazolidinone Synthesis from Aziridines and CO2 (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=consensus_all, gap=4.0)
  Winner advantage: Group A provides an excellent local neighborhood that perfectly captures the target's specific research topic: the synthesis of oxazolidinones from aziridines and CO2. Several papers in Group A discuss this exact reaction, including different catalytic and mechanistic aspects. Group B is completely incorrect, as it exclusively contains papers on the reaction of CO2 with epoxides to form cyclic carbonates, which is a different, though related, field.
  Loser failure mode: Group A provides an excellent local neighborhood that perfectly captures the target's specific research topic: the synthesis of oxazolidinones from aziridines and CO2. Several papers in Group A discuss this exact reaction, including different catalytic and mechanistic aspects. Group B is completely incorrect, as it exclusively contains papers on the reaction of CO2 with epoxides to form cyclic carbonates, which is a different, though related, field.
- `W2766569351` The reactions of dimethyl carbonate and its derivatives (field_15_k06_sum_minus_emb_vs_consensus_bank_n48, k=6, label=sum_minus_emb, gap=2.0)
  Winner advantage: Group B provides a perfectly coherent neighborhood focused on the reactions and applications of dimethyl carbonate (DMC), which directly matches the target paper's topic. Group A is also strong at the top, but it becomes noisy by including papers on the synthesis of DMC and a different compound (diethyl carbonate), which are less central to the target's immediate context.
  Loser failure mode: Group B provides a perfectly coherent neighborhood focused on the reactions and applications of dimethyl carbonate (DMC), which directly matches the target paper's topic. Group A is also strong at the top, but it becomes noisy by including papers on the synthesis of DMC and a different compound (diethyl carbonate), which are less central to the target's immediate context.

### broad_context_noise
- `W2012804551` Application of Poly(ethylene glycol)-based Aqueous Biphasic Systems as Reaction and Reactive Extraction Media (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=consensus_all, gap=4.0)
  Winner advantage: Group A provides an excellent local neighborhood. The top-ranked papers focus specifically on the partitioning and phase diagrams of Poly(ethylene glycol)-based Aqueous Biphasic Systems (PEG-ABS), which is the exact system studied in the target paper.
  Loser failure mode: Group B is almost entirely focused on a different class of solvents, Deep Eutectic Solvents (DES), and fails to capture the target's core topic of ABS, making it a poor representation of the immediate research context.
- `W2067783257` The Catalytic Use of Onion-Like Carbon Materials for Styrene Synthesis by Oxidative Dehydrogenation of Ethylbenzene (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=consensus_all, gap=4.0)
  Winner advantage: Group B provides an excellent and highly coherent neighborhood. Nearly every paper discusses the exact research topic of the target: using nanocarbon materials as catalysts for oxidative dehydrogenation, often specifically for the ethylbenzene to styrene reaction.
  Loser failure mode: Group A is a very poor collection of papers broadly related to catalysis but with no specific connection to the target's catalyst type, reaction, or substrate, making it noisy and uninformative.

### coherent_refinement
- `W2505965475` The Influence of Affective Variables on the Complexity, Accuracy, and Fluency in L2 Oral Production: The Contribution of Task Repetition (field_12_k06_sum_minus_emb_vs_consensus_bank_n48, k=6, label=consensus_all, gap=2.0)
  Winner advantage: Group A provides an excellent, coherent neighborhood that precisely mirrors the target paper's research question, which is the intersection of affective variables (like anxiety), task repetition, and CAF measures in L2 oral production.
  Loser failure mode: Group B is less focused; it includes papers on task repetition and CAF, and separate papers on affective variables, but fails to capture the crucial link between these components that defines the target's contribution.
- `W2017076541` Hierarchical structured α-Al<sub>2</sub>O<sub>3</sub> supported S-promoted Fe catalysts for direct conversion of syngas to lower olefins (field_15_k06_sum_minus_emb_vs_consensus_bank_n48, k=6, label=consensus_all, gap=1.0)
  Winner advantage: Both groups provide excellent neighborhoods focused on iron-based Fischer-Tropsch catalysts for lower olefins.
  Loser failure mode: Group A is slightly better because it includes a paper (rank 7) that specifically investigates sulfur promoters, a key component of the target paper's catalyst. This makes the local context in Group A slightly more specific and directly relevant to the target's contribution.

### material_family_coherence
- `W2021514999` Tunable thermomorphism and applications of ionic liquid analogues of Girard's reagents (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=consensus_all, gap=4.0)
  Winner advantage: Group A provides an excellent and highly coherent neighborhood focused on the target's specific topic: using ionic liquids that exhibit temperature-dependent phase behavior (thermomorphism, LCST/UCST) for homogeneous liquid-liquid extraction.
  Loser failure mode: Group B is a very broad and noisy collection of papers generally related to 'ionic liquids' but covering disparate applications like crystallization, electrolytes, and general synthesis, completely missing the target's specific context.
- `W2024795243` Chemical Absorption of Sulfur Dioxide in Room-Temperature Ionic Liquids (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=consensus_all, gap=4.0)
  Winner advantage: Group A provides a perfectly coherent local neighborhood where every paper is directly about the target's specific topic: SO2 absorption in ionic liquids. Group B is extremely weak; after the first paper, it diverges into papers about different gases (CO2, H2S) or the general properties of ionic liquids, completely missing the target's core research problem.
  Loser failure mode: Group A provides a perfectly coherent local neighborhood where every paper is directly about the target's specific topic: SO2 absorption in ionic liquids. Group B is extremely weak; after the first paper, it diverges into papers about different gases (CO2, H2S) or the general properties of ionic liquids, completely missing the target's core research problem.

### method_family_coherence
- `W2033302412` Combinatorial Rheology of Branched Polymer Melts (field_15_k06_sum_minus_emb_vs_consensus_bank_n48, k=6, label=sum_minus_emb, gap=3.0)
  Winner advantage: Group A provides an excellent, coherent neighborhood. Its top-ranked papers focus on the same core problem as the target: developing general algorithms and computational models to predict the linear rheology of complex branched polymers. The lower-ranked papers in A provide relevant context by focusing on specific architectures like comb and star polymers, which the target paper explicitly mentions. Group B is significantly weaker, as it buries the most relevant papers at the bottom of its ranking and includes a clear topical mismatch (rank 6, polymer film surface dynamics) which is distinct from the target's focus on bulk melts.
  Loser failure mode: Group A provides an excellent, coherent neighborhood. Its top-ranked papers focus on the same core problem as the target: developing general algorithms and computational models to predict the linear rheology of complex branched polymers. The lower-ranked papers in A provide relevant context by focusing on specific architectures like comb and star polymers, which the target paper explicitly mentions. Group B is significantly weaker, as it buries the most relevant papers at the bottom of its ranking and includes a clear topical mismatch (rank 6, polymer film surface dynamics) which is distinct from the target's focus on bulk melts.
- `W1510246799` Estimation of Age Using Alveolar Bone Loss: Forensic and Anthropological Applications (field_12_k06_sum_minus_emb_vs_consensus_bank_n48, k=6, label=consensus_all, gap=2.0)
  Winner advantage: Group B provides a much more coherent and specific local neighborhood. Nearly all its papers focus on dental age estimation methods, which is the precise topic of the target paper. Group A is much broader, mixing in many papers on skeletal age estimation (e.g., from the pubic bone or ilium) and ranking a purely skeletal paper first, which dilutes the target's specific odontological context.
  Loser failure mode: Group B provides a much more coherent and specific local neighborhood. Nearly all its papers focus on dental age estimation methods, which is the precise topic of the target paper. Group A is much broader, mixing in many papers on skeletal age estimation (e.g., from the pubic bone or ilium) and ranking a purely skeletal paper first, which dilutes the target's specific odontological context.

### over_regularized_consensus
- `W1968106432` Binuclear chromium–salan complex catalyzed alternating copolymerization of epoxides and cyclic anhydrides (field_15_k06_sum_minus_emb_vs_consensus_bank_n48, k=6, label=sum_minus_emb, gap=3.0)
  Winner advantage: Group A provides an excellent neighborhood by immediately focusing on the correct reaction (epoxide/anhydride copolymerization) with the correct catalyst family (chromium-salen complexes).
  Loser failure mode: Group B's top-ranked papers are significant mismatches, focusing on catalysts with different metals (Mg, Zn, Mn), which dilutes the target's specific context. Although Group B contains a highly relevant paper on dinuclear chromium catalysts at rank 5, its poor ranking and the noise at the top make it a much weaker neighborhood overall.
- `W2503430913` A Vygotskian sociocultural perspective on immersion education (field_12_k06_sum_minus_emb_vs_consensus_bank_n48, k=6, label=sum_minus_emb, gap=3.0)
  Winner advantage: Group A provides an excellent, highly coherent neighborhood focused on the target's specific topic: the use of L1 in L2 learning from a sociocultural perspective. Its top-ranked papers are directly relevant, and it even includes a paper on the target's specific application context of 'immersion education'.
  Loser failure mode: Group B is much broader and more diffuse, focusing on the general sociocultural theory in second language acquisition rather than the target's specific research question, and it completely misses the immersion education context.

### semantic_drift
- `W2747960663` The phenomenology of performance: exploring musicians' perceptions and experiences (field_12_k06_sum_minus_emb_vs_consensus_bank_n48, k=6, label=consensus_all, gap=3.0)
  Winner advantage: Group A provides an excellent neighborhood that directly matches the target's focus on the phenomenology and subjective experience of musical performance. Its top-ranked papers explore musicians' thoughts, perceptions, and anxieties using similar qualitative approaches. Group B is significantly weaker, as it shifts the focus to quantitative studies on interventions, self-efficacy, and exam outcomes, which is a different research paradigm from the target's exploratory investigation.
  Loser failure mode: Group A provides an excellent neighborhood that directly matches the target's focus on the phenomenology and subjective experience of musical performance. Its top-ranked papers explore musicians' thoughts, perceptions, and anxieties using similar qualitative approaches. Group B is significantly weaker, as it shifts the focus to quantitative studies on interventions, self-efficacy, and exam outcomes, which is a different research paradigm from the target's exploratory investigation.
- `W3022226786` PARASOCIAL MEDIA EFFECTS (field_12_k06_sum_minus_emb_vs_consensus_bank_n48, k=6, label=consensus_all, gap=2.0)
  Winner advantage: Group A is an excellent neighborhood, with every paper directly addressing the core concepts of parasocial interaction and relationships, just like the target paper. The top-ranked papers in A are theoretical reviews and meta-analyses, which perfectly match the target's described focus. Group B is much weaker; its top-ranked paper is a niche case study on the 'Twilight' phenomenon, which is a poor starting point for a theoretical overview, and the group is generally noisier and less focused than A.
  Loser failure mode: Group A is an excellent neighborhood, with every paper directly addressing the core concepts of parasocial interaction and relationships, just like the target paper. The top-ranked papers in A are theoretical reviews and meta-analyses, which perfectly match the target's described focus. Group B is much weaker; its top-ranked paper is a niche case study on the 'Twilight' phenomenon, which is a poor starting point for a theoretical overview, and the group is generally noisier and less focused than A.

### single_cue_specificity
- `W2060442727` Carbon Paste Electrode Modified with Functionalized Nanoporous Silica Gel as a New Sensor for Determination of Silver Ion (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=sum_minus_cc, gap=3.0)
  Winner advantage: Group A provides an excellent and highly coherent neighborhood. Its top-ranked papers are directly relevant, focusing on either the same method (modified carbon paste electrodes) or the same target analyte (silver ion).
  Loser failure mode: Group B is much broader and noisier, with many of its top-ranked papers focusing on the determination of organic drug molecules, which is a different application area from the target's focus on metal ion sensing.
- `W2112002317` A single-site hydroxyapatite-bound zinc catalyst for highly efficient chemical fixation of carbon dioxide with epoxides (field_15_k30_sum_minus_cc_vs_consensus_bank_n48, k=30, label=sum_minus_cc, gap=3.0)
  Winner advantage: Group A provides an excellent local neighborhood by correctly identifying both the target's specific chemical reaction (CO2 fixation with epoxides) and the specific catalyst family (zinc-based catalysts).
  Loser failure mode: Group B identifies the general reaction but completely misses the crucial zinc catalyst aspect, instead providing a diverse and less relevant set of catalysts (Ni, Ru, Cr, organocatalysts) and even including mismatched papers about polymerization.
