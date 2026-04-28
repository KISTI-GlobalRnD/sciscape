# Submission Strategy

## Primary And Backup Targets

- Primary target: `Journal of Informetrics`
- Backup target: `Scientometrics`

## Why `Journal of Informetrics`

This paper fits `Journal of Informetrics` best because its core contribution is
an evaluation and interpretation framework for bibliometric graph combination.
The manuscript is strongest when it emphasizes:

- local neighborhood correctness as an informetric target
- boundary-sensitive combination rather than generic clustering improvement
- science-mapping relevance rather than pure graph algorithm novelty

## What To Emphasize

- The paper studies how multi-layer agreement changes the local semantic
  neighborhood of a target paper.
- The corrected evaluation protocol is conservative:
  order balancing, tie preservation, and explicit non-tie reporting are central
  to the credibility of the result.
- The strongest empirical message is regime dependence:
  consensus helps most when local neighborhoods diverge sharply and overlap
  weakly.
- Taxonomy should be framed as a descriptive mechanism analysis, not as an
  absolute ontology.

## What To De-Emphasize

- Do not pitch the work as a universal new clustering algorithm.
- Do not rely on AMI/NMI as the headline validation.
- Do not overclaim that LLM review is ground truth.
- Do not mix the `dendrogram` project into this manuscript.

## Backup Reframing For `Scientometrics`

If the paper moves to `Scientometrics`, keep the evidence package unchanged but
shift the framing slightly:

- emphasize topic delineation and scientific mapping practice
- foreground cross-field differences and application relevance
- make the field mapping explicit in the main text:
  - `field_15` functions as the chemistry/materials case family
  - `field_12` functions as the education and learning-science case family
- foreground representative cases accordingly:
  - chemistry: `W2067783257`, `W2021514999`, `W2112002317`, `W3088126537`
  - education: `W3016217815`
- allow more space for field-specific interpretation in the discussion

## Submission Checklist

- Align title, abstract, and introduction with an informetrics audience in the
  first page.
- Convert final manuscript prose out of markdown/code style:
  remove backticks around method names, counts, and journal titles in the
  submission draft.
- Use only corrected order-balanced `gemini_v3` outputs in the manuscript.
- Keep five main figures and four main tables as the stable narrative spine.
- Report slice counts as `baseline / consensus / tie` everywhere.
- State the limitation that validation is LLM-based and order-balanced, not
  human annotated.
- State the scope limitation that the corrected validation centers on two
  fields and three canonical reviewed slices.
