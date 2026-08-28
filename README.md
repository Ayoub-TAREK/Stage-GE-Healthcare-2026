<!-- ▼ LOGO — drop the file in figures/ and keep the name below, or change it -->
<p align="center">
  <img src="figures/ge_healthcare_logo.png" alt="GE HealthCare" width="220">
</p>
<!-- ▲ LOGO -->

# Foundation Models & Unsupervised Domain Adaptation for 2D Probe Pose Estimation

**Research internship — GE HealthCare, Interventional Image Processing R&D team (Buc, France) · May – August 2026**
Supervisors: Vincent Jugnon (architect), Gauthier Miralles (PhD candidate) · Engineering school: IMT Nord Europe

> **About this repository.** It documents the problem, the approach and my methodological
> contributions. It contains **no code, no data and no results** — the pipeline, the datasets
> and the measured performance belong to GE HealthCare. Everything below stays at the level
> of publicly explainable method.

---

## 1. The clinical problem

In *structural* interventional cardiology — transcatheter aortic valve implantation, edge-to-edge
mitral repair, left atrial appendage closure — the procedure is performed on a beating heart,
without opening the chest, and needs two imaging modalities at once because neither is
sufficient alone:

- **Fluoroscopy** (continuous low-dose X-ray) shows the instruments and the prostheses, but is
  blind to soft tissue: the operator sees neither the valve to cross nor the septum to puncture.
- **Transesophageal echocardiography** shows those structures, but renders metallic instruments
  poorly.

Two specialists therefore work in parallel on two screens that have no geometric link between
them, and mentally reconstruct the correspondence during the procedure.

**Unified imaging** — overlaying the ultrasound cone at the right place on the X-ray image —
removes that mental step. It requires knowing, at every instant, the **pose of the ultrasound
probe** (its position and orientation) in the X-ray reference frame.

## 2. The existing approach: pose by template matching

The team had a deep-learning pipeline estimating this pose from a **single** fluoroscopy image,
with no additional tracking sensor. It does not regress the angles directly. Instead:

1. The X-ray image is recentred, aligned and resized, so translation, in-plane rotation and
   scale are already resolved upstream. Only the two out-of-plane rotation angles remain.
2. An **encoder** maps the image to a fixed-size latent vector.
3. During training, decoders reconstruct from that vector the *template* corresponding to the
   probe pose — a clean, noise-free rendering of the probe at a known angle.
4. At inference, the same vector is compared by **cosine similarity** against a bank of
   templates with known angles. The model outputs a **ranked list of candidates**, not a pose.

Quality is read as **Top-1 / Top-10 success** — is the correct template ranked first, or within
the first ten. A ranking is also inspected qualitatively as a similarity **heatmap** over the
angle space: a discriminative encoder produces a sharp peak on the true pose, a flat map means
it cannot separate poses even when its argmax happens to land in the right place.

## 3. The structural constraint that shapes everything

No sensor measures the probe pose during a real procedure. **Clinical images can therefore never
be annotated.** Training data is generated instead, by compositing a rendered probe at a known
angle onto a real clinical background — exact ground truth in a visually realistic context.

Validation runs on two sets built on the same probe angles, so they are directly comparable: a
synthetic one, and acquisitions made in a test room on anatomical phantoms — the only setting
offering both a physically acquired image and a controlled pose.

Tracking both in parallel, throughout training, measures the **domain gap** between the data the
model learns from and the data it will meet. That gap is the thread running through the whole
internship: axis 1 tries to shrink it with a better representation, axis 2 attacks it directly.

---

## 4. Axis 1 — Finding the best encoder to represent pose

The starting point was an in-house CNN encoder that already performed well. The question was
whether recent **foundation models** could match or beat it inside the same pipeline. This was
exploration and technology watch, not the repair of a known defect.

**What I did**

- **Structured literature review.** Rather than testing architectures as I read about them, I
  built a comparison spreadsheet: ~30 papers screened, 10 2D encoders retained, scored on the
  criteria that actually matter here — latent vector dimension against the one the pipeline
  expects, parameter count and inference cost, and above all the **nature of the pretraining
  data**.
- **Selected candidates on domain fit, not on headline benchmarks.** *RAD-DINO*, a
  self-supervised vision transformer pretrained mostly on chest radiographs — the same
  anatomical region as our use case. *FluoroSAM*, a Segment Anything variant trained on
  fluoroscopy — the exact image type of the pipeline. Later, on my supervisors' advice,
  *DINOv2*, a generalist model whose preprocessing recipe is fully documented.
- **Built one common protocol.** A single training pipeline and a single evaluation
  methodology, so that every encoder is compared on equal terms: same template bank, periodic
  evaluation on both domains during training, and heatmaps for qualitative inspection.
- **Swept the adaptation strategies**, from cheapest to most expensive: zero-shot → frozen
  backbone with a trained projection head → parameter-efficient fine-tuning with **LoRA**
  (sweeping rank and alpha) → full fine-tuning.

**The methodological lesson**

Pretrained encoders are extremely sensitive to **input preprocessing fidelity**. Their published
recipes are often incomplete, and reproducing the exact normalisation, resizing and channel
handling seen during pretraining turned out to matter more than the choice of adaptation
strategy — enough to reorder the ranking of the methods entirely. Diagnosing this, tracking down
the faithful recipe across reference implementations and re-running the comparison was the
turning point of the first axis.

Halfway through the internship I presented this work at the site's internal **CIFRE day**
(15-minute talk + Q&A, to an audience of researchers, PhD candidates and interns). The questions
raised there — on preprocessing reliability and on whether LoRA was the right tool — directly
redirected what I did next.

---

## 5. Axis 2 — Unsupervised domain adaptation

### The setting

**Unsupervised domain adaptation (UDA)** applies when you hold **labelled** data from a *source*
domain and **unlabelled** data from a different *target* domain, where the model will actually be
used. Same task, different image distributions — noise, texture, contrast, visual elements
present in the image. A model trained on the source silently learns to lean on
source-specific cues, and degrades as soon as it meets the target.

The goal is to make the encoder produce **domain-invariant latent representations**: if the
latent vector no longer carries the information of *which domain the image came from*, while
still carrying the information needed for the task, then a prediction head trained on the source
remains valid on the target.

### Why our case is not the textbook case

The overwhelming majority of published UDA work addresses **classification**, where the output is
one label among finitely many. Here the task is a **regression**: the latent vector feeds a
decoder that must reconstruct a template, scored by a continuous loss. Alignment mechanisms
designed to move decision boundaries between classes do not transpose mechanically to a
continuous output. The adaptation had to be formulated *and evaluated* in a regression setting.

### Designing a controlled benchmark — my contribution

The domain we ultimately care about is the clinical one, but it has no ground truth, so there is
no way to measure whether adaptation actually worked. I needed a source/target pair whose gap I
controlled and whose angles I knew on **both** sides.

I built that pair from a single template base, varying only **which physical parts of the probe
are present** in the image: the *insertion tube* on the source side; the *ring* and the
*shielding* of a newer probe model on the target side. Those parts produce opaque,
high-contrast structures that overlay the probe and partially mask the geometric landmarks the
encoder relies on. The shift is representative of what happens clinically, yet fully controlled
and reduced to a single variable. The target angles exist by construction but are **never
given to the model during training** — they serve only for post-hoc evaluation.

Stated plainly, the objective becomes: the model must stop keying on these parts, whichever
domain they come from, and keep only the geometry of the probe itself.

### The method

The adaptation builds on **FARR** (*Feature Alignment and Redundancy Reduction*), from my second
supervisor's work (paper under review, reference implementation public). The backbone is split at
a chosen block: the lower blocks form a **shared extractor ψ**, the upper blocks the **task head
f**, joined by the reconstruction **decoder g** and an **adversarial branch f_adv**, cloned from
the upper blocks at initialisation and reading the same intermediate output of ψ.

<!-- ▼ FIGURE — UDA architecture. Drop the file in figures/ and keep the name below. -->

<p align="center">
  <img src="figures/uda_architecture.png" alt="UDA architecture: shared feature extractor, self-adversarial representation heads, adversarial redundancy reduction, task-adaptable learning head" width="100%">
</p>

<p align="center"><em>
  Figure 1 — <!-- TODO: one-line caption of the diagram -->
</em></p>

**Reading the diagram.**
<!-- TODO: your own walkthrough, panel by panel. Suggested skeleton:
  1) Inputs / feature extraction — what x^S and x^T are, where ψ stops
  2) Self-adversarial representation heads — why f_adv is a clone of f at initialisation
  3) Adversarial redundancy reduction — what the cross-correlation matrices compare
  4) Task-adaptable learning head — what g reconstructs and against what label
-->

<!-- ▲ FIGURE -->

Each iteration chains three updates: the task path is updated on the regression loss, computed on
the labelled source only; the adversarial branch is trained to extract information decorrelated
from what the task head exploits; then ψ is optimised to satisfy the regression loss *and* the
alignment terms computed across both domains. f_adv surfaces what still distinguishes the
domains, ψ learns to erase it — a genuinely adversarial dynamic, and a delicate one to tune.

### What my experimental work consisted of

The method was theoretically grounded and validated on classification and segmentation tasks, on
other datasets. Nothing guaranteed it would behave as expected on a regression task, with our
backbone and our definition of domain shift — and there were no reference hyperparameter values
and no published results to compare against. Only experiments could settle it, and that is where
my contribution sits.

- **Integration** of the paper's architecture into the existing pose estimation pipeline,
  initialised from the best encoder configuration found in axis 1.
- **Automated hyperparameter search** with **Optuna**, Bayesian TPE sampling, run in two
  successive campaigns — loss weightings first at fixed learning rates, then the per-component
  learning rates at the best weightings found. Decomposing this way sharply reduces the
  dimensionality explored per campaign. Unpromising trials pruned on a median criterion;
  diverging trials recorded with a zero score so the sampler learns the divergence boundaries;
  the whole study persisted to a database so a search survives interruption.
- **Reasoning about the adversarial balance rather than only sweeping it.** The split depth
  between ψ and f_adv is not a neutral partition of layers: it decides how much capacity each
  side of a zero-sum game holds. Two similarly sized networks are hard to stabilise, and ψ has
  the harder job — staying good at the task *while* fooling its adversary. Moving the split much
  deeper, leaving the adversary a single block, exploits a known property of vision transformers
  (deep features are more abstract, hence already partly free of low-level domain cues) while
  keeping enough capacity to detect residual differences. Pairing this with a lower learning rate
  on the adversarial side follows standard adversarial-training practice.
- **Cross-architecture transfer.** I applied the same protocol to the in-house CNN encoder to
  test whether the gains generalise across encoder families, deriving the deepest admissible
  split point from its architecture so that the comparison with the transformer reproduced the
  same capacity logic.

---

## 6. What I took from it

- Working on a problem where **the ground truth cannot exist** changes the whole engineering
  posture: the design of the evaluation protocol becomes as much of a deliverable as the model.
- Reproducing a pretrained model faithfully is a research task in itself, and skipping it
  invalidates any comparison drawn on top of it.
- Adversarial training is less about the loss terms than about the **balance of capacity and
  learning speed** between the two sides.
- Rigorous experimental method —
