# Bird Egg Species Classification — Royal Alberta Museum (RAM)

**Team Just Imagine** | NorQuest College — Faculty of Business, Environment and Technology  
**Course:** CMPT 3835: Machine Learning Work Integrated Project II  
**Instructor:** Palwasha Afsar  
**Team Members:** David Barahona, Paula Frossard, Mubarak Farah, Jewel Gonzalez  
**Date:** March 15, 2026

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
   - [Data Collection](#data-collection)
   - [Species Overview](#species-overview)
   - [Data Cleaning](#data-cleaning)
   - [Class Imbalance](#class-imbalance)
3. [Data Labeling](#data-labeling)
4. [Dataset Splitting](#dataset-splitting)
   - [Approach 1: Augmentation-First (Image-Based)](#approach-1-augmentation-first-image-based)
   - [Approach 2: Split-First (Object-Based)](#approach-2-split-first-object-based)
5. [Data Augmentation](#data-augmentation)
6. [Machine Learning Pipeline](#machine-learning-pipeline)
   - [Model Selection](#model-selection)
   - [Training Configuration](#training-configuration)
   - [Evaluation Metrics and Results](#evaluation-metrics-and-results)
   - [Model Interpretation](#model-interpretation)
   - [Limitations](#limitations)
   - [Future Improvements](#future-improvements)
7. [References](#references)

---

## Project Overview

This project was developed over a four-month period in collaboration with the **Royal Alberta Museum (RAM)**. The goal is to build a machine learning model capable of classifying bird egg species from photographs. The classification scope was reduced from 300 species (held in the RAM collection) to **21 priority species** — selected because their eggs exhibit high visual variance, making them difficult to classify even for specialists, and therefore meaningful targets for automated classification.

The project covers the full machine learning pipeline: data collection, cleaning, labeling, splitting, augmentation, model training, and evaluation.

---

## Dataset

### Data Collection

The dataset was assembled from two main sources:

1. **Bird Eggs of Canada website** — Images were scraped from this website, which covers 405 bird species across 62 families. In some cases, the team also had access to higher-quality versions of the same photographs, resulting in an initial dataset that sometimes contained visually identical images at different quality levels.

2. **Royal Alberta Museum (RAM) physical collection** — At least one team representative visited the museum and photographed bird eggs directly from the physical collection. Images were captured from multiple angles using **5 different smartphones and one digital camera**, introducing variation in perspective, lighting conditions, and image quality to support better model generalization during training.

Due to the four-month project timeline, the classification scope was reduced from 300 to **21 priority species**. Our team was specifically assigned the following **5 species**:

---

### Species Overview

Our team's 5 assigned species, their egg attributes, and sources are described below. All species descriptions, egg characteristics, and typical clutch sizes are sourced from the Cornell Lab of Ornithology (n.d.) via the All About Birds database.

#### 1. *Agelaius phoeniceus* — Red-winged Blackbird

| Attribute | Description |
|---|---|
| Egg color | Pale blue-green to bluish-white |
| Markings | Dark brown, black, and purple scrawls and spots, concentrated at the larger end |
| Shape | Oval to short-oval |
| Size | Approximately 24 length × 17 mm width |
| Clutch size | 3–4 eggs |
| Notable features | Highly variable markings; eggs can look quite different within the same clutch |

#### 2. *Certhia americana* — Brown Creeper

| Attribute | Description |
|---|---|
| Egg color | White to creamy white |
| Markings | Small reddish-brown spots, often concentrated at the larger end |
| Shape | Oval |
| Size | Approximately 15 × 12 mm |
| Clutch size | 5–6 eggs |
| Notable features | Relatively small and plain; spots can be sparse and faint |

#### 3. *Cistothorus palustris* — Marsh Wren

| Attribute | Description |
|---|---|
| Egg color | Pale brown to cinnamon-brown |
| Markings | Dark brown speckles covering most of the surface, making the ground color difficult to see |
| Shape | Oval |
| Size | Approximately 16 × 12 mm |
| Clutch size | 4–6 eggs |
| Notable features | Heavily speckled appearance; among the darker-looking eggs in the dataset |

#### 4. *Euphagus cynocephalus* — Brewer's Blackbird

| Attribute | Description |
|---|---|
| Egg color | Pale gray to pale greenish-gray |
| Markings | Irregular brown and gray blotches and spots distributed across the surface |
| Shape | Oval |
| Size | Approximately 26 × 18 mm |
| Clutch size | 4–6 eggs |
| Notable features | Variable markings; visually similar to other blackbird species, making classification challenging |

#### 5. *Geothlypis tolmiei* — MacGillivray's Warbler

| Attribute | Description |
|---|---|
| Egg color | White to creamy white |
| Markings | Small reddish-brown to dark brown spots and specks, often concentrated at the larger end |
| Shape | Oval |
| Size | Approximately 19 × 14 mm |
| Clutch size | 3–5 eggs |
| Notable features | Relatively plain background with subtle markings |

---

### Data Cleaning

Before preparing the dataset for training, all images were manually inspected. Images were removed if they met any of the following criteria:

- **Duplicates** — Two or more images in the same species folder were visually identical
- **Incomplete framing** — The egg was partially cut off at the edge of the frame
- **Poor visibility** — The egg was not clearly visible due to blur, poor focus, or poor lighting
- **Incorrect content** — The image did not clearly show a bird's egg

Manual quality assurance checks were also performed to verify image clarity and confirm that eggs were correctly visible. In some cases, **manual cropping** was applied to improve framing before further processing.

#### Image Counts Per Assigned Species (After Cleaning, Before Augmentation)

| Species | Common Name | Images Before Augmentation | Target After Augmentation |
|---|---|---|---|
| *Agelaius phoeniceus* | Red-winged Blackbird | 112 | 150 |
| *Certhia americana* | Brown Creeper | 104 | 150 |
| *Cistothorus palustris* | Marsh Wren | 117 | 150 |
| *Euphagus cynocephalus* | Brewer's Blackbird | 269 | 150 |
| *Geothlypis tolmiei* | MacGillivray's Warbler | 114 | 150 |

---

### Class Imbalance

**What is class imbalance?**  
Class imbalance occurs when some classes in a dataset have significantly more samples than others (Goodfellow et al., 2016). When a model sees some species far more often during training, it becomes biased toward those species and performs poorly on under-represented ones.

**Why it affects this project:**  
Some species are more commonly photographed and more extensively documented in zoological literature, while others are rarer or less frequently studied. This made the raw dataset inherently imbalanced.

**How it was addressed — two strategies:**

1. **Setting a uniform target** — A target of 150 images per species was established for all 21 classes.
2. **Over- and under-sampling** — For species with more than 150 images after cleaning, excess images were removed. For species with fewer than 150, augmentation was applied to generate additional samples.
3. **Data augmentation** — Augmentation techniques were used to artificially increase the number of samples for under-represented species by generating new, slightly varied versions of existing images (detailed in the [Data Augmentation](#data-augmentation) section).

---

## Data Labeling

### Scientific Names vs. Common Names

**Scientific names** (binomial names) are standardized two-part names used in biology to uniquely identify a species, following the system established by Carl Linnaeus in the 18th century (Mayr, 1997). They consist of the genus name followed by the species epithet (e.g., *Agelaius phoeniceus*). Scientific names are universal across all languages and eliminate ambiguity.

**Common names** are everyday names used in a particular language or region (e.g., "Red-winged Blackbird"). They are not standardized — the same species can have different common names in different regions, and the same common name can refer to different species in different parts of the world (Sibley, 2000).

**Decision:** Scientific names were used for all class labels throughout this project to ensure consistency and avoid ambiguity arising from regional or informal common names, consistent with practice followed in Davie (1898) and standard ornithological literature.

---

### Labeling Tool: LabelImg

Annotations were created using **LabelImg** (Tzutalin, 2015), a free, open-source graphical image annotation tool designed for object detection tasks. It supports drawing bounding boxes and exporting in standard formats, including the **YOLO format** required by this project. LabelImg operates as a standalone desktop application with no internet connection required, protecting client data privacy.

### Step-by-Step Annotation Process

**1. Installation and Setup**  
LabelImg was installed and launched as a desktop application. The YOLO output format was selected in tool settings to ensure compatibility with the Ultralytics YOLO framework (Jocher et al., 2023).

**2. Class Label Configuration**  
A `classes.txt` file was created containing all 21 species scientific names, one per line. This file was shared between all teams to ensure class IDs were consistent across all annotation files.

**3. Image Loading**  
Images for each assigned species were loaded into LabelImg by pointing the tool to the relevant species directory.

**4. Bounding Box Annotation**  
For each image, bounding boxes were drawn around every visible egg following these rules:
- Every egg visible in the image received its own bounding box
- Boxes were drawn as tightly as possible around the egg without cutting off any part of it
- Images containing multiple eggs received one separate bounding box and label per egg
- The correct species scientific name was assigned to each box from the class label list
- Consistent box tightness was maintained across all annotators to ensure uniformity

**5. Saving and Export**  
LabelImg automatically saves each image's annotations as a `.txt` file in YOLO format in the same directory as the image. Each annotation file contains one line per annotated object. All coordinate values are normalized between 0 and 1 relative to image dimensions (Redmon et al., 2016).

**Example annotation line:**
```
0 0.512 0.487 0.243 0.391
```
This indicates: class `0` (first species in the class list), with the bounding box centered at 51.2% from the left and 48.7% from the top of the image, with a width of 24.3% and height of 39.1% of total image dimensions.

---

## Dataset Splitting

The project's total target was **3,150 images** across all 21 species (150 per species), and **750 images** across our 5 assigned species. Two distinct dataset preparation workflows were applied and compared.

### Approach 1: Augmentation-First (Image-Based)

This approach was followed at the team-level dataset preparation stage, as directed by the course instructor. Each team was responsible only for their assigned species (5 in our case).

**Key characteristics:**
1. The dataset was treated at the **image level** — each image file was considered a separate, independent sample.
2. Images that appeared to be the same photograph at different quality levels were kept as separate samples.
3. Augmentation was applied **before** splitting.
4. To prevent original images appearing in both train and validation/test sets, original images and their augmented versions were kept in the same split.
5. Scope was limited to the 5 species assigned to our team.

**Split Proportions:**

| Split | Proportion | Contains |
|---|---|---|
| Training | 70% | Original + augmented images |
| Validation | 10% | Original + augmented images |
| Test | 20% | Original images + augmented ones |

---

### Approach 2: Split-First (Object-Based)

This approach was developed independently by our team based on a review of best practices for machine learning dataset preparation, guided by industry-standard best practices for YOLOv8 (Lövström, 2024). It was applied to the **entire dataset of 21 species**, not just our assigned five, and involved reorganizing and re-cleaning the full dataset from scratch.

**Key characteristics:**
1. The dataset was treated at the **object (egg) level** — individual annotated eggs, not image files, were the unit of analysis.
2. The split was performed **before** augmentation to prevent data leakage.
3. Images were cropped so that each image contained only a single egg. In cases where eggs were clustered closely together, some images still contained partial views of neighboring eggs.
4. Augmentation was applied **only to the training set** — validation and test sets contain only original, real images.
5. A critical constraint was applied: **all eggs from the same original photograph were always kept in the same split**, preventing the model from being evaluated on images visually identical to ones seen during training.
   - *Example:* An image of *Agelaius phoeniceus* originally containing three eggs, after cropping, becomes three separate images — all three must remain within the same data split.
6. A known compromise: images of the same egg taken from very similar angles and lighting conditions but with different camera devices could potentially lead to data leakage. This was accepted as a necessary trade-off given the limited dataset and project timeline.
7. Scope covered all **21 species** in the shared dataset.

**Split Proportions:**

| Split | Proportion | Contains |
|---|---|---|
| Training | 80% | Original + augmented images |
| Validation | 10% | Original images only |
| Test | 10% | Original images only |

#### Species Example: *Setophaga ruticilla* (American Redstart)

*Setophaga ruticilla* had the smallest number of egg pictures in the dataset at the time of preparation.

| Stage | Count |
|---|---|
| Original images | 29 |
| Annotated eggs | 59 |
| Target egg samples | 150 |
| Assigned to validation (10%) | 15 eggs (original only, no augmentation) |
| Assigned to test (10%) | 15 eggs (original only, no augmentation) |
| Assigned for training (to be augmented) | 29 eggs |
| Training eggs after augmentation | 120 eggs |
| Total final dataset | 150 eggs |

**Key rule:** Eggs coming from the same original image must remain in the same dataset split to prevent the model from being tested on data visually identical to training images.

---

## Data Augmentation

### Image Counts Before Augmentation (Full Dataset — Selected Species)

| Species | Common Name | Eggs Before Augmentation |
|---|---|---|
| *Agelaius phoeniceus* | Red-winged Blackbird | 227 |
| *Certhia americana* | Brown Creeper | 215 |
| *Cistothorus palustris* | Marsh Wren | 229 |
| *Euphagus cynocephalus* | Brewer's Blackbird | ≈ 300 |
| *Geothlypis tolmiei* | MacGillivray's Warbler | 237 |
| *Setophaga ruticilla* (reference) | American Redstart | 59 (minimum in full dataset) |

### What is Augmentation?

Data augmentation is a technique used to increase the size and diversity of a training dataset by applying controlled transformations to existing images, generating new samples that are visually varied but still representative of the same class (Shorten & Khoshgoftaar, 2019). It is a widely accepted strategy for addressing limited dataset sizes in image classification and object detection (LeCun et al., 2015).

Under **Approach 2**, augmentation was applied only to the training set (split-first). Under **Approach 1**, augmentation was applied before splitting, as directed by the course instructor. The target for all species was **150 samples**.

### Why 150 Images Per Species?

The target was established based on:
1. **Data availability** — The total number of unique, high-quality images available for many species was limited, often falling below 100 per species.
2. **Project timeline** — With less than four months available, collecting or manually cleaning several hundred images per species was not feasible.
3. **Balancing quality and quantity** — 150 images per species was determined to be a practical minimum that, combined with augmentation, could support meaningful model training, consistent with recommendations for small-dataset deep learning scenarios (Shorten & Khoshgoftaar, 2019).

### Augmentation Techniques Applied

All techniques were selected to introduce realistic visual variation while preserving the key features used for species identification — specifically egg color, surface markings, and shape — as identified by Davie (1898) as the primary distinguishing characteristics across all North American bird egg species.

| Augmentation Type | Purpose | Source |
|---|---|---|
| **Rotation** | Simulates different orientations; bird eggs are roughly oval with no fixed orientation in photographs | Shorten & Khoshgoftaar (2019) |
| **Flipping** (horizontal and/or vertical) | Doubles viewpoints; valid for symmetric objects with no inherent directional orientation | Shorten & Khoshgoftaar (2019) |
| **Brightness and Contrast Adjustments** | Simulates differences in lighting conditions (e.g., controlled museum lighting at RAM vs. outdoor/natural light) | Shorten & Khoshgoftaar (2019) |
| **Zoom Variations** (random zoom-in/out) | Simulates variation in camera distance; realistic given images were captured using 6 different devices at varying distances | Shorten & Khoshgoftaar (2019) |

---

## Machine Learning Pipeline

### Model Selection

#### From Object Detection to Image Classification

The pipeline underwent a significant methodological transition during development. Initially, the team explored **object detection using YOLOv8** to identify multiple eggs within a single image. However, this approach introduced considerable complexity:
- Many images contained multiple eggs requiring precise bounding box annotations.
- EXIF orientation inconsistencies across images from six different devices made annotation error-prone and time-consuming.

Given the strict four-month timeline, the problem was reformulated as an **image classification task**. Images were cropped so each contained a single egg, converting the multi-object detection problem into a single-label image classification problem. This eliminated the need for bounding box annotations in the final model and significantly reduced pipeline complexity.

#### Architecture Selection: ResNet-50 with Transfer Learning

Three model architectures were evaluated: **ResNet-50**, **EfficientNet-B0**, and **MobileNetV3**. Transfer learning was selected over a custom CNN for two primary reasons:
1. The limited dataset size made training from scratch infeasible.
2. Transfer learning allowed the team to leverage feature representations previously learned from the ImageNet dataset (over 1.2 million labeled images across 1,000 categories) (LeCun et al., 2015).

**ResNet-50** was selected as the primary model. Its residual (skip) connections allow it to train deep networks without the vanishing gradient problem, making it well-suited for learning fine-grained visual features that distinguish bird egg species — particularly color patterns, surface texture, and marking distribution (He et al., 2016).

EfficientNet-B0 and MobileNetV3 were included as lightweight alternatives optimized for efficiency, enabling a performance-versus-complexity trade-off analysis.

All three models were initialized with pre-trained ImageNet weights. The classification head was replaced with a fully connected output layer sized to match the **21-species class count**.

| Architecture | Type | Pre-trained | Rationale |
|---|---|---|---|
| ResNet-50 | Deep residual network | ImageNet | Primary model; strong feature learning; handles vanishing gradients |
| EfficientNet-B0 | Lightweight CNN | ImageNet | Efficient baseline; fast training and inference |
| MobileNetV3 | Mobile-optimized CNN | ImageNet | Lightweight alternative; useful for deployment comparison |

---

### Training Configuration

#### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Batch size | 4 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Input image size | 224 × 224 pixels |
| Early stopping patience | 7 epochs |

The **Adam optimizer** was selected for its adaptive learning rate capabilities, which are particularly effective in small-dataset scenarios (Kingma & Ba, 2015). An input size of **224 × 224 pixels** is the standard for ImageNet-pretrained models and was used consistently across all three architectures.

#### Training Environment

Training was conducted locally on a **GPU-enabled workstation** using VS Code as the development environment. Each training run took approximately **1 to 1.5 hours** to complete. No cloud infrastructure was used.

#### Regularization and Fine-Tuning

- **Early stopping** was applied with a patience of 7 epochs to prevent overfitting.
- Layer freezing and progressive fine-tuning were **not applied**, as initial training results were already strong.
- No additional regularization techniques such as dropout or weight decay were incorporated in the current iteration. These omissions are identified as known limitations (see [Limitations](#limitations)).

#### Dataset Strategy

Training was conducted using the **Approach 2 (Split-First, Object-Based)** dataset. Each training image contained a single cropped egg, augmented images were used only in training, and validation and test sets contained exclusively original, unmodified images.

---

### Evaluation Metrics and Results

#### Metrics Used

- **Accuracy** — Overall proportion of correctly classified samples
- **Precision (Macro)** — Average precision across all classes, weighted equally
- **Recall (Macro)** — Average recall across all classes, weighted equally
- **F1 Score (Macro)** — Harmonic mean of precision and recall
- **mAP@0.5** — Mean Average Precision at an IoU threshold of 0.5
- **mAP@0.5:0.95** — Mean Average Precision averaged across IoU thresholds from 0.5 to 0.95
- **Mean IoU** — Average Intersection over Union across all predictions

#### Overall Performance — ResNet-50 (Primary Model)

| Metric | Score |
|---|---|
| Overall Accuracy | 97% |
| Precision (Macro) | 97% |
| Recall (Macro) | 97% |
| F1 Score (Macro) | 97% |
| mAP@0.5 | 99% |
| mAP@0.5:0.95 | 91% |
| Mean IoU | 91% |

Validation and test performance were consistent, with no significant performance gap observed between the two splits, indicating the model generalized well to unseen data and that the split-first methodology was effective in preventing data leakage.

#### Per-Class Performance Highlights

| Performance Level | Species | Accuracy |
|---|---|---|
| Best performing | White-throated Sparrow (*Zonotrichia albicollis*) | 100% |
| Most challenging | MacGillivray's Warbler (*Geothlypis tolmiei*) | 83% |

The lower accuracy for MacGillivray's Warbler is consistent with its egg attributes (plain white-to-creamy white background with subtle markings) that are visually similar to several other species in the dataset.

#### Comparative Architecture Results

ResNet-50 achieved the highest overall accuracy. EfficientNet-B0 and MobileNetV3 offered faster inference times at a modest reduction in classification performance — a relevant trade-off for future deployment scenarios where computational resources may be limited.

---

### Model Interpretation

#### Confusion Matrix and Precision-Recall Curves

A confusion matrix was generated to identify misclassification patterns between species. Precision-Recall (PR) curves were produced on a per-class basis as well as an average to evaluate performance trade-offs between false positives and false negatives for each species. These visualizations confirmed that the majority of misclassifications occurred between species with overlapping visual characteristics — particularly among the blackbird species (*Passer domesticus* and *Molothrus ater*), whose eggs share similar oval shapes and muted color palettes with variable brown markings.

#### Background Bias

A notable finding during model evaluation was that the model appeared to rely not exclusively on the egg itself but also on background features present in images. This suggests **background bias** — a known failure mode in image classification models trained on datasets where the background is not sufficiently varied (Ribeiro et al., 2016). Because museum specimen images tend to share similar neutral-grey or white backgrounds, the model may have learned to associate these background patterns with certain species rather than relying solely on egg morphology. This highlights the need for greater background diversity in future data collection efforts.

---

### Limitations

- **No layer freezing or fine-tuning strategy** — All layers were trained simultaneously from the start. A progressive unfreezing approach could allow the model to better adapt pretrained features to the egg classification domain.
- **No regularization** — The absence of dropout or weight decay increases the risk of overfitting, particularly given the small dataset size.
- **Limited dataset size** — With a target of only 150 samples per species, the model's generalization may be constrained. Deep learning models typically benefit from substantially larger datasets.
- **Background bias** — The model may be relying on background context rather than egg morphology alone, which would reduce reliability in real-world deployment scenarios.
- **Single-device evaluation** — All test images were drawn from the same pool of photographs taken during museum visits. Performance on images captured in different environments or with different lighting conditions has not been evaluated.

---

### Future Improvements

1. **Implement layer freezing and progressive fine-tuning** — Freezing early layers during initial training and gradually unfreezing them can improve adaptation of pretrained features to the egg classification domain.
2. **Add regularization techniques** — Incorporating dropout layers and weight decay into the training pipeline would reduce overfitting risk.
3. **Evaluate additional architectures** — Testing models such as EfficientNet-B3 or Vision Transformers (ViT) could yield performance improvements, particularly as dataset size grows.
4. **Increase dataset size and background diversity** — Collecting more images — particularly from varied lighting conditions and backgrounds — would reduce background bias and improve generalization.
5. **Incorporate more Explainable AI** — Implementing SHAP would allow the team and museum staff to verify that the model is classifying eggs based on correct morphological features, increasing interpretability and stakeholder trust.
6. **Expand to the full 300-species scope** — As data collection matures and the model is validated on the current 21-species benchmark, the classification scope can be progressively expanded toward the museum's full collection.

---

## References

Cornell Lab of Ornithology. (2024). *All about birds: Online bird guide*. Cornell University. https://www.allaboutbirds.org

Davie, O. (1898). *Nests and eggs of North American birds* (5th ed.). David McKay.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. https://doi.org/10.1109/CVPR.2016.90

Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (Version 8.0). Ultralytics. https://github.com/ultralytics/ultralytics

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1412.6980

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature, 521*(7553), 436–444. https://doi.org/10.1038/nature14539

Lövström, O. (2024, April 24). YOLOv8: Best practices for training. *Medium*. https://medium.com/internet-of-technology/yolov8-best-practices-for-training-cdb6eacf7e4f

Mayr, E. (1997). *This is biology: The science of the living world*. Harvard University Press.

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 779–788. https://doi.org/10.1109/CVPR.2016.91

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144. https://doi.org/10.1145/2939672.2939778

Sibley, D. A. (2000). *The Sibley guide to birds*. Alfred A. Knopf.

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data, 6*(1), 1–48. https://doi.org/10.1186/s40537-019-0197-0

Tzutalin. (2015). *LabelImg: Image annotation tool* [Computer software]. GitHub. https://github.com/tzutalin/labelImg
