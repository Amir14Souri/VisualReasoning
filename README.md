# Question-Conditioned Attention Boxing

The provided model learns to localize answer-relevant image regions from VQA attention maps and converts them into bounding boxes for visual grounding and downstream evaluation.

## Pipeline

**It is recommended to run scripts in the following order to have a complete experience of the project.** You could also use the synthesized datasets and trained `final_model.pth` to to run the evaluations only, and see the results.

**Note:** All the arguments have default values and you can skip them, except for the `-k` in evaluation scripts. It is recommended to set it to `best_k` value obtained from the `tune_k` script.

### 1. Generate Training Data

Creates the training, validation, and test datasets for the `AttentionBoxer` model.

```bash
python -m scripts.generate_data.py --data-dir data/synthetic""
```

---

### 2. Train the Attention Boxer

Trains a model that predicts heatmaps directly from input image patch embeddings and question features. `clip-vit-base-patch16` model is used for feature extraction.

```bash
python -m scripts.train_attn_boxer.py --batch-size 16 --lr 1e-3 --weight-decay 1e-4 --num-epochs 30 --save-dir training/logs
```

**Note:** Additional to the `best_model.pth` saved in `save_dir` for each training process, a `final_model.pth` is also saved in the project's root directory.

---

### 3. Tune the Heatmap Threshold (`k`)

Searches for the best threshold parameter used during heatmap-to-bounding-box conversion. The exact applied rule is mentioned at the end.

```bash
python -m scripts.tune_k.py --batch-size 16 --model-path final_model.pth --save-dir evaluation/val_results
```

---

### 4. Evaluate on the Test Set

Evaluates localization performance using the trained model and tuned threshold on the test dataset. It also preforms a comparison between the model, random box baseline, and center box baseline overally and per question family. Reported metrics are **Mean IoU**, **IoU@0.3**, **IoU@0.5**, and **center-in-target**.

```bash
python -m scripts.evaluate_test.py -k 9 --batch-size 16 --model-path final_model.pth --save-dir evaluation/test_results
```

---

### 5. Evaluate on VQA

Evaluates the effect of the predicted regions on downstream VQA performance. `Qwen2.5-VL-3B-Instruct` model is used for this evaluation. Accuracy is reported for 3 different settings overally and per question family: **Full image**, **Attention-boxed image**, and **Oracle-boxed image**.

```bash
python -m scripts.evaluate_vqa.py -k 9 --batch-size 16 --model-path final_model.pth --save-dir downstream/vqa_results
```

---

## Heatmap to Bounding Box

The `heatmap_to_bbox()` function converts a soft attention heatmap into a single bounding box.

### Algorithm

1. Normalize the attention heatmap (softmax in AttentionBoxer).
2. Sort pixels by attention score.
3. Keep the `k` highest-scoring patches and ignore the rest.
4. Starting from the highest-scoring remaining patch, select all other patches that are neigbors to at least one of the current selected patches.
5. Repeat step 4 until including all possible remaining patches.
7. Compute the smallest enclosing rectangle.
8. Return `(x_min, y_min, x_max, y_max)` as bounding-box.

## Report

### 1. Results of validation-tuned hyper-parameter for heatmap-to-bbox conversion (`tune_k`)

Image for all validation data are saved in `--save-dir` with the ground-truth green boxes and predicted blue boxes. This is only for visualization and not used in any other scripts.

| k | Avg IoU | Avg CiT | IoU@0.3 | IoU@0.5 |
|:-:|:-:|:-:|:-:|:-:|
|  1 | 0.1053 | 0.57 | 0.03 | 0.00 |
|  4 | 0.3186 | 0.78 | 0.52 | 0.20 |
|  9 | 0.4149 | 0.90 | 0.75 | 0.34 |
| 16 | 0.3306 | 0.88 | 0.58 | 0.10 |
| 25 | 0.2430 | 0.61 | 0.34 | 0.04 |
| 36 | 0.1508 | 0.32 | 0.07 | 0.00 |

### 2. Results of grounding evaluation and baselines (`evaluate_test`)

|  | Attention Prediction | Random Box | Center Box |
|:-:|:--|:--|:--|
| Overall | Mean IoU = 0.417<br>IoU@0.3 = 0.765<br>IoU@0.5 = 0.39<br>Center-in-Target = 0.905 | Mean IoU = 0.003<br>IoU@0.3 = 0.0<br>IoU@0.5 = 0.0<br>Center-in-Target = 0.0 | Mean IoU = 0.028<br>IoU@0.3 = 0.02<br>IoU@0.5 = 0.0<br>Center-in-Target = 0.14 |
| Question Family<br>`Attribute` | Mean IoU = 0.397<br>IoU@0.3 = 0.723<br>IoU@0.5 = 0.356<br>Center-in-Target = 0.861 | Mean IoU = 0.003<br>IoU@0.3 = 0.0<br>IoU@0.5 = 0.0<br>Center-in-Target = 0.0 | Mean IoU = 0.029<br>IoU@0.3 = 0.02<br>IoU@0.5 = 0.0<br>Center-in-Target = 0.149 |
| Question Family<br>`Text-in-Shape` | Mean IoU = 0.436<br>IoU@0.3 = 0.808<br>IoU@0.5 = 0.424<br>Center-in-Target = 0.949 | Mean IoU = 0.002<br>IoU@0.3 = 0.0<br>IoU@0.5 = 0.0<br>Center-in-Target = 0.0 | Mean IoU = 0.026<br>IoU@0.3 = 0.02<br>IoU@0.5 = 0.0<br>Center-in-Target = 0.131 |

### 3. Results of downstream VQA evaluation (`evaluate_vqa`)

Two images for all 200 test data are saved in `--save-dir`: One with model-predicted box, and the other with ground-truth box (oracle box). These two images are used for prompting to the LLM. Accuracy values are reported in the table below.

|  | Full Image | Attention Box | Oracle Box |
|:-:|:-:|:-:|:-:|
| Overall | 79.5% | 81.5% | 67.5% |
| Question Family<br>`Attribute` | 65.3% | 71.3% | 53.5% |
| Question Family<br>`Text-in-Shape` | 93.9% | 91.9% | 81.8% |

It seems like marking the ground-truth box on the image is not helping the LLM at all. It might be because of the tight boxes around objects, and perhaps the box is not letting the model see some details it wants.

Comparing with raw images, marking attention-predicted boxes on the image does a tiny improvement to the overall accuracy (2%). Values say that it enhances reasoning in attribute questions better, with an approximately 6% improvement, but it does no good to the text-in-shape questions which are experiencing a 2% decrease in accuracy.
