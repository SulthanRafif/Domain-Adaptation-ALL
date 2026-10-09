# Classifier baseline tanpa Attention-CycleGAN

Baseline ini melatih classifier langsung pada `crop_foreground` C-NMC tanpa
translasi. Train dan validation source dipisah berdasarkan subjek; test
Taleqani dipisah berdasarkan `field_id`. Hasil target test hanya dipakai untuk
laporan akhir.

## 1. Susun split

Jalankan dari root project. Contoh memakai C-NMC bright dan ekstraksi Taleqani
CLAHE-Otsu yang sudah ada:

```bash
python scripts/build_dataset.py \
  --source-root data/preprocessed/CNMC_bright/crop_foreground \
  --source-mask-root data/preprocessed/CNMC_bright/crop_mask \
  --source-id-mode cnmc \
  --target-cells data/preprocessed/Taleqani_fullfield_clahe_otsu/crop_foreground \
  --target-report data/preprocessed/Taleqani_fullfield_clahe_otsu/extraction_report.csv \
  --out datasets/cnmc_bright2taleqani_foreground_baseline \
  --seed 42
```

Folder yang digunakan untuk baseline classifier:

```text
datasets/cnmc_bright2taleqani_foreground_baseline/clf/train  # source foreground
datasets/cnmc_bright2taleqani_foreground_baseline/clf/val    # source foreground
datasets/cnmc_bright2taleqani_foreground_baseline/clf/test   # target foreground
```

Builder menormalkan nama kelas target `Malignant/Benign` menjadi `ALL/Normal`,
agar mapping kelas sama untuk ResNet34. Ia juga mencatat pembagian pada
`split_manifest.csv`. Pemeriksaan manifest penting untuk memastikan tidak ada
subjek source yang terbagi antara train dan val, atau field target antara
adaptation dan test.

## 2. ResNet34 source-only

```bash
python eval/run_classifier.py \
  --train-dir datasets/cnmc_bright2taleqani_foreground_baseline/clf/train \
  --val-dir datasets/cnmc_bright2taleqani_foreground_baseline/clf/val \
  --test-dir datasets/cnmc_bright2taleqani_foreground_baseline/clf/test \
  --tag cnmc_foreground_source_only_resnet34 \
  --arch resnet34 \
  --img-size 128 \
  --seed 0
```

## 3. SVM dan Random Forest source-only

```bash
python eval/run_traditional_classifier.py \
  --train-dir datasets/cnmc_bright2taleqani_foreground_baseline/clf/train \
  --val-dir datasets/cnmc_bright2taleqani_foreground_baseline/clf/val \
  --test-dir datasets/cnmc_bright2taleqani_foreground_baseline/clf/test \
  --out-dir results/cnmc_foreground_source_only \
  --tag cnmc_foreground_source_only_svm_rf \
  --seed 0
```

SVM memakai `StandardScaler` yang di-fit hanya pada train. Parameter SVM dan
Random Forest dipilih dari validation dengan balanced accuracy. Kedua model
kemudian dievaluasi pada target test.

Untuk perbandingan adil dengan domain adaptation, gunakan split dataset yang
sama. Bandingkan baseline ini dengan eksperimen translasi pada target test yang
sama; jangan gunakan test untuk memilih epoch, fitur, atau hyperparameter.
