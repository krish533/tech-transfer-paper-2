# Cross-repository replication verification

- Paper 1 source commit: `25a9472b34334825b6d6c6a334f5b88eb00695b5`
- Paper 2 base commit: `88cf4a1c02540b136adb8beaa35e212625ac755e`
- Paper 1 canonical panel SHA-256: `694c21acc07d2a50ed27199d0e7ec01bb6974f08f843cbce2d7da4318f864198`
- Paper 2 merged dataset SHA-256: `070f10179b34a0730d1f09ea3322d21e8f32e20d3036760df2adfef182aa2d4b`
- Verified replication dataset SHA-256: `de5377b98c1261c5de9a5b4df21efb86af04bbdce28cb3676ec37c61456fcdf6`

## Paper 1 checks

- Full canonical panel: **4,296 rows, 150 institutions**.
- Primary 1944-2025 panel: **4,277 rows, 150 institutions**.
- Directly observed 1944-2025 records: **480**.

## Cross-repository merge checks

- Paper 2 institution-name key selected: `Institution_pci`.
- Paper 2 rows with PCI: **2,564**.
- Rows matched to Paper 1 on standardized institution + calendar year: **2,564**.
- Compared fields: Mean_Tone_Score, Median_Tone_Score, Tone_Index, Clarity_Index, Legal_Load_Index, n_sentences, n_words, Source_Year, Is_Carried_Forward.
- Field mismatch counts: `{"Clarity_Index": 0, "Is_Carried_Forward": 0, "Legal_Load_Index": 0, "Mean_Tone_Score": 0, "Median_Tone_Score": 0, "Source_Year": 0, "Tone_Index": 0, "n_sentences": 0, "n_words": 0}`.

## Paper 2 baseline re-estimation

- Lag-1 PCSI coefficient on ln(1 + new patent applications): **0.615866**.
- Clustered SE: **0.523368**.
- p-value: **0.240590**.
- N: **2,115**.

## New analysis-ready dataset

`data/verified_replication_dataset.csv` contains **3,507 rows and 46 columns**. It is rebuilt from the original Paper 2 merge, uses calendar-year lags from the Paper 2 replication code, and includes commit provenance plus a row-level Paper 1 key-verification flag.
