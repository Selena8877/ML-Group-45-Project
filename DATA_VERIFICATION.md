Milestone 6 – Dataset and Preprocessing Verification

Group: 45
Date: November 27, 2026
Responsibility: Person 1 – Dataset and Preprocessing
Repository commit tested: 48dda1b

Objective
Please ensure that the currently implemented preprocessing pipeline generates consistent datasets for all four languages using the same previously processed dataset files.

Verification Procedure

Examined the language-specific cleaning documents and src/data_pipeline.py.

Checked sizes and distribution of training, validation, and test sets.

Re-created both masked and unmasked datasets in a temporary directory using the pipeline.

Ran text overlap, Chinese user overlap, and test set balancing tests from the pipeline.

Performed a comparison between the generated CSV files and those found in the repository using pandas.testing.assert_frame_equal().

Results
Component                       Passed        Failed
Arabic Datasets                   6              0
Chinese Datasets                  6              0
Portuguese Datasets               6              0
Thai Datasets                     6              0   
pipeline logs                     2              0
split summaries                   2              0


Conclusion

Each of the 28 newly generated CSV files contained the same data as those previously existing. Masked and unmasked preprocessors successfully ran with leakage test included. The verification has been done within the existing local Python installation without modifying any dataset.

Preprocessing code has proved to be stable after these tests; no changes have been done to its implementation. Fresh-clone environment verification is still included in the final review of the repository.