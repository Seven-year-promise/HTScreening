## Drug screening based on PMR of zebrafish larvae

## This code is implemented by Python, and uses the following (parts) libraries:

- torch==1.4.0
- scikit-learn==1.0.2
- seaborn==0.12.2
- scikit-image==0.19.3
- scipy==1.6.2

## Exmplanation of some concepts:

trinary codes: 

- 'actions': the mode patterns from the biological knowlodge
- 'effect code': the mode patterns generated by the code-algorithm
- 'feture code': the trinary code for each feature

## How to use this code

pipeline:
rename --> data_clean --> combine_files --> FT_feature_select or Feature Integrations --> Feature_binary_coe_from_quantile or from integration