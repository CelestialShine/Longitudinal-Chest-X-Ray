# Longitudinal-Chest-X-Ray
This is the implementation of Utilizing Longitudinal Chest X-Rays and Reports to Pre-Fill Radiology Reports.

The preprocess data of MIMIC-CXR is from https://github.com/cuhksz-nlp/R2Gen.

The structure of annotation.json is as follows:

{
    "train": [
        {
            "id": "XXXXXX",
            "study_id": XXXXXX,
            "subject_id": XXXXX,
            "report": "XXXXXXXXXX",
            "image_path": ["p10/p10000032/s50414267/XXXXXXXXX.jpg"],
            "split": "train"
        }
    ]
}

Wr arrange data based on the "StudyDate" from https://physionet.org/content/mimic-cxr-jpg/2.0.0/ Metadata files.

Run bash run_mimic_cxr.sh to train a model on the MIMIC-CXR data.
