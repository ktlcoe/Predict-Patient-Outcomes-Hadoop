# Predict-Patient-Outcomes-Hadoop
# Inputs: CSV files containing anonymized patient data in the format (patient_id, event_id, event_description, timestamp, value) and mortality events in the format (patient_id, timestamp, label)
event id: Clinical event identifiers. For example, DRUG19122121 means that a drug
with RxNorm code as 19122121 was prescribed to the patient. DIAG319049 means
the patient was diagnosed of disease with SNOMED code of 319049 and LAB3026361
means that the laboratory test with a LOINC code of 3026361 was conducted on the
patient.

event description: Shows the description of the clinical event. For example, DIAG319049
is the code for Acute respiratory failure and DRUG19122121 is the code for Insulin.
# Output: Classifiers built using Hive and Pig, to predict patient mortality based on patient events
