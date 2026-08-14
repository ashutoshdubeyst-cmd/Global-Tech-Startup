# Test data

These CSV files are small synthetic fixtures. They contain no production or
personal data.

- `startups_cleaned.csv` contains balanced labelled rows for training,
  validation, and test splitting.
- `new_startups_cleaned.csv` contains unlabelled rows for inference and includes
  categories not seen during training to verify that the encoder handles them.
