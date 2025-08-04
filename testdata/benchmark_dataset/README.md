# Benchmark Dataset

This directory contains benchmark datasets for evaluating ensemble judge performance.

## Files

- `sample_benchmark_dataset.json`: Generated test dataset with 500 questions across math, science, and general domains

## Dataset Structure

The benchmark dataset follows this JSON structure:

```json
{
  "questions": [
    {
      "id": "q0",
      "question": "What is 6 + 1?",
      "candidate_answers": [
        {"id": "q0_a1", "content": "7"},
        {"id": "q0_a2", "content": "8"}
      ],
      "ground_truth_answer_id": "q0_a1",
      "domain": "math",
      "difficulty": "easy"
    }
  ],
  "metadata": {
    "name": "Sample Benchmark Dataset",
    "version": "1.0",
    "license": "MIT",
    "source": "synthetic",
    "question_count": 500
  }
}
```

## Legal Notice

**IMPORTANT**: This is a synthetic dataset generated for testing purposes. For production benchmarks, a properly licensed dataset must be sourced and undergo legal review for license compatibility.

## Usage

Load the dataset using the benchmark utilities:

```go
dataset, err := testutils.LoadBenchmarkDataset("testdata/benchmark_dataset/sample_benchmark_dataset.json")
if err != nil {
    log.Fatal(err)
}
```

## Dataset Statistics

- **Total Questions**: 500
- **Domains**: math (167), science (167), general (166)
- **Difficulties**: easy (167), medium (167), hard (166)
- **Average Answers per Question**: 4.00
