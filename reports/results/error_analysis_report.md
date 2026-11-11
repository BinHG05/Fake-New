# Error Analysis Report

- Overall accuracy: **87.96%**
- REAL precision/recall: **91.99% / 91.03%**
- FAKE precision/recall: **78.33% / 80.34%**

## Accuracy by Graph Size

- `0` comments: 87.37% accuracy on 190 samples
- `1-10` comments: 83.93% accuracy on 112 samples
- `>10` comments: 93.33% accuracy on 105 samples

## Confidence Bands

- `low`: 0.00% accuracy, avg confidence 0.00%, count 0
- `medium`: 69.85% accuracy, avg confidence 61.66%, count 136
- `high`: 97.05% accuracy, avg confidence 83.74%, count 271

## Top False Positives

- `4jzad2` | conf=0.8200 | comments=2 | depth=1 | text=`holly madison reveals secrets from her relationship with hugh hefner this is disgusting consider us all scarred for life after hearing this this is disturbing`
- `6n3zgv` | conf=0.7570 | comments=0 | depth=0 | text=`beautiful rock cairn found on the trail of tears signifying the burial place of an ancient native american warchief`
- `cet6zi` | conf=0.7492 | comments=0 | depth=0 | text=`the master at her canvas mom just brought in a pretty averagelooking rock from the woods that she says she will use in a project`
- `8r9up8` | conf=0.7272 | comments=0 | depth=0 | text=`in post net neutrality america ajit pai seeks a new life in the pizza industry`
- `6pkhv5` | conf=0.7264 | comments=0 | depth=0 | text=`bobby riggs knocks out billie jean king after losing straight sets in the famous battle of the sexes tennis match`

## Top False Negatives

- `143c94` | conf=0.8031 | comments=0 | depth=0 | text=`the shadow`
- `23foyn` | conf=0.8031 | comments=0 | depth=0 | text=`facebomb`
- `4eax09` | conf=0.7394 | comments=1 | depth=1 | text=`wolf next to dead bison`
- `b1r4mh` | conf=0.6610 | comments=1 | depth=1 | text=`blunt smoking son of alleged college admissions cheat gets his rap cd reviewed by nyp`
- `cht36p` | conf=0.6566 | comments=0 | depth=0 | text=`we can do a lot more navy vet walks to raise awareness of veteran suicide homelessness`

## Interpretation

The report is designed to answer whether the checkpoint is biased toward one class,
whether larger cascades actually reduce error, and whether the confidence score is meaningful.

```json
{
  "overall_accuracy": 0.8796068796068796,
  "confusion": {
    "tn": 264,
    "fp": 26,
    "fn": 23,
    "tp": 94
  },
  "class_metrics": {
    "real": {
      "precision": 0.9198606271777003,
      "recall": 0.9103448275862069
    },
    "fake": {
      "precision": 0.7833333333333333,
      "recall": 0.8034188034188035
    }
  },
  "accuracy_by_graph_size": {
    "0": {
      "count": 190,
      "accuracy": 0.8736842105263158
    },
    "1-10": {
      "count": 112,
      "accuracy": 0.8392857142857143
    },
    ">10": {
      "count": 105,
      "accuracy": 0.9333333333333333
    }
  },
  "confidence_bands": {
    "low": {
      "count": 0,
      "accuracy": 0.0,
      "avg_confidence": 0.0
    },
    "medium": {
      "count": 136,
      "accuracy": 0.6985294117647058,
      "avg_confidence": 0.616627198370064
    },
    "high": {
      "count": 271,
      "accuracy": 0.9704797047970479,
      "avg_confidence": 0.8374148382911822
    }
  },
  "false_positive_cases": [
    {
      "id": "4jzad2",
      "expected": "REAL",
      "predicted": "FAKE",
      "confidence": 0.82,
      "prob_real": 0.18,
      "prob_fake": 0.82,
      "comments": 2,
      "depth": 1,
      "branching_factor": 2.0,
      "average_degree": 1.333,
      "text_preview": "holly madison reveals secrets from her relationship with hugh hefner this is disgusting consider us all scarred for life after hearing this this is disturbing"
    },
    {
      "id": "6n3zgv",
      "expected": "REAL",
      "predicted": "FAKE",
      "confidence": 0.757,
      "prob_real": 0.243,
      "prob_fake": 0.757,
      "comments": 0,
      "depth": 0,
      "branching_factor": 0.0,
      "average_degree": 2.0,
      "text_preview": "beautiful rock cairn found on the trail of tears signifying the burial place of an ancient native american warchief"
    },
    {
      "id": "cet6zi",
      "expected": "REAL",
      "predicted": "FAKE",
      "confidence": 0.7492,
      "prob_real": 0.2508,
      "prob_fake": 0.7492,
      "comments": 0,
      "depth": 0,
      "branching_factor": 0.0,
      "average_degree": 2.0,
      "text_preview": "the master at her canvas mom just brought in a pretty averagelooking rock from the woods that she says she will use in a project"
    },
    {
      "id": "8r9up8",
      "expected": "REAL",
      "predicted": "FAKE",
      "confidence": 0.7272,
      "prob_real": 0.2728,
      "prob_fake": 0.7272,
      "comments": 0,
      "depth": 0,
      "branching_factor": 0.0,
      "average_degree": 2.0,
      "text_preview": "in post net neutrality america ajit pai seeks a new life in the pizza industry"
    },
    {
      "id": "6pkhv5",
      "expected": "REAL",
      "predicted": "FAKE",
      "confidence": 0.7264,
      "prob_real": 0.2736,
      "prob_fake": 0.7264,
      "comments": 0,
      "depth": 0,
      "branching_factor": 0.0,
      "average_degree": 2.0,
      "text_preview": "bobby riggs knocks out billie jean king after losing straight sets in the famous battle of the sexes tennis match"
    }
  ],
  "false_negative_cases": [
    {
      "id": "143c94",
      "expected": "FAKE",
      "predicted": "REAL",
      "confidence": 0.8031,
      "prob_real": 0.8031,
      "prob_fake": 0.1969,
      "comments": 0,
      "depth": 0,
      "branching_factor": 0.0,
      "average_degree": 2.0,
      "text_preview": "the shadow"
    },
    {
      "id": "23foyn",
      "expected": "FAKE",
      "predicted": "REAL",
      "confidence": 0.8031,
      "prob_real": 0.8031,
      "prob_fake": 0.1969,
      "comments": 0,
      "depth": 0,
      "branching_factor": 0.0,
      "average_degree": 2.0,
      "text_preview": "facebomb"
    },
    {
      "id": "4eax09",
      "expected": "FAKE",
      "predicted": "REAL",
      "confidence": 0.7394,
      "prob_real": 0.7394,
      "prob_fake": 0.2606,
      "comments": 1,
      "depth": 1,
      "branching_factor": 1.0,
      "average_degree": 1.0,
      "text_preview": "wolf next to dead bison"
    },
    {
      "id": "b1r4mh",
      "expected": "FAKE",
      "predicted": "REAL",
      "confidence": 0.661,
      "prob_real": 0.661,
      "prob_fake": 0.339,
      "comments": 1,
      "depth": 1,
      "branching_factor": 1.0,
      "average_degree": 1.0,
      "text_preview": "blunt smoking son of alleged college admissions cheat gets his rap cd reviewed by nyp"
    },
    {
      "id": "cht36p",
      "expected": "FAKE",
      "predicted": "REAL",
      "confidence": 0.6566,
      "prob_real": 0.6566,
      "prob_fake": 0.3434,
      "comments": 0,
      "depth": 0,
      "branching_factor": 0.0,
      "average_degree": 2.0,
      "text_preview": "we can do a lot more navy vet walks to raise awareness of veteran suicide homelessness"
    }
  ]
}
```
