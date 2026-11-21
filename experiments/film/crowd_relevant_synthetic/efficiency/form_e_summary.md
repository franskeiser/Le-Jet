| Model | Model size (MB) | Peak memory (MB) | Runtime, all (ms/msg) | Runtime, with link | Runtime, no link |
|---|---|---|---|---|---|
| Proposed (CNN–RF + FiLM) | 9.96 ± 1.69 | 49.7 ± 5.5 | 12.317 ± 1.140 | 13.736 ± 1.226 | 1.049 ± 0.091 |
| CNN-only (text) | 6.37 ± 0.01 | 20.8 ± 0.1 | 0.561 ± 0.025 | 0.559 ± 0.024 | 0.571 ± 0.028 |
| RF-only (Link RF) | 3.56 ± 1.69 | 28.4 ± 3.4 | 11.584 ± 0.278 | 11.584 ± 0.278 | - |
| Concatenation fusion | 9.92 ± 1.69 | 51.0 ± 3.4 | 11.631 ± 0.881 | 13.015 ± 1.031 | 0.679 ± 0.057 |
| KNN-RF-ETC voting (text only) | 19.90 ± 0.43 | 39.8 ± 1.4 | 33.616 ± 3.498 | 33.604 ± 3.525 | 33.717 ± 3.288 |
| Proposed, forest also computes the 37 contributions (extra) | 9.96 ± 1.69 | 55.1 ± 6.1 | 64.262 ± 2.862 | 72.235 ± 3.037 | 1.022 ± 0.035 |
| CNN-only, text + raw 37 features (extra) | 6.39 ± 0.01 | 21.4 ± 0.1 | 0.780 ± 0.020 | 0.779 ± 0.017 | 0.783 ± 0.041 |

Values are mean ± SD over the seeds (runs averaged within each seed).
