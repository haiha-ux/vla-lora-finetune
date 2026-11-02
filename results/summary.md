| arm | task | successes | rate | 95% CI (Wilson) | per eval seed (successes/episodes) | mean of seeds |
|---|---|---|---|---|---|---|
| lora_s0 | 5 | 39/90 | 43.3% | 33.6-53.6 | 2001: 14/30, 3001: 11/30, 4001: 14/30 | 43.3% |
| lora_s1 | 5 | 43/90 | 47.8% | 37.8-58.0 | 2001: 12/30, 3001: 12/30, 4001: 19/30 | 47.8% |
| pretrained | 5 | 40/90 | 44.4% | 34.6-54.7 | 2001: 15/30, 3001: 11/30, 4001: 14/30 | 44.4% |

| comparison | task | rate difference | 95% bootstrap CI | per eval seed |
|---|---|---|---|---|
| lora_s0-pretrained | 5 | -1.1 pts | -15.6 to +13.3 | 2001: -3, 3001: +0, 4001: +0 |
| lora_s1-pretrained | 5 | +3.3 pts | -11.1 to +17.8 | 2001: -10, 3001: +3, 4001: +17 |
