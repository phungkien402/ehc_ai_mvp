# Source-Miss Analysis (455-case run)

- Total evaluated: 455
- Source misses: 25
- Source miss rate: 5.49%

## Miss Class Distribution
- near_duplicate_intent_wrong_ticket: 18
- partially_relevant_wrong_ticket: 3
- no_source_returned: 2
- off_target_answer: 2

## Most Frequent Expected IDs in Misses
- expected #19136: 1
- expected #19085: 1
- expected #19023: 1
- expected #18813: 1
- expected #18811: 1
- expected #18801: 1
- expected #18689: 1
- expected #18666: 1
- expected #18584: 1
- expected #18579: 1

## Most Frequent Predicted IDs in Misses
- predicted #: 2
- predicted #17941: 2
- predicted #18917: 1
- predicted #18035: 1
- predicted #18814: 1
- predicted #18812: 1
- predicted #18511: 1
- predicted #17901: 1
- predicted #18581: 1
- predicted #19086: 1

## Frequent Wrong Pair Mappings
- expected #19136 -> predicted #18917: 1
- expected #19085 -> predicted #18035: 1
- expected #19023 -> predicted #(none): 1
- expected #18813 -> predicted #18814: 1
- expected #18811 -> predicted #18812: 1
- expected #18801 -> predicted #17941: 1
- expected #18689 -> predicted #17941: 1
- expected #18666 -> predicted #18511: 1
- expected #18584 -> predicted #17901: 1
- expected #18579 -> predicted #18581: 1
- expected #18568 -> predicted #19086: 1
- expected #18503 -> predicted #18504: 1
- expected #18444 -> predicted #18445: 1
- expected #18336 -> predicted #(none): 1
- expected #18267 -> predicted #18268: 1

## 25 Miss Cases (Compact)
- ticket_18240: exp #18240 -> pred #18243 | class=near_duplicate_intent_wrong_ticket | sem=0.8253 | cov=0.5000 | ovl=0.5556
- ticket_18012: exp #18012 -> pred #18267 | class=near_duplicate_intent_wrong_ticket | sem=0.8112 | cov=0.6667 | ovl=1.0000
- ticket_18267: exp #18267 -> pred #18268 | class=near_duplicate_intent_wrong_ticket | sem=0.8453 | cov=0.7143 | ovl=1.0000
- ticket_17793: exp #17793 -> pred #17944 | class=near_duplicate_intent_wrong_ticket | sem=0.8199 | cov=0.8235 | ovl=0.6667
- ticket_18055: exp #18055 -> pred #18131 | class=near_duplicate_intent_wrong_ticket | sem=0.8641 | cov=0.7500 | ovl=0.7500
- ticket_18444: exp #18444 -> pred #18445 | class=near_duplicate_intent_wrong_ticket | sem=0.8668 | cov=0.7778 | ovl=0.6364
- ticket_18801: exp #18801 -> pred #17941 | class=near_duplicate_intent_wrong_ticket | sem=0.8847 | cov=0.7692 | ovl=0.7500
- ticket_18085: exp #18085 -> pred #19023 | class=near_duplicate_intent_wrong_ticket | sem=0.8479 | cov=0.8571 | ovl=0.6250
- ticket_17834: exp #17834 -> pred #18261 | class=near_duplicate_intent_wrong_ticket | sem=0.9280 | cov=0.7273 | ovl=0.6000
- ticket_19136: exp #19136 -> pred #18917 | class=near_duplicate_intent_wrong_ticket | sem=0.8509 | cov=0.8750 | ovl=0.6000
- ticket_18666: exp #18666 -> pred #18511 | class=near_duplicate_intent_wrong_ticket | sem=0.8476 | cov=0.9000 | ovl=0.7692
- ticket_18813: exp #18813 -> pred #18814 | class=near_duplicate_intent_wrong_ticket | sem=0.8654 | cov=0.8824 | ovl=0.5000
- ticket_19085: exp #19085 -> pred #18035 | class=near_duplicate_intent_wrong_ticket | sem=0.8860 | cov=0.9091 | ovl=0.8571
- ticket_18689: exp #18689 -> pred #17941 | class=near_duplicate_intent_wrong_ticket | sem=0.8910 | cov=0.9167 | ovl=0.4286
- ticket_18067: exp #18067 -> pred #17804 | class=near_duplicate_intent_wrong_ticket | sem=0.8820 | cov=0.9500 | ovl=0.7778
- ticket_17913: exp #17913 -> pred #18684 | class=near_duplicate_intent_wrong_ticket | sem=0.8606 | cov=1.0000 | ovl=0.5000
- ticket_18568: exp #18568 -> pred #19086 | class=near_duplicate_intent_wrong_ticket | sem=0.9043 | cov=0.9286 | ovl=0.7500
- ticket_18584: exp #18584 -> pred #17901 | class=near_duplicate_intent_wrong_ticket | sem=0.8713 | cov=1.0000 | ovl=0.6000
- ticket_19023: exp #19023 -> pred #(none) | class=no_source_returned | sem=0.4567 | cov=0.0833 | ovl=0.1429
- ticket_18336: exp #18336 -> pred #(none) | class=no_source_returned | sem=0.7906 | cov=0.5385 | ovl=0.6364
- ticket_18811: exp #18811 -> pred #18812 | class=off_target_answer | sem=0.6242 | cov=0.4444 | ovl=0.3333
- ticket_18579: exp #18579 -> pred #18581 | class=off_target_answer | sem=0.6940 | cov=0.5000 | ovl=0.2500
- ticket_17817: exp #17817 -> pred #18044 | class=partially_relevant_wrong_ticket | sem=0.7177 | cov=0.5000 | ovl=0.6667
- ticket_18503: exp #18503 -> pred #18504 | class=partially_relevant_wrong_ticket | sem=0.7650 | cov=0.5385 | ovl=0.6000
- ticket_17804: exp #17804 -> pred #18180 | class=partially_relevant_wrong_ticket | sem=0.7826 | cov=0.8000 | ovl=0.7500
