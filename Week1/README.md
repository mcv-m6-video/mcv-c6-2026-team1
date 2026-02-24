EXPLAIN DATA:

Extract AICity_data.zip
copy ai_challenge_s03_c010-full_annotation.xml
For the moment, results_opticalflow_kitti.zip is not used => DIR

RUN CODE:
CODE MUST BE EXECUTED FROM WEEK1 FOLDER!

TASK 1:

- Slide 1: Explain formula and how we do the modeling + mean/variance of the modeling
- Slide 2: Show with/without preprocessing (a video / frame) + post-processing w/ mathematical morphology
  Show result with opening for removing noise (disconnected regions)
  Show result with closing for connecting regions
  Show result with dilation for expanding surviving regions
- Slide 3: Show shadow removal + NMS of BBoxes
  Min + max coordinates of connected components for the bounding box (of the FG mask)

- TODO: Improve noise reduction, try to increase the opening filter!

TASK 1.2:
- Slide 1: Explain how annotation is used (show outside/occlusion GTs) + parked cars (explain why) + all to 1 class (OBJECT)
- Slide 2: Best alpha based on mAP (quantitative results + plot?) -> explain a bit which alphas were tried AND WHICH IS THE BEST!
- Slide 3: Qualitative (baseline + best alpha) + some discussion + future work

TASK 2.1:
- Slide 1: Explain formula and how we do the modeling + mean/variance of the modeling (now mean + variance video) + hyperparams procedure
- Slide 2: Quantitative results (best alpha based on mAP) -> explain which parameters tried (if needed) AND WHICH IS THE BEST!
- Slide 3: Shadow adaptation: we not only adapt to grayscale, but also to the RGB mean so that shadow is adapted? (See comparison)

TASK 2.2:
- Slide 1: Quantitative: best hyperparams, compare mAP with precision/recall curve and explain theoretical differences + common limit
- Slide 2: Non-adaptive vs adaptive masks + video w/GTs (explain visual differences + similarity in single detection) => NO TO STATIC OBJS VIDEO
- Slide 3: Show video stopped and see how they differ. Also both single gaussian, better model changes we could use GMMs (see SOTA) + explain how?

TASK 3:
- Slide 1: Explain mog2 + lbsp + how it differs with our approaches. mAP + relative with ours (+-) and analyze why + algun frame qualitatiu
- Slide 2: Mas de lo mismo con rvm (oh...no cars due to training to people. Dir DL models depends on "training") + transcd
- Slide 3: Comparison w/ our best. "Original" video w/ GTs + "Best ours + mAP" (mask det) + "Best SOTA + mAP" (mask det). DISCUSSION AT THE BOTTOM!

## Week Structure

```bash
Week1/
├── README.md
├── data/
│   ├── AICity_data/
│   └── ai_challenge_s03_c010-full_annotation.xml
├── src/
│   ├── TransCD/
│   ├── color_utils.py
│   ├── ...
│   └── TODO
```
