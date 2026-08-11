Noted on the bug.

For the next training run, I was thinking to limit to datasets with <=2 cameras, is there a CLI to filter by number of cameras, similar to --fps? I think this will help with unequal lengths in the same batch size. Can you do a quick count on the curated datasets to see how many episodes we'd lose? 

Before we make any other perf changes, let's compute the wandb eval table for just 12 examples instead of 36. However, I would like to add a new table eval/samples-all-fields (?) where we sample the same examples but conditioned on generating all the fields (to see what the model generates on eval samples), we don't care about MAE obvs for those, but we should still render the action charts.
