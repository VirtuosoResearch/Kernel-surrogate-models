import torch
from pathlib import Path
from tqdm import tqdm
from src.data import get_dataloader
from src.models import construct_rn9

ckpt_files = list(Path('./checkpoints').rglob('*.pt'))
ckpts = [torch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

# Replace with your choice of model constructor
model = construct_rn9().to(memory_format=torch.channels_last).cuda().eval()

# Replace with your choice of data loader (should be deterministic ordering)
loader_train = get_dataloader(batch_size=128, split='train')

from trak import TRAKer

traker = TRAKer(model=model,
                task='image_classification',
                train_set_size=len(loader_train.dataset))

for model_id, ckpt in enumerate(tqdm(ckpts)):
    # TRAKer loads the provided checkpoint and also associates
    # the provided (unique) model_id with the checkpoint.
    traker.load_checkpoint(ckpt, model_id=model_id)

    for batch in loader_train:
        batch = [x.cuda() for x in batch]
        # TRAKer computes features corresponding to the batch of examples,
        # using the checkpoint loaded above.
        traker.featurize(batch=batch, num_samples=batch[0].shape[0])

# Tells TRAKer that we've given it all the information, at which point
# TRAKer does some post-processing to get ready for the next step
# (scoring target examples).
traker.finalize_features()

loader_targets = get_dataloader(batch_size=128, split='val', augment=False)

for model_id, ckpt in enumerate(tqdm(ckpts)):
    traker.start_scoring_checkpoint(exp_name='quickstart',
                                    checkpoint=ckpt,
                                    model_id=model_id,
                                    num_targets=len(loader_targets.dataset))
    for batch in loader_targets:
        batch = [x.cuda() for x in batch]
        traker.score(batch=batch, num_samples=batch[0].shape[0])

scores = traker.finalize_scores(exp_name='quickstart')
print(scores.shape)