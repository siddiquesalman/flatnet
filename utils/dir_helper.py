"""
 Helper to create directories

@py37+
"""

import logging


def dir_init(args, is_local_rank_0: bool = True):
    """
    Creates paths for

    : save_filename
    : runs/[train,val]
    """
    if is_local_rank_0:
        logging.info("Initialising folders ...")
        ckpt_dir = args.ckpt_dir / args.exp_name
        tensorboard_dump = args.run_dir / args.exp_name

        for dir in [ckpt_dir, tensorboard_dump]:
            if not dir.is_dir():
                logging.info(f"Creating {dir.resolve()}")
                dir.mkdir(parents=True, exist_ok=True)
