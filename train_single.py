from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.data.utils import (
    TestSplitMode,
    ValSplitMode,
)
from anomalib.post_processing import PostProcessor
import argparse
import shutil
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="amd", help="name of correct label")
    parser.add_argument("--input", type=str, required=True, help="name of training dataset")
    parser.add_argument("--error", type=str, default=None, help="name of error dataset")
    args = parser.parse_args()
    sname = args.name

    isWin = os.name == 'nt'

    # Create the datamodule
    datamodule = Folder(
        name=sname,
        root=".",
        normal_dir=args.input,
        abnormal_dir=args.error,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        num_workers=0 if isWin else 16,
        train_batch_size=16,
        eval_batch_size=16
    )

    # Setup the datamodule
    datamodule.setup()

    post_processor = PostProcessor(
        image_sensitivity=0.6,
        pixel_sensitivity=0.6,
    )
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer1", "layer2", "layer3"],
        num_neighbors=6,
        post_processor=post_processor,
        # evaluator=False,
        visualizer=False
    )
    engine = Engine()
    engine.train(datamodule=datamodule, model=model)

    src_ckpt = os.path.join("results", "Patchcore", sname, "v0", "weights", "lightning", "model.ckpt")
    dst_dir = os.path.join("weights", sname)
    dst_ckpt = os.path.join(dst_dir, "model.ckpt")
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src_ckpt, dst_ckpt)
    shutil.rmtree(os.path.join("results", "Patchcore"))
