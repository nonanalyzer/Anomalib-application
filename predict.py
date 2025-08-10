from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.post_processing import PostProcessor
import argparse
import os
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="name of correct label")
    parser.add_argument("--input", type=str, required=True, help="directory of test dataset")
    parser.add_argument("--output", type=str, default="results", help="output directory for results")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path")
    args = parser.parse_args()
    sname = args.name

    post_processor = PostProcessor(
        image_sensitivity=0.6,
        pixel_sensitivity=0.6,
    )
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer1", "layer2", "layer3"],
        num_neighbors=6,
        post_processor=post_processor,
        evaluator=False
    )
    engine = Engine()

    predictions = engine.predict(
        data_path=args.input,
        model=model,
        ckpt_path=args.ckpt if args.ckpt != None else os.path.join('weights', sname, 'model.ckpt')
    )
    
    src_dir = os.path.join("results", "Patchcore", "latest", "images")
    dst_dir = os.path.join(args.output, sname)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    shutil.rmtree(os.path.join("results", "Patchcore"))

    print('Finished.')
