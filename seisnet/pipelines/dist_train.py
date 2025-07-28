from seisnet.pipelines.training import random_train_model_cli, sparse_train_model_cli
import sys

if __name__=="__main__":
    try:
        # random_train_model_cli()
        sparse_train_model_cli()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)