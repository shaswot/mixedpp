import sys
import os
import logging

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model

from utils import prepare_mnist, get_kernel_shape
from utils import quantizer_float_range_is_safe
from mixed_precision import SAFE_FLOAT_RANGE
from mp_mat import create_random_mpmat, mpmat_from_csv, mpmat_to_csv
from objective import replace_dense_with_mp_dense, accuracy_on_dataset


MODEL_FILE = "fc_net.h5"
MPMAT_FILE = "mpmat.csv"
LAYER_IDX = -1       # index of FC layer chosen to be in mixed-precision
                     # -1 means the last layer

def main():
    if os.path.exists(MODEL_FILE):
        orig_model = load_model(MODEL_FILE)
    else:
        raise RuntimeError(f"Model file {MODEL_FILE} does not exists. "
                           "Run `train.py` first!")

    # check if the predefined SAFE_FLOAT_RANGE is sufficient
    if not quantizer_float_range_is_safe(orig_model):
        logging.warning(f"{SAFE_FLOAT_RANGE=} does not cover the full range of "
                        f"weight value")

    if os.path.exists(MPMAT_FILE):
        mp_mat = mpmat_from_csv(MPMAT_FILE)
    else:
        print(f"Mixed-precision matrix file not found -> randomly create one")
        w_shape = get_kernel_shape(orig_model.get_layer(index=LAYER_IDX))
        mp_mat = create_random_mpmat(w_shape)
        mpmat_to_csv(mp_mat, MPMAT_FILE)

    new_model = replace_dense_with_mp_dense(mp_mat, orig_model, LAYER_IDX)

    _, test_set = prepare_mnist()
    orig_acc = accuracy_on_dataset(orig_model, test_set)
    new_acc = accuracy_on_dataset(new_model, test_set)

    logging.info(f"Original model acc: {orig_acc}")
    logging.info(f"Mixed-precision model acc: {new_acc}")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    main()
