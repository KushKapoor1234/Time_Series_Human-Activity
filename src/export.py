# src/export.py
import os, argparse, numpy as np, tensorflow as tf
from . import data as D
from .utils import set_seeds, ensure_dir, zscore_train_stats, zscore_apply, timeit_inference

def export_main(args=None):
    p=argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--art_dir', default='artifacts/export')
    p.add_argument('--seed', type=int, default=42)
    a=p.parse_args(args) if args is not None else p.parse_args()

    set_seeds(a.seed); ensure_dir(a.art_dir)
    Xtr, ytr, _, Xte, yte, _ = D.get_train_test()
    mu, sd = zscore_train_stats(Xtr); Xte = zscore_apply(Xte, mu, sd)

    model = tf.keras.models.load_model(a.model_path, compile=False)
    # Model size / params
    params = model.count_params()
    with open(os.path.join(a.art_dir,"model_profile.txt"), "w") as f:
        f.write(f"Params: {params}\n")

    # ONNX (optional)
    try:
        import tf2onnx
        spec = (tf.TensorSpec((None,)+model.input_shape[1:], tf.float32, name="input"),)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        onnx_path = os.path.join(a.art_dir, "model.onnx")
        with open(onnx_path, "wb") as f: f.write(onnx_model.SerializeToString())
    except Exception as e:
        with open(os.path.join(a.art_dir,"onnx_error.txt"), "w") as f: f.write(str(e))

    # TFLite + dynamic range quantization
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = conv.convert()
    with open(os.path.join(a.art_dir,"model_fp.tflite"), "wb") as f: f.write(tflite_model)

    conv_opt = tf.lite.TFLiteConverter.from_keras_model(model)
    conv_opt.optimizations=[tf.lite.Optimize.DEFAULT]
    tflite_q = conv_opt.convert()
    with open(os.path.join(a.art_dir,"model_int8.tflite"), "wb") as f: f.write(tflite_q)

    # Latency benchmark (keras)
    mean_ms, std_ms = timeit_inference(model, Xte, runs=30, batch_size=1)
    with open(os.path.join(a.art_dir,"inference_latency_ms.csv"), "w") as f:
        f.write("mean_ms,std_ms\n{:.3f},{:.3f}\n".format(mean_ms, std_ms))
    print("Export & benchmark artifacts saved.")

if __name__=='__main__':
    export_main()
