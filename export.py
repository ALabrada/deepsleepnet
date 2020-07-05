import os
import tensorflow as tf

from deepsleep.data_loader import SeqDataLoader
from deepsleep.model import DeepSleepNet
from deepsleep.nn import *
from deepsleep.sleep_stage import (NUM_CLASSES,
                                   EPOCH_SEC_LEN,
                                   SAMPLING_RATE)
from deepsleep.utils import iterate_batch_seq_minibatches
from predict import  CustomDeepSleepNet
from prepare_physionet import class_dict

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('fold_idx', 0,
                            """Index of cross-validation fold to export.""")

def export(model_dir, fold_idx, output_dir):
    export_path = os.path.join(
        tf.compat.as_bytes(output_dir),
        tf.compat.as_bytes("fold{}".format(fold_idx)),
        tf.compat.as_bytes('saved model')
    )

    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        network = CustomDeepSleepNet(
            batch_size=1,
            input_dims=EPOCH_SEC_LEN * 100,
            n_classes=NUM_CLASSES,
            seq_length=25,
            n_rnn_layers=2,
            return_last=False,
            is_train=False,
            reuse_params=False,
            use_dropout_feature=True,
            use_dropout_sequence=True
        )

        # Initialize parameters
        network.init_ops()

        checkpoint_path = os.path.join(
            model_dir,
            "fold{}".format(fold_idx),
            "deepsleepnet"
        )

        # Restore the trained model
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        print("Model restored from: {}\n".format(tf.train.latest_checkpoint(checkpoint_path)))

        fw_state = sess.run(network.fw_initial_state)
        bw_state = sess.run(network.bw_initial_state)

        print('Exporting trained model to', export_path)
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)

        # Build the signature_def_map.

        feed_dict = {'epochs': tf.compat.v1.saved_model.utils.build_tensor_info(network.input_var)}

        for i, (c, h) in enumerate(network.fw_initial_state):
            feed_dict['fw_ltsm_{}_c'.format(i)] = tf.compat.v1.saved_model.utils.build_tensor_info(c)
            feed_dict['fw_ltsm_{}_h'.format(i)] = tf.compat.v1.saved_model.utils.build_tensor_info(h)

        for i, (c, h) in enumerate(network.bw_initial_state):
            feed_dict['bw_ltsm_{}_c'.format(i)] = tf.compat.v1.saved_model.utils.build_tensor_info(c)
            feed_dict['bw_ltsm_{}_h'.format(i)] = tf.compat.v1.saved_model.utils.build_tensor_info(h)

        output_dict = {
            'stages': tf.compat.v1.saved_model.utils.build_tensor_info(network.pred_op),
            'logits': tf.compat.v1.saved_model.utils.build_tensor_info(network.logits)
        }

        for i, (c, h) in enumerate(network.fw_final_state):
            output_dict['fw_ltsm_{}_c'.format(i)] = tf.compat.v1.saved_model.utils.build_tensor_info(c)
            output_dict['fw_ltsm_{}_h'.format(i)] = tf.compat.v1.saved_model.utils.build_tensor_info(h)

        for i, (c, h) in enumerate(network.bw_final_state):
            output_dict['bw_ltsm_{}_c'.format(i)] = tf.compat.v1.saved_model.utils.build_tensor_info(c)
            output_dict['bw_ltsm_{}_h'.format(i)] = tf.compat.v1.saved_model.utils.build_tensor_info(h)

        prediction_signature = (
            tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                inputs=feed_dict,
                outputs=output_dict,
                method_name=tf.compat.v1.saved_model.signature_constants
                    .PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_stages':
                    prediction_signature
            },
            strip_default_attrs=True)

        builder.save()


def main(argv=None):
    # # Makes the random numbers predictable
    # np.random.seed(0)
    # tf.set_random_seed(0)

    # Output dir
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    n_subjects = 20
    n_subjects_per_fold = 1
    export(
        model_dir=FLAGS.model_dir,
        fold_idx=FLAGS.fold_idx,
        output_dir=FLAGS.output_dir
    )


if __name__ == "__main__":
    tf.compat.v1.app.run()