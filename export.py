import os
import shutil
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python import ops
from tensorflow.tools.graph_transforms import TransformGraph

from deepsleep.data_loader import SeqDataLoader
from deepsleep.model import DeepSleepNet
from deepsleep.nn import *
from deepsleep.sleep_stage import (NUM_CLASSES,
                                   EPOCH_SEC_LEN,
                                   SAMPLING_RATE)
from deepsleep.utils import iterate_batch_seq_minibatches
from predict import CustomDeepSleepNet, CustomSeq2SeqNet, MultiChannelDeepFeatureNet
from prepare_physionet import class_dict

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('fold_idx', 0,
                            """Index of cross-validation fold to export.""")


def get_graph_def_from_file(graph_filepath):
    with ops.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def


def export(model_dir, fold_idx, output_dir, n_channels=1, n_features=0, n_rnn_layers=2):
    export_path = os.path.join(
        tf.compat.as_bytes(output_dir),
        tf.compat.as_bytes("fold{}".format(fold_idx)),
        tf.compat.as_bytes('saved model')
    )

    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        if n_features > 0:
            network = CustomSeq2SeqNet(
                batch_size=1,
                input_dims=n_features,
                n_classes=NUM_CLASSES,
                seq_length=10,
                n_rnn_layers=2,
                n_fc_layers=2,
                return_last=False,
                is_train=False,
                reuse_params=False,
                use_dropout=True,
            )
        elif n_rnn_layers > 0:
            network = CustomDeepSleepNet(
                batch_size=None,
                input_dims=EPOCH_SEC_LEN * 100,
                n_channels=n_channels,
                n_classes=NUM_CLASSES,
                seq_length=5,
                n_rnn_layers=2,
                return_last=False,
                is_train=False,
                reuse_params=False,
                use_dropout_feature=True,
                use_dropout_sequence=True
            )
        else:
            network = MultiChannelDeepFeatureNet(
                batch_size=None,
                input_dims=EPOCH_SEC_LEN * 100,
                n_channels=n_channels,
                n_classes=NUM_CLASSES,
                is_train=False,
                reuse_params=False,
                use_dropout=True
            )

        # Initialize parameters
        network.init_ops()

        checkpoint_path = os.path.join(
            model_dir,
            "fold{}".format(fold_idx),
            network.name
        )

        if os.path.exists(export_path):
            shutil.rmtree(export_path, ignore_errors=True)

        # Restore the trained model
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        print("Model restored from: {}\n".format(tf.train.latest_checkpoint(checkpoint_path)))

        # fw_state = sess.run(network.fw_initial_state)
        # bw_state = sess.run(network.bw_initial_state)

        print('Exporting trained model to', export_path)
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)

        # Build the signature_def_map.

        feed_dict = {'epochs': tf.compat.v1.saved_model.utils.build_tensor_info(network.input_var)}

        if n_rnn_layers > 0:
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

        if n_rnn_layers > 0:
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
        print("Exported to Saved Model format")

        output_file_name = network.name + ".pb"
        inputs = [n.name.split(sep=':')[0] for n in feed_dict.values()]
        outputs = [n.name.split(sep=':')[0] for n in output_dict.values()]

        graph_def = freeze_graph.freeze_graph(
            input_graph=None,
            input_saver=False,
            input_binary=False,
            input_checkpoint=None,
            output_node_names=",".join(outputs),
            restore_op_name=None,
            filename_tensor_name=None,
            # output_graph=os.path.join(tf.compat.as_bytes(output_dir), output_file_name.encode()),
            output_graph=None,
            clear_devices=True,
            initializer_nodes="",
            variable_names_whitelist="",
            variable_names_blacklist="",
            input_meta_graph=False,
            input_saved_model_dir=export_path,
            saved_model_tags=tf.compat.v1.saved_model.tag_constants.SERVING)

        print("Graph freezed!")

        transforms = [
            'remove_nodes(op=Identity)',
            'merge_duplicate_nodes',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
            'strip_unused_nodes',
            # 'quantize_nodes',
            # 'quantize_weights'
        ]

        graph_def = TransformGraph(
            graph_def,
            inputs,
            outputs,
            transforms)

        print('Graph optimized!')

        tf.train.write_graph(graph_def,
                             as_text=False,
                             logdir=output_dir,
                             name=output_file_name)

        print('Result saved to {}'.format(output_file_name))


def main(argv=None):
    # # Makes the random numbers predictable
    # np.random.seed(0)
    # tf.set_random_seed(0)

    # Output dir
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    export(
        model_dir=FLAGS.model_dir,
        fold_idx=FLAGS.fold_idx,
        output_dir=FLAGS.output_dir,
        n_channels=FLAGS.n_channels,
        n_features=FLAGS.n_features
    )


if __name__ == "__main__":
    tf.compat.v1.app.run()
