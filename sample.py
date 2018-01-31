import tensorflow as tf
import model as m
from utils import *
from config import *

data_loader = DataLoader(args.batch_size, args.T, args.data_scale,
                         chars=args.chars, points_per_char=args.points_per_char)
args.U = len(args.str)
args.c_dimension = len(data_loader.chars) + 1
args.T = 1
args.batch_size = 1
args.action = 'sample'

# Save new checkpoint to choose the model epoch
newCheckpointContent = "model_checkpoint_path: "
newCheckpointContent += "\"" + os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            'save_%s' % args.mode,
                                            "model_synthesis.tfmodel-" + str(args.epoch_model)) + "\""
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'save_%s' % args.mode, "checkpoint"), 'w') as checkpointFile:
    checkpointFile.write(newCheckpointContent)

model = m.Model(args)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('save_%s' % args.mode)

with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    if args.mode == 'predict':
        strokes = model.sample(sess, 800)
    if args.mode == 'synthesis':
        str_vec = vectorization(args.str, data_loader.char_to_indices)
        strokes = model.sample(sess, len(args.str) * args.points_per_char, str=str_vec, verbose=False)
    # print strokes
    draw_strokes_random_color(strokes, factor=0.1, svg_filename=os.path.join("export", args.output_file_name))
